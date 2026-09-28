#!/usr/bin/env python3
"""处理流水线：定位 A 模板 -> 重投影 B -> 瓦片推理 -> 汇总检测。

单文件：直接处理该 B。
文件夹：递归收集其下所有 FITS 作为 B，逐一处理。
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from pair_infer import PairModel, to_uint8
from reproject import (
    build_celestial_wcs,
    crop_tile,
    iter_tiles,
    load_fits_data,
    reproject_b_to_a_wcs,
)
from template_resolver import is_fits, resolve_reference

LogCb = Optional[Callable[[str], None]]


def collect_fits_files(path: str | os.PathLike) -> List[str]:
    """文件 -> [文件]；文件夹 -> 递归下所有 FITS；其它 -> []。"""
    p = Path(path)
    if p.is_file():
        return [str(p)] if is_fits(p) else []
    if p.is_dir():
        return [str(f) for f in sorted(p.rglob("*")) if f.is_file() and is_fits(f)]
    return []


def _downscale_u8(arr_u8: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    import cv2

    h, w = arr_u8.shape
    s = min(1.0, float(max_side) / float(max(h, w)))
    if s >= 1.0:
        return arr_u8, 1.0
    out = cv2.resize(
        arr_u8, (max(1, int(round(w * s))), max(1, int(round(h * s)))),
        interpolation=cv2.INTER_AREA,
    )
    return out, s


def _downscale_f32(arr: np.ndarray, max_side: int) -> np.ndarray:
    """把原始浮点帧降采样为预览用数组（供局部/全图自适应拉伸）。

    NaN（B 未覆盖）用掩码保留，避免 INTER_AREA 把 NaN 扩散到邻域。
    """
    import cv2

    h, w = arr.shape
    s = min(1.0, float(max_side) / float(max(h, w)))
    src = np.asarray(arr, dtype=np.float32)
    if s >= 1.0:
        return src.copy()

    nw = max(1, int(round(w * s)))
    nh = max(1, int(round(h * s)))
    finite = np.isfinite(src)
    filled = np.where(finite, src, 0.0).astype(np.float32)
    data_small = cv2.resize(filled, (nw, nh), interpolation=cv2.INTER_AREA)
    mask_small = cv2.resize(
        finite.astype(np.float32), (nw, nh), interpolation=cv2.INTER_AREA
    )
    out = data_small.astype(np.float32)
    out[mask_small < 0.5] = np.nan
    return out


def _dedup(dets: List[Dict], radius: float) -> List[Dict]:
    """按分数降序贪心去重（跨瓦片同一目标只保留最高分）。"""
    if not dets:
        return []
    r2 = float(radius) ** 2
    kept: List[Dict] = []
    for d in sorted(dets, key=lambda x: -x["score"]):
        if any((d["x"] - k["x"]) ** 2 + (d["y"] - k["y"]) ** 2 <= r2 for k in kept):
            continue
        kept.append(d)
    return kept


def process_b_file(
    b_path: str,
    model: PairModel,
    template_root: str,
    tile_size: int,
    overlap: float,
    dedup_radius: float,
    reproject_chunk_rows: int,
    fill_invalid_with_a: bool = True,
    preview_max_side: int = 1600,
    log_cb: LogCb = None,
) -> Dict:
    """处理单个 B 文件，返回结果字典（含预览缩略图与检测列表）。"""

    def log(msg: str) -> None:
        if log_cb:
            log_cb(msg)

    t0 = time.perf_counter()
    result: Dict = {
        "b_path": b_path,
        "a_path": None,
        "width": 0,
        "height": 0,
        "detections": [],
        "n_tiles": 0,
        "mean_dx": 0.0,
        "mean_dy": 0.0,
        "mean_roll": 0.0,
        "a_u8": None,
        "b_raw": None,
        "preview_scale": 1.0,
        "elapsed": 0.0,
        "error": None,
    }

    a_path = resolve_reference(b_path, template_root)
    if not a_path:
        result["error"] = "未找到参考模板 A"
        result["elapsed"] = time.perf_counter() - t0
        log(f"[跳过] 未定位到 A: {os.path.basename(b_path)}")
        return result
    result["a_path"] = a_path
    log(f"参考 A: {a_path}")

    a_data, a_header = load_fits_data(a_path)
    b_data, b_header = load_fits_data(b_path)
    wcs_a = build_celestial_wcs(a_header)
    wcs_b = build_celestial_wcs(b_header)

    log(f"重投影 B -> A 网格 ({a_data.shape[1]}x{a_data.shape[0]}) ...")
    b_rep = reproject_b_to_a_wcs(
        a_data.shape, b_data, wcs_a, wcs_b, chunk_rows=reproject_chunk_rows
    )
    valid = np.isfinite(b_rep)
    if fill_invalid_with_a:
        b_filled = np.where(valid, b_rep, a_data)
    else:
        b_filled = b_rep

    a_u8 = to_uint8(a_data)
    b_u8 = to_uint8(b_filled)
    h, w = a_u8.shape
    result["width"], result["height"] = w, h

    tiles = list(iter_tiles(h, w, tile_size=tile_size, overlap=overlap))
    result["n_tiles"] = len(tiles)
    log(f"瓦片数: {len(tiles)} (tile={tile_size}, overlap={overlap:.0%})")

    batch_a: List[np.ndarray] = []
    batch_b: List[np.ndarray] = []
    origins: List[tuple] = []
    dets: List[Dict] = []
    poses: List[tuple] = []

    def flush() -> None:
        if not batch_a:
            return
        out = model.infer_tiles(batch_a, batch_b, tile_size)
        for (x0, y0), r in zip(origins, out):
            poses.append((r["dx"], r["dy"], r["roll"]))
            for pk in r["peaks"]:
                fx = x0 + pk["x"]
                fy = y0 + pk["y"]
                xi = min(w - 1, max(0, int(round(fx))))
                yi = min(h - 1, max(0, int(round(fy))))
                if not valid[yi, xi]:
                    continue  # B 未覆盖区域，忽略
                dets.append(
                    {
                        "x": float(fx),
                        "y": float(fy),
                        "score": float(pk["score"]),
                        "cls": int(pk["cls"]),
                        "dx": float(r["dx"]),
                        "dy": float(r["dy"]),
                        "roll": float(r["roll"]),
                        "tile_x": int(x0),
                        "tile_y": int(y0),
                    }
                )
        batch_a.clear()
        batch_b.clear()
        origins.clear()

    for (x0, y0) in tiles:
        batch_a.append(crop_tile(a_u8, x0, y0, tile_size))
        batch_b.append(crop_tile(b_u8, x0, y0, tile_size))
        origins.append((x0, y0))
        if len(batch_a) >= model.batch_size:
            flush()
    flush()

    dets = _dedup(dets, dedup_radius)
    dets.sort(key=lambda d: -d["score"])
    result["detections"] = dets
    if poses:
        arr = np.asarray(poses, dtype=np.float64)
        result["mean_dx"] = float(np.mean(np.abs(arr[:, 0])))
        result["mean_dy"] = float(np.mean(np.abs(arr[:, 1])))
        result["mean_roll"] = float(np.mean(np.abs(arr[:, 2])))

    a_prev, s = _downscale_u8(a_u8, preview_max_side)
    result["a_u8"] = a_prev
    result["b_raw"] = _downscale_f32(b_filled, preview_max_side)
    result["preview_scale"] = s
    result["elapsed"] = time.perf_counter() - t0
    log(
        f"完成 {os.path.basename(b_path)}: 检测 {len(dets)} 个, "
        f"|pose| dx={result['mean_dx']:.2f} dy={result['mean_dy']:.2f} "
        f"roll={result['mean_roll']:.2f}°, 用时 {result['elapsed']:.1f}s"
    )
    return result
