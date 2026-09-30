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
from valid_region import compute_valid_mask, extract_valid_polygons

LogCb = Optional[Callable[[str], None]]

#: A 模板有效区多边形缓存（同一模板在处理多个 B 时复用）
_VALID_POLY_CACHE: Dict[str, list] = {}
#: A 模板有效区掩码缓存（供 A∩B 重叠过滤）
_A_VALID_CACHE: Dict[str, np.ndarray] = {}


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


def _downscale_mask(mask: np.ndarray, max_side: int) -> np.ndarray:
    """把布尔覆盖掩码降采样为预览尺度 uint8(0/255)。"""
    import cv2

    h, w = mask.shape
    s = min(1.0, float(max_side) / float(max(h, w)))
    m = np.asarray(mask, dtype=np.float32)
    if s >= 1.0:
        return ((m > 0.5).astype(np.uint8)) * 255
    nw = max(1, int(round(w * s)))
    nh = max(1, int(round(h * s)))
    small = cv2.resize(m, (nw, nh), interpolation=cv2.INTER_AREA)
    return ((small >= 0.5).astype(np.uint8)) * 255


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
    valid_overlap_filter: bool = True,
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
        "n_keep": 0,
        "n_total": 0,
        "mean_dx": 0.0,
        "mean_dy": 0.0,
        "mean_roll": 0.0,
        "a_u8": None,
        "b_raw": None,
        "sat_prev": None,
        "cov_prev": None,
        "a_valid_polys": None,
        "b_valid_polys": None,
        "preview_scale": 1.0,
        "fill_invalid_with_a": bool(fill_invalid_with_a),
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

    # A 模板有效区(星空/空白)边界多边形（多项式坐标，A 网格）
    a_valid = None
    try:
        cached = _VALID_POLY_CACHE.get(a_path)
        if cached is None:
            cached = extract_valid_polygons(a_data)
            _VALID_POLY_CACHE[a_path] = cached
        result["a_valid_polys"] = cached
        if cached:
            log(f"A有效区边界: {len(cached)} 个多边形, 最大顶点数 "
                f"{max(len(p) for p in cached)}")
        # A 有效区掩码（供 A∩B 重叠过滤）
        cached_mask = _A_VALID_CACHE.get(a_path)
        if cached_mask is None:
            cached_mask = compute_valid_mask(a_data).astype(bool)
            _A_VALID_CACHE[a_path] = cached_mask
        a_valid = cached_mask
    except Exception as ex:  # noqa: BLE001
        log(f"A有效区提取失败: {ex}")

    log(f"重投影 B -> A 网格 ({a_data.shape[1]}x{a_data.shape[0]}) ...")
    b_rep = reproject_b_to_a_wcs(
        a_data.shape, b_data, wcs_a, wcs_b, chunk_rows=reproject_chunk_rows
    )
    valid = np.isfinite(b_rep)
    if fill_invalid_with_a:
        b_filled = np.where(valid, b_rep, a_data)
    else:
        b_filled = b_rep

    # B(重投影后) 有效区(星空/空白)边界多边形
    try:
        result["b_valid_polys"] = extract_valid_polygons(b_rep)
        if result["b_valid_polys"]:
            log(f"B有效区边界: {len(result['b_valid_polys'])} 个多边形, 最大顶点数 "
                f"{max(len(p) for p in result['b_valid_polys'])}")
    except Exception as ex:  # noqa: BLE001
        log(f"B有效区提取失败: {ex}")

    a_u8 = to_uint8(a_data)
    b_u8 = to_uint8(b_filled)
    h, w = a_u8.shape
    result["width"], result["height"] = w, h

    tiles = list(iter_tiles(h, w, tile_size=tile_size, overlap=overlap))
    result["n_tiles"] = len(tiles)
    log(f"瓦片数: {len(tiles)} (tile={tile_size}, overlap={overlap:.0%})")

    # 卫星掩码在预览尺度上累加，供叠加显示
    prev_scale = min(1.0, float(preview_max_side) / float(max(h, w)))
    ph = max(1, int(round(h * prev_scale)))
    pw = max(1, int(round(w * prev_scale)))
    sat_prev = None
    cov_prev = _downscale_mask(valid, preview_max_side)

    batch_a: List[np.ndarray] = []
    batch_b: List[np.ndarray] = []
    origins: List[tuple] = []
    dets: List[Dict] = []
    poses: List[tuple] = []

    def _paste(mask2d: np.ndarray, target: np.ndarray, channel: int,
               x0: int, y0: int) -> None:
        """把瓦片尺寸掩码缩到预览尺度并 max 合并进 target。"""
        import cv2

        x0p = int(round(x0 * prev_scale))
        y0p = int(round(y0 * prev_scale))
        x1p = min(pw, max(x0p + 1, int(round((x0 + tile_size) * prev_scale))))
        y1p = min(ph, max(y0p + 1, int(round((y0 + tile_size) * prev_scale))))
        mc = cv2.resize(
            mask2d.astype(np.float32), (x1p - x0p, y1p - y0p),
            interpolation=cv2.INTER_AREA,
        )
        mcu = np.clip(mc, 0, 255).astype(np.uint8)
        if target.ndim == 2:
            np.maximum(target[y0p:y1p, x0p:x1p], mcu,
                       out=target[y0p:y1p, x0p:x1p])
        else:
            np.maximum(target[y0p:y1p, x0p:x1p, channel], mcu,
                       out=target[y0p:y1p, x0p:x1p, channel])

    def flush() -> None:
        nonlocal sat_prev
        if not batch_a:
            return
        out = model.infer_tiles(batch_a, batch_b, tile_size)
        for i, ((x0, y0), r) in enumerate(zip(origins, out)):
            poses.append((r["dx"], r["dy"], r["roll"]))
            # 卫星掩码（PairRegNet ob 通道 2）
            m = r.get("masks")
            if m is not None and m.shape[0] >= 3:
                if sat_prev is None:
                    sat_prev = np.zeros((ph, pw), dtype=np.uint8)
                _paste((np.asarray(m[2]) * 255.0).astype(np.uint8), sat_prev, 0, x0, y0)
            for pk in r["peaks"]:
                fx = x0 + pk["x"]
                fy = y0 + pk["y"]
                xi = min(w - 1, max(0, int(round(fx))))
                yi = min(h - 1, max(0, int(round(fy))))
                status = pk.get("status", "keep")
                if status == "keep":
                    if not valid[yi, xi]:
                        status = "b_uncovered"
                    elif (valid_overlap_filter and a_valid is not None
                          and not a_valid[yi, xi]):
                        status = "a_invalid"
                dets.append(
                    {
                        "x": float(fx),
                        "y": float(fy),
                        "score": float(pk["score"]),
                        "cls": int(pk["cls"]),
                        "status": status,
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

    # 跨瓦片去重：把重复标记为 dedup（保留在结果里，便于排查）
    r2 = float(dedup_radius) ** 2
    dets.sort(key=lambda d: -d["score"])
    kept: List[Dict] = []
    for d in dets:
        if d["status"] == "keep":
            if any((d["x"] - k["x"]) ** 2 + (d["y"] - k["y"]) ** 2 <= r2 for k in kept):
                d["status"] = "dedup"
            else:
                kept.append(d)
    dets.sort(key=lambda d: -d["score"])
    result["detections"] = dets
    result["n_total"] = len(dets)
    result["n_keep"] = sum(1 for d in dets if d["status"] == "keep")
    if poses:
        arr = np.asarray(poses, dtype=np.float64)
        result["mean_dx"] = float(np.mean(np.abs(arr[:, 0])))
        result["mean_dy"] = float(np.mean(np.abs(arr[:, 1])))
        result["mean_roll"] = float(np.mean(np.abs(arr[:, 2])))

    a_prev, s = _downscale_u8(a_u8, preview_max_side)
    result["a_u8"] = a_prev
    result["b_raw"] = _downscale_f32(b_filled, preview_max_side)
    result["sat_prev"] = sat_prev
    result["cov_prev"] = cov_prev
    result["preview_scale"] = s

    result["elapsed"] = time.perf_counter() - t0
    log(
        f"完成 {os.path.basename(b_path)}: 命中 {result['n_keep']}/{result['n_total']} "
        f"(原始峰值), |pose| dx={result['mean_dx']:.2f} dy={result['mean_dy']:.2f} "
        f"roll={result['mean_roll']:.2f}°, 用时 {result['elapsed']:.1f}s"
    )
    return result
