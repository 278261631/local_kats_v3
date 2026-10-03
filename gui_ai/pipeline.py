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
from valid_region import (
    background_stats,
    compute_valid_mask,
    extract_b2_polygons,
    extract_valid_polygons,
    filled_region,
    inner_band,
    shading_mask,
)

LogCb = Optional[Callable[[str], None]]

#: A 模板有效区多边形缓存（同一模板在处理多个 B 时复用）
_VALID_POLY_CACHE: Dict[str, list] = {}
#: A 模板有效区掩码缓存（供 A∩B 重叠过滤）
_A_VALID_CACHE: Dict[str, np.ndarray] = {}
#: A 模板数据/WCS 缓存（同模板多 B 时免重复读盘）
_A_DATA_CACHE: Dict[str, tuple] = {}
#: A 模板"像素->天球"分块缓存（供重投影复用）
_A_WORLD_CACHE: Dict[str, dict] = {}


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


def _downscale_mask_any(mask: np.ndarray, max_side: int, thresh: float = 0.02) -> np.ndarray:
    """降采样布尔掩码为预览尺度 uint8(0/255)，保留细带（任一覆盖即可）。"""
    import cv2

    h, w = mask.shape
    m = np.asarray(mask, dtype=np.float32)
    s = min(1.0, float(max_side) / float(max(h, w)))
    if s >= 1.0:
        return ((m > 0.5).astype(np.uint8)) * 255
    nw = max(1, int(round(w * s)))
    nh = max(1, int(round(h * s)))
    small = cv2.resize(m, (nw, nh), interpolation=cv2.INTER_AREA)
    return ((small > float(thresh)).astype(np.uint8)) * 255


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


def _blob_size(img: np.ndarray, xi: int, yi: int, win: int, thresh: float) -> int:
    """以 (xi,yi) 为中心窗口内，阈值化后包含中心像素的连通块大小。"""
    import cv2

    half = max(1, int(win) // 2)
    h, w = img.shape
    x0 = max(0, xi - half)
    y0 = max(0, yi - half)
    x1 = min(w, xi + half + 1)
    y1 = min(h, yi + half + 1)
    sub = np.asarray(img[y0:y1, x0:x1], dtype=np.float32)
    m = (sub > float(thresh)).astype(np.uint8)
    cy, cx = yi - y0, xi - x0
    if cy < 0 or cx < 0 or cy >= m.shape[0] or cx >= m.shape[1] or m[cy, cx] == 0:
        return 0
    n, lab = cv2.connectedComponents(m, 8)
    return int((lab == lab[cy, cx]).sum())


def _det_metrics(img: np.ndarray, xi: int, yi: int, r: int, R: int,
                 search: int = 5):
    """先在小窗口内重新定位局部峰值，再返回 (SNR, conc, sigma)。

    - 重新定位可消除检测点 (x,y) 的 1~2px 偏差，避免 conc 被误压低；
    - conc = 峰值净通量 / 孔径净通量；sigma 由孔径内二阶矩得到(FWHM=2.355σ)。
    """
    h, w = img.shape
    half = max(1, int(R) + int(search))
    x0 = max(0, xi - half)
    y0 = max(0, yi - half)
    x1 = min(w, xi + half + 1)
    y1 = min(h, yi + half + 1)
    sub = np.asarray(img[y0:y1, x0:x1], dtype=np.float32)
    if sub.size == 0:
        return 0.0, 0.0, 0.0
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d2 = (xx - xi) ** 2 + (yy - yi) ** 2
    ann = (d2 > float(r) ** 2) & (d2 <= float(R) ** 2)
    bv = sub[ann]
    bv = bv[np.isfinite(bv)]
    if bv.size < 5:
        return 0.0, 0.0, 0.0
    bg = float(np.median(bv))

    # 在 search 半径内重新定位峰值
    m = (d2 <= float(search) ** 2) & np.isfinite(sub)
    if not m.any():
        return 0.0, 0.0, 0.0
    py, px = np.unravel_index(int(np.argmax(np.where(m, sub, -np.inf))), sub.shape)
    gpx = float(xx[py, px])
    gpy = float(yy[py, px])
    peak = float(sub[py, px]) - bg

    d2p = (xx - gpx) ** 2 + (yy - gpy) ** 2
    app = d2p <= float(r) ** 2
    ap = sub[app]
    ap = ap[np.isfinite(ap)]
    if ap.size == 0:
        return 0.0, 0.0, 0.0
    flux = float(np.sum(ap - bg))
    conc = peak / flux if flux > 1e-6 else 0.0

    wgt = np.clip(sub[app] - bg, 0.0, None)
    if wgt.sum() > 0:
        axx = xx[app].astype(np.float64)
        ayy = yy[app].astype(np.float64)
        cxm = float((wgt * axx).sum() / wgt.sum())
        cym = float((wgt * ayy).sum() / wgt.sum())
        var = float((wgt * ((axx - cxm) ** 2 + (ayy - cym) ** 2)).sum() / wgt.sum())
        sigma = var ** 0.5 if var > 0 else 0.0
    else:
        sigma = 0.0

    # SNR：以峰值点为中心的外环估噪声
    ann2 = (d2p > float(r) ** 2) & (d2p <= float(R) ** 2)
    bv2 = sub[ann2]
    bv2 = bv2[np.isfinite(bv2)]
    if bv2.size >= 5:
        bg2 = float(np.median(bv2))
        sig2 = 1.4826 * float(np.median(np.abs(bv2 - bg2)))
        if sig2 > 0:
            flux2 = float(np.sum(ap - bg2))
            snr = flux2 / (sig2 * (ap.size ** 0.5))
        else:
            snr = 0.0
    else:
        snr = 0.0
    return float(snr), float(conc), float(sigma)


def _aperture_snr(img: np.ndarray, xi: int, yi: int, r: int, R: int,
                  search: int = 5) -> float:
    return _det_metrics(img, xi, yi, r, R, search)[0]


def _shape_metrics(img: np.ndarray, xi: int, yi: int, r: int, R: int,
                   search: int = 5):
    _, conc, sig = _det_metrics(img, xi, yi, r, R, search)
    return conc, sig


def _block_mean_nan(arr: np.ndarray, factor: int) -> np.ndarray:
    """按 factor×factor 块求均值（忽略 NaN），用于降分辨率做边界检测。"""
    f = int(factor)
    a = np.asarray(arr, dtype=np.float32)
    if f <= 1:
        return a
    h, w = a.shape
    h2, w2 = (h // f) * f, (w // f) * f
    if h2 == 0 or w2 == 0:
        return a
    blk = a[:h2, :w2].reshape(h2 // f, f, w2 // f, f)
    fin = np.isfinite(blk)
    s = np.where(fin, blk, 0.0).sum(axis=(1, 3))
    c = fin.sum(axis=(1, 3))
    return np.where(c > 0, s / np.maximum(c, 1), np.nan).astype(np.float32)


def _scale_polys(polys: list, factor: float) -> list:
    return [(np.asarray(p, dtype=np.float32) * float(factor)) for p in polys]


def _upsample_mask(mask_small: np.ndarray, factor: int, shape) -> np.ndarray:
    f = int(factor)
    m = np.repeat(np.repeat(np.asarray(mask_small).astype(np.uint8), f, 0), f, 1)
    out = np.zeros(shape, dtype=np.uint8)
    hh = min(shape[0], m.shape[0])
    ww = min(shape[1], m.shape[1])
    out[:hh, :ww] = m[:hh, :ww]
    return out.astype(bool)


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
    snr_filter: bool = True,
    shading_k: float = 3.0,
    b2_ksize: int = 21,
    noise_k: float = 3.0,
    aperture_radius: int = 3,
    aperture_annulus: int = 6,
    aperture_snr_min: float = 4.0,
    det_center_search: int = 5,
    shape_conc_max: float = 0.5,
    shape_fwhm_min: float = 0.8,
    shape_fwhm_max: float = 0.0,
    edge_band: int = 5,
    boundary_scale: int = 4,
    isolated_filter: bool = True,
    isolated_win: int = 7,
    isolated_k: float = 3.0,
    isolated_min_px: int = 2,
    median_filter: bool = False,
    median_ksize: int = 3,
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
        "edge_prev": None,
        "final_prev": None,
        "a_valid_polys": None,
        "b_valid_polys": None,
        "b2_valid_polys": None,
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

    # A 模板：数据/WCS 缓存（同一模板处理多个 B 时复用）
    cached_a = _A_DATA_CACHE.get(a_path)
    if cached_a is None:
        a_data, a_header = load_fits_data(a_path)
        wcs_a = build_celestial_wcs(a_header)
        if len(_A_DATA_CACHE) >= 2:
            _A_DATA_CACHE.pop(next(iter(_A_DATA_CACHE)))
        _A_DATA_CACHE[a_path] = (a_data, wcs_a)
    else:
        a_data, wcs_a = cached_a
    b_data, b_header = load_fits_data(b_path)
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
    if a_path not in _A_WORLD_CACHE:
        _A_WORLD_CACHE.clear()
        _A_WORLD_CACHE[a_path] = {}
    b_rep = reproject_b_to_a_wcs(
        a_data.shape, b_data, wcs_a, wcs_b, chunk_rows=reproject_chunk_rows,
        world_cache=_A_WORLD_CACHE[a_path],
    )
    valid = np.isfinite(b_rep)
    if fill_invalid_with_a:
        b_filled = np.where(valid, b_rep, a_data)
    else:
        b_filled = b_rep

    # 边界检测在降分辨率上进行（边界平滑，省 10x+）
    bs = max(1, int(boundary_scale))
    b_small = _block_mean_nan(b_rep, bs) if bs > 1 else b_rep
    valid_small = np.isfinite(b_small)

    # B(重投影后) 有效区(星空/空白)边界多边形
    try:
        polys = extract_valid_polygons(b_small)
        result["b_valid_polys"] = _scale_polys(polys, bs) if bs > 1 else polys
        if result["b_valid_polys"]:
            log(f"B有效区边界: {len(result['b_valid_polys'])} 个多边形")
    except Exception as ex:  # noqa: BLE001
        log(f"B有效区提取失败: {ex}")

    # B 二次有效区：剔除内层暗边/渐晕后的边界 + 全局背景统计/暗边掩码
    b2_polys = []
    shaded = None
    g_bg, g_sig = 0.0, 1.0
    try:
        ks = max(3, int(b2_ksize) // bs)
        b2_small = extract_b2_polygons(b_small, valid_small, k=shading_k, ksize=ks)
        result["b2_valid_polys"] = _scale_polys(b2_small, bs) if bs > 1 else b2_small
        shaded_small = shading_mask(b_small, valid_small, k=shading_k, ksize=ks)
        shaded = _upsample_mask(shaded_small, bs, b_rep.shape) if bs > 1 else shaded_small
        g_bg, g_sig = background_stats(b_rep, valid)
        if result["b2_valid_polys"]:
            log(f"B二级有效区边界: {len(result['b2_valid_polys'])} 个多边形, 暗边占比 "
                f"{100.0*float(shaded.mean()):.1f}%")
    except Exception as ex:  # noqa: BLE001
        log(f"B二级有效区提取失败: {ex}")

    # 边界内边带：命中落在 A/B/B二级 有效区边界 width 像素内也过滤
    band_a = band_b = band_b2 = None
    if edge_band and int(edge_band) > 0:
        try:
            band_b = inner_band(filled_region(valid), int(edge_band))
            if shaded is not None:
                band_b2 = inner_band(
                    filled_region(valid & ~shaded), int(edge_band))
            if a_valid is not None:
                band_a = inner_band(filled_region(a_valid), int(edge_band))
            band_any = band_b
            for b in (band_b2, band_a):
                if b is not None:
                    band_any = b if band_any is None else (band_any | b)
            if band_any is not None:
                result["edge_prev"] = _downscale_mask_any(
                    band_any, preview_max_side)
            # 最终计算边框：A∩B 有效区 再去掉内边带后的核心区
            usable = filled_region(valid)
            if valid_overlap_filter and a_valid is not None:
                usable = usable & filled_region(a_valid)
            core = usable & ~inner_band(usable, int(edge_band))
            result["final_prev"] = _downscale_mask(core, preview_max_side)
        except Exception as ex:  # noqa: BLE001
            log(f"边带计算失败: {ex}")

    a_u8 = to_uint8(a_data)
    b_u8 = to_uint8(b_filled)
    if median_filter and int(median_ksize) >= 3:
        import cv2
        b_u8 = cv2.medianBlur(b_u8, int(median_ksize) | 1)
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
            # 瓦片局部统计（供低信噪过滤）
            t_bg, t_sig = g_bg, g_sig
            tile_noisy = False
            if snr_filter:
                traw = crop_tile(b_filled, x0, y0, tile_size)
                tv = traw[np.isfinite(traw)]
                if tv.size:
                    t_bg = float(np.median(tv))
                    t_sig = 1.4826 * float(np.median(np.abs(tv - t_bg)))
                    if t_sig <= 0:
                        t_sig = g_sig
                    tile_noisy = t_sig > noise_k * g_sig
            for pk in r["peaks"]:
                fx = x0 + pk["x"]
                fy = y0 + pk["y"]
                xi = min(w - 1, max(0, int(round(fx))))
                yi = min(h - 1, max(0, int(round(fy))))
                status = pk.get("status", "keep")
                asnr, conc, sig = _det_metrics(
                    b_filled, xi, yi, aperture_radius, aperture_annulus,
                    det_center_search)
                if status == "keep":
                    if not valid[yi, xi]:
                        status = "b_uncovered"
                    elif (valid_overlap_filter and a_valid is not None
                          and not a_valid[yi, xi]):
                        status = "a_invalid"
                    elif ((band_b is not None and band_b[yi, xi])
                          or (valid_overlap_filter and band_a is not None and band_a[yi, xi])
                          or (band_b2 is not None and band_b2[yi, xi])):
                        status = "edge"
                    elif snr_filter:
                        fwhm = 2.355 * sig
                        if ((shaded is not None and shaded[yi, xi]) or tile_noisy
                                or asnr < aperture_snr_min):
                            status = "low_snr"
                        elif (conc > shape_conc_max
                              or (shape_fwhm_min > 0 and fwhm < shape_fwhm_min)
                              or (shape_fwhm_max > 0 and fwhm > shape_fwhm_max)):
                            status = "spike"
                    if status == "keep" and isolated_filter:
                        blobs = _blob_size(
                            b_filled, xi, yi, isolated_win,
                            t_bg + isolated_k * t_sig)
                        if blobs < int(isolated_min_px):
                            status = "isolated"
                dets.append(
                    {
                        "x": float(fx),
                        "y": float(fy),
                        "score": float(pk["score"]),
                        "cls": int(pk["cls"]),
                        "status": status,
                        "snr": float(asnr),
                        "conc": float(conc),
                        "fwhm": float(2.355 * sig),
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
