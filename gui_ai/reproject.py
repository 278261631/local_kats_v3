#!/usr/bin/env python3
"""WCS 重投影（B -> A 网格）与带 overlap 的瓦片切分。

重投影方式与 E:/github/misaligned_fits/reproject_wcs_and_export_stars.py 一致：
    输出网格 = A 的像素网格；
    每个输出像素 -> wcs_a 像素转天球 -> wcs_b 天球转像素 -> 在 B 上采样。
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Tuple

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import map_coordinates


def load_fits_data(path: str | Path) -> Tuple[np.ndarray, "fits.Header"]:
    """读取 FITS 主 HDU 的二维数据（float32）与头。"""
    with fits.open(str(path)) as hdul:
        header = hdul[0].header
        data = hdul[0].data
    if data is None:
        raise ValueError(f"FITS 无数据: {path}")
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"FITS 数据不是二维: {path} shape={arr.shape}")
    return arr, header


def build_celestial_wcs(header) -> WCS:
    return WCS(header).celestial


def _safe_source(arr: np.ndarray) -> np.ndarray:
    """把源图中非有限值替换为中位数，避免 map_coordinates 传播 nan。"""
    out = np.array(arr, dtype=np.float32, copy=True)
    finite = np.isfinite(out)
    if not finite.all():
        fill = float(np.median(out[finite])) if finite.any() else 0.0
        out[~finite] = fill
    return out


def reproject_b_to_a_wcs(
    a_shape: Tuple[int, int],
    b_data: np.ndarray,
    wcs_a: WCS,
    wcs_b: WCS,
    chunk_rows: int = 256,
) -> np.ndarray:
    """把 B 重投影到 A 的像素网格上。

    Returns:
        float32 (H, W) 数组，落在 B 覆盖范围外的像素为 NaN。
    """
    h, w = int(a_shape[0]), int(a_shape[1])
    src = _safe_source(b_data)
    out = np.full((h, w), np.nan, dtype=np.float32)
    xx = np.arange(w, dtype=float)

    step = max(1, int(chunk_rows))
    for y0 in range(0, h, step):
        y1 = min(h, y0 + step)
        yy = np.arange(y0, y1, dtype=float)
        gx, gy = np.meshgrid(xx, yy)
        lon, lat = wcs_a.pixel_to_world_values(gx.ravel(), gy.ravel())
        src_x, src_y = wcs_b.world_to_pixel_values(lon, lat)
        block = map_coordinates(
            src,
            [src_y.reshape(gx.shape), src_x.reshape(gx.shape)],
            order=1,
            mode="constant",
            cval=np.nan,
            prefilter=False,
        )
        out[y0:y1, :] = block.astype(np.float32)
    return out


def iter_tiles(
    height: int,
    width: int,
    tile_size: int = 256,
    overlap: float = 0.10,
) -> Iterator[Tuple[int, int]]:
    """生成瓦片左上角 (x0, y0)。

    overlap 为占瓦片宽度的比例（0.10 => 步长 = tile_size * 0.9）。
    始终覆盖到图像右下角；图像小于 tile 时只产生一个 (0, 0)。
    """
    tile = int(tile_size)
    step = max(1, int(round(tile * (1.0 - float(overlap)))))
    if width <= tile:
        xs = [0]
    else:
        xs = list(range(0, width - tile + 1, step))
        if xs[-1] != width - tile:
            xs.append(width - tile)
    if height <= tile:
        ys = [0]
    else:
        ys = list(range(0, height - tile + 1, step))
        if ys[-1] != height - tile:
            ys.append(height - tile)
    for y0 in ys:
        for x0 in xs:
            yield x0, y0


def crop_tile(arr: np.ndarray, x0: int, y0: int, tile_size: int) -> np.ndarray:
    """裁 tile_size×tile_size，越界用边缘填充（保证尺寸恒定）。"""
    tile = int(tile_size)
    h, w = arr.shape
    x0 = int(x0)
    y0 = int(y0)
    x1, y1 = x0 + tile, y0 + tile

    pad_l = max(0, -x0)
    pad_t = max(0, -y0)
    pad_r = max(0, x1 - w)
    pad_b = max(0, y1 - h)
    xs, ys = max(0, x0), max(0, y0)
    xe, ye = min(w, x1), min(h, y1)

    sub = arr[ys:ye, xs:xe]
    if sub.size == 0:
        return np.zeros((tile, tile), dtype=arr.dtype)
    if pad_l or pad_r or pad_t or pad_b:
        sub = np.pad(sub, ((pad_t, pad_b), (pad_l, pad_r)), mode="edge")
    if sub.shape != (tile, tile):
        sub = np.pad(
            sub,
            ((0, max(0, tile - sub.shape[0])), (0, max(0, tile - sub.shape[1]))),
            mode="edge",
        )
        sub = sub[:tile, :tile]
    return sub
