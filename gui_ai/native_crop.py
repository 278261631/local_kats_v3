#!/usr/bin/env python3
"""按需从原始 FITS 读取原生分辨率裁块（避免预览降采样导致的星点模糊）。

预览用的 a_u8/b_raw 是降采样到 preview_max_side 的，用它放大做裁切会糊。
这里直接：
    * A：从 A 文件 memmap 切片；
    * B：把该窗口重投影到 A 网格（只读取 B 需要的子块，内存/IO 都很小）。
"""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import map_coordinates


class FitsCache:
    """按路径缓存已打开的 FITS（memmap），限制缓存数量。"""

    def __init__(self, max_items: int = 2) -> None:
        self._cache: "OrderedDict[str, tuple]" = OrderedDict()
        self.max_items = max(1, int(max_items))

    def get(self, path: str):
        key = str(path)
        item = self._cache.get(key)
        if item is not None:
            self._cache.move_to_end(key)
            return item

        hdul = None
        data = None
        try:
            hdul = fits.open(key, memmap=True)
            data = hdul[0].data  # 可能在此时抛错（BZERO/BSCALE/BLANK 不能 memmap）
        except Exception:
            if hdul is not None:
                try:
                    hdul.close()
                except Exception:
                    pass
            hdul = fits.open(key, memmap=False)
            data = hdul[0].data
        hdu = hdul[0]
        if data is not None and data.ndim == 3:
            data = data[0]
        wcs = WCS(hdu.header).celestial
        self._cache[key] = (hdul, data, wcs)
        while len(self._cache) > self.max_items:
            _k, (h, _d, _w) = self._cache.popitem(last=False)
            try:
                h.close()
            except Exception:
                pass
        return self._cache[key]

    def close(self) -> None:
        for (_h, _d, _w) in self._cache.values():
            try:
                _h.close()
            except Exception:
                pass
        self._cache.clear()


def _crop_window(arr, x0: int, y0: int, size: int) -> np.ndarray:
    """从 (H,W) 数组裁 size×size（越界边缘填充）；返回 float32。"""
    h, w = arr.shape
    x1, y1 = x0 + size, y0 + size
    pad_l, pad_t = max(0, -x0), max(0, -y0)
    pad_r, pad_b = max(0, x1 - w), max(0, y1 - h)
    xs, ys = max(0, x0), max(0, y0)
    xe, ye = min(w, x1), min(h, y1)
    if xe <= xs or ye <= ys:
        return np.zeros((size, size), dtype=np.float32)
    sub = np.asarray(arr[ys:ye, xs:xe], dtype=np.float32)
    if pad_l or pad_r or pad_t or pad_b:
        sub = np.pad(sub, ((pad_t, pad_b), (pad_l, pad_r)), mode="edge")
    if sub.shape != (size, size):
        out = np.zeros((size, size), dtype=np.float32)
        ih, iw = min(size, sub.shape[0]), min(size, sub.shape[1])
        out[:ih, :iw] = sub[:ih, :iw]
        sub = out
    return sub


def load_native_pair_crops(
    cache: FitsCache,
    a_path: str,
    b_path: str,
    centers: List[Tuple[float, float]],
    size: int,
    fill_invalid_with_a: bool = True,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """返回 [(a_crop(float32), b_crop(float32)), ...]，均为 size×size 原生裁块。"""
    size = int(size)
    half = size // 2
    out: List[Tuple[np.ndarray, np.ndarray]] = []
    if not centers:
        return out

    _ah, a_data, wcs_a = cache.get(a_path)
    _bh, b_data, wcs_b = cache.get(b_path)
    H, W = a_data.shape
    Bh, Bw = b_data.shape
    margin = max(8, size // 8)
    off_x = np.arange(size, dtype=float)
    off_y = np.arange(size, dtype=float)

    for (cx, cy) in centers:
        x0 = int(round(cx)) - half
        y0 = int(round(cy)) - half
        a_crop = _crop_window(a_data, x0, y0, size)

        # 窗口内每个 A 像素 -> B 像素
        xx, yy = np.meshgrid(x0 + off_x, y0 + off_y)
        lon, lat = wcs_a.pixel_to_world_values(xx, yy)
        sx, sy = wcs_b.world_to_pixel_values(lon, lat)
        finite = np.isfinite(sx) & np.isfinite(sy)

        b_crop = np.full((size, size), np.nan, dtype=np.float32)
        if finite.any():
            fxs, fys = sx[finite], sy[finite]
            bx0 = max(0, int(math.floor(fxs.min())) - margin)
            bx1 = min(Bw, int(math.ceil(fxs.max())) + margin + 1)
            by0 = max(0, int(math.floor(fys.min())) - margin)
            by1 = min(Bh, int(math.ceil(fys.max())) + margin + 1)
            if bx1 > bx0 and by1 > by0:
                b_sub = np.ascontiguousarray(
                    b_data[by0:by1, bx0:bx1], dtype=np.float32
                )
                sxc = np.where(finite, sx - bx0, -1.0e9)
                syc = np.where(finite, sy - by0, -1.0e9)
                b_crop = map_coordinates(
                    b_sub, [syc, sxc], order=1, mode="constant",
                    cval=np.nan, prefilter=False,
                ).astype(np.float32)

        # A 范围之外的像素置 nan
        valid = np.zeros((size, size), dtype=bool)
        ax0, ay0 = max(0, -x0), max(0, -y0)
        ax1, ay1 = min(size, W - x0), min(size, H - y0)
        if ax1 > ax0 and ay1 > ay0:
            valid[ay0:ay1, ax0:ax1] = True
        b_crop = np.where(valid, b_crop, np.nan)

        if fill_invalid_with_a:
            b_crop = np.where(np.isfinite(b_crop), b_crop, a_crop)
        out.append((a_crop, b_crop))
    return out
