#!/usr/bin/env python3
"""从单张 A 模板直接分析"星空有效区 / 空白区"边界。

不做多矩形精确重建（模板只是单一结果，没有各曝光 footprint 信息）。
用局部方差区分"有噪声的真实星空"与"恒定的空白填充"：

    blank  = 局部方差 ≈ 0（或非有限值）
    valid  = ~blank
边界是多张曝光覆盖并集形成的多边形，用轮廓 + Douglas–Peucker 简化为直线段。
"""

from __future__ import annotations

from typing import List

import numpy as np
import cv2


def compute_valid_mask(
    data: np.ndarray,
    ksize: int = 9,
    var_eps: float = 1e-6,
    clean: int = 7,
) -> np.ndarray:
    """返回 uint8 (0/1) 有效区掩码。"""
    x = np.asarray(data, dtype=np.float32)
    finite = np.isfinite(x)
    xf = np.where(finite, x, 0.0).astype(np.float32)
    k = int(ksize) | 1
    m = cv2.boxFilter(xf, -1, (k, k))
    m2 = cv2.boxFilter(xf * xf, -1, (k, k))
    var = np.clip(m2 - m * m, 0.0, None)
    valid = (finite & (var > var_eps)).astype(np.uint8)
    # 兜底：若方差法几乎全空/全满，退回"非有限/非零"规则
    frac = float(valid.mean())
    if frac < 0.05 or frac > 0.999:
        valid = (finite & (x != 0)).astype(np.uint8)
    if clean > 1:
        kk = np.ones((int(clean), int(clean)), np.uint8)
        valid = cv2.morphologyEx(valid, cv2.MORPH_CLOSE, kk)
        valid = cv2.morphologyEx(valid, cv2.MORPH_OPEN, kk)
    return valid


def extract_valid_polygons(
    data: np.ndarray,
    ksize: int = 9,
    var_eps: float = 1e-6,
    approx_frac: float = 0.002,
    min_area_frac: float = 1e-4,
    clean: int = 7,
) -> List[np.ndarray]:
    """返回有效区边界多边形列表，每个为 (N,2) float32 的 (x, y) 像素坐标。"""
    valid = compute_valid_mask(data, ksize=ksize, var_eps=var_eps, clean=clean)
    H, W = valid.shape
    contours, _ = cv2.findContours(valid, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    total = float(H * W)
    polys: List[np.ndarray] = []
    for c in contours:
        if cv2.contourArea(c) < min_area_frac * total:
            continue
        peri = cv2.arcLength(c, True)
        ap = cv2.approxPolyDP(c, approx_frac * peri, True)
        if len(ap) >= 3:
            polys.append(ap.reshape(-1, 2).astype(np.float32))
    polys.sort(key=lambda p: -abs(cv2.contourArea(p.reshape(-1, 1, 2).astype(np.int32))))
    return polys
