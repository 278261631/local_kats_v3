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


def filled_region(mask: np.ndarray, clean: int = 5, largest: bool = True) -> np.ndarray:
    """把区域清理成单一、无内部空洞的整体（去掉星洞/碎块）。"""
    m = np.asarray(mask).astype(np.uint8)
    if clean > 1:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((int(clean), int(clean)), np.uint8))
    if largest and m.any():
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        if n > 1:
            idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            m = (lab == idx).astype(np.uint8)
    if m.any():
        h, w = m.shape
        ff = m.copy()
        ff_mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(ff, ff_mask, (0, 0), 1)
        m = m | (ff == 0).astype(np.uint8)
    return m.astype(bool)


def inner_band(mask: np.ndarray, width: int) -> np.ndarray:
    """掩码的"内边带"（距边界 width 像素内的区域，bool）。"""
    m = np.asarray(mask).astype(np.uint8)
    w = int(width)
    if w <= 0 or m.size == 0:
        return np.zeros(m.shape, dtype=bool)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * w + 1, 2 * w + 1))
    er = cv2.erode(m, k)
    return (m.astype(bool) & ~er.astype(bool))


def background_stats(data: np.ndarray, valid: np.ndarray):
    """有效区的背景中值 bg 与 MAD 噪声 sigma。"""
    x = np.asarray(data, dtype=np.float32)
    m = np.asarray(valid).astype(bool) & np.isfinite(x)
    v = x[m]
    if v.size == 0:
        return 0.0, 1.0
    bg = float(np.median(v))
    sig = 1.4826 * float(np.median(np.abs(v - bg)))
    if sig <= 0:
        sig = 1.0
    return bg, sig


def shading_mask(data: np.ndarray, valid: np.ndarray,
                 k: float = 3.0, ksize: int = 21) -> np.ndarray:
    """有效区内"细暗边/黑框"掩码（bool）。

    用**黑帽变换** (closing - image) 提取比结构元更细的暗特征，
    可压掉大尺度渐晕/噪声包络，只留那条细黑边。
    """
    x = np.asarray(data, dtype=np.float32)
    finite = np.isfinite(x)
    bg, sig = background_stats(x, valid)
    xf = np.where(finite, x, bg).astype(np.float32)
    s = int(ksize) | 1
    kk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (s, s))
    bh = cv2.morphologyEx(xf, cv2.MORPH_BLACKHAT, kk)
    return np.asarray(valid).astype(bool) & (bh > float(k) * sig)


def extract_b2_polygons(
    data: np.ndarray,
    valid: np.ndarray,
    k: float = 3.0,
    ksize: int = 21,
    approx_frac: float = 0.002,
    min_area_frac: float = 1e-4,
    clean: int = 7,
) -> List[np.ndarray]:
    """二次有效区边界：在第一次有效区内剔除细黑边(黑帽)后取外轮廓。"""
    shaded = shading_mask(data, valid, k=k, ksize=ksize)
    # 把细黑边扩成带，避免边界锯齿
    if shaded.any():
        shaded = cv2.dilate(shaded.astype(np.uint8),
                            np.ones((5, 5), np.uint8)).astype(bool)
    v2 = (np.asarray(valid).astype(bool) & ~shaded).astype(np.uint8)
    if v2.sum() == 0:
        return []
    # 保留最大连通域（去掉黑边打散的小块）
    n, lab, stats, _ = cv2.connectedComponentsWithStats(v2, 8)
    if n > 1:
        idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        v2 = (lab == idx).astype(np.uint8)
    if clean > 1:
        kk = np.ones((int(clean), int(clean)), np.uint8)
        v2 = cv2.morphologyEx(v2, cv2.MORPH_CLOSE, kk)
        v2 = cv2.morphologyEx(v2, cv2.MORPH_OPEN, kk)
    contours, _ = cv2.findContours(v2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total = float(v2.shape[0] * v2.shape[1])
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
