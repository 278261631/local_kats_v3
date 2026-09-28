#!/usr/bin/env python3
"""从 B（新报告）文件名解析望远镜/天区，并按原版逻辑定位参考帧 A。

原版逻辑（gui/filename_parser.py）：
    1. parse_filename(B) 得到 tel_name(GY1..GY6) 与 k_full(如 K001-1)；
    2. find_template_file(template_root, tel_name, k_full)：
       在 <template_root>/<tel 小写>/ 下找以 k_full 开头、且其后为分隔符/结束的 FITS。

本模块为自包含实现，不导入 gui/。
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, Optional

FITS_EXTS = (".fits", ".fit", ".fts")

#: GY5_K053-1_No Filter_..._UTC..._.fit  /  GY5_K053_...
_B_NAME_RE = re.compile(r"(GY[1-6])[_\-\s](K\d{3})-?(\d*)", re.IGNORECASE)


def is_fits(path: str | os.PathLike) -> bool:
    return str(path).lower().endswith(FITS_EXTS)


def parse_b_name(path: str | os.PathLike) -> Dict[str, str]:
    """解析 B 文件名 -> {tel_name, k_number, k_full}；失败返回 {}。"""
    base = os.path.basename(str(path))
    m = _B_NAME_RE.search(base)
    if not m:
        return {}
    tel = m.group(1).upper()
    k_number = m.group(2).upper()
    suffix = m.group(3)
    k_full = f"{k_number}-{suffix}" if suffix else k_number
    return {"tel_name": tel, "k_number": k_number, "k_full": k_full}


def _name_matches_region(name_without_ext: str, k_full: str) -> bool:
    """k_full 精确匹配文件名前缀，避免 K053-1 误配 K053-10。"""
    n = name_without_ext.lower()
    k = k_full.lower()
    if not n.startswith(k):
        return False
    return len(n) == len(k) or n[len(k)] in ("_", "-", ".", " ")


def find_template_file(
    template_root: str | os.PathLike,
    tel_name: str,
    k_full: str,
) -> Optional[str]:
    """在 <template_root>/<tel 小写>/ 下查找天区匹配的模板 FITS。"""
    root = Path(template_root)
    if not root.is_dir():
        return None
    tel_low = str(tel_name).lower()
    for item in sorted(os.listdir(root)):
        sub = root / item
        if not sub.is_dir() or item.lower() != tel_low:
            continue
        for filename in sorted(os.listdir(sub)):
            if not is_fits(filename):
                continue
            stem = os.path.splitext(filename)[0]
            if _name_matches_region(stem, k_full):
                return str(sub / filename)
        return None
    return None


def resolve_reference(
    b_path: str | os.PathLike,
    template_root: str | os.PathLike,
) -> Optional[str]:
    """由 B 文件路径解析并返回参考帧 A 的路径；无法定位时返回 None。"""
    info = parse_b_name(b_path)
    if not info:
        return None
    return find_template_file(template_root, info["tel_name"], info["k_full"])
