#!/usr/bin/env python3
"""处理结果持久化：每个 B 写 `<B>.gui_ai.json`（检测/参数/多边形）
+ `<B>.gui_ai.npz`（预览数组，供恢复预览与叠加）。

文件跟随数据（写在 B 同目录），可按日期目录汇总成总表。
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

SUFFIX_JSON = ".gui_ai.json"
SUFFIX_NPZ = ".gui_ai.npz"

#: 需要持久化的预览数组（降采样尺度）
_PREVIEW_KEYS = ("a_u8", "b_raw", "cov_prev", "sat_prev", "edge_prev", "final_prev")

_DATE_RE = re.compile(r"^\d{8}$")
_TEL_RE = re.compile(r"^[Gg][Yy][1-6]$")
_REGION_RE = re.compile(r"^K\d{3}$")


def result_json_path(b_path: str) -> str:
    return str(b_path) + SUFFIX_JSON


def result_npz_path(b_path: str) -> str:
    return str(b_path) + SUFFIX_NPZ


def save_result(res: Dict, params: Dict | None = None) -> None:
    """写出结果 json + 预览 npz（失败不抛）。"""
    b = res.get("b_path")
    if not b:
        return
    try:
        meta = {
            "b_path": res.get("b_path"),
            "a_path": res.get("a_path"),
            "width": res.get("width"),
            "height": res.get("height"),
            "n_keep": res.get("n_keep"),
            "n_total": res.get("n_total"),
            "mean_dx": res.get("mean_dx"),
            "mean_dy": res.get("mean_dy"),
            "mean_roll": res.get("mean_roll"),
            "elapsed": res.get("elapsed"),
            "detections": res.get("detections", []),
            "a_valid_polys": [np.asarray(p).tolist() for p in (res.get("a_valid_polys") or [])],
            "b_valid_polys": [np.asarray(p).tolist() for p in (res.get("b_valid_polys") or [])],
            "b2_valid_polys": [np.asarray(p).tolist() for p in (res.get("b2_valid_polys") or [])],
            "preview_scale": res.get("preview_scale"),
            "fill_invalid_with_a": res.get("fill_invalid_with_a", True),
            "params": params or {},
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        }
        Path(result_json_path(b)).write_text(
            json.dumps(meta, ensure_ascii=False), encoding="utf-8"
        )
        arrays = {k: res[k] for k in _PREVIEW_KEYS if res.get(k) is not None}
        if arrays:
            np.savez_compressed(result_npz_path(b), **arrays)
    except Exception:
        pass


def load_result(json_path: str) -> Dict:
    """由 json（+同名 npz）重建结果字典。"""
    d = json.loads(Path(json_path).read_text(encoding="utf-8"))
    b = d.get("b_path")
    res: Dict = {
        "b_path": b,
        "a_path": d.get("a_path"),
        "width": d.get("width"),
        "height": d.get("height"),
        "n_keep": d.get("n_keep"),
        "n_total": d.get("n_total"),
        "mean_dx": d.get("mean_dx"),
        "mean_dy": d.get("mean_dy"),
        "mean_roll": d.get("mean_roll"),
        "elapsed": d.get("elapsed"),
        "detections": d.get("detections", []),
        "a_valid_polys": [np.asarray(p, dtype=np.float32) for p in d.get("a_valid_polys", [])],
        "b_valid_polys": [np.asarray(p, dtype=np.float32) for p in d.get("b_valid_polys", [])],
        "b2_valid_polys": [np.asarray(p, dtype=np.float32) for p in d.get("b2_valid_polys", [])],
        "preview_scale": d.get("preview_scale", 1.0),
        "fill_invalid_with_a": d.get("fill_invalid_with_a", True),
        "a_u8": None,
        "b_raw": None,
        "cov_prev": None,
        "sat_prev": None,
        "edge_prev": None,
        "final_prev": None,
        "error": None,
        "_loaded": True,
        "_saved_at": d.get("saved_at"),
    }
    if b:
        nz = result_npz_path(b)
        if os.path.exists(nz):
            try:
                with np.load(nz) as z:
                    for k in _PREVIEW_KEYS:
                        if k in z:
                            res[k] = z[k]
            except Exception:
                pass
    return res


def scan_results(path: str | os.PathLike) -> List[str]:
    """返回 path（文件或目录）下所有结果 json 路径。"""
    p = Path(path)
    if p.is_file():
        j = str(p) + SUFFIX_JSON
        return [j] if os.path.exists(j) else []
    if p.is_dir():
        return [str(f) for f in sorted(p.rglob("*" + SUFFIX_JSON))]
    return []


def parse_path_info(json_path: str) -> Dict[str, str]:
    """从路径推断 日期/系统/天区。"""
    tel = date = region = ""
    for s in Path(json_path).parts:
        if _DATE_RE.match(s):
            date = s
        elif _TEL_RE.match(s):
            tel = s.upper()
        elif _REGION_RE.match(s):
            region = s
    return {"tel": tel, "date": date, "region": region}
