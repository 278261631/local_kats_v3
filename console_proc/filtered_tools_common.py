#!/usr/bin/env python3
"""
console_proc 筛选工具公共函数。
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


CSV_NAME = "variable_candidates_nonref_only_inner_border.csv"


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(out.get(k), dict) and isinstance(v, dict):
            out[k] = merge_dict(out[k], v)
        else:
            out[k] = v
    return out


def default_filter_profiles() -> Dict[str, Dict[str, Any]]:
    return {
        "A": {
            "flux_min": 30.0,
            "flux_max": 800.0,
            "variable_count_mode": "=-1",
            "mpc_count_mode": "=-1",
            "ai_class_mode": "=0",
            "skip_mode": "=0",
            "skip_large_csv": False,
            "large_csv_max_rows": 200,
        },
        "B": {
            "flux_min": 30.0,
            "flux_max": 800.0,
            "variable_count_mode": "=0",
            "mpc_count_mode": "=0",
            "ai_class_mode": "=1",
            "skip_mode": "=0",
            "skip_large_csv": False,
            "large_csv_max_rows": 200,
        },
    }


def load_filter_profile(cfg: Dict[str, Any], profile_name: str) -> Dict[str, Any]:
    merged_profiles = merge_dict(default_filter_profiles(), cfg.get("filter_profiles", {}))
    key = str(profile_name).strip().upper() or "A"
    if key not in merged_profiles:
        raise ValueError(f"未知筛选配置: {profile_name}，可选: {', '.join(sorted(merged_profiles.keys()))}")

    p = dict(merged_profiles[key])
    p["name"] = key
    return p


def validate_filter_profile(profile: Dict[str, Any]) -> None:
    flux_min = float(profile.get("flux_min", 0.0))
    flux_max = float(profile.get("flux_max", 0.0))
    if not (flux_min < flux_max):
        raise ValueError(f"筛选配置 {profile.get('name', '?')} 的 flux 区间非法: {flux_min} !< {flux_max}")
    if str(profile.get("variable_count_mode", "")) not in {"=0", "=-1", ">0"}:
        raise ValueError("variable_count_mode 仅支持 =0 / =-1 / >0")
    if str(profile.get("mpc_count_mode", "")) not in {"=0", "=-1", ">0"}:
        raise ValueError("mpc_count_mode 仅支持 =0 / =-1 / >0")
    if str(profile.get("ai_class_mode", "")).lower() not in {"=0", "=1", "<0", "all"}:
        raise ValueError("ai_class_mode 仅支持 =0 / =1 / <0 / all")
    if str(profile.get("skip_mode", "")).lower() not in {"=0", "=1", "all"}:
        raise ValueError("skip_mode 仅支持 =0 / =1 / all")
    if int(profile.get("large_csv_max_rows", 200)) <= 0:
        raise ValueError("large_csv_max_rows 必须 > 0")


def discover_candidate_csvs(
    diff_output_root: Path,
    date_text: str,
    telescope: Optional[str] = None,
    region: Optional[str] = None,
) -> List[Path]:
    if not diff_output_root.exists():
        return []

    tel_filter = str(telescope).strip().upper() if telescope else None
    region_filter = normalize_region(region) if region else None
    matched: List[Path] = []
    for csv_path in diff_output_root.rglob(CSV_NAME):
        try:
            rel = csv_path.relative_to(diff_output_root)
        except Exception:
            continue
        parts = rel.parts
        # 期望: <tel>/<date>/<region>/<output_dir>/variable_candidates_nonref_only_inner_border.csv
        if len(parts) < 5:
            continue
        tel = str(parts[0]).upper()
        date_part = str(parts[1])
        reg = str(parts[2]).upper()
        if date_part != date_text:
            continue
        if tel_filter and tel != tel_filter:
            continue
        if region_filter and reg != region_filter:
            continue
        matched.append(csv_path)
    return sorted(matched)


def normalize_region(region: str) -> str:
    r = str(region).strip().upper()
    if not r.startswith("K"):
        r = f"K{r}"
    if re.fullmatch(r"K\d{1,3}", r):
        r = f"K{int(r[1:]):03d}"
    if not re.fullmatch(r"K\d{3}", r):
        raise ValueError(f"非法天区格式: {region}")
    return r


def load_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def write_csv_rows(csv_path: Path, rows: List[Dict[str, str]]) -> None:
    if rows:
        fieldnames = list(rows[0].keys())
    else:
        fieldnames = []
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def ensure_csv_default_fields(csv_path: Path) -> Dict[str, int]:
    """为 CSV 中缺失的 variable_count/mpc_count/ai_class/skip_flag 设置默认值并回写。

    默认值:
        variable_count → "-1"  (尚未做变星匹配)
        mpc_count      → "-1"  (尚未做 MPC 匹配)
        ai_class       → "0"   (尚未做 AI 分类)
        skip_flag      → "0"   (未被跳过)

    Returns:
        {"total_rows": 总行数, "changed_rows": 被修改行数}
    """
    rows = load_csv_rows(csv_path)
    if not rows:
        return {"total_rows": 0, "changed_rows": 0}

    changed_rows = 0
    for row in rows:
        modified = False
        if is_missing_csv_value(row.get("variable_count")):
            row["variable_count"] = "-1"
            modified = True
        if is_missing_csv_value(row.get("mpc_count")):
            row["mpc_count"] = "-1"
            modified = True
        if is_missing_csv_value(row.get("ai_class")):
            row["ai_class"] = "0"
            modified = True
        if is_missing_csv_value(row.get("skip_flag")):
            row["skip_flag"] = "0"
            modified = True
        if modified:
            changed_rows += 1

    if changed_rows > 0:
        write_csv_rows(csv_path, rows)

    return {"total_rows": len(rows), "changed_rows": changed_rows}


def try_parse_int_from_csv_value(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(float(str(value).strip()))
    except Exception:
        return None


def is_missing_csv_value(value: Any) -> bool:
    if value is None:
        return True
    return str(value).strip() == ""


def try_get_float_from_row(row: Dict[str, Any], key_candidates: List[str]) -> Optional[float]:
    for key in key_candidates:
        if key in row:
            v = str(row.get(key, "")).strip()
            if not v:
                continue
            m = re.search(r"[-+]?\d+(?:\.\d+)?", v)
            if not m:
                continue
            try:
                return float(m.group(0))
            except Exception:
                continue
    return None


def csv_count_mode_match(value: Any, mode: str) -> bool:
    iv = try_parse_int_from_csv_value(value)
    # 兼容历史CSV：计数字段缺失时按 -1 处理。
    if iv is None and is_missing_csv_value(value):
        iv = -1
    if iv is None:
        return False
    if mode == "=0":
        return iv == 0
    if mode == "=-1":
        return iv == -1
    if mode == ">0":
        return iv > 0
    return False


def csv_ai_class_mode_match(value: Any, mode: str) -> bool:
    m = str(mode).strip().lower()
    if m == "all":
        return True
    iv = try_parse_int_from_csv_value(value)
    if iv is None:
        iv = 0
    if m == "=0":
        return iv == 0
    if m == "=1":
        return iv == 1
    if m == "<0":
        return iv < 0
    return False


def csv_skip_mode_match(value: Any, mode: str) -> bool:
    m = str(mode).strip().lower()
    if m == "all":
        return True
    iv = try_parse_int_from_csv_value(value)
    if iv is None:
        iv = 0
    if m == "=0":
        return iv == 0
    if m == "=1":
        return iv == 1
    return False


def row_matches_filter(row: Dict[str, Any], profile: Dict[str, Any]) -> bool:
    flux_min = float(profile.get("flux_min", 0.0))
    flux_max = float(profile.get("flux_max", 0.0))
    var_mode = str(profile.get("variable_count_mode", "=0"))
    mpc_mode = str(profile.get("mpc_count_mode", "=0"))
    ai_mode = str(profile.get("ai_class_mode", "=0"))
    skip_mode = str(profile.get("skip_mode", "=0"))

    flux = try_get_float_from_row(row, ["median_flux_norm"])
    if flux is None or not (flux_min < flux < flux_max):
        return False
    if not csv_count_mode_match(row.get("variable_count"), var_mode):
        return False
    if not csv_count_mode_match(row.get("mpc_count"), mpc_mode):
        return False
    if not csv_ai_class_mode_match(row.get("ai_class"), ai_mode):
        return False
    if not csv_skip_mode_match(row.get("skip_flag", row.get("skip")), skip_mode):
        return False
    return True


def find_primary_aligned_fits_in_output_dir(output_dir: Path) -> Optional[Path]:
    if not output_dir.exists():
        return None
    rp_files = sorted(output_dir.glob("*.02rp.fit"))
    if rp_files:
        return rp_files[0]
    fallback = sorted(output_dir.glob("*.fit"))
    if fallback:
        return fallback[0]
    return None
