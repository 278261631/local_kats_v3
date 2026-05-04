#!/usr/bin/env python3
"""
console_proc 独立网页 ZIP 导出器（按筛选命中导出局部图、index.html、meta.json）。
"""

from __future__ import annotations

import argparse
from datetime import datetime
import html
import json
import logging
from pathlib import Path
import re
import shutil
from typing import Any, Dict, List, Optional, Tuple
import zipfile

import numpy as np
from astropy.io import fits
from PIL import Image

from filtered_tools_common import (
    discover_candidate_csvs,
    find_primary_aligned_fits_in_output_dir,
    load_csv_rows,
    load_filter_profile,
    load_json,
    row_matches_filter,
    try_get_float_from_row,
    try_parse_int_from_csv_value,
    validate_filter_profile,
)


Hit = Tuple[Path, Path, int, Dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按 console_proc 筛选配置导出网页 ZIP")
    parser.add_argument("--config", default="console_proc/config.json", help="配置文件路径")
    parser.add_argument("--date", required=True, help="日期，YYYYMMDD")
    parser.add_argument("--telescope", help="仅处理指定系统，例如 GY1")
    parser.add_argument("--region", help="仅处理指定天区，例如 K019")
    parser.add_argument("--profile", default="zip", help="筛选配置名，默认 zip")
    parser.add_argument("--zip-root", help="ZIP 输出根目录；默认使用配置 paths.zip_output_directory")
    parser.add_argument("--patch-size", type=int, default=512, help="局部图尺寸像素，默认 512")
    parser.add_argument("--hist-level", choices=["low", "medium", "high"], default="high", help="局部图拉伸强度")
    parser.add_argument("--max-csv", type=int, default=0, help="最多扫描 CSV 数量，0=不限制")
    parser.add_argument("--verbose", action="store_true", help="输出调试日志")
    return parser.parse_args()


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")


def validate_date(date_text: str) -> bool:
    return bool(re.fullmatch(r"\d{8}", date_text))


def sanitize_output_name(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r'[<>:"/\\|?*\s]+', "_", text)
    text = re.sub(r"_+", "_", text).strip("._")
    return text or "item"


def get_skip_flag(row: Dict[str, Any]) -> int:
    raw_value = row.get("skip_flag")
    if raw_value is None and "skip" in row:
        raw_value = row.get("skip")
    iv = try_parse_int_from_csv_value(raw_value)
    return 1 if iv == 1 else 0


def condition_summary(profile: Dict[str, Any]) -> str:
    flux_min = float(profile.get("flux_min", 0.0))
    flux_max = float(profile.get("flux_max", 0.0))
    return (
        f"{flux_min:g}<median_flux_norm<{flux_max:g} AND "
        f"variable_count{profile.get('variable_count_mode', '=0')} AND "
        f"mpc_count{profile.get('mpc_count_mode', '=0')} AND "
        f"ai_class{profile.get('ai_class_mode', '=0')} AND "
        f"skip_flag{profile.get('skip_mode', '=0')}"
    )


def load_fits_image_and_header(fits_path: Path) -> Tuple[Optional[np.ndarray], Any]:
    try:
        with fits.open(str(fits_path)) as hdul:
            header = hdul[0].header
            data = hdul[0].data
        if data is None:
            return None, None
        arr = data.astype(np.float64)
        if len(arr.shape) == 3:
            arr = arr[0]
        return arr, header
    except Exception as e:
        logging.warning("读取 FITS 失败: %s (%s)", fits_path, e)
        return None, None


def resolve_candidate_pixel_xy(row: Dict[str, Any], header: Any) -> Tuple[Optional[float], Optional[float]]:
    x = try_get_float_from_row(row, ["x", "pixel_x", "x_px", "xpix", "target_x", "cx", "col", "img_x"])
    y = try_get_float_from_row(row, ["y", "pixel_y", "y_px", "ypix", "target_y", "cy", "row", "img_y"])
    if x is not None and y is not None:
        return x, y

    ra = try_get_float_from_row(row, ["ra", "ra_deg", "ra_degree", "target_ra"])
    dec = try_get_float_from_row(row, ["dec", "dec_deg", "dec_degree", "target_dec"])
    if ra is None or dec is None or header is None:
        return None, None
    try:
        from astropy.wcs import WCS

        pixel_coords = WCS(header).all_world2pix([[ra, dec]], 0)
        return float(pixel_coords[0][0]), float(pixel_coords[0][1])
    except Exception:
        return None, None


def extract_local_patch(data: np.ndarray, x: float, y: float, half_size: int) -> Optional[np.ndarray]:
    if data is None or x is None or y is None:
        return None
    h, w = data.shape[0], data.shape[1]
    xi = int(round(x))
    yi = int(round(y))
    x0 = max(0, xi - half_size)
    x1 = min(w, xi + half_size)
    y0 = max(0, yi - half_size)
    y1 = min(h, yi + half_size)
    patch = data[y0:y1, x0:x1]
    if patch.size == 0:
        return None
    return patch


def stretch_patch_to_uint8(patch: np.ndarray, level: str) -> np.ndarray:
    if patch is None:
        return np.zeros((1, 1), dtype=np.uint8)
    arr = np.asarray(patch, dtype=np.float64)
    if arr.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)

    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros(arr.shape, dtype=np.uint8)

    valid = arr[finite]
    params = {
        "low": (5.0, 99.5, 1.0, 3.0),
        "medium": (2.0, 99.8, 0.9, 5.0),
        "high": (0.5, 99.95, 0.75, 8.0),
    }
    p_low, p_high, gamma, asinh_scale = params.get(str(level).lower(), params["high"])
    lo = float(np.percentile(valid, p_low))
    hi = float(np.percentile(valid, p_high))
    if not np.isfinite(lo):
        lo = float(np.min(valid))
    if not np.isfinite(hi):
        hi = float(np.max(valid))
    if hi <= lo:
        hi = lo + 1e-8

    norm = (np.clip(arr, lo, hi) - lo) / (hi - lo)
    norm = np.clip(norm, 0.0, 1.0)
    if gamma != 1.0:
        norm = np.power(norm, gamma)
    if asinh_scale > 0:
        norm = np.arcsinh(norm * asinh_scale) / np.arcsinh(asinh_scale)
    out = np.zeros_like(arr, dtype=np.float64)
    out[finite] = np.clip(norm[finite], 0.0, 1.0)
    return np.round(out * 255.0).astype(np.uint8)


def format_ra_dec_hms_dms_text(ra_deg: Optional[float], dec_deg: Optional[float]) -> str:
    if ra_deg is None or dec_deg is None:
        return ""
    try:
        ra_norm = float(ra_deg) % 360.0
        ra_total_seconds = ra_norm / 15.0 * 3600.0
        ra_h = int(ra_total_seconds // 3600)
        ra_m = int((ra_total_seconds % 3600) // 60)
        ra_s = ra_total_seconds % 60

        dec_val = float(dec_deg)
        dec_sign = "+" if dec_val >= 0 else "-"
        dec_abs = abs(dec_val)
        dec_total_seconds = dec_abs * 3600.0
        dec_d = int(dec_total_seconds // 3600)
        dec_m = int((dec_total_seconds % 3600) // 60)
        dec_s = dec_total_seconds % 60
        return f"{ra_h:02d}:{ra_m:02d}:{ra_s:05.2f}  {dec_sign}{dec_d:02d}:{dec_m:02d}:{dec_s:05.2f}"
    except Exception:
        return ""


def build_filtered_web_export_html(
    items: List[Dict[str, Any]],
    *,
    summary: str,
    patch_size_px: int,
    hist_level: str,
    empty_message: str,
) -> str:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for item in items:
        dir_tag = str(item.get("output_tag", "") or "").strip()
        group_key = dir_tag[:8] if dir_tag else "UNGROUPED"
        grouped.setdefault(group_key, []).append(item)

    group_blocks = []
    for group_key in sorted(grouped.keys()):
        cards = []
        for item in grouped[group_key]:
            cards.append(
                '<div class="card">'
                f'<a href="{html.escape(item["img_rel"])}" target="_blank"><img src="{html.escape(item["img_rel"])}" alt="patch"></a>'
                '<div class="meta">'
                f'<div>rank: {html.escape(str(item.get("rank", "")))}</div>'
                f'<div>ai_class: {html.escape(str(item.get("ai_class", "")))}</div>'
                f'<div>skip_flag: {html.escape(str(item.get("skip_flag", "0")))}</div>'
                f'<div>ra/dec: {html.escape(str(item.get("ra_dec_deg", "")))}</div>'
                f'<div>HMS/DMS: {html.escape(str(item.get("hms_dms", "")))}</div>'
                f'<div>median_flux_norm: {html.escape(str(item.get("median_flux_norm", "")))}</div>'
                f'<div>row: {html.escape(str(item.get("raw_idx", "")))}</div>'
                f'<div>dir: {html.escape(str(item.get("output_tag", "")))}</div>'
                "</div></div>"
            )
        group_blocks.append(
            '<section class="group">'
            f'<h3>分组: {html.escape(group_key)} <span class="count">({len(grouped[group_key])})</span></h3>'
            f'<div class="grid">{"".join(cards)}</div>'
            "</section>"
        )

    groups_html = "\n".join(group_blocks)
    if not groups_html:
        groups_html = (
            '<section class="group">'
            '<h3>筛选结果为 0</h3>'
            f'<p>{html.escape(empty_message or "当前筛选条件没有命中任何 CSV 行。")}</p>'
            "</section>"
        )

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CSV筛选导出</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 16px; }}
    .summary {{ background: #f6f8fa; padding: 10px 12px; border-radius: 8px; margin-bottom: 12px; }}
    .summary pre {{ white-space: pre-wrap; margin: 6px 0 0 0; }}
    .group {{ margin-bottom: 18px; }}
    .group h3 {{ margin: 10px 0 8px 0; }}
    .group .count {{ color: #666; font-weight: normal; font-size: 12px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 10px; }}
    .card {{ border: 1px solid #ddd; border-radius: 8px; overflow: hidden; background: #fff; }}
    .card img {{ width: 100%; height: auto; display: block; background: #000; }}
    .meta {{ font-size: 12px; line-height: 1.45; padding: 8px; }}
  </style>
</head>
<body>
  <h2>CSV筛选导出网页</h2>
  <div class="summary">
    <div>导出时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
    <div>命中导出: {len(items)}</div>
    <div>patch尺寸: {patch_size_px}px，拉伸: {html.escape(hist_level)}</div>
    <pre>条件: {html.escape(summary)}</pre>
  </div>
  {groups_html}
</body>
</html>"""


def collect_hits(csv_paths: List[Path], profile: Dict[str, Any]) -> Tuple[List[Hit], int, int]:
    hits: List[Hit] = []
    skipped_large_csv = 0
    scanned_csv = 0
    skip_large_csv = bool(profile.get("skip_large_csv", False))
    large_csv_max_rows = int(profile.get("large_csv_max_rows", 200))
    for csv_path in csv_paths:
        rows = load_csv_rows(csv_path)
        scanned_csv += 1
        if skip_large_csv and len(rows) > large_csv_max_rows:
            skipped_large_csv += 1
            continue
        for raw_idx, row in enumerate(rows):
            if row_matches_filter(row, profile):
                hits.append((csv_path, csv_path.parent, raw_idx, row))
    return hits, skipped_large_csv, scanned_csv


def resolve_zip_base(args: argparse.Namespace, cfg: Dict[str, Any], diff_output_root: Path) -> Path:
    if args.zip_root:
        return Path(args.zip_root).resolve()
    paths_cfg = cfg.get("paths", {})
    zip_root = str(paths_cfg.get("zip_output_directory", "") or "").strip()
    if not zip_root:
        zip_root = str(paths_cfg.get("zip_output_root", "") or "").strip()
    if zip_root:
        return Path(zip_root).resolve()
    parent_dir = diff_output_root.resolve().parent
    if parent_dir == diff_output_root.resolve():
        raise ValueError("diff_output_root 没有可用父目录，无法创建 out_zip_yyyymmdd。")
    return parent_dir


def export_web_zip(
    hits: List[Hit],
    *,
    out_zip_base: Path,
    summary: str,
    patch_size_px: int,
    hist_level: str,
) -> Tuple[int, Path, Path, str]:
    out_zip_dir = out_zip_base / f"out_zip_{datetime.now().strftime('%Y%m%d')}"
    out_zip_dir.mkdir(parents=True, exist_ok=True)
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    stage_dir = out_zip_dir / f"web_export_{run_tag}"
    assets_dir = stage_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    half = max(1, int(round(max(64, min(2048, patch_size_px)) / 2.0)))
    items: List[Dict[str, Any]] = []
    cached_output_dir: Optional[Path] = None
    aligned_data = None
    aligned_header = None
    empty_message = "当前筛选条件没有命中任何 CSV 行。" if not hits else ""

    try:
        for seq_idx, (_csv_path, output_dir, raw_idx, row) in enumerate(hits):
            if output_dir != cached_output_dir:
                aligned_fits = find_primary_aligned_fits_in_output_dir(output_dir)
                if not aligned_fits:
                    cached_output_dir = None
                    aligned_data = None
                    aligned_header = None
                    continue
                aligned_data, aligned_header = load_fits_image_and_header(aligned_fits)
                if aligned_data is None:
                    cached_output_dir = None
                    aligned_header = None
                    continue
                cached_output_dir = output_dir

            if aligned_data is None or aligned_header is None:
                continue
            x, y = resolve_candidate_pixel_xy(row, aligned_header)
            if x is None or y is None:
                continue
            patch = extract_local_patch(aligned_data, x, y, half_size=half)
            if patch is None:
                continue
            u8 = stretch_patch_to_uint8(patch, level=hist_level)

            rank_raw = str(row.get("rank", "") or "").strip() or str(seq_idx + 1)
            frame_tag = sanitize_output_name(output_dir.name)
            fname = f"{frame_tag}_rank{sanitize_output_name(rank_raw)}_{len(items):04d}.png"
            out_path = assets_dir / fname
            try:
                Image.fromarray(u8.astype(np.uint8)).save(str(out_path))
            except Exception as e:
                logging.warning("写入 PNG 失败: %s (%s)", out_path, e)
                continue

            ra_deg = try_get_float_from_row(row, ["ra", "ra_deg", "ra_degree", "target_ra"])
            dec_deg = try_get_float_from_row(row, ["dec", "dec_deg", "dec_degree", "target_dec"])
            median_flux_norm = try_get_float_from_row(row, ["median_flux_norm"])
            items.append(
                {
                    "img_rel": f"assets/{fname}",
                    "rank": rank_raw,
                    "ai_class": str(row.get("ai_class", "")).strip(),
                    "skip_flag": str(get_skip_flag(row)),
                    "ra_dec_deg": "" if ra_deg is None or dec_deg is None else f"{float(ra_deg):.8f}, {float(dec_deg):.8f}",
                    "hms_dms": format_ra_dec_hms_dms_text(ra_deg, dec_deg),
                    "median_flux_norm": "" if median_flux_norm is None else f"{float(median_flux_norm):.6g}",
                    "raw_idx": int(raw_idx),
                    "output_tag": frame_tag,
                }
            )

        if not items and hits:
            empty_message = "没有可导出的候选（缺少 aligned FITS、坐标解析失败或写入失败）。"
            logging.warning(empty_message)
            print(empty_message)

        html_text = build_filtered_web_export_html(
            items,
            summary=summary,
            patch_size_px=patch_size_px,
            hist_level=hist_level,
            empty_message=empty_message,
        )
        (stage_dir / "index.html").write_text(html_text, encoding="utf-8")
        (stage_dir / "meta.json").write_text(
            json.dumps(
                {
                    "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "condition_summary": summary,
                    "patch_size_px": patch_size_px,
                    "hist_level": hist_level,
                    "count": len(items),
                    "empty_message": empty_message,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        result_tag = "filter_results_0_" if not items else ""
        zip_path = out_zip_dir / f"csv_filtered_web_{result_tag}{run_tag}.zip"
        with zipfile.ZipFile(str(zip_path), "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(stage_dir.rglob("*")):
                if p.is_file():
                    zf.write(str(p), arcname=str(p.relative_to(stage_dir)))
        return len(items), zip_path.resolve(), out_zip_dir.resolve(), empty_message
    finally:
        shutil.rmtree(stage_dir, ignore_errors=True)


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    if not validate_date(args.date):
        raise SystemExit(f"日期格式错误: {args.date}，应为 YYYYMMDD")

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise SystemExit(f"配置文件不存在: {cfg_path}")
    cfg = load_json(cfg_path)

    paths_cfg = cfg.get("paths", {})
    diff_output_root = Path(str(paths_cfg.get("diff_output_root", "")))
    if not diff_output_root.exists():
        raise SystemExit(f"diff 输出目录不存在: {diff_output_root}")

    profile = load_filter_profile(cfg, args.profile)
    validate_filter_profile(profile)
    summary = condition_summary(profile)

    csv_paths = discover_candidate_csvs(
        diff_output_root=diff_output_root,
        date_text=args.date,
        telescope=args.telescope,
        region=args.region,
    )
    if args.max_csv > 0:
        csv_paths = csv_paths[: args.max_csv]

    hits, skipped_large_csv, scanned_csv = collect_hits(csv_paths, profile)
    if skipped_large_csv:
        summary += f"\n跳过大 CSV 的文件数：{skipped_large_csv}"

    logging.info(
        "筛选扫描完成: csv=%d, 命中行=%d, 跳过大CSV=%d, profile=%s",
        scanned_csv,
        len(hits),
        skipped_large_csv,
        profile["name"],
    )

    out_zip_base = resolve_zip_base(args, cfg, diff_output_root)
    n, zip_path, out_dir, empty_message = export_web_zip(
        hits,
        out_zip_base=out_zip_base,
        summary=summary,
        patch_size_px=max(64, min(2048, int(args.patch_size))),
        hist_level=args.hist_level,
    )
    logging.info("导出完成: 导出目标=%d, ZIP=%s, 目录=%s", n, zip_path, out_dir)
    if empty_message:
        logging.info("空结果原因: %s", empty_message)
    print(f"已导出 {n} 个目标并打包为ZIP。")
    print(f"ZIP: {zip_path}")
    print(f"目录: {out_dir}")


if __name__ == "__main__":
    main()
