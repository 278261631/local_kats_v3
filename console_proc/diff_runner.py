#!/usr/bin/env python3
"""
console_proc 独立 diff pipeline 执行器

从 files_data/<date>/index.json 读取任务清单，定位下载后的 FITS 文件，
匹配模板文件后按 diff_pipeline_settings 执行命令链。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从 files_data 清单执行 diff pipeline（console 独立版）"
    )
    parser.add_argument("--config", default="console_proc/config.json", help="配置文件路径")
    parser.add_argument("--date", required=True, help="日期，YYYYMMDD")
    parser.add_argument("--files-data-root", default="files_data", help="files_data 根目录")
    parser.add_argument("--telescope", help="仅处理指定系统，例如 GY1")
    parser.add_argument("--region", help="仅处理指定天区，例如 K019")
    parser.add_argument("--max-files", type=int, default=0, help="每个天区最多处理文件数，0=不限制")
    parser.add_argument("--dry-run", action="store_true", help="仅打印命令，不实际执行")
    parser.add_argument("--verbose", action="store_true", help="输出调试日志")
    return parser.parse_args()


def validate_date(date_text: str) -> bool:
    return bool(re.fullmatch(r"\d{8}", date_text))


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_region(region: str) -> str:
    r = region.strip().upper()
    if not r.startswith("K"):
        r = f"K{r}"
    if re.fullmatch(r"K\d{1,3}", r):
        r = f"K{int(r[1:]):03d}"
    if not re.fullmatch(r"K\d{3}", r):
        raise ValueError(f"非法天区格式: {region}")
    return r


def sanitize_output_name(name: str) -> str:
    # 复用 GUI 的思路：输出名仅保留安全字符
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def is_fits_file(path: Path) -> bool:
    return path.suffix.lower() in {".fits", ".fit", ".fts"}


def parse_k_from_filename(filename: str) -> Optional[str]:
    # 支持 K053-1 或 K053
    m = re.search(r"(K\d{3}(?:-\d+)?)", filename, flags=re.IGNORECASE)
    return m.group(1).upper() if m else None


def find_case_insensitive_subdir(root: Path, name: str) -> Optional[Path]:
    if not root.exists():
        return None
    target = name.lower()
    for item in root.iterdir():
        if item.is_dir() and item.name.lower() == target:
            return item
    return None


def find_template_file(template_root: Path, telescope: str, region: str, source_fits: Path) -> Optional[Path]:
    tel_dir = find_case_insensitive_subdir(template_root, telescope)
    if not tel_dir:
        return None

    fits_files = [p for p in tel_dir.iterdir() if p.is_file() and is_fits_file(p)]
    if not fits_files:
        return None

    k_full = parse_k_from_filename(source_fits.name) or region
    candidates = [k_full, region]
    if "-" in k_full:
        candidates.append(k_full.split("-", 1)[0])

    def match_prefix(stem: str, key: str) -> bool:
        stem_l = stem.lower()
        key_l = key.lower()
        if not stem_l.startswith(key_l):
            return False
        if len(stem_l) == len(key_l):
            return True
        return stem_l[len(key_l)] in {"_", "-", ".", " "}

    for key in candidates:
        for f in fits_files:
            if match_prefix(f.stem, key):
                return f
    return None


def default_diff_pipeline_settings() -> Dict[str, Any]:
    return {
        "script_paths": {
            "export_fits_stars": "D:/github/misaligned_fits/export_fits_stars.py",
            "recommended_pipeline_console": "D:/github/fits_data_view_process_3d/std_process/recommended_pipeline_console.py",
            "reproject_wcs_and_export_stars": "D:/github/misaligned_fits/reproject_wcs_and_export_stars.py",
            "solve_alignment_from_stars": "D:/github/misaligned_fits/solve_alignment_from_stars.py",
            "render_alignment_outputs": "D:/github/misaligned_fits/render_alignment_outputs.py",
            "rank_variable_candidates": "D:/github/misaligned_fits/rank_variable_candidates.py",
            "crossmatch_nonref_candidates": "D:/github/misaligned_fits/crossmatch_nonref_candidates.py",
            "export_nonref_candidate_ab_cutouts": "D:/github/misaligned_fits/export_nonref_candidate_ab_cutouts.py",
        },
        "export_uniform_grid_x": 7,
        "export_uniform_grid_y": 7,
        "export_uniform_per_cell": 100,
        "preprocess_box": 48,
        "preprocess_clip_sigma": 3.0,
        "preprocess_median_ksize": 3,
        "preprocess_denoise_sigma": 2.0,
        "preprocess_mix_alpha": 0.7,
        "reproject_max_stars": 5000,
        "reproject_uniform_grid_x": 7,
        "reproject_uniform_grid_y": 7,
        "reproject_uniform_per_cell": 100,
        "solve_radii": [24, 32, 40],
        "rank_min_observations": 2,
        "enable_crossmatch_nonref_candidates": True,
        "enable_export_nonref_candidate_ab_cutouts": False,
        "nonref_candidate_cutout_size": 128,
        "fast_mode": True,
    }


def merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(out.get(k), dict) and isinstance(v, dict):
            out[k] = merge_dict(out[k], v)
        else:
            out[k] = v
    return out


def build_pipeline_commands(
    source_file: Path,
    template_file: Path,
    output_dir: Path,
    settings: Dict[str, Any],
) -> List[Tuple[str, List[str]]]:
    py = sys.executable or "python"

    script_paths = settings["script_paths"]
    export_fits_stars_script = script_paths["export_fits_stars"]
    preprocess_script = script_paths["recommended_pipeline_console"]
    reproject_script = script_paths["reproject_wcs_and_export_stars"]
    solve_script = script_paths["solve_alignment_from_stars"]
    render_script = script_paths["render_alignment_outputs"]
    rank_script = script_paths["rank_variable_candidates"]
    crossmatch_script = script_paths["crossmatch_nonref_candidates"]
    export_nonref_cutouts_script = script_paths["export_nonref_candidate_ab_cutouts"]

    template_base = sanitize_output_name(template_file.stem)
    target_base = sanitize_output_name(source_file.stem)
    template_dir = template_file.parent

    template_stars = str(template_dir / f"{template_base}.stars.npz")
    template_stars_all = str(template_dir / f"{template_base}.stars.all.npz")
    proc_fit = str(output_dir / f"{target_base}.01proc.fit")
    rp_fit = str(output_dir / f"{target_base}.02rp.fit")
    rp_stars = str(output_dir / f"{target_base}.rp.stars.npz")
    rp_stars_all = str(output_dir / f"{target_base}.rp.stars.all.npz")
    align_npz = str(output_dir / f"{target_base}.rp.align.npz")
    out_csv_rank = str(output_dir / "variable_candidates_rank.csv")
    out_csv_nonref = str(output_dir / "variable_candidates_nonref_only.csv")
    out_csv_nonref_inner_border = str(output_dir / "variable_candidates_nonref_only_inner_border.csv")
    out_csv_ref_missing = str(output_dir / "variable_candidates_ref_only_missing_in_targets.csv")
    out_png_rank = str(output_dir / "variable_candidates_rank.png")
    out_overlap_expr = str(output_dir / "ref_target_overlap_polygon_expr.json")
    out_overlap_expr_png = str(output_dir / "ref_target_overlap_polygon_expr.png")
    ref_valid_region_json = str(template_dir / f"{template_base}.effective.json")
    ref_valid_region_png = str(template_dir / f"{template_base}.effective.png")
    out_find_mpc_csv = str(output_dir / "find_mpc.csv")

    solve_radii_raw = settings.get("solve_radii", [24, 32, 40])
    if not isinstance(solve_radii_raw, list) or not solve_radii_raw:
        solve_radii_raw = [24, 32, 40]
    solve_radii = [str(v) for v in solve_radii_raw]

    commands: List[Tuple[str, List[str]]] = [
        (
            "导出模板星点",
            [
                py, export_fits_stars_script,
                "--fits", str(template_file),
                "--out", template_stars,
                "--out-all", template_stars_all,
                "--out-valid-region", ref_valid_region_json,
                "--out-valid-region-png", ref_valid_region_png,
                "--uniform-grid-x", str(settings.get("export_uniform_grid_x", 7)),
                "--uniform-grid-y", str(settings.get("export_uniform_grid_y", 7)),
                "--uniform-per-cell", str(settings.get("export_uniform_per_cell", 100)),
            ],
        ),
        (
            "预处理目标图",
            [
                py, preprocess_script,
                "-i", str(source_file),
                "-o", proc_fit,
                "--box", str(settings.get("preprocess_box", 48)),
                "--clip-sigma", str(settings.get("preprocess_clip_sigma", 3.0)),
                "--median-ksize", str(settings.get("preprocess_median_ksize", 3)),
                "--denoise-sigma", str(settings.get("preprocess_denoise_sigma", 2.0)),
                "--mix-alpha", str(settings.get("preprocess_mix_alpha", 0.7)),
                "--overwrite",
            ],
        ),
        (
            "WCS重投影并导出目标星点",
            [
                py, reproject_script,
                "--a", str(template_file),
                "--b", proc_fit,
                "--out-fits", rp_fit,
                "--out-stars", rp_stars,
                "--out-stars-all", rp_stars_all,
                "--skip-median-filter",
                "--max-stars", str(settings.get("reproject_max_stars", 5000)),
                "--uniform-grid-x", str(settings.get("reproject_uniform_grid_x", 7)),
                "--uniform-grid-y", str(settings.get("reproject_uniform_grid_y", 7)),
                "--uniform-per-cell", str(settings.get("reproject_uniform_per_cell", 100)),
            ],
        ),
        (
            "求解对齐",
            [
                py, solve_script,
                "--a-stars", template_stars,
                "--b-stars", rp_stars,
                "--out", align_npz,
                "--radii", *solve_radii,
            ],
        ),
        (
            "生成候选目标CSV",
            [
                py, rank_script,
                "--ref-stars-all", template_stars_all,
                "--target-stars-all", rp_stars_all,
                "--target-align", align_npz,
                "--out-csv", out_csv_rank,
                "--out-csv-nonref", out_csv_nonref,
                "--out-overlap-expr", out_overlap_expr,
                "--out-overlap-expr-png", out_overlap_expr_png,
                "--out-csv-nonref-inner-border", out_csv_nonref_inner_border,
                "--ref-valid-region", ref_valid_region_json,
                "--out-csv-ref-missing", out_csv_ref_missing,
                "--out-png", out_png_rank,
                "--min-observations", str(settings.get("rank_min_observations", 2)),
            ],
        ),
    ]

    if not bool(settings.get("fast_mode", True)):
        commands.insert(
            4,
            (
                "渲染对齐结果",
                [
                    py, render_script,
                    "--a", str(template_file),
                    "--b", rp_fit,
                    "--align", align_npz,
                    "--outdir", str(output_dir),
                ],
            ),
        )

    if bool(settings.get("enable_crossmatch_nonref_candidates", True)):
        commands.append(
            (
                "交叉匹配非参考候选",
                [
                    py, crossmatch_script,
                    "--input-csv", out_csv_nonref_inner_border,
                    "--find-mpc-csv", out_find_mpc_csv,
                    "--ref-fits", str(template_file),
                ],
            )
        )

    if bool(settings.get("enable_export_nonref_candidate_ab_cutouts", False)):
        commands.append(
            (
                "导出非参考候选AB切图",
                [
                    py, export_nonref_cutouts_script,
                    "--input-csv", out_csv_nonref_inner_border,
                    "--a-fits", str(template_file),
                    "--b-fits", rp_fit,
                    "--a-stars-all", template_stars_all,
                    "--b-stars-all", rp_stars_all,
                    "--align-npz", align_npz,
                    "--cutout-size", str(settings.get("nonref_candidate_cutout_size", 128)),
                ],
            )
        )

    return commands


def run_command(title: str, cmd: List[str], log_file: Path, dry_run: bool) -> Tuple[bool, str]:
    cmd_text = subprocess.list2cmdline(cmd)
    with log_file.open("a", encoding="utf-8") as fp:
        fp.write(f"\n[STEP] {title}\n")
        fp.write(f"[CMD ] {cmd_text}\n")
    logging.info("执行步骤: %s", title)
    logging.debug("命令: %s", cmd_text)

    if dry_run:
        return True, "dry-run"

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    with log_file.open("a", encoding="utf-8") as fp:
        fp.write(proc.stdout or "")
        fp.write(f"\n[EXIT] {proc.returncode}\n")

    if proc.returncode != 0:
        return False, f"步骤失败: {title}, exit={proc.returncode}"
    return True, "ok"


def collect_region_files(item: Dict[str, Any]) -> List[Path]:
    download_dir = Path(str(item.get("download_dir", "")))
    if not download_dir.exists():
        return []

    files_txt_val = item.get("files_txt")
    if files_txt_val:
        files_txt = Path(str(files_txt_val))
        if files_txt.exists():
            result: List[Path] = []
            for line in files_txt.read_text(encoding="utf-8").splitlines():
                name = line.strip()
                if not name:
                    continue
                p = download_dir / name
                if p.exists() and is_fits_file(p):
                    result.append(p)
            return result

    return [p for p in download_dir.iterdir() if p.is_file() and is_fits_file(p)]


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
    template_root = Path(str(paths_cfg.get("template_root", "")))
    diff_output_root = Path(str(paths_cfg.get("diff_output_root", "diff_output")))
    diff_output_root.mkdir(parents=True, exist_ok=True)
    if not template_root.exists():
        raise SystemExit(f"模板目录不存在: {template_root}")

    pipeline_settings = merge_dict(default_diff_pipeline_settings(), cfg.get("diff_pipeline_settings", {}))

    date_dir = Path(args.files_data_root) / args.date
    index_path = date_dir / "index.json"
    if not index_path.exists():
        raise SystemExit(f"未找到 files_data 索引: {index_path}")
    index_data = load_json(index_path)
    items = index_data.get("items", [])
    if not isinstance(items, list):
        raise SystemExit("index.json 格式错误: items 不是数组")

    if args.telescope:
        items = [x for x in items if str(x.get("telescope", "")).upper() == args.telescope.upper()]
    if args.region:
        norm_region = normalize_region(args.region)
        items = [x for x in items if str(x.get("region", "")).upper() == norm_region]

    if not items:
        logging.warning("无匹配处理项")
        raise SystemExit(0)

    run_result_dir = date_dir / "diff_results"
    run_result_dir.mkdir(parents=True, exist_ok=True)

    all_results: List[Dict[str, Any]] = []
    success = 0
    failed = 0
    skipped = 0

    for item in items:
        tel = str(item.get("telescope", "")).upper()
        region = str(item.get("region", "")).upper()
        date_text = str(item.get("date", args.date))
        region_files = collect_region_files(item)
        if args.max_files and args.max_files > 0:
            region_files = region_files[: args.max_files]

        logging.info("[%s/%s/%s] 待处理文件数: %d", tel, date_text, region, len(region_files))
        for source_file in region_files:
            template_file = find_template_file(template_root, tel, region, source_file)
            if not template_file:
                skipped += 1
                all_results.append(
                    {
                        "status": "skipped",
                        "reason": "template_not_found",
                        "date": date_text,
                        "telescope": tel,
                        "region": region,
                        "source_file": str(source_file),
                    }
                )
                logging.warning("跳过（未找到模板）: %s", source_file.name)
                continue

            output_dir = diff_output_root / tel / date_text / region / sanitize_output_name(source_file.stem)
            output_dir.mkdir(parents=True, exist_ok=True)
            log_file = output_dir / "pipeline.log"

            commands = build_pipeline_commands(
                source_file=source_file,
                template_file=template_file,
                output_dir=output_dir,
                settings=pipeline_settings,
            )

            ok = True
            fail_reason = ""
            for title, cmd in commands:
                step_ok, msg = run_command(title, cmd, log_file, args.dry_run)
                if not step_ok:
                    ok = False
                    fail_reason = msg
                    break

            if ok:
                success += 1
                all_results.append(
                    {
                        "status": "success",
                        "date": date_text,
                        "telescope": tel,
                        "region": region,
                        "source_file": str(source_file),
                        "template_file": str(template_file),
                        "output_dir": str(output_dir),
                        "log_file": str(log_file),
                    }
                )
            else:
                failed += 1
                all_results.append(
                    {
                        "status": "failed",
                        "reason": fail_reason,
                        "date": date_text,
                        "telescope": tel,
                        "region": region,
                        "source_file": str(source_file),
                        "template_file": str(template_file),
                        "output_dir": str(output_dir),
                        "log_file": str(log_file),
                    }
                )
                logging.error("处理失败: %s (%s)", source_file.name, fail_reason)

    summary = {
        "date": args.date,
        "item_count": len(items),
        "success": success,
        "failed": failed,
        "skipped": skipped,
        "dry_run": bool(args.dry_run),
        "results": all_results,
    }
    summary_path = run_result_dir / "diff_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logging.info(
        "完成: success=%d, failed=%d, skipped=%d, summary=%s",
        success,
        failed,
        skipped,
        summary_path,
    )


if __name__ == "__main__":
    main()
