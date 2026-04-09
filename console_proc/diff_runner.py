#!/usr/bin/env python3
"""
console_proc 独立 diff pipeline 执行器

从 download_root/<telescope>/<date>/<region> 发现下载后的 FITS 文件，
匹配模板文件后按 diff_pipeline_settings 执行命令链。
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从 download_root 目录执行 diff pipeline（console 独立版）"
    )
    parser.add_argument("--config", default="console_proc/config.json", help="配置文件路径")
    parser.add_argument("--date", required=True, help="日期，YYYYMMDD")
    parser.add_argument("--telescope", help="仅处理指定系统，例如 GY1")
    parser.add_argument("--region", help="仅处理指定天区，例如 K019")
    parser.add_argument("--max-files", type=int, default=0, help="每个天区最多处理文件数，0=不限制")
    parser.add_argument("--force-rerun", action="store_true", help="忽略完成标记，强制重跑")
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


def relocate_preprocess_output_to_output_dir(source_file: Path, output_dir: Path) -> Optional[str]:
    """
    recommended_pipeline_console.py 实际会把 .01proc.fit 写入输入目录。
    这里统一搬运到 output_dir，保证后续步骤与产物目录一致。
    """
    target_base = sanitize_output_name(source_file.stem)
    actual_path = source_file.parent / f"{target_base}.01proc.fit"
    expected_path = output_dir / f"{target_base}.01proc.fit"

    if expected_path.exists():
        if actual_path.exists() and actual_path != expected_path:
            try:
                actual_path.unlink()
            except Exception:
                pass
        return None

    if not actual_path.exists():
        return f"预处理输出不存在: {actual_path}"

    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        actual_path.replace(expected_path)
    except Exception as ex:
        return f"搬运预处理输出失败: {ex}"
    return None


def build_last_command_signature(commands: List[Tuple[str, List[str]]]) -> Dict[str, str]:
    if not commands:
        return {"last_step_title": "", "last_cmd": "", "last_cmd_sha256": ""}
    last_title, last_cmd = commands[-1]
    last_cmd_text = subprocess.list2cmdline(last_cmd)
    return {
        "last_step_title": last_title,
        "last_cmd": last_cmd_text,
        "last_cmd_sha256": hashlib.sha256(last_cmd_text.encode("utf-8")).hexdigest(),
    }


def should_skip_by_done_marker(done_path: Path, expected_sig: Dict[str, str]) -> bool:
    if not done_path.exists():
        return False
    try:
        payload = load_json(done_path)
    except Exception:
        return False

    if payload.get("status") != "success":
        return False
    return str(payload.get("last_cmd_sha256", "")) == expected_sig["last_cmd_sha256"]


def write_done_marker(
    done_path: Path,
    source_file: Path,
    template_file: Path,
    output_dir: Path,
    command_sig: Dict[str, str],
) -> None:
    marker = {
        "status": "success",
        "source_file": str(source_file),
        "template_file": str(template_file),
        "output_dir": str(output_dir),
        "last_step_title": command_sig["last_step_title"],
        "last_cmd": command_sig["last_cmd"],
        "last_cmd_sha256": command_sig["last_cmd_sha256"],
    }
    with done_path.open("w", encoding="utf-8") as fp:
        json.dump(marker, fp, ensure_ascii=False, indent=2)


def get_diff_runner_max_workers(cfg: Dict[str, Any]) -> int:
    runner_cfg = cfg.get("diff_runner", {})
    value = runner_cfg.get("max_workers", 2)
    try:
        n = int(value)
    except Exception:
        n = 2
    return max(1, n)


def get_template_lock(
    template_locks: Dict[str, threading.Lock],
    template_locks_guard: threading.Lock,
    template_file: Path,
) -> threading.Lock:
    key = str(template_file.resolve()).lower()
    with template_locks_guard:
        lock = template_locks.get(key)
        if lock is None:
            lock = threading.Lock()
            template_locks[key] = lock
        return lock


def run_one_task(
    task: Dict[str, Any],
    pipeline_settings: Dict[str, Any],
    force_rerun: bool,
    dry_run: bool,
    template_locks: Dict[str, threading.Lock],
    template_locks_guard: threading.Lock,
) -> Dict[str, Any]:
    tel = str(task["telescope"])
    region = str(task["region"])
    date_text = str(task["date"])
    source_file = Path(str(task["source_file"]))
    template_file = Path(str(task["template_file"]))
    output_dir = Path(str(task["output_dir"]))
    log_file = output_dir / "pipeline.log"
    done_path = output_dir / ".done.json"

    commands = build_pipeline_commands(
        source_file=source_file,
        template_file=template_file,
        output_dir=output_dir,
        settings=pipeline_settings,
    )
    command_sig = build_last_command_signature(commands)

    if not force_rerun and should_skip_by_done_marker(done_path, command_sig):
        return {
            "status": "skipped",
            "reason": "already_done",
            "date": date_text,
            "telescope": tel,
            "region": region,
            "source_file": str(source_file),
            "template_file": str(template_file),
            "output_dir": str(output_dir),
            "done_file": str(done_path),
        }

    ok = True
    fail_reason = ""
    for title, cmd in commands:
        if title == "导出模板星点":
            template_lock = get_template_lock(template_locks, template_locks_guard, template_file)
            with template_lock:
                step_ok, msg = run_command(title, cmd, log_file, dry_run)
        else:
            step_ok, msg = run_command(title, cmd, log_file, dry_run)
        if not step_ok:
            ok = False
            fail_reason = msg
            break
        if title == "预处理目标图" and not dry_run:
            move_err = relocate_preprocess_output_to_output_dir(source_file, output_dir)
            if move_err:
                ok = False
                fail_reason = move_err
                break

    if ok:
        if not dry_run:
            write_done_marker(done_path, source_file, template_file, output_dir, command_sig)
        return {
            "status": "success",
            "date": date_text,
            "telescope": tel,
            "region": region,
            "source_file": str(source_file),
            "template_file": str(template_file),
            "output_dir": str(output_dir),
            "log_file": str(log_file),
            "done_file": str(done_path),
        }

    return {
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


def discover_download_items(download_root: Path, date_text: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if not download_root.exists():
        return items

    tel_dirs = sorted([p for p in download_root.iterdir() if p.is_dir()], key=lambda p: p.name.lower())
    for tel_dir in tel_dirs:
        date_dir = tel_dir / date_text
        if not date_dir.exists() or not date_dir.is_dir():
            continue

        region_dirs = sorted([p for p in date_dir.iterdir() if p.is_dir()], key=lambda p: p.name.lower())
        for region_dir in region_dirs:
            try:
                region = normalize_region(region_dir.name)
            except ValueError:
                continue
            items.append(
                {
                    "date": date_text,
                    "telescope": tel_dir.name.upper(),
                    "region": region,
                    "download_dir": str(region_dir),
                }
            )
    return items


def collect_region_files(download_dir: Path) -> List[Path]:
    if not download_dir.exists():
        return []
    return sorted([p for p in download_dir.iterdir() if p.is_file() and is_fits_file(p)], key=lambda p: p.name.lower())


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
    download_root_val = paths_cfg.get("download_root", "")
    download_root = Path(str(download_root_val))
    template_root = Path(str(paths_cfg.get("template_root", "")))
    diff_output_root = Path(str(paths_cfg.get("diff_output_root", "diff_output")))
    diff_output_root.mkdir(parents=True, exist_ok=True)
    if not download_root.exists():
        raise SystemExit(f"下载目录不存在: {download_root}")
    if not template_root.exists():
        raise SystemExit(f"模板目录不存在: {template_root}")

    pipeline_settings = merge_dict(default_diff_pipeline_settings(), cfg.get("diff_pipeline_settings", {}))
    max_workers = get_diff_runner_max_workers(cfg)

    items = discover_download_items(download_root, args.date)

    if args.telescope:
        items = [x for x in items if str(x.get("telescope", "")).upper() == args.telescope.upper()]
    if args.region:
        norm_region = normalize_region(args.region)
        items = [x for x in items if str(x.get("region", "")).upper() == norm_region]

    if not items:
        logging.warning("无匹配处理项")
        raise SystemExit(0)

    run_result_dir = diff_output_root / "_meta" / args.date / "diff_results"
    run_result_dir.mkdir(parents=True, exist_ok=True)

    all_results: List[Dict[str, Any]] = []
    success = 0
    failed = 0
    skipped = 0
    tasks: List[Dict[str, Any]] = []

    for item in items:
        tel = str(item.get("telescope", "")).upper()
        region = str(item.get("region", "")).upper()
        date_text = str(item.get("date", args.date))
        download_dir = Path(str(item.get("download_dir", "")))
        region_files = collect_region_files(download_dir)
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
            tasks.append(
                {
                    "date": date_text,
                    "telescope": tel,
                    "region": region,
                    "source_file": str(source_file),
                    "template_file": str(template_file),
                    "output_dir": str(output_dir),
                }
            )

    template_locks: Dict[str, threading.Lock] = {}
    template_locks_guard = threading.Lock()
    logging.info("开始执行任务: %d, 并发线程: %d", len(tasks), max_workers)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_map = {
            pool.submit(
                run_one_task,
                task,
                pipeline_settings,
                bool(args.force_rerun),
                bool(args.dry_run),
                template_locks,
                template_locks_guard,
            ): task
            for task in tasks
        }
        for future in as_completed(future_map):
            task = future_map[future]
            try:
                result = future.result()
            except Exception as ex:
                failed += 1
                result = {
                    "status": "failed",
                    "reason": f"worker_exception: {ex}",
                    "date": str(task["date"]),
                    "telescope": str(task["telescope"]),
                    "region": str(task["region"]),
                    "source_file": str(task["source_file"]),
                    "template_file": str(task["template_file"]),
                    "output_dir": str(task["output_dir"]),
                }

            status = str(result.get("status", "failed"))
            if status == "success":
                success += 1
            elif status == "skipped":
                skipped += 1
                logging.info("跳过: %s", str(result.get("source_file", "")))
            else:
                failed += 1
                logging.error(
                    "处理失败: %s (%s)",
                    str(result.get("source_file", "")),
                    str(result.get("reason", "unknown")),
                )
            all_results.append(result)

    summary = {
        "date": args.date,
        "download_root": str(download_root),
        "max_workers": max_workers,
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
