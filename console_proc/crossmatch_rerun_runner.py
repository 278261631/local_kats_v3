#!/usr/bin/env python3
"""
console_proc 独立 Crossmatch 重跑执行器（按筛选命中批量 --only-rank）。
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Dict, List, Tuple

from filtered_tools_common import (
    discover_candidate_csvs,
    load_csv_rows,
    load_filter_profile,
    load_json,
    row_matches_filter,
    try_parse_int_from_csv_value,
    validate_filter_profile,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按筛选配置批量重跑 crossmatch_nonref_candidates（--only-rank）"
    )
    parser.add_argument("--config", default="console_proc/config.json", help="配置文件路径")
    parser.add_argument("--date", required=True, help="日期，YYYYMMDD")
    parser.add_argument("--telescope", help="仅处理指定系统，例如 GY1")
    parser.add_argument("--region", help="仅处理指定天区，例如 K019")
    parser.add_argument("--profile", default="B", help="筛选配置名，默认 B")
    parser.add_argument("--max-csv", type=int, default=0, help="最多处理 CSV 数量，0=不限制")
    parser.add_argument("--max-workers", type=int, default=0, help="并发线程数，0=使用配置")
    parser.add_argument("--dry-run", action="store_true", help="仅打印命令，不实际执行")
    parser.add_argument("--verbose", action="store_true", help="输出调试日志")
    return parser.parse_args()


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")


def validate_date(date_text: str) -> bool:
    return bool(re.fullmatch(r"\d{8}", date_text))


def collect_tasks_for_csv(
    csv_path: Path,
    profile: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    rows = load_csv_rows(csv_path)
    skip_large_csv = bool(profile.get("skip_large_csv", False))
    large_csv_max_rows = int(profile.get("large_csv_max_rows", 200))
    if skip_large_csv and len(rows) > large_csv_max_rows:
        return [], {
            "matched_rows": 0,
            "invalid_rank_rows": 0,
            "dedup_rank_rows": 0,
            "skipped_large_csv_count": 1,
        }

    tasks: List[Dict[str, Any]] = []
    seen_ranks = set()
    matched_rows = 0
    invalid_rank_rows = 0
    dedup_rank_rows = 0
    for row in rows:
        if not row_matches_filter(row, profile):
            continue
        matched_rows += 1
        rank_value = try_parse_int_from_csv_value(row.get("rank"))
        if rank_value is None or rank_value <= 0:
            invalid_rank_rows += 1
            continue
        if rank_value in seen_ranks:
            dedup_rank_rows += 1
            continue
        seen_ranks.add(rank_value)
        tasks.append(
            {
                "csv_path": str(csv_path),
                "rank": int(rank_value),
            }
        )

    return tasks, {
        "matched_rows": matched_rows,
        "invalid_rank_rows": invalid_rank_rows,
        "dedup_rank_rows": dedup_rank_rows,
        "skipped_large_csv_count": 0,
    }


def run_csv_tasks(
    csv_path: Path,
    ranks: List[int],
    crossmatch_script: Path,
    dry_run: bool,
) -> Dict[str, int]:
    py = sys.executable or "python"
    ok_count = 0
    failed_count = 0
    for rank in ranks:
        cmd = [
            py,
            str(crossmatch_script),
            "--input-csv",
            str(csv_path),
            "--only-rank",
            str(rank),
        ]
        logging.info("Crossmatch命令: %s", subprocess.list2cmdline(cmd))
        if dry_run:
            ok_count += 1
            continue
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if proc.returncode == 0:
            ok_count += 1
        else:
            failed_count += 1
            logging.error(
                "Crossmatch执行失败: csv=%s rank=%s exit=%s\n%s",
                str(csv_path),
                rank,
                proc.returncode,
                (proc.stdout or "").strip(),
            )
    return {
        "task_ok": ok_count,
        "task_failed": failed_count,
    }


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
    logging.info("使用筛选配置 %s: %s", profile["name"], profile)

    script_paths = cfg.get("diff_pipeline_settings", {}).get("script_paths", {})
    crossmatch_script = Path(str(script_paths.get("crossmatch_nonref_candidates", "")))
    if not crossmatch_script.exists():
        raise SystemExit(f"crossmatch 脚本不存在: {crossmatch_script}")

    rerun_cfg = cfg.get("crossmatch_rerun", {})
    max_workers = int(args.max_workers) if args.max_workers > 0 else int(rerun_cfg.get("max_workers", 3))
    max_workers = max(1, min(max_workers, 16))

    csv_paths = discover_candidate_csvs(
        diff_output_root=diff_output_root,
        date_text=args.date,
        telescope=args.telescope,
        region=args.region,
    )
    if args.max_csv > 0:
        csv_paths = csv_paths[: args.max_csv]
    if not csv_paths:
        logging.warning("未找到可处理 CSV")
        raise SystemExit(0)
    logging.info("待扫描 CSV: %d", len(csv_paths))

    grouped_tasks: Dict[str, List[int]] = {}
    stats_total = {
        "csv_scanned": 0,
        "matched_rows": 0,
        "invalid_rank_rows": 0,
        "dedup_rank_rows": 0,
        "skipped_large_csv_count": 0,
        "task_total": 0,
        "task_ok": 0,
        "task_failed": 0,
    }

    for csv_path in csv_paths:
        tasks, one_stats = collect_tasks_for_csv(csv_path, profile)
        stats_total["csv_scanned"] += 1
        stats_total["matched_rows"] += int(one_stats["matched_rows"])
        stats_total["invalid_rank_rows"] += int(one_stats["invalid_rank_rows"])
        stats_total["dedup_rank_rows"] += int(one_stats["dedup_rank_rows"])
        stats_total["skipped_large_csv_count"] += int(one_stats["skipped_large_csv_count"])
        if tasks:
            grouped_tasks[str(csv_path)] = [int(t["rank"]) for t in tasks]
            stats_total["task_total"] += len(tasks)

    if not grouped_tasks:
        logging.warning("筛选命中后无可执行任务")
        raise SystemExit(0)

    logging.info(
        "即将执行 crossmatch 任务: csv=%d, tasks=%d, workers=%d, dry_run=%s",
        len(grouped_tasks),
        stats_total["task_total"],
        max_workers,
        bool(args.dry_run),
    )

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_map = {
            pool.submit(
                run_csv_tasks,
                Path(csv_text),
                ranks,
                crossmatch_script,
                bool(args.dry_run),
            ): csv_text
            for csv_text, ranks in grouped_tasks.items()
        }
        for fut in as_completed(future_map):
            one = fut.result()
            stats_total["task_ok"] += int(one.get("task_ok", 0))
            stats_total["task_failed"] += int(one.get("task_failed", 0))

    logging.info(
        "Crossmatch重跑完成: 扫描CSV=%d, 命中行=%d, 无效rank=%d, 重复rank=%d, 跳过大CSV=%d, "
        "任务总数=%d, 成功=%d, 失败=%d",
        stats_total["csv_scanned"],
        stats_total["matched_rows"],
        stats_total["invalid_rank_rows"],
        stats_total["dedup_rank_rows"],
        stats_total["skipped_large_csv_count"],
        stats_total["task_total"],
        stats_total["task_ok"],
        stats_total["task_failed"],
    )


if __name__ == "__main__":
    main()

