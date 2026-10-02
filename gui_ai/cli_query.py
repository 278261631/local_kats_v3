#!/usr/bin/env python3
"""控制台批量查询（无 UI）：默认查询数据源根目录下所有结果，带进度显示。

先全部查变星(VSX)，再全部查 MPC。用法:
    python cli_query.py
    python cli_query.py --root E:/fix_data/download
    python cli_query.py --force        # 不跳过已查询
"""

from __future__ import annotations

import argparse
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from config import DEFAULTS, load_settings  # noqa: E402
import query_core  # noqa: E402
import results_io  # noqa: E402


def _bar(done: int, total: int, width: int = 30) -> str:
    total = max(1, total)
    frac = min(1.0, done / total)
    n = int(round(frac * width))
    return "[" + "#" * n + "-" * (width - n) + f"] {done}/{total} {frac*100:5.1f}%"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=None, help="数据源根目录")
    ap.add_argument("--force", action="store_true", help="不跳过已查询")
    args = ap.parse_args()

    st = load_settings()
    root = args.root or st.get("root") or DEFAULTS["root"]
    if not os.path.isdir(root):
        print(f"[错误] 数据源目录不存在: {root}")
        return 2

    results = query_core.load_all_results(root)
    print("=" * 70)
    print(f"数据源: {root}")
    print(f"结果文件: {len(results)}")
    if not results:
        print("没有找到结果 (*.gui_ai.json)")
        return 0
    tasks, n_no_epoch = query_core.build_tasks(results)
    print(f"命中检测(可查): {len(tasks)}  (无观测时间 {n_no_epoch})")
    print("=" * 70)

    cfg = {
        "radius": float(DEFAULTS["query_radius_arcsec"]),
        "mag_limit": float(DEFAULTS["query_mag_limit"]),
        "vsx_timeout": float(DEFAULTS["vsx_timeout"]),
        "mpc_timeout": float(DEFAULTS["mpc_timeout"]),
        "threads": max(1, int(DEFAULTS.get("query_threads", 3))),
        "skip_done": bool(DEFAULTS["query_skip_done"]) and not args.force,
        "vsx_host": DEFAULTS["vsx_host"], "vsx_port": int(DEFAULTS["vsx_port"]),
        "mpc_host": DEFAULTS["mpc_host"], "mpc_port": int(DEFAULTS["mpc_port"]),
        "vsx_url": f"{DEFAULTS['vsx_host']}:{DEFAULTS['vsx_port']}",
        "mpc_url": f"{DEFAULTS['mpc_host']}:{DEFAULTS['mpc_port']}",
    }

    last_phase = {"v": None}

    def log(m: str) -> None:
        sys.stdout.write("\r" + " " * 80 + "\r")
        print(m)

    def progress(phase: str, done: int, total: int) -> None:
        if last_phase["v"] != phase:
            sys.stdout.write("\n")
            last_phase["v"] = phase
        label = "变星" if phase == "var" else "MPC "
        sys.stdout.write(f"\r{label} {_bar(done, total)}")
        sys.stdout.flush()
        if total and done >= total:
            sys.stdout.write("\n")

    query_core.run_queries(tasks, cfg, log, progress)

    n = 0
    for res in results:
        try:
            results_io.save_result(res, st)
            n += 1
        except Exception:
            pass
    print(f"已回写 {n} 个结果")
    if n_no_epoch:
        print(f"MPC 跳过: {n_no_epoch} 个目标因 B 头缺少观测时间(MJD-OBS/JD/DATE-OBS)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
