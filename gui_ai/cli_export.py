#!/usr/bin/env python3
"""控制台导出网页 ZIP (V4)：读数据源下所有结果并导出。

用法:
    python cli_export.py
    python cli_export.py --root E:/fix_data/download --out E:/kats_sync
    python cli_export.py --all          # 含被过滤结果
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
import web_export  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=None, help="数据源根目录")
    ap.add_argument("--out", default=None, help="ZIP 输出根目录")
    ap.add_argument("--all", action="store_true", help="包含被过滤结果")
    ap.add_argument("--patch-size", type=int, default=None)
    ap.add_argument("--group-radius", type=float, default=None)
    ap.add_argument("--snr-split", type=float, default=10.0)
    args = ap.parse_args()

    st = load_settings()
    root = args.root or st.get("root") or DEFAULTS["root"]
    out = args.out or st.get("web_zip_root") or DEFAULTS["web_zip_root"]
    patch = args.patch_size or int(st.get("crop_size", DEFAULTS["crop_size"]))
    gr = (args.group_radius if args.group_radius is not None
          else float(st.get("web_group_radius_px", DEFAULTS["web_group_radius_px"])))

    if not os.path.isdir(root):
        print(f"[错误] 数据源目录不存在: {root}")
        return 2
    results = query_core.load_all_results(root, with_preview=False)
    print("=" * 70)
    print(f"数据源: {root}")
    print(f"结果文件: {len(results)}")
    print(f"输出根: {out}   补丁={patch}px   聚类半径={gr:g}px")
    print("=" * 70)
    if not results:
        print("没有找到结果 (*.gui_ai.json)")
        return 0

    n, zip_path, out_dir = web_export.export_results_web(
        results,
        out_root=out,
        patch_size=patch,
        keep_only=not args.all,
        tag=str(DEFAULTS["web_zip_tag"]),
        snr_split=args.snr_split,
        group_radius_px=gr,
        manual_exclude=tuple(DEFAULTS.get(
            "manual_reject_classes", ["m-noise", "m-pix-shift"])),
        log=print,
    )
    print(f"导出 {n} 个目标并打包。")
    print(f"ZIP: {zip_path}")
    print(f"目录: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
