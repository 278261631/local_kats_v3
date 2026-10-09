#!/usr/bin/env python3
"""控制台批量处理（无 UI）：默认处理数据源根目录下所有 FITS，带进度显示。

用法:
    python cli_process.py                 # 用 settings.json 的数据源/参数
    python cli_process.py --root E:/fix_data/download
    python cli_process.py --force         # 不跳过已有结果
"""

from __future__ import annotations

import argparse
import os
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from config import DEFAULTS, load_settings  # noqa: E402
from pair_infer import PairModel  # noqa: E402
from pipeline import collect_fits_files, process_b_file  # noqa: E402
import results_io  # noqa: E402


def _bar(done: int, total: int, width: int = 30) -> str:
    total = max(1, total)
    frac = min(1.0, done / total)
    n = int(round(frac * width))
    return "[" + "#" * n + "-" * (width - n) + f"] {done}/{total} {frac*100:5.1f}%"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=None, help="数据源根目录")
    ap.add_argument("--template", default=None, help="模板根目录")
    ap.add_argument("--model", default=None, help="模型目录")
    ap.add_argument("--device", default=None, help="auto/cpu/cuda")
    ap.add_argument("--threshold", type=float, default=None, help="检测阈值")
    ap.add_argument("--batch", type=int, default=None, help="批大小")
    ap.add_argument("--force", action="store_true", help="不跳过已有结果")
    args = ap.parse_args()

    st = load_settings()

    def pick(key, arg, default):
        if arg is not None:
            return arg
        return st.get(key, default)

    root = pick("root", args.root, DEFAULTS["root"])
    template_root = pick("template_root", args.template, DEFAULTS["template_root"])
    model_dir = pick("model_dir", args.model, DEFAULTS["model_dir"])
    device = pick("device", args.device, DEFAULTS["device"])
    threshold = float(pick("det_threshold", args.threshold, DEFAULTS["det_threshold"]))
    batch = int(pick("batch_size", args.batch, DEFAULTS["batch_size"]))
    overlap = float(st.get("overlap", DEFAULTS["overlap"]))
    tile = int(st.get("tile_size", DEFAULTS["tile_size"]))
    fill = bool(st.get("fill_invalid_with_a", DEFAULTS["fill_invalid_with_a"]))
    valid_overlap = bool(st.get("valid_overlap_filter", DEFAULTS["valid_overlap_filter"]))
    skip = DEFAULTS["skip_existing"] and not args.force

    if not os.path.isdir(root):
        print(f"[错误] 数据源目录不存在: {root}")
        return 2
    if not os.path.isdir(template_root):
        print(f"[错误] 模板根目录不存在: {template_root}")
        return 2

    files = collect_fits_files(root)
    print("=" * 70)
    print(f"数据源: {root}")
    print(f"模板  : {template_root}")
    print(f"模型  : {model_dir}  设备={device}  阈值={threshold}  batch={batch}")
    print(f"待处理 B 文件: {len(files)}  跳过已有结果={skip}")
    print("=" * 70)
    if not files:
        return 0

    model = None
    ab_model = None
    ab_filter = bool(st.get("ab_filter", DEFAULTS["ab_filter"]))
    ab_keep = st.get("ab_keep_classes", DEFAULTS["ab_keep_classes"])
    ab_noise_max = float(st.get("ab_noise_max", DEFAULTS["ab_noise_max"]))
    ab_shift_max = float(st.get("ab_pixelshift_max", DEFAULTS["ab_pixelshift_max"]))
    anomaly_min_keep = int(st.get("file_anomaly_min_keep", DEFAULTS["file_anomaly_min_keep"]))
    if ab_filter:
        print(f"A/B分类过滤: 开启 (noise<={ab_noise_max}, pixelshift<={ab_shift_max})")
    print(f"整文件异常判定: n_keep >= {anomaly_min_keep}")
    n_ok = n_skip = n_fail = 0
    t_start = time.perf_counter()
    for i, f in enumerate(files, 1):
        name = os.path.basename(f)
        prefix = f"[{i}/{len(files)}]"
        if skip and os.path.exists(results_io.result_json_path(f)):
            n_skip += 1
            print(f"{prefix} 跳过(已有结果) {name[:60]}")
            continue
        t0 = time.perf_counter()
        try:
            if model is None:
                model = PairModel(model_dir, device=device,
                                  det_threshold=threshold, batch_size=batch)
            if ab_filter and ab_model is None:
                from ab_classify import ABModel
                ab_model = ABModel(
                    st.get("ab_model_dir", DEFAULTS["ab_model_dir"]), device=device)
                print(f"A/B分类器已加载 (类别={ab_model.classes})")
            res = process_b_file(
                f, model, template_root=template_root, tile_size=tile,
                overlap=overlap, dedup_radius=float(DEFAULTS["dedup_radius"]),
                reproject_chunk_rows=int(DEFAULTS["reproject_chunk_rows"]),
                fill_invalid_with_a=fill, valid_overlap_filter=valid_overlap,
                snr_filter=bool(st.get("snr_filter", DEFAULTS["snr_filter"])),
                shading_k=float(st.get("shading_k", DEFAULTS["shading_k"])),
                b2_ksize=int(st.get("b2_ksize", DEFAULTS["b2_ksize"])),
                noise_k=float(st.get("noise_k", DEFAULTS["noise_k"])),
                aperture_radius=int(st.get("aperture_radius", DEFAULTS["aperture_radius"])),
                aperture_annulus=int(st.get("aperture_annulus", DEFAULTS["aperture_annulus"])),
                aperture_snr_min=float(st.get("aperture_snr_min", DEFAULTS["aperture_snr_min"])),
                det_center_search=int(st.get("det_center_search", DEFAULTS["det_center_search"])),
                edge_band=int(st.get("edge_band", DEFAULTS["edge_band"])),
                boundary_scale=int(st.get("boundary_scale", DEFAULTS["boundary_scale"])),
                isolated_filter=bool(st.get("isolated_filter", DEFAULTS["isolated_filter"])),
                shape_conc_max=float(st.get("shape_conc_max", DEFAULTS["shape_conc_max"])),
                shape_fwhm_min=float(st.get("shape_fwhm_min", DEFAULTS["shape_fwhm_min"])),
                shape_fwhm_max=float(st.get("shape_fwhm_max", DEFAULTS["shape_fwhm_max"])),
                median_filter=bool(st.get("median_filter", DEFAULTS["median_filter"])),
                median_ksize=int(st.get("median_ksize", DEFAULTS["median_ksize"])),
                ab_model=ab_model,
                ab_filter=ab_filter,
                ab_keep_classes=ab_keep,
                ab_noise_max=ab_noise_max,
                ab_pixelshift_max=ab_shift_max,
                ab_patch=int(st.get("ab_patch", DEFAULTS["ab_patch"])),
                anomaly_min_keep=anomaly_min_keep,
            )
            if res.get("error"):
                n_fail += 1
                print(f"{prefix} {name[:60]}  {res['error']}")
                continue
            results_io.save_result(res, st)
            n_ok += 1
            mark = "  [异常图像]" if res.get("anomaly") else ""
            print(f"{prefix} {name[:60]}  命中 {res['n_keep']}/{res['n_total']}"
                  f"  {time.perf_counter()-t0:.1f}s{mark}")
        except Exception as ex:  # noqa: BLE001
            n_fail += 1
            print(f"{prefix} {name[:60]}  失败: {ex}")

    print(_bar(len(files), len(files)))
    print(f"完成: 成功 {n_ok}, 跳过 {n_skip}, 失败 {n_fail}, 用时 {time.perf_counter()-t_start:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
