#!/usr/bin/env python3
"""
console_proc 独立 PairRegNet 执行器（按候选裁 reference/aligned 小块推理）。

参考 continuous-learning-diff-area/data_viewer/viewer.py 的用法：
对 CSV 中每个候选 (x, y)，分别从
    A = 模板参考帧（reference，从 .done.json 或 pipeline_commands.txt 解析）
    B = 对齐目标帧（aligned，*.02rp.fit）
裁一块同中心的 cutout_size×cutout_size 原生小图，缩放到模型输入尺寸(256)后送模型。

回写列：
    pair_class  : 命中检测类别（0=appear / 1=dim / 2=satellite），低于阈值或无法推理为 -1
    pair_score  : 该类别的检测分数
    pair_dx     : 局部位姿 x（原生像素，cutout 尺度换算）
    pair_dy     : 局部位姿 y（原生像素）
    pair_droll  : 局部滚转角（度）

说明：
    * 不做 WCS 解析；候选坐标直接取 CSV 的 x/y。
    * A、B 共用同一像素网格，所以同一 (x,y) 裁两块即可。
    * 不做垂直翻转。
    * 不修改网页 ZIP 导出（export_filtered_web_zip_runner）。
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.util
import json
import logging
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from astropy.io import fits

from filtered_tools_common import (
    discover_candidate_csvs,
    ensure_csv_default_fields,
    find_primary_aligned_fits_in_output_dir,
    load_csv_rows,
    load_filter_profile,
    load_json,
    try_get_float_from_row,
    validate_filter_profile,
    write_csv_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按 output 目录对每个候选裁 reference/aligned 小块运行 PairRegNet，回写 pair_* 列"
    )
    parser.add_argument("--config", default="console_proc/config.json", help="配置文件路径")
    parser.add_argument("--date", required=True, help="日期，YYYYMMDD")
    parser.add_argument("--telescope", help="仅处理指定系统，例如 GY1")
    parser.add_argument("--region", help="仅处理指定天区，例如 K019")
    parser.add_argument("--profile", default="A", help="筛选配置名（仅用于跳过大CSV阈值），默认 A")
    parser.add_argument("--max-csv", type=int, default=0, help="最多处理 CSV 数量，0=不限制")
    parser.add_argument("--max-workers", type=int, default=0, help="并发线程数，0=使用配置")
    parser.add_argument("--dry-run", action="store_true", help="仅统计，不回写")
    parser.add_argument("--verbose", action="store_true", help="输出调试日志")
    return parser.parse_args()


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")


def validate_date(date_text: str) -> bool:
    return bool(re.fullmatch(r"\d{8}", date_text))


def resolve_path(path_text: str, base_dir: Path) -> Path:
    p = Path(str(path_text))
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def load_model_module(model_dir: Path):
    """动态加载 models 目录下的 model.py。"""
    model_py = model_dir / "model.py"
    if not model_py.exists():
        raise FileNotFoundError(f"PairRegNet 模型代码不存在: {model_py}")
    spec = importlib.util.spec_from_file_location("pair_reg_model", str(model_py))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模型模块: {model_py}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pair_reg_model"] = mod
    spec.loader.exec_module(mod)
    return mod


def resolve_model_size(model_dir: Path, configured: int) -> int:
    """模型输入网格尺寸：优先配置，其次 best_info.json，最后 256。"""
    if isinstance(configured, int) and configured > 0:
        return int(configured)
    side = model_dir / "best_info.json"
    if side.exists():
        try:
            return int(json.loads(side.read_text(encoding="utf-8"))["model_in"])
        except Exception:
            pass
    return 256


def load_pair_model(model_dir: Path, device: str):
    """加载 best.pt（纯 state_dict 或含 model 的 checkpoint）。"""
    mod = load_model_module(model_dir)
    ckpt = model_dir / "best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"PairRegNet 权重不存在: {ckpt}")
    obj = torch.load(str(ckpt), map_location=device)
    sd = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    net = mod.build_model_from_state(sd)
    net.to(device)
    net.eval()
    return mod, net


def resolve_template_file(output_dir: Path) -> Optional[Path]:
    """解析模板参考帧（A）路径。

    优先读取 .done.json 的 template_file；若不存在则回退解析同目录下
    pipeline_commands.txt 的 "# Template FITS:" 行（兼容没有 .done.json 的输出目录）。
    """
    done = output_dir / ".done.json"
    if done.exists():
        try:
            data = json.loads(done.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        tf = data.get("template_file")
        if tf:
            p = Path(str(tf))
            if p.exists():
                return p

    cmd_txt = output_dir / "pipeline_commands.txt"
    if cmd_txt.exists():
        try:
            text = cmd_txt.read_text(encoding="utf-8", errors="replace")
        except Exception:
            text = ""
        m = re.search(r"^#\s*Template FITS:\s*(.+?)\s*$", text, flags=re.MULTILINE)
        if m:
            p = Path(m.group(1).strip())
            if p.exists():
                return p
        m = re.search(r"--fits\s+(\S+?\.fits?)", text)
        if m:
            p = Path(m.group(1).strip().strip('"'))
            if p.exists():
                return p
    return None


def load_fits_gray(path: Path) -> Optional[np.ndarray]:
    try:
        with fits.open(str(path)) as hdul:
            data = hdul[0].data
    except Exception:
        return None
    if data is None:
        return None
    arr = data.astype(np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2:
        return None
    return arr


def to_uint8(arr: np.ndarray) -> np.ndarray:
    """百分位拉伸到 uint8（与参考 viewer._cutout_to_u8 一致：1.0/99.5）。"""
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros(arr.shape, dtype=np.uint8)
    valid = arr[finite]
    lo = float(np.percentile(valid, 1.0))
    hi = float(np.percentile(valid, 99.5))
    if hi - lo <= 1e-6:
        lo = float(np.min(valid))
        hi = float(np.max(valid))
    if hi - lo <= 1e-6:
        return np.zeros(arr.shape, dtype=np.uint8)
    norm = (np.nan_to_num(arr, nan=lo) - lo) / (hi - lo)
    return np.clip(norm * 255.0, 0.0, 255.0).astype(np.uint8)


def cutout(img: np.ndarray, cx: float, cy: float, size: int) -> np.ndarray:
    """以 (cx, cy) 为中心裁 size×size 原生小块，越界用边缘填充。"""
    size = int(size)
    half = size // 2
    x0 = int(round(float(cx))) - half
    y0 = int(round(float(cy))) - half
    x1 = x0 + size
    y1 = y0 + size
    h, w = img.shape

    pad_l = max(0, -x0)
    pad_t = max(0, -y0)
    pad_r = max(0, x1 - w)
    pad_b = max(0, y1 - h)
    xs = max(0, x0)
    ys = max(0, y0)
    xe = min(w, x1)
    ye = min(h, y1)

    out = img[ys:ye, xs:xe]
    if out.size == 0:
        return np.zeros((size, size), dtype=img.dtype)
    if pad_l or pad_r or pad_t or pad_b:
        out = np.pad(out, ((pad_t, pad_b), (pad_l, pad_r)), mode="edge")
    return out


def infer_candidate(
    a_arr: np.ndarray,
    b_arr: np.ndarray,
    x: float,
    y: float,
    model_mod,
    model,
    device: str,
    model_size: int,
    cutout_size: int,
    det_threshold: float,
) -> Tuple[int, float, Optional[float], Optional[float], Optional[float]]:
    """对单个候选裁块推理。

    Returns: (pair_class, pair_score, pair_dx, pair_dy, pair_droll)
    """
    import cv2

    a_cut = to_uint8(cutout(a_arr, x, y, cutout_size).astype(np.float32))
    b_cut = to_uint8(cutout(b_arr, x, y, cutout_size).astype(np.float32))
    a_r = cv2.resize(a_cut, (model_size, model_size), interpolation=cv2.INTER_AREA)
    b_r = cv2.resize(b_cut, (model_size, model_size), interpolation=cv2.INTER_AREA)

    pair_t = model_mod.preprocess(a_r, b_r).unsqueeze(0).to(device)
    with torch.no_grad():
        pose, det = model(pair_t)
        dx_m, dy_m, roll_deg = model_mod.decode(pose)[0].tolist()
        if det is not None:
            prob = torch.sigmoid(det)[0]
            conf = prob.reshape(prob.shape[0], -1).amax(dim=1).tolist()
        else:
            conf = []

    scale = float(model_size) / float(max(1, int(cutout_size)))
    pair_dx = float(dx_m) / scale
    pair_dy = float(dy_m) / scale
    pair_droll = float(roll_deg)

    if conf:
        pair_class = int(max(range(len(conf)), key=lambda i: conf[i]))
        pair_score = float(conf[pair_class])
        if pair_score < float(det_threshold):
            pair_class = -1
    else:
        pair_class, pair_score = -1, 0.0
    return pair_class, pair_score, pair_dx, pair_dy, pair_droll


def process_one_csv(
    csv_path: Path,
    model_mod,
    model,
    device: str,
    model_size: int,
    cutout_size: int,
    det_threshold: float,
    profile: Dict[str, Any],
    dry_run: bool,
) -> Dict[str, int]:
    stats = {
        "csv_count": 1,
        "written_csv": 0,
        "skipped_large_csv": 0,
        "processed_rows": 0,
        "matched_rows": 0,
        "error_frames": 0,
    }

    if not dry_run:
        ensure_csv_default_fields(csv_path)
    rows = load_csv_rows(csv_path)
    if not rows:
        return stats

    skip_large_csv = bool(profile.get("skip_large_csv", False))
    large_csv_max_rows = int(profile.get("large_csv_max_rows", 200))
    if skip_large_csv and len(rows) > large_csv_max_rows:
        stats["skipped_large_csv"] = 1
        return stats

    output_dir = csv_path.parent
    template_file = resolve_template_file(output_dir)
    aligned_fits = find_primary_aligned_fits_in_output_dir(output_dir)
    if template_file is None or aligned_fits is None:
        logging.warning(
            "跳过（缺少模板或对齐帧）: %s (template=%s, aligned=%s)",
            output_dir,
            template_file,
            aligned_fits,
        )
        stats["error_frames"] = 1
        return stats

    if dry_run:
        return stats

    a_arr = load_fits_gray(template_file)
    b_arr = load_fits_gray(aligned_fits)
    if a_arr is None or b_arr is None:
        logging.warning("跳过（读取帧失败）: %s", output_dir)
        stats["error_frames"] = 1
        return stats

    changed = False
    for row in rows:
        x = try_get_float_from_row(row, ["x", "pixel_x", "x_px", "xpix", "target_x", "cx", "col", "img_x"])
        y = try_get_float_from_row(row, ["y", "pixel_y", "y_px", "ypix", "target_y", "cy", "row", "img_y"])
        if x is None or y is None:
            pair_class, pair_score = -1, 0.0
            pair_dx = pair_dy = pair_droll = None
        else:
            try:
                pair_class, pair_score, pair_dx, pair_dy, pair_droll = infer_candidate(
                    a_arr, b_arr, x, y, model_mod, model, device,
                    model_size, cutout_size, det_threshold,
                )
            except Exception as ex:
                logging.debug("候选推理失败: %s", ex)
                pair_class, pair_score = -1, 0.0
                pair_dx = pair_dy = pair_droll = None

        stats["processed_rows"] += 1
        if pair_class >= 0:
            stats["matched_rows"] += 1

        new_class = str(int(pair_class))
        new_score = f"{pair_score:.6f}"
        new_dx = "" if pair_dx is None else f"{pair_dx:.4f}"
        new_dy = "" if pair_dy is None else f"{pair_dy:.4f}"
        new_droll = "" if pair_droll is None else f"{pair_droll:.4f}"
        if (
            str(row.get("pair_class", "")).strip() != new_class
            or str(row.get("pair_score", "")).strip() != new_score
            or str(row.get("pair_dx", "")).strip() != new_dx
            or str(row.get("pair_dy", "")).strip() != new_dy
            or str(row.get("pair_droll", "")).strip() != new_droll
        ):
            row["pair_class"] = new_class
            row["pair_score"] = new_score
            row["pair_dx"] = new_dx
            row["pair_dy"] = new_dy
            row["pair_droll"] = new_droll
            changed = True

    if changed:
        write_csv_rows(csv_path, rows)
        stats["written_csv"] = 1

    return stats


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    if not validate_date(args.date):
        raise SystemExit(f"日期格式错误: {args.date}，应为 YYYYMMDD")

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise SystemExit(f"配置文件不存在: {cfg_path}")
    cfg = load_json(cfg_path)
    cfg_base = cfg_path.parent

    paths_cfg = cfg.get("paths", {})
    diff_output_root = Path(str(paths_cfg.get("diff_output_root", "")))
    if not diff_output_root.exists():
        raise SystemExit(f"diff 输出目录不存在: {diff_output_root}")

    profile = load_filter_profile(cfg, args.profile)
    validate_filter_profile(profile)

    pair_cfg = cfg.get("pair_tools", {})
    model_dir = resolve_path(str(pair_cfg.get("model_dir", "../gui/models_tr")), cfg_base)
    if not model_dir.exists():
        raise SystemExit(f"PairRegNet 模型目录不存在: {model_dir}")
    configured_model_in = int(pair_cfg.get("model_in", 0) or 0)
    model_size = resolve_model_size(model_dir, configured_model_in)
    cutout_size = int(pair_cfg.get("cutout_size", 128) or 128)
    det_threshold = float(pair_cfg.get("det_threshold", 0.35))
    max_workers = int(args.max_workers) if args.max_workers > 0 else int(pair_cfg.get("max_workers", 1))
    max_workers = max(1, max_workers)

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
    logging.info(
        "待扫描 CSV: %d, 模型输入=%d, cutout=%d, 阈值=%.3f",
        len(csv_paths), model_size, cutout_size, det_threshold,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_mod = None
    model = None
    if not args.dry_run:
        model_mod, model = load_pair_model(model_dir, device)
        logging.info("PairRegNet 已加载: %s (device=%s)", model_dir, device)

    stats_total = {
        "csv_count": 0,
        "written_csv": 0,
        "skipped_large_csv": 0,
        "processed_rows": 0,
        "matched_rows": 0,
        "error_frames": 0,
    }

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(
                process_one_csv,
                csv_path,
                model_mod,
                model,
                device,
                model_size,
                cutout_size,
                det_threshold,
                profile,
                bool(args.dry_run),
            )
            for csv_path in csv_paths
        ]
        for fut in as_completed(futures):
            item = fut.result()
            for k in stats_total.keys():
                stats_total[k] += int(item.get(k, 0))

    logging.info(
        "PairRegNet 完成: csv=%d, 写回CSV=%d, 跳过大CSV=%d, 处理行=%d, 命中行=%d, 错误帧=%d, dry_run=%s",
        stats_total["csv_count"],
        stats_total["written_csv"],
        stats_total["skipped_large_csv"],
        stats_total["processed_rows"],
        stats_total["matched_rows"],
        stats_total["error_frames"],
        bool(args.dry_run),
    )


if __name__ == "__main__":
    main()
