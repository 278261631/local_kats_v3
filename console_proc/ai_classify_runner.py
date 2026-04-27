#!/usr/bin/env python3
"""
console_proc 独立 AI 分类执行器（按筛选命中批量回写 ai_class/ai_label）。
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from astropy.io import fits
from PIL import Image
import torch
from torchvision.models import ResNet18_Weights, resnet18

from filtered_tools_common import (
    discover_candidate_csvs,
    find_primary_aligned_fits_in_output_dir,
    load_csv_rows,
    load_filter_profile,
    load_json,
    row_matches_filter,
    try_get_float_from_row,
    validate_filter_profile,
    write_csv_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按筛选配置批量执行 AI 分类并回写 variable_candidates_nonref_only_inner_border.csv"
    )
    parser.add_argument("--config", default="console_proc/config.json", help="配置文件路径")
    parser.add_argument("--date", required=True, help="日期，YYYYMMDD")
    parser.add_argument("--telescope", help="仅处理指定系统，例如 GY1")
    parser.add_argument("--region", help="仅处理指定天区，例如 K019")
    parser.add_argument("--profile", default="A", help="筛选配置名，默认 A")
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


class ResnetJoblibClassifier:
    def __init__(self, model_path: Path) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"AI 模型文件不存在: {model_path}")
        saved = joblib.load(str(model_path))
        model = saved.get("model")
        classes = saved.get("classes", [])
        if model is None:
            raise RuntimeError("模型文件缺少 model")
        self.model = model
        self.classes = [str(c).strip().lower() for c in list(classes or [])]

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        weights = ResNet18_Weights.IMAGENET1K_V1
        self.backbone = resnet18(weights=weights)
        self.backbone.fc = torch.nn.Identity()
        self.backbone.eval()
        self.backbone.to(self.device)
        self.preprocess = weights.transforms()

    def predict_patch_u8(self, patch_u8: np.ndarray) -> Tuple[int, str]:
        img = Image.fromarray(patch_u8.astype(np.uint8), mode="L").convert("RGB")
        x = self.preprocess(img).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            feat = self.backbone(x).cpu().numpy().astype(np.float32)

        pred = str(self.model.predict(feat)[0]).strip()
        label = pred.lower()
        if label == "good":
            return 1, pred
        neg_labels = [c for c in self.classes if c and c != "good"]
        if label in neg_labels:
            return -(neg_labels.index(label) + 1), pred
        return -99, pred


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
    except Exception:
        return None, None


def resolve_candidate_pixel_xy(row: Dict[str, Any], header: Any) -> Tuple[Optional[float], Optional[float]]:
    x = try_get_float_from_row(row, ["x", "pixel_x", "x_px", "xpix", "target_x", "cx", "col", "img_x"])
    y = try_get_float_from_row(row, ["y", "pixel_y", "y_px", "ypix", "target_y", "cy", "row", "img_y"])
    if x is not None and y is not None:
        return x, y

    ra = try_get_float_from_row(row, ["ra", "ra_deg", "ra_degree", "target_ra"])
    dec = try_get_float_from_row(row, ["dec", "dec_deg", "dec_degree", "target_dec"])
    if ra is None or dec is None:
        return None, None
    if header is None:
        return None, None
    try:
        from astropy.wcs import WCS

        wcs = WCS(header)
        pixel_coords = wcs.all_world2pix([[ra, dec]], 0)
        return float(pixel_coords[0][0]), float(pixel_coords[0][1])
    except Exception:
        return None, None


def extract_local_patch(data: np.ndarray, x: float, y: float, half_size: int) -> Optional[np.ndarray]:
    if data is None:
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


def stretch_patch_to_uint8(patch: np.ndarray) -> np.ndarray:
    arr = np.asarray(patch, dtype=np.float64)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros(arr.shape, dtype=np.uint8)

    valid = arr[finite]
    lo = float(np.percentile(valid, 0.5))
    hi = float(np.percentile(valid, 99.95))
    if hi <= lo:
        hi = lo + 1e-8
    clipped = np.clip(arr, lo, hi)
    norm = (clipped - lo) / (hi - lo)
    norm = np.power(np.clip(norm, 0.0, 1.0), 0.75)
    norm = np.arcsinh(norm * 8.0) / np.arcsinh(8.0)
    out = np.zeros_like(arr, dtype=np.float64)
    out[finite] = np.clip(norm[finite], 0.0, 1.0)
    return np.round(out * 255.0).astype(np.uint8)


def process_one_csv(
    csv_path: Path,
    profile: Dict[str, Any],
    classifier: Optional[ResnetJoblibClassifier],
    patch_half_size: int,
    dry_run: bool,
) -> Dict[str, int]:
    rows = load_csv_rows(csv_path)
    output_dir = csv_path.parent
    skip_large_csv = bool(profile.get("skip_large_csv", False))
    large_csv_max_rows = int(profile.get("large_csv_max_rows", 200))
    if skip_large_csv and len(rows) > large_csv_max_rows:
        return {
            "csv_count": 1,
            "hit_rows": 0,
            "processed_rows": 0,
            "predicted_rows": 0,
            "fallback_zero_rows": 0,
            "written_csv": 0,
            "skipped_large_csv": 1,
        }

    aligned_fits = find_primary_aligned_fits_in_output_dir(output_dir)
    data = None
    header = None
    if aligned_fits is not None:
        data, header = load_fits_image_and_header(aligned_fits)

    changed = False
    hit_rows = 0
    processed_rows = 0
    predicted_rows = 0
    fallback_zero_rows = 0
    for row in rows:
        if not row_matches_filter(row, profile):
            continue
        hit_rows += 1
        if dry_run:
            continue

        ai_class = 0
        ai_label = "error_no_aligned_fits"
        try:
            if classifier is None:
                ai_class, ai_label = 0, "error_no_model"
            elif data is None:
                ai_class, ai_label = 0, "error_no_aligned_fits"
            else:
                x, y = resolve_candidate_pixel_xy(row, header)
                if x is None or y is None:
                    ai_class, ai_label = 0, "error_xy"
                else:
                    patch = extract_local_patch(data, x, y, patch_half_size)
                    if patch is None:
                        ai_class, ai_label = 0, "error_patch"
                    else:
                        patch_u8 = stretch_patch_to_uint8(patch)
                        ai_class, ai_label = classifier.predict_patch_u8(patch_u8)
                        predicted_rows += 1
        except Exception:
            ai_class, ai_label = 0, "error_predict"

        processed_rows += 1
        if ai_class == 0:
            fallback_zero_rows += 1

        if str(row.get("ai_class", "")).strip() != str(int(ai_class)) or str(row.get("ai_label", "")).strip() != str(ai_label):
            row["ai_class"] = str(int(ai_class))
            row["ai_label"] = str(ai_label)
            changed = True

    if changed and not dry_run:
        if "ai_class" not in rows[0]:
            for r in rows:
                r.setdefault("ai_class", "0")
        if "ai_label" not in rows[0]:
            for r in rows:
                r.setdefault("ai_label", "")
        write_csv_rows(csv_path, rows)
        written_csv = 1
    else:
        written_csv = 0

    return {
        "csv_count": 1,
        "hit_rows": hit_rows,
        "processed_rows": processed_rows,
        "predicted_rows": predicted_rows,
        "fallback_zero_rows": fallback_zero_rows,
        "written_csv": written_csv,
        "skipped_large_csv": 0,
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
    cfg_base = cfg_path.parent

    paths_cfg = cfg.get("paths", {})
    diff_output_root = Path(str(paths_cfg.get("diff_output_root", "")))
    if not diff_output_root.exists():
        raise SystemExit(f"diff 输出目录不存在: {diff_output_root}")

    profile = load_filter_profile(cfg, args.profile)
    validate_filter_profile(profile)
    logging.info("使用筛选配置 %s: %s", profile["name"], profile)

    ai_tools_cfg = cfg.get("ai_tools", {})
    model_path = resolve_path(str(ai_tools_cfg.get("model_path", "gui/classifier_model.joblib")), cfg_base)
    patch_half_size = int(ai_tools_cfg.get("patch_half_size", 50))
    max_workers = int(args.max_workers) if args.max_workers > 0 else int(ai_tools_cfg.get("max_workers", 1))
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
    logging.info("待扫描 CSV: %d", len(csv_paths))

    classifier: Optional[ResnetJoblibClassifier] = None
    if not args.dry_run:
        classifier = ResnetJoblibClassifier(model_path=model_path)

    stats_total = {
        "csv_count": 0,
        "hit_rows": 0,
        "processed_rows": 0,
        "predicted_rows": 0,
        "fallback_zero_rows": 0,
        "written_csv": 0,
        "skipped_large_csv": 0,
    }

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(
                process_one_csv,
                csv_path,
                profile,
                classifier,
                patch_half_size,
                bool(args.dry_run),
            )
            for csv_path in csv_paths
        ]
        for fut in as_completed(futures):
            item = fut.result()
            for k in stats_total.keys():
                stats_total[k] += int(item.get(k, 0))

    logging.info(
        "AI分类完成: csv=%d, 命中=%d, 处理=%d, 成功分类=%d, 回写0=%d, 写回CSV=%d, 跳过大CSV=%d, dry_run=%s",
        stats_total["csv_count"],
        stats_total["hit_rows"],
        stats_total["processed_rows"],
        stats_total["predicted_rows"],
        stats_total["fallback_zero_rows"],
        stats_total["written_csv"],
        stats_total["skipped_large_csv"],
        bool(args.dry_run),
    )


if __name__ == "__main__":
    main()

