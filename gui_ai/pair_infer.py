#!/usr/bin/env python3
"""PairRegNet 加载与瓦片批量推理。

模型：models_256_final/best.pt（n_cls=1 appear 检测头 + n_mask=3 掩码头）。
输入：一对同网格的 256×256 灰度瓦片 (A=参考, B=新报告重投影后)。
输出：每瓦片的 det 峰值、相对位姿 (dx,dy,roll)、掩码。

模型自带的 preprocess()（逐帧 1%/99.5% 百分位拉伸）为网络必需，不可省略；
此处仅做“灰度->uint8”这一步，不涉及校准/flat/去噪等本地预处理。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent


def _load_model_module():
    """加载本目录内的 model.py。"""
    model_py = BASE_DIR / "model.py"
    spec = importlib.util.spec_from_file_location("pair_reg_model", str(model_py))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模型代码: {model_py}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pair_reg_model"] = mod
    spec.loader.exec_module(mod)
    return mod


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def to_uint8(arr: np.ndarray) -> np.ndarray:
    """百分位(1/99.5)拉伸到 uint8；NaN 视为背景。"""
    x = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros(x.shape, dtype=np.uint8)
    valid = x[finite]
    lo = float(np.percentile(valid, 1.0))
    hi = float(np.percentile(valid, 99.5))
    if hi - lo <= 1e-6:
        lo = float(np.min(valid))
        hi = float(np.max(valid))
    if hi - lo <= 1e-6:
        return np.zeros(x.shape, dtype=np.uint8)
    norm = (np.nan_to_num(x, nan=lo) - lo) / (hi - lo)
    return np.clip(norm * 255.0, 0.0, 255.0).astype(np.uint8)


class PairModel:
    """封装模型加载、批量瓦片推理与峰值提取。"""

    def __init__(
        self,
        model_dir: str | Path,
        device: str = "auto",
        det_threshold: float = 0.35,
        batch_size: int = 16,
    ) -> None:
        self.mod = _load_model_module()
        self.model_dir = Path(model_dir)
        self.device = resolve_device(device)
        self.det_threshold = float(det_threshold)
        self.batch_size = max(1, int(batch_size))
        self.model_size = self._resolve_model_size(self.model_dir)
        ckpt = self.model_dir / "best.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"模型权重不存在: {ckpt}")
        obj = torch.load(str(ckpt), map_location=self.device)
        sd = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
        self.net = self.mod.build_model_from_state(sd)
        self.net.to(self.device)
        self.net.eval()

    @staticmethod
    def _resolve_model_size(model_dir: Path) -> int:
        side = model_dir / "best_info.json"
        if side.exists():
            try:
                return int(json.loads(side.read_text(encoding="utf-8"))["model_in"])
            except Exception:
                pass
        return 256

    def _prepare(self, a_u8: np.ndarray, b_u8: np.ndarray) -> np.ndarray:
        """(256,256) uint8 对 -> (2,H,W) float 张量。"""
        import cv2

        size = self.model_size
        if a_u8.shape != (size, size):
            a_u8 = cv2.resize(a_u8, (size, size), interpolation=cv2.INTER_AREA)
        if b_u8.shape != (size, size):
            b_u8 = cv2.resize(b_u8, (size, size), interpolation=cv2.INTER_AREA)
        return self.mod.preprocess(a_u8, b_u8)

    @torch.no_grad()
    def infer_tiles(
        self,
        a_tiles_u8: List[np.ndarray],
        b_tiles_u8: List[np.ndarray],
        tile_size: int,
    ) -> List[Dict]:
        """对一批瓦片推理，返回每个瓦片的结果字典。"""
        results: List[Dict] = []
        n = len(a_tiles_u8)
        scale = float(tile_size) / float(self.model_size)

        for i in range(0, n, self.batch_size):
            a_chunk = a_tiles_u8[i : i + self.batch_size]
            b_chunk = b_tiles_u8[i : i + self.batch_size]
            pair = torch.stack(
                [self._prepare(a, b) for a, b in zip(a_chunk, b_chunk)]
            ).to(self.device)

            pose, det, ob = self.net(pair)
            dxdy = self.mod.decode(pose).cpu().numpy()  # (N,3) dx,dy,roll(deg)
            peaks_per = (
                self.mod.heat_to_peaks(torch.sigmoid(det), thresh=self.det_threshold)
                if det is not None
                else [[] for _ in range(len(a_chunk))]
            )
            masks = torch.sigmoid(ob).cpu().numpy() if ob is not None else None

            for j in range(len(a_chunk)):
                peaks = peaks_per[j]
                masks_j = masks[j] if masks is not None else None
                # 仅保留卫星通道（掩码通道 2 = B-satellite）
                sat = None
                if masks_j is not None and masks_j.shape[0] >= 3:
                    sat = masks_j[2] > 0.5

                kept = []
                for cls, xm, ym, sc in peaks:
                    status = "keep"
                    if sat is not None:
                        xi = min(masks_j.shape[2] - 1, max(0, int(round(xm))))
                        yi = min(masks_j.shape[1] - 1, max(0, int(round(ym))))
                        if sat[yi, xi]:
                            status = "satellite"
                    kept.append(
                        {
                            "cls": int(cls),
                            "x": float(xm * scale),
                            "y": float(ym * scale),
                            "score": float(sc),
                            "status": status,
                        }
                    )

                results.append(
                    {
                        "peaks": kept,
                        "dx": float(dxdy[j, 0]) * scale,
                        "dy": float(dxdy[j, 1]) * scale,
                        "roll": float(dxdy[j, 2]),
                        "masks": masks_j,
                    }
                )
        return results
