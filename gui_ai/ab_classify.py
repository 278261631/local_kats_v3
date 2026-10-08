#!/usr/bin/env python3
"""A/B 16x16 分类器推理封装（用于检测列表过滤）。

模型：ab16/models_ab/best.pt（ABClassifier，3 类 noise / pixelshift / target）。
输入：一对同位置的 16x16 原生素材 —— ch0 = A（模板裁块），ch1 = B（新帧裁块）；
      两者量纲不同，normalise() 各自归一到可比单位后再喂网络（与训练一致）。
输出：每个裁块的类别与概率。

与训练数据生成一致：A 取模板 a_data 在检测点 (x,y) 的原生裁块，
B 取 b_data 在对应天球坐标映射回 B 网格后的原生裁块。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent
AB_DIR = BASE_DIR / "ab16"
DEFAULT_MODEL_DIR = AB_DIR / "models_ab"


def _load_model_module():
    """加载 ab16/model_ab.py（与 gui_ai 同名模块隔离）。"""
    model_py = AB_DIR / "model_ab.py"
    spec = importlib.util.spec_from_file_location("ab_classifier_model", str(model_py))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载分类模型代码: {model_py}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ab_classifier_model"] = mod
    spec.loader.exec_module(mod)
    return mod


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _robust_scale(x: np.ndarray) -> float:
    x = np.nan_to_num(x, nan=0.0)
    s = float(np.percentile(np.abs(x), 99.0))
    if not np.isfinite(s) or s <= 1e-6:
        s = float(np.max(np.abs(x))) if np.max(np.abs(x)) > 0 else 1.0
    return s if s > 1e-6 else 1.0


def normalise(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """A（模板）与 B（新帧）各自归一到可比量纲（与 make_split_dataset 一致）。

    只处理 A/B 之间的量纲/背景差异：A 直接按稳健尺度缩放；B 先减中位数再缩放。
    """
    a = np.nan_to_num(np.asarray(a, dtype=np.float32), nan=0.0)
    a = np.clip(a / _robust_scale(a), -8.0, 8.0)
    b = np.asarray(b, dtype=np.float32)
    med = float(np.median(b)) if np.isfinite(np.median(b)) else 0.0
    b = np.nan_to_num(b, nan=med)
    b = b - med
    b = np.clip(b / _robust_scale(b), -8.0, 8.0)
    return a.astype(np.float32), b.astype(np.float32)


class ABModel:
    """封装 ABClassifier 的加载与批量推理。"""

    def __init__(
        self,
        model_dir: str | Path | None = None,
        device: str = "auto",
        batch_size: int = 128,
    ) -> None:
        self.mod = _load_model_module()
        self.model_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
        self.device = resolve_device(device)
        self.batch_size = max(1, int(batch_size))
        ckpt = self.model_dir / "best.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"分类模型权重不存在: {ckpt}")
        ck = torch.load(str(ckpt), map_location=self.device)
        sd = ck.get("model", ck) if isinstance(ck, dict) else ck
        self.classes: List[str] = list(
            ck.get("classes", ["noise", "pixelshift", "target"])
            if isinstance(ck, dict) else ["noise", "pixelshift", "target"]
        )
        self.use_snr = bool(ck.get("use_snr", False)) if isinstance(ck, dict) else False
        base = int(ck.get("base", 32)) if isinstance(ck, dict) else 32
        self.net = self.mod.ABClassifier(
            n_cls=len(self.classes), base=base,
            n_extra=1 if self.use_snr else 0,
        )
        self.net.load_state_dict(sd)
        self.net.to(self.device)
        self.net.eval()

    @torch.inference_mode()
    def classify(
        self,
        a_crops: Sequence[np.ndarray],
        b_crops: Sequence[np.ndarray],
        snrs: Optional[Sequence[float]] = None,
    ) -> List[Dict]:
        """a_crops/b_crops: 原始（未归一化）同尺寸数组；snrs: 每对 SNR（可选）。

        返回 [{"label": str, "score": float, "probs": {cls: p}}]。
        """
        n = len(a_crops)
        out: List[Dict] = []
        for i in range(0, n, self.batch_size):
            ac = a_crops[i:i + self.batch_size]
            bc = b_crops[i:i + self.batch_size]
            xs = []
            for a, b in zip(ac, bc):
                an, bn = normalise(a, b)
                xs.append(np.stack([an, bn]))
            x = torch.from_numpy(np.asarray(xs, dtype=np.float32)).to(self.device)
            extra = None
            if self.use_snr:
                sv = snrs[i:i + self.batch_size] if snrs is not None else [0.0] * len(ac)
                ev = [np.log1p(max(float(s), 0.0)) if s is not None else 0.0 for s in sv]
                extra = torch.tensor(ev, dtype=torch.float32).reshape(-1, 1).to(self.device)
            logits = self.net(x, extra)
            probs = torch.softmax(logits, 1).cpu().numpy()
            for j in range(len(ac)):
                p = probs[j]
                k = int(p.argmax())
                out.append(
                    {
                        "label": self.classes[k],
                        "score": float(p[k]),
                        "probs": {c: float(p[ci]) for ci, c in enumerate(self.classes)},
                    }
                )
        return out
