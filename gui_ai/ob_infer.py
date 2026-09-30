#!/usr/bin/env python3
"""单帧 OB/不可用区分割器（train_ob 的 OBNet）加载与瓦片推理。

与 PairRegNet 不同，OBNet 只看单张 256×256 帧，输出 64×64 的
不可用区(光学黑边/遮光带) 概率；这里逐瓦片推理并上采样回瓦片尺寸。
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import torch

from model_ob import OBNet


class OBModel:
    def __init__(self, model_dir: str | Path, device: str = "auto",
                 batch_size: int = 64, threshold: float = 0.5) -> None:
        self.model_dir = Path(model_dir)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.batch_size = max(1, int(batch_size))
        self.threshold = float(threshold)
        ckpt = self.model_dir / "best.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"OBNet 权重不存在: {ckpt}")
        obj = torch.load(str(ckpt), map_location=self.device)
        base = int(obj["base"]) if isinstance(obj, dict) and "base" in obj else 32
        sd = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
        self.net = OBNet(base=base)
        self.net.load_state_dict(sd)
        self.net.to(self.device)
        self.net.eval()
        self.model_size = 256

    @torch.no_grad()
    def infer_tiles(self, tiles_u8: List[np.ndarray], tile_size: int) -> List[np.ndarray]:
        """输入一批 (H,W) uint8 瓦片，返回同尺寸 uint8(0/255) 不可用区掩码。"""
        import cv2

        size = self.model_size
        out: List[np.ndarray] = []
        n = len(tiles_u8)
        for i in range(0, n, self.batch_size):
            chunk = tiles_u8[i : i + self.batch_size]
            arr = []
            for t in chunk:
                if t.shape != (size, size):
                    t = cv2.resize(t, (size, size), interpolation=cv2.INTER_AREA)
                arr.append(t.astype(np.float32) / 255.0)
            x = torch.from_numpy(np.stack(arr)[:, None]).to(self.device)
            prob = torch.sigmoid(self.net(x))[:, 0].cpu().numpy()  # (N,64,64)
            for k in range(prob.shape[0]):
                mu = cv2.resize(
                    prob[k], (tile_size, tile_size), interpolation=cv2.INTER_LINEAR
                )
                out.append(((mu > self.threshold).astype(np.uint8)) * 255)
        return out
