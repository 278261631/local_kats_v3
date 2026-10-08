#!/usr/bin/env python3
"""Small dual-channel CNN for the 16x16 A/B candidate classifier.

Input  : (B, 2, 16, 16)   channel 0 = A (signed difference), channel 1 = B (raw)
Optional scalar features (e.g. header SNR) are concatenated before the head.
Output : (B, n_cls) logits.
"""

from __future__ import annotations

import torch
from torch import nn


def conv_bn(cin: int, cout: int, pool: bool = False) -> nn.Sequential:
    layers = [nn.Conv2d(cin, cout, 3, padding=1, bias=False),
              nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class ABClassifier(nn.Module):
    def __init__(self, n_cls: int = 3, base: int = 32, n_extra: int = 0,
                 drop: float = 0.3) -> None:
        super().__init__()
        b = int(base)
        self.features = nn.Sequential(
            conv_bn(2, b), conv_bn(b, b, pool=True),        # 8
            conv_bn(b, b * 2), conv_bn(b * 2, b * 2, pool=True),  # 4
            conv_bn(b * 2, b * 4), conv_bn(b * 4, b * 4),    # 4
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.n_extra = int(n_extra)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(drop),
            nn.Linear(b * 4 + self.n_extra, 128), nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(128, n_cls),
        )

    def forward(self, x: torch.Tensor, extra: torch.Tensor | None = None) -> torch.Tensor:
        f = self.pool(self.features(x)).flatten(1)
        if self.n_extra > 0:
            if extra is None:
                extra = torch.zeros(x.shape[0], self.n_extra, device=x.device)
            f = torch.cat([f, extra], dim=1)
        return self.head(f)
