# -*- coding: utf-8 -*-
"""经典时间序列FCN：三层一维卷积、批归一化和全局平均池化。"""

from __future__ import annotations

import torch
import torch.nn as nn


class CNN1D(nn.Module):
    """Wang等人的FCN基线的PyTorch一维实现。"""

    def __init__(self, in_channels: int, dropout: float, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            self._block(in_channels, 128, 8),
            self._block(128, 256, 5),
            self._block(256, 128, 3),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )
        self._initialize()

    @staticmethod
    def _block(in_channels: int, out_channels: int, kernel_size: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv1d(
                in_channels, out_channels, kernel_size,
                padding="same", bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(x)).squeeze(-1))
