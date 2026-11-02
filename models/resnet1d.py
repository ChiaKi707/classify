# -*- coding: utf-8 -*-
"""经典时间序列ResNet：三组8/5/3一维卷积残差块。"""

from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.residual = nn.Sequential(
            self._conv(in_channels, out_channels, 8, activate=True),
            self._conv(out_channels, out_channels, 5, activate=True),
            self._conv(out_channels, out_channels, 3, activate=False),
        )
        self.shortcut = (
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm1d(out_channels),
            )
            if in_channels != out_channels
            else nn.BatchNorm1d(in_channels)
        )
        self.activation = nn.ReLU(inplace=True)

    @staticmethod
    def _conv(
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        activate: bool,
    ) -> nn.Sequential:
        layers: list[nn.Module] = [
            nn.Conv1d(
                in_channels, out_channels, kernel_size,
                padding="same", bias=False,
            ),
            nn.BatchNorm1d(out_channels),
        ]
        if activate:
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.residual(x) + self.shortcut(x))


class ResNet1D(nn.Module):
    """Wang等人的时间序列ResNet基线的PyTorch实现。"""

    def __init__(self, in_channels: int, dropout: float, num_classes: int):
        super().__init__()
        self.blocks = nn.Sequential(
            ResidualBlock1D(in_channels, 64),
            ResidualBlock1D(64, 128),
            ResidualBlock1D(128, 128),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )
        self._initialize()

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.blocks(x)).squeeze(-1))
