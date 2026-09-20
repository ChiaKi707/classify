# -*- coding: utf-8 -*-
"""网络的基础模块：SE通道注意力、多尺度残差块。仅依赖PyTorch。"""

from __future__ import annotations

import torch
import torch.nn as nn


class SEBlock1d(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 全局平均响应经小网络得到0～1权重，再乘回每个特征通道。
        return x * self.gate(x).unsqueeze(-1)


class MultiScaleResidualBlock(nn.Module):
    """并行窄/中/宽感受野，融合峰尖、峰群和背景形状。"""

    def __init__(self, in_channels: int, out_channels: int, dropout: float):
        super().__init__()
        branch_channels = max(out_channels // 3, 4)
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        in_channels,
                        branch_channels,
                        kernel_size=kernel,
                        padding=kernel // 2,
                        bias=False,
                    ),
                    nn.BatchNorm1d(branch_channels),
                    nn.GELU(),
                )
                for kernel in (3, 7, 15)
            ]
        )
        merged_channels = 3 * branch_channels
        self.fuse = nn.Sequential(
            nn.Conv1d(merged_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        )
        self.activation = nn.GELU()
        self.se = SEBlock1d(out_channels)
        self.dropout = nn.Dropout1d(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 三个分支分别观察窄峰、中等宽度峰和较宽的峰群。
        merged = torch.cat([branch(x) for branch in self.branches], dim=1)
        out = self.activation(self.fuse(merged) + self.skip(x))
        return self.dropout(self.se(out))

