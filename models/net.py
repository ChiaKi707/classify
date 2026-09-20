# -*- coding: utf-8 -*-
"""模型本体：从多通道一维光谱得到[MG1, R6G]两个logit。

该文件不读取CSV、不计算损失、不设置有无阈值。
只定义网络结构与forward，其他工程也能单独导入models使用。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import MultiScaleResidualBlock


class DualPresenceMSNet(nn.Module):
    """自定义多尺度残差网络，输出 [MG1存在, R6G存在] 两个 logit。"""

    def __init__(self, in_channels: int = 3, dropout: float = 0.15):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 24, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(24),
            nn.GELU(),
        )
        self.encoder = nn.Sequential(
            MultiScaleResidualBlock(24, 32, dropout),
            nn.MaxPool1d(2),
            MultiScaleResidualBlock(32, 48, dropout),
            nn.MaxPool1d(2),
            MultiScaleResidualBlock(48, 64, dropout),
        )
        self.attention_score = nn.Conv1d(64, 1, kernel_size=1)
        self.shared = nn.Sequential(
            nn.Linear(64 * 3, 64),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.mg_head = nn.Linear(64, 1)
        self.r6g_head = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入：[批量, 输入通道, 波数点数]；模型3对应[批量, 3, 524]。
        # 两次最大池化后，524个点变成131个位置，每个位置有64维特征。
        features = self.encoder(self.stem(x))
        # 编码后同时提取平均响应、最强响应和模型重点关注区域。
        avg_pool = features.mean(dim=-1)
        max_pool = features.amax(dim=-1)
        attention = torch.softmax(self.attention_score(features), dim=-1)
        attention_pool = (features * attention).sum(dim=-1)
        shared = self.shared(torch.cat([avg_pool, max_pool, attention_pool], dim=1))
        # 输出是两个未经过 sigmoid 的 logit，顺序固定为 [MG1, R6G]。
        return torch.cat([self.mg_head(shared), self.r6g_head(shared)], dim=1)

