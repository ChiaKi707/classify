# -*- coding: utf-8 -*-
"""验证损失监控：最低损失权重与早停参考值分别维护。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict


# 旧规则未记录版本；版本2将权重选优与min_delta早停解耦。
# 与checkpoint格式分开，旧权重仍可推理，但不能复用为新规则实验。
TRAINING_POLICY_VERSION = 2


@dataclass(frozen=True)
class ValidationDecision:
    checkpoint_updated: bool
    early_stopping_improved: bool
    should_stop: bool


class ValidationMonitor:
    """只监控验证交叉熵；相同最低值保留最早一轮。

    best_loss追踪每次严格下降；reference_loss只在下降超过min_delta时
    更新。微小下降可以相对同一个参考值累积，且不会阻止最佳权重更新。
    """

    def __init__(self, min_delta: float, patience: int):
        if not math.isfinite(min_delta) or min_delta < 0:
            raise ValueError("min_delta必须是大于等于0的有限数")
        if not isinstance(patience, int) or isinstance(patience, bool) or patience < 1:
            raise ValueError("patience必须是大于等于1的整数")
        self.min_delta = float(min_delta)
        self.patience = patience
        self.best_loss = math.inf
        self.best_epoch = 0
        self.reference_loss = math.inf
        self.reference_epoch = 0
        self.stale_epochs = 0
        self.epochs_completed = 0

    def update(self, epoch: int, val_loss: float) -> ValidationDecision:
        if epoch != self.epochs_completed + 1:
            raise ValueError("验证轮次必须从1开始连续递增")
        if not math.isfinite(val_loss) or val_loss < 0:
            raise ValueError(f"第{epoch}轮验证交叉熵不是有效的非负有限数：{val_loss}")

        checkpoint_updated = val_loss < self.best_loss
        if checkpoint_updated:
            self.best_loss = float(val_loss)
            self.best_epoch = epoch

        early_stopping_improved = val_loss < self.reference_loss - self.min_delta
        if early_stopping_improved:
            self.reference_loss = float(val_loss)
            self.reference_epoch = epoch
            self.stale_epochs = 0
        else:
            self.stale_epochs += 1
        self.epochs_completed = epoch
        return ValidationDecision(
            checkpoint_updated=checkpoint_updated,
            early_stopping_improved=early_stopping_improved,
            should_stop=self.stale_epochs >= self.patience,
        )

    def summary(self) -> Dict[str, Any]:
        if self.epochs_completed == 0:
            raise RuntimeError("尚未监控任何验证轮次")
        return {
            "training_policy_version": TRAINING_POLICY_VERSION,
            "monitor": "validation_cross_entropy",
            "checkpoint_selection": "strict_minimum_earliest_tie",
            "early_stopping": "absolute_min_delta_from_last_significant_improvement",
            "min_delta": self.min_delta,
            "patience": self.patience,
            "best_epoch": self.best_epoch,
            "best_validation_loss": self.best_loss,
            "early_stopping_reference_epoch": self.reference_epoch,
            "early_stopping_reference_loss": self.reference_loss,
            "epochs_without_significant_improvement": self.stale_epochs,
            "epochs_completed": self.epochs_completed,
            "stopped_early": self.stale_epochs >= self.patience,
        }
