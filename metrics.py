# -*- coding: utf-8 -*-
"""三分类推理、核心指标和精简诊断图。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

# 无需弹出窗口，在终端、服务器上也能生成图片。
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from config import CLASS_ANALYTE_TARGETS, CLASS_NAMES

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


CLASS_TO_ANALYTES = np.asarray(CLASS_ANALYTE_TARGETS, dtype=np.int64)


def class_ids_to_analytes(class_ids: np.ndarray) -> np.ndarray:
    """把三个互斥类别转换成[MG1存在, R6G存在]，仅用于派生指标。"""
    ids = np.asarray(class_ids, dtype=np.int64).reshape(-1)
    if np.any(ids < 0) or np.any(ids >= len(CLASS_NAMES)):
        raise ValueError("类别编号超出三分类范围")
    return CLASS_TO_ANALYTES[ids]


def class_probabilities_to_analyte_probabilities(probabilities: np.ndarray) -> np.ndarray:
    """由三类概率求每种成分的边缘存在概率。"""
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[1] != len(CLASS_NAMES):
        raise ValueError(f"类别概率必须是[N,{len(CLASS_NAMES)}]形状")
    return probs @ CLASS_TO_ANALYTES


@torch.no_grad()
def predict_loader(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Tuple[np.ndarray, Optional[np.ndarray], float]:
    model.eval()
    probabilities: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    losses: List[float] = []
    criterion = nn.CrossEntropyLoss(reduction="mean")
    for batch in loader:
        if isinstance(batch, (tuple, list)):
            x_batch, y_batch = batch
            y_batch = y_batch.to(device, non_blocking=True)
        else:
            x_batch, y_batch = batch, None
        logits = model(x_batch.to(device, non_blocking=True))
        probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
        if y_batch is not None:
            targets.append(y_batch.cpu().numpy())
            losses.append(float(criterion(logits, y_batch).item()) * len(x_batch))
    probs = np.concatenate(probabilities, axis=0)
    if not targets:
        return probs, None, float("nan")
    true = np.concatenate(targets, axis=0)
    return probs, true, sum(losses) / len(true)


def compute_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> Dict[str, Any]:
    true_class = np.asarray(y_true, dtype=np.int64).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape != (len(true_class), len(CLASS_NAMES)):
        raise ValueError(f"类别概率必须是[{len(true_class)},{len(CLASS_NAMES)}]形状")
    pred_class = probabilities.argmax(axis=1).astype(np.int64)
    labels = list(range(len(CLASS_NAMES)))
    cm = confusion_matrix(true_class, pred_class, labels=labels)
    one_hot = np.eye(len(CLASS_NAMES), dtype=np.float64)[true_class]
    return {
        "n_samples": int(len(true_class)),
        "n_errors": int(np.count_nonzero(pred_class != true_class)),
        "brier_score": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "exact_match_accuracy": float(accuracy_score(true_class, pred_class)),
        "macro_f1_three_class": float(
            f1_score(true_class, pred_class, labels=labels, average="macro", zero_division=0)
        ),
        "confusion_matrix": cm.tolist(),
    }


def plot_diagnostics(
    history: List[Dict[str, Any]],
    test_matrix: Sequence[Sequence[int]],
    robustness_matrix: Optional[Sequence[Sequence[int]]],
    path: Path,
    best_epoch: Optional[int] = None,
) -> None:
    frame = pd.DataFrame(history)
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes[0, 0].plot(frame["epoch"], frame["train_loss"], label="训练集")
    axes[0, 0].plot(frame["epoch"], frame["val_loss"], label="验证集")
    if best_epoch is not None:
        axes[0, 0].axvline(
            best_epoch, color="gray", linestyle="--", label=f"最佳：{best_epoch}"
        )
    axes[0, 0].set(xlabel="轮次", ylabel="交叉熵", title="训练与验证损失")
    axes[0, 0].legend()
    axes[0, 1].plot(frame["epoch"], frame["val_accuracy"], label="正确率")
    axes[0, 1].plot(frame["epoch"], frame["val_macro_f1"], label="Macro-F1")
    axes[0, 1].set(xlabel="轮次", ylabel="分数", ylim=(0, 1.02), title="验证集")
    axes[0, 1].legend()
    _draw_confusion(axes[1, 0], test_matrix, "原始测试谱")
    if robustness_matrix is None:
        axes[1, 1].axis("off")
    else:
        _draw_confusion(axes[1, 1], robustness_matrix, "增强鲁棒性测试")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _draw_confusion(ax: Any, matrix: Sequence[Sequence[int]], title: str) -> None:
    cm = np.asarray(matrix)
    ax.imshow(cm, cmap="Blues")
    ax.set(
        xticks=np.arange(len(CLASS_NAMES)),
        yticks=np.arange(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel="预测类别",
        ylabel="真实类别",
        title=title,
    )
    for row in range(len(CLASS_NAMES)):
        for column in range(len(CLASS_NAMES)):
            color = "white" if cm[row, column] > cm.max() / 2 else "black"
            ax.text(column, row, str(cm[row, column]), ha="center", va="center", color=color)


def make_prediction_frame(
    metadata: pd.DataFrame,
    probabilities: np.ndarray,
    y_true: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    class_ids = np.asarray(probabilities).argmax(axis=1).astype(np.int64)
    binary = class_ids_to_analytes(class_ids)
    analyte_probabilities = class_probabilities_to_analyte_probabilities(probabilities)
    frame = metadata.reset_index(drop=True).copy()
    frame["仅MG1类别概率"] = probabilities[:, 0]
    frame["仅R6G类别概率"] = probabilities[:, 1]
    frame["两者都有类别概率"] = probabilities[:, 2]
    frame["MG1存在概率"] = analyte_probabilities[:, 0]
    frame["R6G存在概率"] = analyte_probabilities[:, 1]
    frame["预测含MG1"] = binary[:, 0]
    frame["预测含R6G"] = binary[:, 1]
    frame["预测类别"] = [CLASS_NAMES[value] for value in class_ids]
    if y_true is not None:
        true_ids = np.asarray(y_true, dtype=np.int64).reshape(-1)
        frame["真实类别"] = [CLASS_NAMES[value] for value in true_ids]
        frame["是否正确"] = class_ids == true_ids
    return frame


def json_default(value: Any):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"无法 JSON 序列化：{type(value)}")
