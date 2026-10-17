# -*- coding: utf-8 -*-
"""三分类批量推理、指标计算、结果表格与中文绘图。"""

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
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    hamming_loss, precision_recall_fscore_support, roc_auc_score,
)
from torch.utils.data import DataLoader

from config import ANALYTE_NAMES, CLASS_ANALYTE_TARGETS, CLASS_NAMES

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
    pred_class = np.asarray(probabilities).argmax(axis=1).astype(np.int64)
    binary_true = class_ids_to_analytes(true_class)
    binary_pred = class_ids_to_analytes(pred_class)
    analyte_probabilities = class_probabilities_to_analyte_probabilities(probabilities)

    precision, recall, f1, support = precision_recall_fscore_support(
        binary_true, binary_pred, average=None, zero_division=0
    )
    per_analyte: Dict[str, Any] = {}
    for column, name in enumerate(ANALYTE_NAMES):
        auc = None
        if len(np.unique(binary_true[:, column])) == 2:
            auc = float(roc_auc_score(binary_true[:, column], analyte_probabilities[:, column]))
        per_analyte[name] = {
            "precision": float(precision[column]),
            "recall": float(recall[column]),
            "f1": float(f1[column]),
            "support_positive": int(support[column]),
            "roc_auc": auc,
        }

    labels = list(range(len(CLASS_NAMES)))
    cm = confusion_matrix(true_class, pred_class, labels=labels)
    report = classification_report(
        true_class,
        pred_class,
        labels=labels,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    return {
        "exact_match_accuracy": float(accuracy_score(true_class, pred_class)),
        "macro_f1_three_class": float(
            f1_score(true_class, pred_class, labels=labels, average="macro", zero_division=0)
        ),
        # 分浓度、分类别时可能只有部分真实类别，因此同时保存组内实际类别平均F1。
        "macro_f1_observed_class": float(
            f1_score(
                true_class,
                pred_class,
                labels=np.unique(true_class),
                average="macro",
                zero_division=0,
            )
        ),
        "observed_classes": [CLASS_NAMES[value] for value in np.unique(true_class)],
        "macro_f1_analyte": float(np.mean(f1)),
        "hamming_loss": float(hamming_loss(binary_true, binary_pred)),
        "per_analyte": per_analyte,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }


def plot_history(
    history: List[Dict[str, Any]], path: Path, best_epoch: Optional[int] = None
) -> None:
    frame = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(frame["epoch"], frame["train_loss"], label="训练集")
    axes[0].plot(frame["epoch"], frame["val_loss"], label="验证集")
    if best_epoch is not None:
        axes[0].axvline(best_epoch, color="gray", linestyle="--",
                       label=f"保留权重：第{best_epoch}轮")
    axes[0].set(xlabel="训练轮次", ylabel="交叉熵损失", title="训练与验证损失")
    axes[0].legend()
    axes[1].plot(frame["epoch"], frame["val_exact_accuracy"], label="完全正确率")
    axes[1].plot(frame["epoch"], frame["val_macro_f1_three_class"], label="三类别平均F1")
    axes[1].plot(frame["epoch"], frame["val_macro_f1_analyte"], label="成分平均F1")
    axes[1].set(xlabel="训练轮次", ylabel="分数", ylim=(0, 1.02), title="验证集指标")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_confusion(matrix: Sequence[Sequence[int]], path: Path) -> None:
    cm = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    image = ax.imshow(cm, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(len(CLASS_NAMES)),
        yticks=np.arange(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel="预测类别",
        ylabel="真实类别",
        title="测试集混淆矩阵",
    )
    for row in range(len(CLASS_NAMES)):
        for column in range(len(CLASS_NAMES)):
            color = "white" if cm[row, column] > cm.max() / 2 else "black"
            ax.text(column, row, str(cm[row, column]), ha="center", va="center", color=color)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


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


def metrics_by_group(
    metadata: pd.DataFrame,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    group_column: str,
) -> pd.DataFrame:
    """生成消融研究所需的浓度/角色分组指标。"""
    meta = metadata.reset_index(drop=True)
    class_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    class_pred = np.asarray(probabilities).argmax(axis=1).astype(np.int64)
    binary_true = class_ids_to_analytes(class_true)
    binary_pred = class_ids_to_analytes(class_pred)
    rows: List[Dict[str, Any]] = []
    for group_value, indices in meta.groupby(group_column, sort=False, dropna=False).indices.items():
        idx = np.asarray(indices, dtype=np.int64)
        group_metrics = compute_metrics(class_true[idx], probabilities[idx])
        row: Dict[str, Any] = {
            "group": str(group_value),
            "n_samples": int(len(idx)),
            "exact_match_accuracy": group_metrics["exact_match_accuracy"],
            "macro_f1_three_class": group_metrics["macro_f1_three_class"],
            "macro_f1_observed_class": group_metrics["macro_f1_observed_class"],
            "macro_f1_analyte": group_metrics["macro_f1_analyte"],
            "hamming_loss": group_metrics["hamming_loss"],
        }
        for column, name in enumerate(ANALYTE_NAMES):
            true_column = binary_true[idx, column]
            pred_column = binary_pred[idx, column]
            positive_mask = true_column == 1
            negative_mask = ~positive_mask
            row[f"{name}_positive_support"] = int(positive_mask.sum())
            row[f"{name}_negative_support"] = int(negative_mask.sum())
            row[f"{name}_recall"] = (
                float(np.mean(pred_column[positive_mask] == 1))
                if positive_mask.any()
                else np.nan
            )
            row[f"{name}_false_positive_rate"] = (
                float(np.mean(pred_column[negative_mask] == 1))
                if negative_mask.any()
                else np.nan
            )
            row[f"{name}_f1"] = group_metrics["per_analyte"][name]["f1"]
        rows.append(row)
    return pd.DataFrame(rows)


def json_default(value: Any):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"无法 JSON 序列化：{type(value)}")
