# -*- coding: utf-8 -*-
"""单模型训练入口：python train.py --model 3，省略编号也默认模型3。

先拆分，再拟合训练集预处理；按验证损失选权重、按验证F1选阈值。
最后评估测试集，保存至pt与output，网络本体位于models文件夹。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import (
    ANALYTE_NAMES, CHANNEL_NAMES_CN, CHANNEL_SETS, CLASS_NAMES, ChineseParser,
    MODELS, OUTPUT_DIR, PT_DIR, TABLE_COLUMNS_CN, choose_device, set_seed, setup_console,
)
from data import (
    SpectralPreprocessor, SpectraDataset, data_fingerprints, grouped_stratified_split,
    load_training_data, parse_data_spec, resolve_data_specs, validate_targets,
)
from metrics import (
    calibrate_thresholds, compute_metrics, json_default, make_prediction_frame,
    metrics_by_group, plot_confusion, plot_history, predict_loader,
)
from models import DualPresenceMSNet


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip: float,
) -> float:
    model.train()
    total_loss = 0.0
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        # 前馈：四/三/二/单通道光谱 -> 两个存在性 logit。
        loss = criterion(model(x_batch), y_batch)
        # 反向传播：计算梯度，再裁剪过大的梯度并更新网络参数。
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += float(loss.item()) * len(x_batch)
    return total_loss / len(loader.dataset)


def command_train(args: argparse.Namespace) -> None:
    """完成一次指定通道模型的读取、训练、选优、评估和保存。"""
    set_seed(args.seed)
    device = choose_device(args.device)
    args.data = resolve_data_specs(args.data)
    args.channel_set = MODELS[args.model]["channel_set"]
    specs = [parse_data_spec(spec) for spec in args.data]
    input_files = data_fingerprints(specs)
    roles = [role for role, _ in specs]
    x_raw, y, axis, metadata = load_training_data(specs)
    validate_targets(y, roles)
    # 先拆分，再拟合预处理器；顺序不能颠倒，否则会发生数据泄漏。
    train_idx, val_idx, test_idx, split_notes = grouped_stratified_split(
        metadata, args.val_fraction, args.test_fraction, args.group_size, args.seed
    )

    # 只有读取、标签与拆分检查通过后，才创建该模型的输出目录。
    folder = MODELS[args.model]["folder"]
    output_dir = Path(args.output_dir).expanduser().resolve() / folder
    weight_dir = Path(args.pt_dir).expanduser().resolve() / folder
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    for directory in (weight_dir, table_dir, figure_dir):
        directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = weight_dir / "best.pt"

    metadata = metadata.copy()
    metadata["split"] = ""
    metadata.loc[train_idx, "split"] = "train"
    metadata.loc[val_idx, "split"] = "validation"
    metadata.loc[test_idx, "split"] = "test"
    metadata.to_csv(table_dir / "dataset_manifest.csv", index=False, encoding="utf-8-sig")
    metadata.rename(columns=TABLE_COLUMNS_CN).to_csv(
        table_dir / "数据拆分清单.csv", index=False, encoding="utf-8-sig"
    )

    # 均值、标准差只来自 train_idx，随后原样应用到验证集和测试集。
    preprocessor = SpectralPreprocessor(
        clip=args.preprocess_clip, channel_set=args.channel_set
    ).fit(x_raw[train_idx], axis)
    x_train = preprocessor.transform(x_raw[train_idx])
    x_val = preprocessor.transform(x_raw[val_idx])
    x_test = preprocessor.transform(x_raw[test_idx])

    train_dataset = SpectraDataset(
        x_train,
        y[train_idx],
        augment=not args.no_augmentation,
        noise_std=args.noise_std,
        shift_points=args.shift_points,
    )
    val_dataset = SpectraDataset(x_val, y[val_idx])
    test_dataset = SpectraDataset(x_test, y[test_idx])
    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size * 2, shuffle=False, **loader_kwargs
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size * 2, shuffle=False, **loader_kwargs
    )

    model_config = {"in_channels": int(x_train.shape[1]), "dropout": args.dropout}
    model = DualPresenceMSNet(**model_config).to(device)
    train_targets = y[train_idx]
    positives = train_targets.sum(axis=0)
    negatives = len(train_targets) - positives
    if np.any(positives == 0) or np.any(negatives == 0):
        raise ValueError("训练拆分中某个存在性标签只有单一取值，请检查数据")
    pos_weight = torch.tensor(negatives / positives, dtype=torch.float32, device=device)
    # 两个输出分别判断 MG1/R6G 是否存在，正样本权重用于减轻数量不平衡。
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=args.learning_rate * 0.03
    )

    print(f"当前模型: {MODELS[args.model]['name']}")
    print(f"设备: {device}")
    print(f"原始样本/特征: {len(x_raw)} / {x_raw.shape[1]}")
    print(f"训练/验证/测试: {len(train_idx)} / {len(val_idx)} / {len(test_idx)}")
    print(f"波数范围: {axis[0]:.3f} ~ {axis[-1]:.3f}")
    total_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数: {total_parameters:,}")
    for note in split_notes:
        print(f"拆分提示: {note}")

    best_loss = math.inf
    best_epoch = 0
    best_state: Optional[Dict[str, torch.Tensor]] = None
    stale_epochs = 0
    history: List[Dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, args.grad_clip
        )
        val_probs, val_true, val_loss = predict_loader(model, val_loader, device)
        assert val_true is not None
        val_metrics = compute_metrics(val_true, val_probs, (0.5, 0.5))
        history.append(
            {
                "epoch": epoch,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_exact_accuracy": val_metrics["exact_match_accuracy"],
                "val_macro_f1_analyte": val_metrics["macro_f1_analyte"],
            }
        )
        scheduler.step()

        # 最佳模型只由验证集损失决定，绝不查看测试集后再选权重。
        improved = val_loss < best_loss - args.min_delta
        if improved:
            best_loss = val_loss
            best_epoch = epoch
            stale_epochs = 0
            # 先复制到内存；全部训练结束后再统一写入 best.pt。
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            stale_epochs += 1

        print(
            f"第 {epoch:03d}/{args.epochs} 轮 | 训练损失 {train_loss:.5f} | "
            f"验证损失 {val_loss:.5f} | 完全正确率 {val_metrics['exact_match_accuracy']:.4f} | "
            f"成分平均F1 {val_metrics['macro_f1_analyte']:.4f}"
        )
        if stale_epochs >= args.patience:
            print(f"验证损失连续 {args.patience} 轮未改善，提前停止。")
            break

    if best_state is None:
        raise RuntimeError("训练未生成有效模型")
    model.load_state_dict(best_state)

    val_probs, val_true, _ = predict_loader(model, val_loader, device)
    assert val_true is not None
    # 权重固定后，再用验证集分别确定 MG1 和 R6G 的“有/无”阈值。
    thresholds = calibrate_thresholds(val_true.astype(int), val_probs)
    test_probs, test_true, test_loss = predict_loader(model, test_loader, device)
    assert test_true is not None
    metrics = compute_metrics(test_true, test_probs, thresholds)
    metrics.update(
        {
            "test_bce_loss": test_loss,
            "best_epoch": best_epoch,
            "best_validation_loss": best_loss,
            "thresholds": dict(zip(ANALYTE_NAMES, thresholds)),
            "blank_samples_available": "none" in roles,
            "channel_set": args.channel_set,
            "channel_names": CHANNEL_SETS[args.channel_set],
            "trainable_parameters": total_parameters,
            "split_notes": split_notes,
            "sample_counts": {
                "all": len(x_raw),
                "train": len(train_idx),
                "validation": len(val_idx),
                "test": len(test_idx),
            },
        }
    )

    saved_args = {
        key: value for key, value in vars(args).items() if key != "func" and not callable(value)
    }
    with (output_dir / "training_config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "training_args": saved_args,
                "data_files": input_files,
                "channel_names": CHANNEL_SETS[args.channel_set],
                "model_config": model_config,
            },
            handle,
            ensure_ascii=False,
            indent=2,
            default=json_default,
        )
    checkpoint = {
        "format_version": 1,
        "model_name": "DualPresenceMSNet",
        "model_config": model_config,
        "model_state": best_state,
        "preprocessor": preprocessor.state_dict(),
        "reference_axis": torch.from_numpy(axis.astype(np.float32)),
        "thresholds": thresholds,
        "analyte_names": ANALYTE_NAMES,
        "class_names": CLASS_NAMES,
        "training_args": saved_args,
        "metrics": metrics,
    }
    torch.save(checkpoint, checkpoint_path)
    history_frame = pd.DataFrame(history)
    history_frame.to_csv(table_dir / "training_history.csv", index=False)
    history_frame.rename(columns=TABLE_COLUMNS_CN).to_csv(
        table_dir / "训练记录.csv", index=False, encoding="utf-8-sig"
    )
    plot_history(history, figure_dir / "training_history.png")
    plot_confusion(metrics["confusion_matrix"], figure_dir / "test_confusion_matrix.png")

    test_frame = make_prediction_frame(
        metadata.iloc[test_idx], test_probs, thresholds, test_true.astype(int)
    )
    test_frame.to_csv(table_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
    test_frame.rename(columns=TABLE_COLUMNS_CN).to_csv(
        table_dir / "测试集逐条预测.csv", index=False, encoding="utf-8-sig"
    )
    test_metadata = metadata.iloc[test_idx].copy()
    concentration_metrics = metrics_by_group(
        test_metadata, test_true.astype(int), test_probs, thresholds, "concentration_group"
    )
    concentration_metrics.to_csv(
        table_dir / "test_metrics_by_concentration.csv", index=False, encoding="utf-8-sig"
    )
    concentration_metrics.rename(columns=TABLE_COLUMNS_CN).to_csv(
        table_dir / "按浓度测试结果.csv", index=False, encoding="utf-8-sig"
    )
    role_metrics = metrics_by_group(
        test_metadata, test_true.astype(int), test_probs, thresholds, "role"
    )
    role_metrics.to_csv(
        table_dir / "test_metrics_by_role.csv", index=False, encoding="utf-8-sig"
    )
    role_metrics.rename(columns=TABLE_COLUMNS_CN).to_csv(
        table_dir / "按类别测试结果.csv", index=False, encoding="utf-8-sig"
    )
    with (output_dir / "test_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2, default=json_default)

    # 额外生成一份不需要阅读 JSON 的中文摘要，方便课程设计和实验记录。
    with (output_dir / "结果摘要.txt").open("w", encoding="utf-8") as handle:
        handle.write(f"模型通道组合：{args.channel_set}\n")
        handle.write(
            "实际输入通道："
            + "、".join(CHANNEL_NAMES_CN[name] for name in CHANNEL_SETS[args.channel_set])
            + "\n"
        )
        handle.write(f"最佳训练轮次：{best_epoch}\n")
        handle.write(f"最佳验证损失：{best_loss:.8f}\n")
        handle.write(f"测试BCE损失：{test_loss:.8f}\n")
        handle.write(f"MG1判定阈值：{thresholds[0]:.3f}\n")
        handle.write(f"R6G判定阈值：{thresholds[1]:.3f}\n")
        handle.write(f"测试集完全正确率：{metrics['exact_match_accuracy']:.4f}\n")
        handle.write(f"测试集四类别平均F1：{metrics['macro_f1_four_class']:.4f}\n")
        handle.write(f"测试集实际类别平均F1：{metrics['macro_f1_observed_class']:.4f}\n")
        handle.write(f"测试集成分平均F1：{metrics['macro_f1_analyte']:.4f}\n")
        handle.write(f"MG1召回率：{metrics['per_analyte']['MG1']['recall']:.4f}\n")
        handle.write(f"R6G召回率：{metrics['per_analyte']['R6G']['recall']:.4f}\n")
        handle.write(f"是否包含真实空白样本：{'是' if 'none' in roles else '否'}\n")

    print("\n训练完成")
    print(f"最佳训练轮次: {best_epoch}")
    print(
        "输入通道: "
        + "、".join(CHANNEL_NAMES_CN[name] for name in CHANNEL_SETS[args.channel_set])
    )
    print(f"校准阈值: MG1={thresholds[0]:.3f}, R6G={thresholds[1]:.3f}")
    print(f"测试集完全正确率: {metrics['exact_match_accuracy']:.4f}")
    print(f"测试集成分平均F1: {metrics['macro_f1_analyte']:.4f}")
    print(f"权重文件: {checkpoint_path}")
    print(f"结果目录: {output_dir}")
    if "none" not in roles:
        print("警告：未提供空白样本，“无”类别没有真实测试样本，不能视为已验证。")


def add_training_arguments(parser: argparse.ArgumentParser) -> None:
    """单模型和四模型入口共用参数，避免相同实验出现不同默认设置。"""
    parser.add_argument("--data", action="append", metavar="角色=CSV",
                        help="可重复传入mg/r6g/both/none；省略则读data中默认文件")
    parser.add_argument("--pt-dir", default=str(PT_DIR), help="权重保存根目录")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="表格、图片及摘要根目录")
    parser.add_argument("--epochs", type=int, default=120, help="最大训练轮数")
    parser.add_argument("--batch-size", type=int, default=32, help="每批训练光谱数")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="初始学习率")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW权重衰减")
    parser.add_argument("--dropout", type=float, default=0.15, help="随机丢弃特征比例")
    parser.add_argument("--patience", type=int, default=20, help="未有效改善多少轮后早停")
    parser.add_argument("--min-delta", type=float, default=1e-4, help="保留新权重所需的最小验证损失降幅")
    parser.add_argument("--grad-clip", type=float, default=5.0, help="梯度范数上限")
    parser.add_argument("--val-fraction", type=float, default=0.15, help="验证集目标比例，按组取整")
    parser.add_argument("--test-fraction", type=float, default=0.20, help="测试集目标比例，按组取整")
    parser.add_argument("--group-size", type=int, default=10, help="同文件同浓度中连续多少行分为一组")
    parser.add_argument("--preprocess-clip", type=float, default=10.0, help="原始标准化通道的截断范围")
    parser.add_argument("--noise-std", type=float, default=0.01, help="训练增强噪声标准差")
    parser.add_argument("--shift-points", type=int, default=2, help="训练增强最大平移点数")
    parser.add_argument("--no-augmentation", action="store_true", help="关闭训练增强")
    parser.add_argument("--seed", type=int, default=66, help="数据拆分和训练随机种子")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="计算设备")
    parser.add_argument("--num-workers", type=int, default=0, help="数据加载进程数，Windows建议0")


def validate_arguments(args: argparse.Namespace) -> None:
    """提前拦截常见的手工参数输入错误。"""
    for name in ("epochs", "batch_size", "patience", "group_size"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} 必须>=1")
    for name in ("learning_rate", "grad_clip", "preprocess_clip"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} 必须是大于0的有限数")
    for name in ("weight_decay", "min_delta", "noise_std"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"--{name.replace('_', '-')} 必须是大于等于0的有限数")
    if not 0 <= args.dropout < 1:
        raise ValueError("--dropout 必须在[0,1)内")
    if not (0 < args.val_fraction < 1 and 0 < args.test_fraction < 1
            and args.val_fraction + args.test_fraction < 1):
        raise ValueError("验证和测试比例必须都大于0，且两者之和小于1")
    if args.num_workers < 0 or args.shift_points < 0:
        raise ValueError("--num-workers和--shift-points必须>=0")
    if not 0 <= args.seed < 2**32:
        raise ValueError("--seed 必须在0到2**32-1之间")


def build_parser() -> argparse.ArgumentParser:
    parser = ChineseParser(description="训练MG1/R6G存在性分类模型")
    parser.add_argument("--model", type=int, choices=tuple(MODELS), default=3,
                        help="1原始；2加SNV；3再加一阶导数；4再加二阶导数")
    add_training_arguments(parser)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    setup_console()
    args = build_parser().parse_args(argv)
    validate_arguments(args)
    command_train(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError, argparse.ArgumentTypeError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        raise SystemExit(2)

