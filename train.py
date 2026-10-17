# -*- coding: utf-8 -*-
"""三分类单模型训练入口：python train.py --model 3，省略编号也默认模型3。

先拆分，再拟合训练集预处理；按验证损失选权重。
最后评价仅MG1、仅R6G、两者都有三个类别并保存结果。
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
    ANALYTE_NAMES, CHANNEL_NAMES_CN, CHANNEL_SETS, CHECKPOINT_FORMAT_VERSION,
    CLASS_NAMES, ChineseParser, MODELS, OUTPUT_DIR, PT_DIR, TABLE_COLUMNS_CN,
    TASK_NAME, choose_device, set_seed, setup_console,
)
from data import (
    SpectralPreprocessor, SpectraDataset, data_fingerprints, grouped_stratified_split,
    load_training_data, parse_data_spec, resolve_data_specs, validate_targets,
)
from metrics import (
    compute_metrics, json_default, make_prediction_frame,
    metrics_by_group, plot_confusion, plot_history, predict_loader,
)
from models import ReagentClassMSNet
from training_control import TRAINING_POLICY_VERSION, ValidationMonitor


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
        # 前馈：四/三/二/单通道光谱 -> 三个互斥类别logit。
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

    train_dataset = SpectraDataset(x_train, y[train_idx])
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

    model_config = {
        "in_channels": int(x_train.shape[1]),
        "dropout": args.dropout,
        "num_classes": len(CLASS_NAMES),
    }
    model = ReagentClassMSNet(**model_config).to(device)
    train_targets = y[train_idx]
    class_counts = np.bincount(train_targets, minlength=len(CLASS_NAMES))
    if np.any(class_counts == 0):
        raise ValueError("训练拆分未同时包含仅MG1、仅R6G、两者都有三个类别")
    class_weights = len(train_targets) / (len(CLASS_NAMES) * class_counts)
    # 三类互斥，使用逆频率类别权重减轻混合样品数量较多造成的不平衡。
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=args.learning_rate * 0.03
    )

    print(f"当前模型: {MODELS[args.model]['name']}")
    print(f"设备: {device}")
    print(f"输入CSV总行数/特征: {len(x_raw)} / {x_raw.shape[1]}")
    print(f"训练/验证/测试: {len(train_idx)} / {len(val_idx)} / {len(test_idx)}")
    print(f"波数范围: {axis[0]:.3f} ~ {axis[-1]:.3f}")
    total_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数: {total_parameters:,}")
    print(
        "离线数据构成: "
        f"原始光谱 {int((~metadata['is_augmented']).sum())} / "
        f"扩充光谱 {int(metadata['is_augmented'].sum())}"
    )
    for note in split_notes:
        print(f"拆分提示: {note}")

    monitor = ValidationMonitor(args.min_delta, args.patience)
    best_state: Optional[Dict[str, torch.Tensor]] = None
    history: List[Dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, args.grad_clip
        )
        val_probs, val_true, val_loss = predict_loader(model, val_loader, device)
        assert val_true is not None
        decision = monitor.update(epoch, val_loss)
        # 任何严格更低的验证损失都保留；min_delta只控制早停计数。
        if decision.checkpoint_updated:
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        val_metrics = compute_metrics(val_true, val_probs)
        history.append(
            {
                "epoch": epoch,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_exact_accuracy": val_metrics["exact_match_accuracy"],
                "val_macro_f1_three_class": val_metrics["macro_f1_three_class"],
                "val_macro_f1_analyte": val_metrics["macro_f1_analyte"],
                "checkpoint_updated": decision.checkpoint_updated,
                "early_stopping_improved": decision.early_stopping_improved,
                "best_validation_loss": monitor.best_loss,
                "early_stopping_reference_loss": monitor.reference_loss,
                "epochs_without_significant_improvement": monitor.stale_epochs,
            }
        )
        scheduler.step()

        print(
            f"第 {epoch:03d}/{args.epochs} 轮 | 训练损失 {train_loss:.6g} | "
            f"验证损失 {val_loss:.6g} | 完全正确率 {val_metrics['exact_match_accuracy']:.4f} | "
            f"成分平均F1 {val_metrics['macro_f1_analyte']:.4f} | "
            f"最低损失轮次 {monitor.best_epoch} | "
            f"早停计数 {monitor.stale_epochs}/{args.patience}"
        )
        if decision.should_stop:
            print(f"验证损失连续 {args.patience} 轮无超过min_delta的有效改善，提前停止。")
            break

    if best_state is None:
        raise RuntimeError("训练未生成有效模型")
    training_control = monitor.summary()
    model.load_state_dict(best_state)

    test_probs, test_true, test_loss = predict_loader(model, test_loader, device)
    assert test_true is not None
    metrics = compute_metrics(test_true, test_probs)
    metrics.update(
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "task": TASK_NAME,
            "class_names": CLASS_NAMES,
            "test_cross_entropy_loss": test_loss,
            **training_control,
            "channel_set": args.channel_set,
            "channel_names": CHANNEL_SETS[args.channel_set],
            "trainable_parameters": total_parameters,
            "split_notes": split_notes,
            "sample_counts": {
                "all_input_rows": len(x_raw),
                "independent_original_spectra": int((~metadata["is_augmented"]).sum()),
                "offline_augmented_spectra": int(metadata["is_augmented"].sum()),
                "train": len(train_idx),
                "validation": len(val_idx),
                "test": len(test_idx),
                "train_parent_spectra": int(
                    metadata.iloc[train_idx][["source_id", "parent_source_row"]]
                    .drop_duplicates().shape[0]
                ),
                "validation_parent_spectra": int(
                    metadata.iloc[val_idx][["source_id", "parent_source_row"]]
                    .drop_duplicates().shape[0]
                ),
                "test_parent_spectra": int(
                    metadata.iloc[test_idx][["source_id", "parent_source_row"]]
                    .drop_duplicates().shape[0]
                ),
            },
        }
    )

    saved_args = {
        key: value for key, value in vars(args).items() if key != "func" and not callable(value)
    }
    with (output_dir / "training_config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "format_version": CHECKPOINT_FORMAT_VERSION,
                "training_policy_version": TRAINING_POLICY_VERSION,
                "task": TASK_NAME,
                "training_control": training_control,
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
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "training_policy_version": TRAINING_POLICY_VERSION,
        "task": TASK_NAME,
        "training_control": training_control,
        "model_name": "ReagentClassMSNet",
        "model_config": model_config,
        "model_state": best_state,
        "preprocessor": preprocessor.state_dict(),
        "reference_axis": torch.from_numpy(axis.astype(np.float32)),
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
    plot_history(history, figure_dir / "training_history.png", best_epoch=monitor.best_epoch)
    plot_confusion(metrics["confusion_matrix"], figure_dir / "test_confusion_matrix.png")

    test_frame = make_prediction_frame(metadata.iloc[test_idx], test_probs, test_true.astype(int))
    test_frame.to_csv(table_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
    test_frame.rename(columns=TABLE_COLUMNS_CN).to_csv(
        table_dir / "测试集逐条预测.csv", index=False, encoding="utf-8-sig"
    )
    test_metadata = metadata.iloc[test_idx].copy()
    concentration_metrics = metrics_by_group(
        test_metadata, test_true.astype(int), test_probs, "concentration_group"
    )
    concentration_metrics.to_csv(
        table_dir / "test_metrics_by_concentration.csv", index=False, encoding="utf-8-sig"
    )
    concentration_metrics.rename(columns=TABLE_COLUMNS_CN).to_csv(
        table_dir / "按浓度测试结果.csv", index=False, encoding="utf-8-sig"
    )
    role_metrics = metrics_by_group(
        test_metadata, test_true.astype(int), test_probs, "role"
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
        handle.write(f"任务格式版本：{CHECKPOINT_FORMAT_VERSION}\n")
        handle.write(f"训练策略版本：{TRAINING_POLICY_VERSION}\n")
        handle.write("权重选择规则：所有已完成轮次的最低验证交叉熵，相同值保留最早轮次\n")
        handle.write("分类任务：仅MG1、仅R6G、两者都有\n")
        handle.write(f"模型通道组合：{args.channel_set}\n")
        handle.write(
            "实际输入通道："
            + "、".join(CHANNEL_NAMES_CN[name] for name in CHANNEL_SETS[args.channel_set])
            + "\n"
        )
        handle.write(f"输入CSV总行数：{len(x_raw)}\n")
        handle.write(f"独立原始光谱数：{int((~metadata['is_augmented']).sum())}\n")
        handle.write(f"离线扩充光谱数：{int(metadata['is_augmented'].sum())}\n")
        handle.write(f"实际训练行数：{len(train_idx)}\n")
        handle.write(f"验证集评价行数：{len(val_idx)}\n")
        handle.write(f"测试集评价行数：{len(test_idx)}\n")
        handle.write(
            "验证集父光谱数："
            f"{metadata.iloc[val_idx][['source_id', 'parent_source_row']].drop_duplicates().shape[0]}\n"
        )
        handle.write(
            "测试集父光谱数："
            f"{metadata.iloc[test_idx][['source_id', 'parent_source_row']].drop_duplicates().shape[0]}\n"
        )
        handle.write(f"实际完成轮数：{monitor.epochs_completed}\n")
        handle.write(f"是否触发早停：{'是' if training_control['stopped_early'] else '否'}\n")
        handle.write(f"最佳训练轮次：{monitor.best_epoch}\n")
        handle.write(f"最佳验证损失：{monitor.best_loss:.10g}\n")
        handle.write(f"早停参考轮次：{monitor.reference_epoch}\n")
        handle.write(f"早停参考损失：{monitor.reference_loss:.10g}\n")
        handle.write(f"早停最小有效降幅：{args.min_delta:.10g}\n")
        handle.write(f"连续无有效改善轮数：{monitor.stale_epochs}\n")
        handle.write(f"测试交叉熵损失：{test_loss:.10g}\n")
        handle.write(f"测试集完全正确率：{metrics['exact_match_accuracy']:.4f}\n")
        handle.write(f"测试集三类别平均F1：{metrics['macro_f1_three_class']:.4f}\n")
        handle.write(f"测试集实际类别平均F1：{metrics['macro_f1_observed_class']:.4f}\n")
        handle.write(f"测试集成分平均F1：{metrics['macro_f1_analyte']:.4f}\n")
        handle.write(f"MG1召回率：{metrics['per_analyte']['MG1']['recall']:.4f}\n")
        handle.write(f"R6G召回率：{metrics['per_analyte']['R6G']['recall']:.4f}\n")
        handle.write("业务约束：待测样品至少含一种目标试剂，不建模二者都无\n")

    print("\n训练完成")
    print(f"实际完成轮数: {monitor.epochs_completed}")
    print(f"最佳训练轮次: {monitor.best_epoch}；最低验证损失: {monitor.best_loss:.10g}")
    print(f"早停参考轮次: {monitor.reference_epoch}；参考损失: {monitor.reference_loss:.10g}")
    print(
        "输入通道: "
        + "、".join(CHANNEL_NAMES_CN[name] for name in CHANNEL_SETS[args.channel_set])
    )
    print(f"测试集完全正确率: {metrics['exact_match_accuracy']:.4f}")
    print(f"测试集三类别平均F1: {metrics['macro_f1_three_class']:.4f}")
    print(f"测试集成分平均F1: {metrics['macro_f1_analyte']:.4f}")
    print(f"权重文件: {checkpoint_path}")
    print(f"结果目录: {output_dir}")


def add_training_arguments(parser: argparse.ArgumentParser) -> None:
    """单模型和四模型入口共用参数，避免相同实验出现不同默认设置。"""
    parser.add_argument("--data", action="append", metavar="角色=CSV",
                        help="可重复传入mg/r6g/both；省略则读data_augmented中默认文件")
    parser.add_argument("--pt-dir", default=str(PT_DIR), help="权重保存根目录")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="表格、图片及摘要根目录")
    parser.add_argument("--epochs", type=int, default=120, help="最大训练轮数")
    parser.add_argument("--batch-size", type=int, default=32, help="每批训练光谱数")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="初始学习率")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW权重衰减")
    parser.add_argument("--dropout", type=float, default=0.15, help="随机丢弃特征比例")
    parser.add_argument("--patience", type=int, default=20, help="未有效改善多少轮后早停")
    parser.add_argument("--min-delta", type=float, default=1e-4,
                        help="重置早停计数所需超过的验证损失降幅；不限制最佳权重更新")
    parser.add_argument("--grad-clip", type=float, default=5.0, help="梯度范数上限")
    parser.add_argument("--val-fraction", type=float, default=0.15, help="验证集目标比例，按组取整")
    parser.add_argument("--test-fraction", type=float, default=0.20, help="测试集目标比例，按组取整")
    parser.add_argument(
        "--group-size", type=int, default=10,
        help="同文件同浓度中连续多少条原始光谱分为一组",
    )
    parser.add_argument("--preprocess-clip", type=float, default=10.0, help="原始标准化通道的截断范围")
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
    for name in ("weight_decay", "min_delta"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"--{name.replace('_', '-')} 必须是大于等于0的有限数")
    if not 0 <= args.dropout < 1:
        raise ValueError("--dropout 必须在[0,1)内")
    if not (0 < args.val_fraction < 1 and 0 < args.test_fraction < 1
            and args.val_fraction + args.test_fraction < 1):
        raise ValueError("验证和测试比例必须都大于0，且两者之和小于1")
    if args.num_workers < 0:
        raise ValueError("--num-workers必须>=0")
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
