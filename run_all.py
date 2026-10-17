# -*- coding: utf-8 -*-
"""四模型实验入口：独立训练、检查相同拆分、生成中文对比表与图。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from config import (
    CHECKPOINT_FORMAT_VERSION, ChineseParser, MODELS, ROOT, TASK_NAME, setup_console,
)
from data import data_fingerprints, parse_data_spec, resolve_data_specs
from metrics import plt
from train import add_training_arguments, validate_arguments
from training_control import TRAINING_POLICY_VERSION

TRAIN_SCRIPT = ROOT / "train.py"
MODEL_NAMES_CN = {item["channel_set"]: item["name"] for item in MODELS.values()}


SUMMARY_COLUMNS_CN = {
    "order": "序号",
    "model": "模型",
    "channel_set": "通道组合",
    "n_channels": "通道数",
    "trainable_parameters": "可训练参数",
    "best_epoch": "最佳训练轮次",
    "best_validation_loss": "最佳验证损失",
    "training_policy_version": "训练策略版本",
    "epochs_completed": "实际完成轮数",
    "stopped_early": "是否触发早停",
    "early_stopping_reference_epoch": "早停参考轮次",
    "early_stopping_reference_loss": "早停参考损失",
    "epochs_without_significant_improvement": "连续无有效改善轮数",
    "test_cross_entropy_loss": "测试交叉熵损失",
    "exact_match_accuracy": "完全正确率",
    "macro_f1_three_class": "三类别平均F1",
    "macro_f1_observed_class": "实际类别平均F1",
    "macro_f1_analyte": "成分平均F1",
    "hamming_loss": "标签错误率",
    "MG1_precision": "MG1精确率",
    "MG1_recall": "MG1召回率",
    "MG1_f1": "MG1的F1",
    "MG1_roc_auc": "MG1的AUC",
    "R6G_precision": "R6G精确率",
    "R6G_recall": "R6G召回率",
    "R6G_f1": "R6G的F1",
    "R6G_roc_auc": "R6G的AUC",
    "group": "分组",
    "n_samples": "样本数",
    "MG1_positive_support": "MG1阳性数",
    "MG1_negative_support": "MG1阴性数",
    "MG1_false_positive_rate": "MG1误报率",
    "R6G_positive_support": "R6G阳性数",
    "R6G_negative_support": "R6G阴性数",
    "R6G_false_positive_rate": "R6G误报率",
}


def to_chinese_table(frame: pd.DataFrame) -> pd.DataFrame:
    """把供程序读取的英文列名转换为方便直接查看的中文列名。"""
    result = frame.copy()
    if "model" in result:
        result["model"] = result["model"].map(MODEL_NAMES_CN).fillna(result["model"])
    return result.rename(columns=SUMMARY_COLUMNS_CN)


def verify_identical_splits(output_dirs: Sequence[Path]) -> None:
    """确认四个模型使用了逐行完全相同的训练/验证/测试集合。"""
    columns = [
        "source_file",
        "row_in_file",
        "parent_source_row",
        "augmentation_index",
        "role",
        "concentration_group",
        "split",
    ]
    reference = pd.read_csv(output_dirs[0] / "tables" / "dataset_manifest.csv")[columns]
    for output_dir in output_dirs[1:]:
        current = pd.read_csv(output_dir / "tables" / "dataset_manifest.csv")[columns]
        try:
            pd.testing.assert_frame_equal(reference, current, check_dtype=False)
        except AssertionError as exc:
            raise RuntimeError(f"{output_dir.name} 的数据拆分与第一个模型不一致") from exc


def collect_results(configs: Sequence[Dict[str, Any]], output_root: Path) -> pd.DataFrame:
    """读取四个模型的测试结果，生成总体、按浓度和按类别汇总。"""
    comparison_dir = output_root / "compare" / "tables"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: List[Dict[str, Any]] = []
    concentration_frames: List[pd.DataFrame] = []
    role_frames: List[pd.DataFrame] = []
    for config in configs:
        output_dir = output_root / config["folder"]
        with (output_dir / "test_metrics.json").open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        if metrics.get("training_policy_version") != TRAINING_POLICY_VERSION:
            raise ValueError(f"{output_dir.name}使用旧训练策略，请重新训练后再汇总")
        summary_rows.append(
            {
                "order": config["order"],
                "model": config["name"],
                "channel_set": config["channel_set"],
                "n_channels": len(metrics["channel_names"]),
                "trainable_parameters": metrics["trainable_parameters"],
                "best_epoch": metrics["best_epoch"],
                "best_validation_loss": metrics["best_validation_loss"],
                **{key: metrics[key] for key in (
                    "training_policy_version", "epochs_completed", "stopped_early",
                    "early_stopping_reference_epoch", "early_stopping_reference_loss",
                    "epochs_without_significant_improvement",
                )},
                "test_cross_entropy_loss": metrics["test_cross_entropy_loss"],
                "exact_match_accuracy": metrics["exact_match_accuracy"],
                "macro_f1_three_class": metrics["macro_f1_three_class"],
                "macro_f1_observed_class": metrics.get(
                    "macro_f1_observed_class", metrics["macro_f1_three_class"]
                ),
                "macro_f1_analyte": metrics["macro_f1_analyte"],
                "hamming_loss": metrics["hamming_loss"],
                "MG1_precision": metrics["per_analyte"]["MG1"]["precision"],
                "MG1_recall": metrics["per_analyte"]["MG1"]["recall"],
                "MG1_f1": metrics["per_analyte"]["MG1"]["f1"],
                "MG1_roc_auc": metrics["per_analyte"]["MG1"]["roc_auc"],
                "R6G_precision": metrics["per_analyte"]["R6G"]["precision"],
                "R6G_recall": metrics["per_analyte"]["R6G"]["recall"],
                "R6G_f1": metrics["per_analyte"]["R6G"]["f1"],
                "R6G_roc_auc": metrics["per_analyte"]["R6G"]["roc_auc"],
            }
        )
        concentration = pd.read_csv(output_dir / "tables" / "test_metrics_by_concentration.csv")
        concentration.insert(0, "model", config["name"])
        concentration_frames.append(concentration)
        role = pd.read_csv(output_dir / "tables" / "test_metrics_by_role.csv")
        role.insert(0, "model", config["name"])
        role_frames.append(role)

    summary = pd.DataFrame(summary_rows).sort_values("order")
    summary.to_csv(comparison_dir / "ablation_summary.csv", index=False, encoding="utf-8-sig")
    to_chinese_table(summary).to_csv(
        comparison_dir / "四模型对比结果.csv", index=False, encoding="utf-8-sig"
    )
    concentration_summary = pd.concat(concentration_frames, ignore_index=True)
    concentration_summary.to_csv(
        comparison_dir / "ablation_by_concentration.csv", index=False, encoding="utf-8-sig"
    )
    to_chinese_table(concentration_summary).to_csv(
        comparison_dir / "各浓度对比结果.csv", index=False, encoding="utf-8-sig"
    )
    role_summary = pd.concat(role_frames, ignore_index=True)
    role_summary.to_csv(
        comparison_dir / "ablation_by_role.csv", index=False, encoding="utf-8-sig"
    )
    to_chinese_table(role_summary).to_csv(
        comparison_dir / "各类别对比结果.csv", index=False, encoding="utf-8-sig"
    )
    return summary


def plot_summary(summary: pd.DataFrame, path: Path) -> None:
    names = [MODEL_NAMES_CN.get(name, name) for name in summary["model"].tolist()]
    x = np.arange(len(names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, summary["exact_match_accuracy"], width, label="完全正确率")
    ax.bar(x, summary["macro_f1_analyte"], width, label="成分平均F1")
    ax.bar(x + width, summary["macro_f1_three_class"], width, label="三类别平均F1")
    ax.set_xticks(x, names, rotation=12)
    ax.set_ylim(0, 1.03)
    ax.set_ylabel("测试集分数")
    ax.set_title("相同测试集上的通道消融对比")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def build_train_command(args: argparse.Namespace, number: int) -> List[str]:
    """完整传递公共参数，保证单模型与四模型的训练设置一致。"""
    command = [sys.executable, "-B", str(TRAIN_SCRIPT), "--model", str(number)]
    for key, value in vars(args).items():
        if key in {"data", "skip_existing"}:
            continue
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                command.append(flag)
        else:
            command.extend([flag, str(value)])
    for spec in args.data:
        command.extend(["--data", spec])
    return command


def can_skip(args: argparse.Namespace, number: int, fingerprints: List[Dict[str, str]]) -> bool:
    """只复用文件齐全、数据内容和训练参数一致的已完成实验。"""
    folder = MODELS[number]["folder"]
    directory = Path(args.output_dir) / folder
    files = [
        Path(args.pt_dir) / folder / "best.pt",
        directory / "training_config.json", directory / "test_metrics.json",
        directory / "结果摘要.txt",
        directory / "figures" / "training_history.png",
        directory / "figures" / "test_confusion_matrix.png",
    ]
    files.extend(directory / "tables" / name for name in (
        "dataset_manifest.csv", "数据拆分清单.csv", "training_history.csv", "训练记录.csv",
        "test_predictions.csv", "测试集逐条预测.csv",
        "test_metrics_by_concentration.csv", "按浓度测试结果.csv",
        "test_metrics_by_role.csv", "按类别测试结果.csv",
    ))
    if not all(path.is_file() for path in files):
        return False
    try:
        with (directory / "training_config.json").open(encoding="utf-8") as handle:
            saved = json.load(handle)
        with (directory / "test_metrics.json").open(encoding="utf-8") as handle:
            metrics = json.load(handle)
    except (OSError, ValueError):
        return False
    if saved.get("data_files") != fingerprints:
        return False
    if (saved.get("format_version") != CHECKPOINT_FORMAT_VERSION
            or saved.get("task") != TASK_NAME):
        return False
    if any(record.get("training_policy_version") != TRAINING_POLICY_VERSION
           for record in (saved, metrics)):
        return False
    old_args = saved.get("training_args", {})
    if old_args.get("model") != number:
        return False
    return all(old_args.get(key) == value for key, value in vars(args).items()
               if key not in {"skip_existing", "data", "pt_dir", "output_dir"})


def build_parser() -> argparse.ArgumentParser:
    parser = ChineseParser(description="依次训练模型1～4，比较逐步增加通道的效果")
    add_training_arguments(parser)
    parser.add_argument("--skip-existing", action="store_true",
                        help="数据和参数一致、结果文件齐全时复用已完成模型；不是断点续训")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    setup_console()
    args = build_parser().parse_args(argv)
    validate_arguments(args)
    # 先检查文件路径，再转成绝对路径，避免从其他工作目录启动子进程时出错。
    specs = [parse_data_spec(spec) for spec in resolve_data_specs(args.data)]
    args.data = [f"{role}={path.resolve()}" for role, path in specs]
    args.pt_dir = str(Path(args.pt_dir).expanduser().resolve())
    args.output_dir = str(Path(args.output_dir).expanduser().resolve())
    fingerprints = data_fingerprints(specs)
    configs = [
        {"order": number, "name": item["channel_set"],
         "folder": item["folder"], "channel_set": item["channel_set"]}
        for number, item in MODELS.items()
    ]
    output_root = Path(args.output_dir)
    output_dirs = []
    # 每个模型从头随机初始化，不接着上一个模型的权重继续训练。
    for number, item in MODELS.items():
        directory = output_root / item["folder"]
        output_dirs.append(directory)
        print(f"\n开始 {item['name']}（{number}/4）", flush=True)
        if args.skip_existing and can_skip(args, number, fingerprints):
            print(f"数据与参数一致，复用完整结果：{directory}", flush=True)
            continue
        subprocess.run(build_train_command(args, number), cwd=ROOT, check=True)

    verify_identical_splits(output_dirs)
    summary = collect_results(configs, output_root)
    figure_dir = output_root / "compare" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    plot_summary(summary, figure_dir / "ablation_summary.png")
    print("\n四个模型已完成，训练/验证/测试拆分完全一致。")
    print(to_chinese_table(summary[[
        "order", "model", "best_validation_loss", "test_cross_entropy_loss",
        "exact_match_accuracy", "macro_f1_three_class",
    ]]).to_string(index=False))
    print(f"中文对比表：{output_root / 'compare' / 'tables' / '四模型对比结果.csv'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError, argparse.ArgumentTypeError,
            subprocess.CalledProcessError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        raise SystemExit(2)
