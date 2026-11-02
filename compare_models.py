# -*- coding: utf-8 -*-
"""读取多个已完成实验目录，校验公平性并单独比较模型。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from config import CLASS_NAMES, ChineseParser, OUTPUT_DIR, ROOT, setup_console
from data import sha256_file
from metrics import plt
from research_protocol import normalize_run_id
from representations import available_representations
from run_experiments import plot_accuracy


IGNORED_TRAINING_ARGUMENTS = {
    "architecture", "model_options", "run_id", "prepared_dir",
    "pt_dir", "output_dir",
}
SHARED_TRAINING_CODE = (
    "config.py", "representations.py", "data.py", "metrics.py",
    "train.py", "training_control.py",
)


def load_config(path: Path) -> Dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not {"comparison_id", "runs"}.issubset(config):
        raise ValueError("比较配置必须包含comparison_id和runs")
    config["comparison_id"] = normalize_run_id(str(config["comparison_id"]))
    runs = config["runs"]
    if (
        not isinstance(runs, list)
        or len(runs) < 2
        or any(not isinstance(value, str) or not value.strip() for value in runs)
        or len(runs) != len(set(runs))
    ):
        raise ValueError("runs必须包含至少两个不重复的实验目录")
    return config


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def protocol_signature(record: Dict[str, Any], split_sha256: str) -> Dict[str, Any]:
    data_hashes = sorted(
        (item["name"], item["sha256"]) for item in record["data_files"]
    )
    return {
        "protocol_version": record["protocol_version"],
        "training_policy_version": record["training_policy_version"],
        "data_hashes": data_hashes,
        "preparation_config_sha256": record["preparation"]["config_sha256"],
        "sample_counts": record["test"]["sample_counts"],
        "split_sha256": split_sha256,
        "shared_code_hashes": {
            name: record["code_fingerprints"][name]
            for name in SHARED_TRAINING_CODE
        },
    }


def training_signature(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in record["training_args"].items()
        if key not in IGNORED_TRAINING_ARGUMENTS
    }


def result_row(
    record: Dict[str, Any],
    run_root: Path,
    metrics_path: Path,
    checkpoint: Path,
) -> Dict[str, Any]:
    experiment = record["experiment"]
    validation = record["validation"]
    test = record["test"]
    robust = record.get("robustness") or {}
    control = record["training_control"]
    return {
        "source_run": run_root.name,
        "architecture": experiment["architecture"],
        "channel_set": experiment["channel_set"],
        "training_seed": int(experiment["training_seed"]),
        "trainable_parameters": record["trainable_parameters"],
        "best_epoch": validation["best_epoch"],
        "epochs_completed": control["epochs_completed"],
        "best_validation_loss": validation["loss"],
        "best_validation_accuracy": validation["accuracy"],
        "best_validation_macro_f1": validation["macro_f1"],
        "test_accuracy": test["exact_match_accuracy"],
        "test_macro_f1": test["macro_f1_three_class"],
        "test_errors": test["n_errors"],
        "test_cross_entropy": test["cross_entropy"],
        "test_brier_score": test["brier_score"],
        "robustness_accuracy": robust.get("exact_match_accuracy", np.nan),
        "robustness_macro_f1": robust.get("macro_f1_three_class", np.nan),
        "robustness_errors": robust.get("n_errors", np.nan),
        "training_seconds": record["training_seconds"],
        "metrics_file": str(metrics_path),
        "checkpoint": str(checkpoint),
    }


def load_completed_runs(
    run_roots: Sequence[Path],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    protocol: Optional[Dict[str, Any]] = None
    signatures: Dict[Tuple[str, int], Dict[str, Any]] = {}
    combinations: set[Tuple[str, str, int]] = set()

    for run_root in run_roots:
        snapshot_path = run_root / "config.json"
        split_path = run_root / "split.csv"
        if not snapshot_path.is_file() or not split_path.is_file():
            raise FileNotFoundError(f"实验目录不完整：{run_root}")
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        required = {"config", "effective_run_id", "training_seeds"}
        if not isinstance(snapshot, dict) or not required.issubset(snapshot):
            raise ValueError(f"实验配置快照格式错误：{snapshot_path}")
        experiment_config = snapshot["config"]
        run_id = str(snapshot["effective_run_id"])
        channels = experiment_config["channel_sets"]
        seeds = snapshot["training_seeds"]
        split_hash = sha256_file(split_path)

        for architecture_item in experiment_config["architectures"]:
            architecture = (
                architecture_item["name"]
                if isinstance(architecture_item, dict)
                else str(architecture_item)
            )
            for channel_set in channels:
                for seed in seeds:
                    key = (architecture, str(channel_set), int(seed))
                    if key in combinations:
                        raise ValueError(f"重复的模型/表示/种子结果：{key}")
                    combinations.add(key)
                    directory = run_root / architecture / str(channel_set) / f"seed_{seed}"
                    metrics_path = directory / "metrics.json"
                    if not metrics_path.is_file():
                        raise FileNotFoundError(f"缺少已训练结果：{metrics_path}")
                    record = json.loads(metrics_path.read_text(encoding="utf-8"))
                    pt_root = Path(record["training_args"]["pt_dir"]).expanduser()
                    if not pt_root.is_absolute():
                        pt_root = ROOT / pt_root
                    checkpoint = (
                        pt_root.resolve() / run_id / architecture / str(channel_set)
                        / f"seed_{seed}" / "best.pt"
                    )
                    if not checkpoint.is_file():
                        raise FileNotFoundError(f"缺少模型权重：{checkpoint}")
                    experiment = record["experiment"]
                    actual = (
                        experiment["architecture"], experiment["channel_set"],
                        int(experiment["training_seed"]),
                    )
                    if actual != key:
                        raise ValueError(f"结果目录与metrics.json身份不一致：{metrics_path}")

                    current_protocol = protocol_signature(record, split_hash)
                    if protocol is None:
                        protocol = current_protocol
                    elif current_protocol != protocol:
                        raise ValueError(f"数据或评价协议不一致：{metrics_path}")

                    fair_key = (str(channel_set), int(seed))
                    current_training = training_signature(record)
                    expected_training = signatures.setdefault(fair_key, current_training)
                    if current_training != expected_training:
                        raise ValueError(f"训练参数不一致：{metrics_path}")
                    rows.append(result_row(record, run_root, metrics_path, checkpoint))

    frame = pd.DataFrame(rows)
    if frame["architecture"].nunique() < 2:
        raise ValueError("至少需要两个不同模型结构才能比较")
    coverage = {
        architecture: set(zip(group["channel_set"], group["training_seed"]))
        for architecture, group in frame.groupby("architecture")
    }
    reference = next(iter(coverage.values()))
    if any(current != reference for current in coverage.values()):
        raise ValueError("各模型的光谱表示和训练种子覆盖不一致")
    assert protocol is not None
    return frame.sort_values(
        ["architecture", "channel_set", "training_seed"]
    ).reset_index(drop=True), protocol


def summarize(results: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summary = results.groupby(
        ["architecture", "channel_set"], as_index=False, sort=False
    ).agg(
        runs=("training_seed", "count"),
        validation_accuracy_mean=("best_validation_accuracy", "mean"),
        validation_accuracy_std=("best_validation_accuracy", lambda x: float(x.std(ddof=0))),
        validation_macro_f1_mean=("best_validation_macro_f1", "mean"),
        validation_macro_f1_std=("best_validation_macro_f1", lambda x: float(x.std(ddof=0))),
        validation_loss_mean=("best_validation_loss", "mean"),
        test_accuracy_mean=("test_accuracy", "mean"),
        test_accuracy_std=("test_accuracy", lambda x: float(x.std(ddof=0))),
        test_macro_f1_mean=("test_macro_f1", "mean"),
        robustness_accuracy_mean=("robustness_accuracy", "mean"),
        robustness_accuracy_std=("robustness_accuracy", lambda x: float(x.std(ddof=0))),
        parameters_mean=("trainable_parameters", "mean"),
        training_seconds_mean=("training_seconds", "mean"),
    )
    ordering = [
        "validation_accuracy_mean", "validation_macro_f1_mean",
        "validation_loss_mean", "parameters_mean",
    ]
    ascending = [False, False, True, True]
    summary = summary.sort_values(ordering, ascending=ascending, kind="stable").reset_index(drop=True)
    summary.insert(0, "validation_rank", np.arange(1, len(summary) + 1))

    model_summary = (
        summary.groupby("architecture", as_index=False, sort=False).head(1)
        .sort_values(ordering, ascending=ascending, kind="stable")
        .reset_index(drop=True)
    )
    model_summary.insert(0, "model_rank", np.arange(1, len(model_summary) + 1))
    return summary, model_summary


def select_model(
    results: pd.DataFrame,
    model_summary: pd.DataFrame,
    comparison_id: str,
) -> Dict[str, Any]:
    winner = model_summary.iloc[0]
    candidates = results[
        (results["architecture"] == winner["architecture"])
        & (results["channel_set"] == winner["channel_set"])
    ].sort_values(
        [
            "best_validation_accuracy", "best_validation_macro_f1",
            "best_validation_loss", "training_seed",
        ],
        ascending=[False, False, True, True],
        kind="stable",
    )
    chosen = candidates.iloc[0]
    return {
        "comparison_id": comparison_id,
        "selection_scope": "validation_only_across_completed_runs",
        "architecture": chosen["architecture"],
        "channel_set": chosen["channel_set"],
        "training_seed": int(chosen["training_seed"]),
        "checkpoint": chosen["checkpoint"],
        "metrics_file": chosen["metrics_file"],
        "note": "先为每个模型选择验证表现最佳的通道组合，再按同一验证规则比较模型；测试集不参与选择",
    }


def _plot_order(
    results: pd.DataFrame,
    model_summary: pd.DataFrame,
) -> Tuple[List[str], List[str]]:
    architectures = model_summary.sort_values("model_rank")["architecture"].astype(str).tolist()
    present_channels = set(results["channel_set"].astype(str))
    channels = [name for name in available_representations() if name in present_channels]
    return architectures, channels


def plot_all_confusions(
    results: pd.DataFrame,
    model_summary: pd.DataFrame,
    path: Path,
) -> None:
    architectures, channels = _plot_order(results, model_summary)
    fig, axes = plt.subplots(
        len(architectures), len(channels),
        figsize=(4.5 * len(channels), 4.1 * len(architectures)),
        squeeze=False, constrained_layout=True,
    )
    image = None
    for model_index, architecture in enumerate(architectures):
        for channel_index, channel_set in enumerate(channels):
            axis = axes[model_index, channel_index]
            subset = results[
                (results["architecture"] == architecture)
                & (results["channel_set"] == channel_set)
            ].sort_values("training_seed")
            if subset.empty:
                raise RuntimeError(f"缺少组合结果：{architecture}/{channel_set}")
            matrices = []
            for metrics_file in subset["metrics_file"]:
                record = json.loads(Path(metrics_file).read_text(encoding="utf-8"))
                matrix = np.asarray(record["test"]["confusion_matrix"], dtype=np.int64)
                if matrix.shape != (len(CLASS_NAMES), len(CLASS_NAMES)) or np.any(matrix < 0):
                    raise ValueError(f"测试混淆矩阵格式错误：{metrics_file}")
                matrices.append(matrix)
            counts = np.sum(matrices, axis=0)
            row_totals = counts.sum(axis=1, keepdims=True)
            percentages = np.divide(
                counts * 100.0,
                row_totals,
                out=np.zeros_like(counts, dtype=np.float64),
                where=row_totals > 0,
            )
            image = axis.imshow(percentages, cmap="Blues", vmin=0, vmax=100)
            axis.set(
                xticks=np.arange(len(CLASS_NAMES)),
                yticks=np.arange(len(CLASS_NAMES)),
                xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES,
                xlabel="预测类别",
                ylabel="真实类别",
                title=f"{architecture} | {channel_set}\n{len(matrices)} 个种子",
            )
            for row in range(len(CLASS_NAMES)):
                for column in range(len(CLASS_NAMES)):
                    value = percentages[row, column]
                    axis.text(
                        column, row, f"{value:.1f}%\n({counts[row, column]})",
                        ha="center", va="center",
                        color="white" if value >= 50 else "black",
                        fontsize=7.5,
                    )
    assert image is not None
    fig.colorbar(image, ax=axes.ravel().tolist(), label="真实类别内比例（%）", shrink=.8)
    fig.suptitle("全部模型 × 通道方案：原始测试集混淆矩阵")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_model_training_curves(
    results: pd.DataFrame,
    model_summary: pd.DataFrame,
    output_dir: Path,
) -> List[Path]:
    architectures, channels = _plot_order(results, model_summary)
    panels = (
        ("train_loss", "训练损失", None),
        ("val_loss", "验证损失", None),
        ("val_accuracy", "验证正确率", (0, 1.03)),
        ("val_macro_f1", "验证 Macro-F1", (0, 1.03)),
    )
    paths = []
    channel_colors = {
        channel: color
        for channel, color in zip(
            channels, plt.get_cmap("tab10")(np.linspace(0, .75, len(channels)))
        )
    }
    for architecture in architectures:
        model_results = results[results["architecture"] == architecture]
        seeds = sorted(model_results["training_seed"].astype(int).unique().tolist())
        fig, axes = plt.subplots(
            len(seeds), len(panels),
            figsize=(4.6 * len(panels), 3.8 * len(seeds)),
            squeeze=False, sharex="row",
        )
        for seed_index, seed in enumerate(seeds):
            for channel_set in channels:
                subset = model_results[
                    (model_results["channel_set"] == channel_set)
                    & (model_results["training_seed"] == seed)
                ]
                if len(subset) != 1:
                    raise RuntimeError(
                        f"{architecture}/{channel_set}/seed_{seed}结果数量应为1"
                    )
                history_path = Path(subset.iloc[0]["metrics_file"]).with_name("history.csv")
                if not history_path.is_file():
                    raise FileNotFoundError(f"缺少训练记录：{history_path}")
                history = pd.read_csv(history_path)
                required = {"epoch", *(item[0] for item in panels)}
                if not required.issubset(history):
                    raise ValueError(f"训练记录字段不完整：{history_path}")
                for panel_index, (column, title, limits) in enumerate(panels):
                    axis = axes[seed_index, panel_index]
                    axis.plot(
                        history["epoch"], history[column],
                        color=channel_colors[channel_set], label=channel_set,
                        linewidth=1.7,
                    )
                    axis.set(
                        title=f"seed_{seed} | {title}",
                        xlabel="轮次", ylabel=title,
                    )
                    if limits is not None:
                        axis.set_ylim(*limits)
                    axis.grid(alpha=.2)
            axes[seed_index, 0].legend(title="通道方案", loc="best")
        fig.suptitle(f"{architecture}：四种通道方案训练过程（每种颜色为一个通道方案）")
        fig.tight_layout(rect=(0, 0, 1, .97))
        path = output_dir / f"training_curves_{architecture}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = ChineseParser(description="单独比较多个已经训练完成的模型实验")
    parser.add_argument("--config", required=True, help="模型比较JSON配置")
    parser.add_argument("--output-dir", help="覆盖比较结果目录")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    setup_console()
    args = build_parser().parse_args(argv)
    config_path = Path(args.config).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"比较配置不存在：{config_path}")
    config = load_config(config_path)
    run_roots = [resolve_path(value) for value in config["runs"]]
    if len(run_roots) != len(set(run_roots)):
        raise ValueError("runs解析后指向重复实验目录")
    output_dir = resolve_path(
        args.output_dir
        or config.get("output_dir", OUTPUT_DIR / config["comparison_id"])
    )
    if output_dir in run_roots:
        raise ValueError("比较输出目录不能与已有训练实验目录相同")
    output_dir.mkdir(parents=True, exist_ok=True)

    results, protocol = load_completed_runs(run_roots)
    summary, model_summary = summarize(results)
    selection = select_model(results, model_summary, config["comparison_id"])

    results.to_csv(output_dir / "results.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "summary.csv", index=False, encoding="utf-8-sig")
    model_summary.to_csv(output_dir / "model_summary.csv", index=False, encoding="utf-8-sig")
    plot_accuracy(summary, output_dir / "comparison.png")
    plot_all_confusions(results, model_summary, output_dir / "confusion_matrices.png")
    plot_model_training_curves(results, model_summary, output_dir)
    legacy_training_plot = output_dir / "training_curves.png"
    if legacy_training_plot.is_file():
        legacy_training_plot.unlink()
    (output_dir / "selected_model.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    snapshot = {
        "comparison_id": config["comparison_id"],
        "source_config": str(config_path),
        "runs": [str(path) for path in run_roots],
        "protocol": protocol,
    }
    (output_dir / "config.json").write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("完成 | 已训练结果协议一致 | 每个模型先按验证集选择最佳通道 | 已生成汇总诊断图")
    display_columns = [
        "model_rank", "architecture", "channel_set",
        "validation_accuracy_mean", "test_accuracy_mean", "parameters_mean",
    ]
    if model_summary["robustness_accuracy_mean"].notna().any():
        display_columns.insert(-1, "robustness_accuracy_mean")
    print(model_summary[display_columns].to_string(index=False))
    print(f"选择: {selection['architecture']}/{selection['channel_set']} | {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError, argparse.ArgumentTypeError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        raise SystemExit(2)
