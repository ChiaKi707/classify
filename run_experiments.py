# -*- coding: utf-8 -*-
"""运行模型结构 × 光谱表示 × 训练种子的统一实验矩阵。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from config import ChineseParser, OUTPUT_DIR, PREPARED_DATA_DIR, PT_DIR, ROOT, setup_console
from data import data_fingerprints, preparation_file_paths, require_prepared_data
from metrics import plt
from models import get_architecture
from representations import available_representations, get_representation
from research_protocol import (
    EXPERIMENT_PROTOCOL_VERSION, code_fingerprints, normalize_run_id,
    preparation_provenance,
)
from train import (
    build_parser as build_train_parser, experiment_directory, resolve_arguments,
    validate_arguments as validate_train_arguments,
)
from training_control import TRAINING_POLICY_VERSION


TRAIN_SCRIPT = ROOT / "train.py"
TRAINING_KEYS = (
    "epochs", "batch_size", "learning_rate", "weight_decay", "dropout",
    "patience", "min_delta", "grad_clip", "preprocess_clip", "device",
    "num_workers",
)


def load_experiment_config(path: Path) -> Dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = ("run_id", "architectures", "channel_sets", "training_seeds")
    if not isinstance(config, dict) or any(key not in config for key in required):
        raise ValueError("实验配置缺少run_id、architectures、channel_sets或training_seeds")
    config["run_id"] = normalize_run_id(str(config["run_id"]))

    architectures = []
    for item in config["architectures"]:
        item = {"name": item, "model_options": {}} if isinstance(item, str) else item
        if not isinstance(item, dict) or "name" not in item:
            raise ValueError("architecture必须是名称或包含name的对象")
        name = str(item["name"])
        get_architecture(name)
        options = item.get("model_options", {})
        if not isinstance(options, dict):
            raise ValueError(f"{name}的model_options必须是对象")
        architectures.append({"name": name, "model_options": options})
    if not architectures:
        raise ValueError("architectures不能为空")
    names = [item["name"] for item in architectures]
    if len(names) != len(set(names)):
        raise ValueError("architectures中的name不能重复；不同结构变体请使用不同注册名")
    config["architectures"] = architectures

    if not isinstance(config["channel_sets"], list) or not config["channel_sets"]:
        raise ValueError("channel_sets不能为空")
    for name in config["channel_sets"]:
        get_representation(str(name))
    if len(config["channel_sets"]) != len(set(config["channel_sets"])):
        raise ValueError("channel_sets不能重复")
    seeds = config["training_seeds"]
    if (
        not isinstance(seeds, list) or not seeds or len(seeds) != len(set(seeds))
        or any(not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**32 for seed in seeds)
    ):
        raise ValueError("training_seeds必须是不重复的有效整数数组")
    training = config.setdefault("training", {})
    if not isinstance(training, dict):
        raise ValueError("training必须是对象")
    unknown = set(training) - set(TRAINING_KEYS)
    if unknown:
        raise ValueError("training包含未知参数：" + "、".join(sorted(unknown)))
    return config


def parse_seeds(value: Optional[str]) -> Optional[List[int]]:
    if value is None:
        return None
    try:
        seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("训练种子必须是逗号分隔整数") from exc
    if (
        not seeds
        or len(seeds) != len(set(seeds))
        or any(not 0 <= seed < 2**32 for seed in seeds)
    ):
        raise argparse.ArgumentTypeError("训练种子必须是不重复的[0,2**32)整数")
    return seeds


def resolve_prepared_dir(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return require_prepared_data(path)


def build_train_arguments(
    config: Dict[str, Any],
    architecture: Dict[str, Any],
    channel_set: str,
    seed: int,
    *,
    run_id: str,
    prepared_dir: Path,
    pt_dir: Path,
    output_dir: Path,
    device: Optional[str],
    epochs: Optional[int],
) -> List[str]:
    arguments = [
        "--architecture", architecture["name"],
        "--channel-set", channel_set,
        "--model-options", json.dumps(architecture["model_options"], separators=(",", ":")),
        "--run-id", run_id,
        "--prepared-dir", str(prepared_dir),
        "--pt-dir", str(pt_dir),
        "--output-dir", str(output_dir),
        "--train-seed", str(seed),
    ]
    training = dict(config.get("training", {}))
    if device is not None:
        training["device"] = device
    if epochs is not None:
        training["epochs"] = epochs
    for key, value in training.items():
        arguments.extend(["--" + key.replace("_", "-"), str(value)])
    return arguments


def parsed_train_arguments(arguments: Sequence[str]) -> argparse.Namespace:
    parsed = build_train_parser().parse_args(list(arguments))
    validate_train_arguments(parsed)
    resolve_arguments(parsed)
    return parsed


def required_result_files(output_dir: Path, checkpoint: Path) -> Tuple[Path, ...]:
    return checkpoint, output_dir / "metrics.json", output_dir / "history.csv", output_dir / "diagnostics.png"


def can_skip(
    parsed: argparse.Namespace,
    prepared_dir: Path,
    output_dir: Path,
    checkpoint: Path,
) -> bool:
    if not all(path.is_file() for path in required_result_files(output_dir, checkpoint)):
        return False
    try:
        saved = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    current_args = dict(vars(parsed))
    files = preparation_file_paths(prepared_dir)
    return (
        saved.get("training_policy_version") == TRAINING_POLICY_VERSION
        and saved.get("protocol_version") == EXPERIMENT_PROTOCOL_VERSION
        and saved.get("training_args") == current_args
        and saved.get("data_files") == data_fingerprints(files)
        and saved.get("preparation") == preparation_provenance(prepared_dir)
        and saved.get("code_fingerprints") == code_fingerprints(ROOT)
    )


def verify_identical_protocols(output_dirs: Sequence[Path]) -> None:
    records = [
        json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
        for directory in output_dirs
    ]
    reference = records[0]
    expected = (
        reference["data_files"],
        reference["preparation"]["config_sha256"],
        reference["test"]["sample_counts"],
    )
    for directory, record in zip(output_dirs[1:], records[1:]):
        current = (
            record["data_files"],
            record["preparation"]["config_sha256"],
            record["test"]["sample_counts"],
        )
        if current != expected:
            raise RuntimeError(f"{directory}的数据协议与其他实验不一致")


def collect_results(
    run_id: str,
    output_root: Path,
    pt_root: Path,
    runs: Sequence[argparse.Namespace],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    run_root = output_root / run_id
    rows = []
    for parsed in runs:
        record = json.loads(
            (experiment_directory(output_root, parsed) / "metrics.json").read_text(encoding="utf-8")
        )
        test, validation = record["test"], record["validation"]
        control = record["training_control"]
        rows.append(
            {
                "architecture": parsed.architecture,
                "channel_set": parsed.channel_set,
                "training_seed": parsed.train_seed,
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
                "training_seconds": record["training_seconds"],
            }
        )
    all_runs = pd.DataFrame(rows).sort_values(["architecture", "channel_set", "training_seed"])
    all_runs.to_csv(run_root / "results.csv", index=False, encoding="utf-8-sig")

    summary = all_runs.groupby(["architecture", "channel_set"], as_index=False, sort=False).agg(
        runs=("training_seed", "count"),
        validation_accuracy_mean=("best_validation_accuracy", "mean"),
        validation_accuracy_std=("best_validation_accuracy", lambda x: float(x.std(ddof=0))),
        validation_macro_f1_mean=("best_validation_macro_f1", "mean"),
        validation_macro_f1_std=("best_validation_macro_f1", lambda x: float(x.std(ddof=0))),
        validation_loss_mean=("best_validation_loss", "mean"),
        test_accuracy_mean=("test_accuracy", "mean"),
        test_accuracy_std=("test_accuracy", lambda x: float(x.std(ddof=0))),
        test_macro_f1_mean=("test_macro_f1", "mean"),
        parameters_mean=("trainable_parameters", "mean"),
        training_seconds_mean=("training_seconds", "mean"),
    )
    summary = summary.sort_values(
        ["validation_accuracy_mean", "validation_macro_f1_mean", "validation_loss_mean", "parameters_mean"],
        ascending=[False, False, True, True],
        kind="stable",
    ).reset_index(drop=True)
    summary.insert(0, "validation_rank", np.arange(1, len(summary) + 1))
    summary.to_csv(run_root / "summary.csv", index=False, encoding="utf-8-sig")

    winner = summary.iloc[0]
    candidates = all_runs[
        (all_runs["architecture"] == winner["architecture"])
        & (all_runs["channel_set"] == winner["channel_set"])
    ].sort_values(
        ["best_validation_accuracy", "best_validation_macro_f1", "best_validation_loss", "training_seed"],
        ascending=[False, False, True, True],
        kind="stable",
    )
    chosen = candidates.iloc[0]
    seed = int(chosen["training_seed"])
    selection = {
        "run_id": run_id,
        "selection_scope": "validation_only",
        "architecture": chosen["architecture"],
        "channel_set": chosen["channel_set"],
        "training_seed": seed,
        "checkpoint": str(pt_root / run_id / chosen["architecture"] / chosen["channel_set"] / f"seed_{seed}" / "best.pt"),
        "output_dir": str(output_root / run_id / chosen["architecture"] / chosen["channel_set"] / f"seed_{seed}"),
        "note": "测试指标只用于最终报告，不参与模型选择",
    }
    (run_root / "selected_model.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot_accuracy(summary, run_root / "comparison.png")
    plot_training_curves(all_runs, run_root)
    return all_runs, summary, selection


def plot_accuracy(summary: pd.DataFrame, path: Path) -> None:
    architectures = summary["architecture"].drop_duplicates().tolist()
    present = set(summary["channel_set"])
    channels = [name for name in available_representations() if name in present]
    x = np.arange(len(channels), dtype=float)
    width = .8 / len(architectures)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for index, architecture in enumerate(architectures):
        subset = summary[summary["architecture"] == architecture].set_index("channel_set")
        values = [subset.loc[name, "test_accuracy_mean"] if name in subset.index else np.nan for name in channels]
        errors = [subset.loc[name, "test_accuracy_std"] if name in subset.index else np.nan for name in channels]
        offset = (index - (len(architectures) - 1) / 2) * width
        ax.bar(x + offset, values, width, yerr=errors, capsize=3, label=architecture)
    ax.set_xticks(x, [get_representation(name).display_name for name in channels], rotation=12)
    ax.set(ylim=(0, 1.03), ylabel="测试集正确率", title="模型结构 × 光谱表示")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_training_curves(results: pd.DataFrame, run_root: Path) -> List[Path]:
    """每个模型一张图；四种通道用不同颜色，多种子逐行显示。"""
    panels = (
        ("train_loss", "训练损失", None),
        ("val_loss", "验证损失", None),
        ("val_accuracy", "验证正确率", (0, 1.03)),
        ("val_macro_f1", "验证 Macro-F1", (0, 1.03)),
    )
    architectures = results["architecture"].drop_duplicates().astype(str).tolist()
    present = set(results["channel_set"].astype(str))
    channels = [name for name in available_representations() if name in present]
    colors = {
        channel: color
        for channel, color in zip(
            channels, plt.get_cmap("tab10")(np.linspace(0, .75, len(channels)))
        )
    }
    paths: List[Path] = []
    for architecture in architectures:
        subset_model = results[results["architecture"] == architecture]
        seeds = sorted(subset_model["training_seed"].astype(int).unique().tolist())
        fig, axes = plt.subplots(
            len(seeds), len(panels),
            figsize=(4.6 * len(panels), 3.8 * len(seeds)),
            squeeze=False, sharex="row",
        )
        for seed_index, seed in enumerate(seeds):
            for channel in channels:
                history_path = (
                    run_root / architecture / channel / f"seed_{seed}" / "history.csv"
                )
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
                        color=colors[channel], label=channel, linewidth=1.8,
                    )
                    axis.set(
                        title=f"seed_{seed} | {title}",
                        xlabel="轮次", ylabel=title,
                    )
                    if limits is not None:
                        axis.set_ylim(*limits)
                    axis.grid(alpha=.2)
            axes[seed_index, 0].legend(title="通道方案", loc="best")
        fig.suptitle(f"{architecture}：四种递增通道训练过程")
        fig.tight_layout(rect=(0, 0, 1, .97))
        path = (
            run_root / "training_curves.png"
            if len(architectures) == 1
            else run_root / f"training_curves_{architecture}.png"
        )
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = ChineseParser(description="运行模型结构×光谱表示×训练种子的实验矩阵")
    parser.add_argument("--config", required=True, help="实验JSON配置")
    parser.add_argument("--run-id", help="覆盖运行编号")
    parser.add_argument("--prepared-dir", help="覆盖准备数据目录")
    parser.add_argument("--pt-dir", default=str(PT_DIR))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--training-seeds", help="逗号分隔，覆盖配置")
    parser.add_argument("--epochs", type=int, help="覆盖最大轮数")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    setup_console()
    args = build_parser().parse_args(argv)
    config_path = Path(args.config).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"实验配置不存在：{config_path}")
    config = load_experiment_config(config_path)
    run_id = normalize_run_id(args.run_id or config["run_id"])
    prepared_dir = resolve_prepared_dir(args.prepared_dir or config.get("prepared_dir", PREPARED_DATA_DIR))
    pt_root = Path(args.pt_dir).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()
    seeds = parse_seeds(args.training_seeds) or config["training_seeds"]

    commands: List[Tuple[List[str], argparse.Namespace]] = []
    for architecture in config["architectures"]:
        for channel_set in config["channel_sets"]:
            for seed in seeds:
                arguments = build_train_arguments(
                    config, architecture, channel_set, seed,
                    run_id=run_id, prepared_dir=prepared_dir, pt_dir=pt_root,
                    output_dir=output_root, device=args.device, epochs=args.epochs,
                )
                commands.append((arguments, parsed_train_arguments(arguments)))
    if args.dry_run:
        print(f"配置有效，共 {len(commands)} 个实验：")
        for arguments, _ in commands:
            print(" ".join([sys.executable, "-B", str(TRAIN_SCRIPT), *arguments]))
        return 0

    run_root = output_root / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    snapshot = {
        "source_config": str(config_path),
        "effective_run_id": run_id,
        "prepared_dir": str(prepared_dir),
        "config": config,
        "training_seeds": seeds,
    }
    (run_root / "config.json").write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    shutil.copy2(prepared_dir / "split_manifest.csv", run_root / "split.csv")

    runs = [parsed for _, parsed in commands]
    output_dirs = [experiment_directory(output_root, parsed) for parsed in runs]
    for index, (arguments, parsed) in enumerate(commands, 1):
        output_dir = experiment_directory(output_root, parsed)
        checkpoint = experiment_directory(pt_root, parsed) / "best.pt"
        print(f"\n[{index}/{len(commands)}] {parsed.architecture} × {parsed.channel_set} × seed_{parsed.train_seed}")
        reusable = can_skip(parsed, prepared_dir, output_dir, checkpoint)
        if args.summary_only:
            if not reusable:
                raise RuntimeError(f"结果不完整或协议不一致：{output_dir}")
        elif args.skip_existing and reusable:
            print(f"严格一致，复用完整结果：{output_dir}")
        else:
            subprocess.run([sys.executable, "-B", str(TRAIN_SCRIPT), *arguments], cwd=ROOT, check=True)

    verify_identical_protocols(output_dirs)
    _, summary, selection = collect_results(run_id, output_root, pt_root, runs)
    print("\n完成 | 显式拆分一致 | 按验证集排名")
    display_columns = [
        "validation_rank", "architecture", "channel_set",
        "validation_accuracy_mean", "test_accuracy_mean",
    ]
    print(summary[display_columns].to_string(index=False))
    print(f"选择: {selection['architecture']}/{selection['channel_set']}/seed_{selection['training_seed']} | {run_root}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError, argparse.ArgumentTypeError, subprocess.CalledProcessError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        raise SystemExit(2)
