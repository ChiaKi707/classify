# -*- coding: utf-8 -*-
"""训练一个“模型结构 × 光谱表示 × 随机种子”组合。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import (
    ANALYTE_NAMES, CHANNEL_SETS, CHECKPOINT_FORMAT_VERSION, CLASS_NAMES,
    ChineseParser, OUTPUT_DIR, PREPARED_DATA_DIR, PT_DIR, ROOT, TASK_NAME,
    choose_device, set_seed, setup_console,
)
from data import (
    SpectralPreprocessor, SpectraDataset, data_fingerprints,
    load_prepared_data, preparation_file_paths,
)
from metrics import compute_metrics, json_default, plot_diagnostics, predict_loader
from models import available_architectures, build_model, get_architecture
from representations import available_representations, get_representation
from research_protocol import (
    EXPERIMENT_PROTOCOL_VERSION, code_fingerprints, environment_snapshot,
    normalize_run_id, preparation_provenance,
)
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
    total = 0.0
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x_batch), y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total += float(loss.item()) * len(x_batch)
    return total / len(loader.dataset)


def parse_model_options(value: str | Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    text = value.strip()
    if text.startswith("@"):
        text = Path(text[1:]).read_text(encoding="utf-8")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--model-options不是有效JSON：{exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("--model-options必须是JSON对象")
    return parsed


def resolve_arguments(args: argparse.Namespace) -> None:
    get_representation(args.channel_set)
    get_architecture(args.architecture)
    args.model_options = parse_model_options(args.model_options)
    args.run_id = normalize_run_id(args.run_id)
    args.prepared_dir = str(Path(args.prepared_dir).expanduser().resolve())


def experiment_directory(root: str | Path, args: argparse.Namespace) -> Path:
    return (
        Path(root).expanduser().resolve()
        / args.run_id / args.architecture / args.channel_set / f"seed_{args.train_seed}"
    )


def _loader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    device: torch.device,
    workers: int,
) -> DataLoader:
    return DataLoader(
        SpectraDataset(x, y), batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=device.type == "cuda",
    )


def command_train(args: argparse.Namespace) -> Dict[str, Any]:
    validate_arguments(args)
    resolve_arguments(args)
    set_seed(args.train_seed)
    device = choose_device(args.device)
    prepared = load_prepared_data(args.prepared_dir)
    train, validation, test = (
        prepared[name] for name in ("train", "validation", "test")
    )
    robustness = prepared.splits.get("robustness")

    output_dir = experiment_directory(args.output_dir, args)
    weight_dir = experiment_directory(args.pt_dir, args)
    output_dir.mkdir(parents=True, exist_ok=True)
    weight_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = weight_dir / "best.pt"

    clean_train = train.metadata["augmentation_index"].to_numpy(np.int64) == 0
    preprocessor = SpectralPreprocessor(args.preprocess_clip, channel_set=args.channel_set)
    preprocessor.fit(train.spectra[clean_train], prepared.axis)
    x_train = preprocessor.transform(train.spectra)
    x_val = preprocessor.transform(validation.spectra)
    x_test = preprocessor.transform(test.spectra)
    x_robust = None if robustness is None else preprocessor.transform(robustness.spectra)

    loader_options = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(
        SpectraDataset(x_train, train.targets),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.train_seed),
        **loader_options,
    )
    val_loader = _loader(x_val, validation.targets, args.batch_size * 2, device, args.num_workers)
    test_loader = _loader(x_test, test.targets, args.batch_size * 2, device, args.num_workers)
    robustness_loader = (
        None if robustness is None else
        _loader(x_robust, robustness.targets, args.batch_size * 2, device, args.num_workers)
    )

    model_config = {
        "in_channels": int(x_train.shape[1]),
        "dropout": args.dropout,
        "num_classes": len(CLASS_NAMES),
    }
    model = build_model(
        args.architecture, **model_config, model_options=args.model_options
    ).to(device)
    class_counts = np.bincount(train.targets, minlength=len(CLASS_NAMES))
    if np.any(class_counts == 0):
        raise ValueError("训练集未同时包含三个类别")
    class_weights = len(train.targets) / (len(CLASS_NAMES) * class_counts)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.learning_rate * .03
    )

    parameters = sum(item.numel() for item in model.parameters() if item.requires_grad)
    data_counts = f"{len(train.targets)} / {len(validation.targets)} / {len(test.targets)}"
    data_names = "训练/验证/测试"
    if robustness is not None:
        data_counts += f" / {len(robustness.targets)}"
        data_names += "/鲁棒性"
    print(
        f"[{args.architecture}/{args.channel_set}/seed_{args.train_seed}] {device} | "
        f"参数 {parameters:,} | {data_counts} ({data_names})"
    )

    monitor = ValidationMonitor(args.min_delta, args.patience)
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_validation: Optional[Dict[str, Any]] = None
    history: List[Dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, args.grad_clip)
        val_probs, val_true, val_loss = predict_loader(model, val_loader, device)
        assert val_true is not None
        val_metrics = compute_metrics(val_true, val_probs)
        decision = monitor.update(epoch, val_loss)
        if decision.checkpoint_updated:
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            best_validation = dict(val_metrics)
        history.append(
            {
                "epoch": epoch,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_metrics["exact_match_accuracy"],
                "val_macro_f1": val_metrics["macro_f1_three_class"],
            }
        )
        scheduler.step()
        if epoch == 1 or epoch % 5 == 0 or decision.should_stop or epoch == args.epochs:
            print(
                f"  epoch {epoch:03d} | train {train_loss:.5g} | val {val_loss:.5g} | "
                f"acc {val_metrics['exact_match_accuracy']:.4f} | wait {monitor.stale_epochs}"
            )
        if decision.should_stop:
            break

    training_seconds = time.perf_counter() - started
    if best_state is None or best_validation is None:
        raise RuntimeError("训练未生成有效模型")
    model.load_state_dict(best_state)
    training_control = monitor.summary()

    test_probs, test_true, test_loss = predict_loader(model, test_loader, device)
    assert test_true is not None
    test_metrics = compute_metrics(test_true, test_probs)
    test_metrics["cross_entropy"] = test_loss
    test_metrics["sample_counts"] = {
        "independent_original_experiments": int(clean_train.sum()) + len(validation.targets) + len(test.targets),
        "train_original": int(clean_train.sum()),
        "train_augmented": int((~clean_train).sum()),
        "train_rows": len(train.targets),
        "validation_original": len(validation.targets),
        "test_original": len(test.targets),
        "robustness_augmented": 0 if robustness is None else len(robustness.targets),
    }
    robustness_metrics = None
    if robustness_loader is not None:
        robust_probs, robust_true, robust_loss = predict_loader(model, robustness_loader, device)
        assert robust_true is not None
        robustness_metrics = compute_metrics(robust_true, robust_probs)
        robustness_metrics["cross_entropy"] = robust_loss

    saved_args = dict(vars(args))
    prepared_files = preparation_file_paths(prepared.directory)
    provenance = preparation_provenance(prepared.directory)
    fingerprints = code_fingerprints(ROOT)
    experiment = {
        "run_id": args.run_id,
        "experiment_id": f"{args.architecture}__{args.channel_set}",
        "architecture": args.architecture,
        "backend": get_architecture(args.architecture).backend,
        "model_options": args.model_options,
        "channel_set": args.channel_set,
        "channel_names": CHANNEL_SETS[args.channel_set],
        "training_seed": args.train_seed,
        "protocol_version": EXPERIMENT_PROTOCOL_VERSION,
    }
    common = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "training_policy_version": TRAINING_POLICY_VERSION,
        "protocol_version": EXPERIMENT_PROTOCOL_VERSION,
        "task": TASK_NAME,
        "experiment": experiment,
        "training_control": training_control,
        "training_args": saved_args,
        "data_files": data_fingerprints(prepared_files),
        "preparation": provenance,
        "code_fingerprints": fingerprints,
        "environment": environment_snapshot(),
        "model_config": model_config,
    }
    result = {
        **common,
        "validation": {
            "best_epoch": monitor.best_epoch,
            "loss": monitor.best_loss,
            "accuracy": best_validation["exact_match_accuracy"],
            "macro_f1": best_validation["macro_f1_three_class"],
        },
        "test": test_metrics,
        "robustness": robustness_metrics,
        "trainable_parameters": parameters,
        "training_seconds": training_seconds,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )
    checkpoint = {
        **common,
        "architecture": args.architecture,
        "model_name": get_architecture(args.architecture).display_name,
        "model_options": args.model_options,
        "model_state": best_state,
        "preprocessor": preprocessor.state_dict(),
        "reference_axis": torch.from_numpy(prepared.axis.copy()),
        "analyte_names": ANALYTE_NAMES,
        "class_names": CLASS_NAMES,
        "metrics": test_metrics,
        "robustness_metrics": robustness_metrics,
    }
    torch.save(checkpoint, checkpoint_path)

    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
    plot_diagnostics(
        history,
        test_metrics["confusion_matrix"],
        None if robustness_metrics is None else robustness_metrics["confusion_matrix"],
        output_dir / "diagnostics.png",
        best_epoch=monitor.best_epoch,
    )
    final_line = (
        f"  done | best {monitor.best_epoch} | val {monitor.best_loss:.3g} | "
        f"test {test_metrics['exact_match_accuracy']:.4f}"
    )
    if robustness_metrics is not None:
        final_line += f" | robust {robustness_metrics['exact_match_accuracy']:.4f}"
    print(final_line)
    print(f"  {output_dir}")
    return {"output_dir": output_dir, "checkpoint_path": checkpoint_path, "metrics": test_metrics}


def add_training_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--prepared-dir", default=str(PREPARED_DATA_DIR), help="准备数据目录")
    parser.add_argument("--pt-dir", default=str(PT_DIR), help="权重根目录")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="结果根目录")
    parser.add_argument("--run-id", default="manual")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=.15)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--preprocess-clip", type=float, default=10.0)
    parser.add_argument("--train-seed", type=int, default=66)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--num-workers", type=int, default=0)


def validate_arguments(args: argparse.Namespace) -> None:
    for name in ("epochs", "batch_size", "patience"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')}必须>=1")
    for name in ("learning_rate", "grad_clip", "preprocess_clip"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"--{name.replace('_', '-')}必须是大于0的有限数")
    for name in ("weight_decay", "min_delta"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"--{name.replace('_', '-')}必须是非负有限数")
    if not 0 <= args.dropout < 1 or args.num_workers < 0:
        raise ValueError("dropout必须在[0,1)，num-workers必须>=0")
    if not 0 <= args.train_seed < 2**32:
        raise ValueError("train-seed必须在[0,2**32)内")


def build_parser() -> argparse.ArgumentParser:
    parser = ChineseParser(description="训练一个模型结构、光谱表示和随机种子组合")
    add_training_arguments(parser)
    parser.add_argument("--architecture", choices=available_architectures(), default="msnet")
    parser.add_argument("--channel-set", choices=available_representations(), default="raw_snv_d1")
    parser.add_argument("--model-options", default="{}", help="模型结构选项JSON")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    setup_console()
    args = build_parser().parse_args(argv)
    command_train(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError, argparse.ArgumentTypeError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        raise SystemExit(2)
