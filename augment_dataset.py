# -*- coding: utf-8 -*-
"""一次完成原始实验分层拆分、训练集离线增强和可追溯CSV导出。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    ChineseParser, PREPARED_DATA_DIR, RAW_DATA_DIR, RAW_SOURCES,
    ROLE_CLASS_IDS, setup_console,
)
from data import (
    CORE_SPLIT_FILES, PREPARED_METADATA, SPLIT_FILES, concentration_group, data_fingerprints,
    load_prepared_data, read_raw_csv, sha256_file,
)


PREPARATION_FORMAT_VERSION = 2


def shift_with_edge_values(spectrum: np.ndarray, shift: int) -> np.ndarray:
    result = np.roll(spectrum, shift).copy()
    if shift > 0:
        result[:shift] = result[shift]
    elif shift < 0:
        result[shift:] = result[shift - 1]
    return result


def _selected(rng: np.random.Generator, probability: float) -> bool:
    return probability >= 1 or (probability > 0 and rng.random() < probability)


def augment_one(
    spectrum: np.ndarray,
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    augmented = np.asarray(spectrum, dtype=np.float64).copy()
    trace: List[Dict[str, Any]] = []
    for step in range(1, args.steps + 1):
        shift = 0
        shift_applied = args.shift_points > 0 and _selected(rng, args.shift_prob)
        if shift_applied:
            shift = int(rng.integers(-args.shift_points, args.shift_points + 1))
            augmented = shift_with_edge_values(augmented, shift)

        gain = 1.0
        gain_applied = _selected(rng, args.gain_prob)
        if gain_applied:
            gain = float(rng.uniform(args.gain_min, args.gain_max))
            augmented *= gain

        noise_std = 0.0
        noise_applied = _selected(rng, args.noise_prob)
        if noise_applied:
            noise_std = float(args.noise_ratio * np.std(augmented))
            augmented += rng.normal(0, noise_std, size=augmented.shape)

        trace.append(
            {
                "step": step,
                "shift_applied": shift_applied,
                "shift_points": shift,
                "gain_applied": gain_applied,
                "gain": gain,
                "noise_applied": noise_applied,
                "noise_std": noise_std,
            }
        )
    return augmented.astype(np.float32), {
        "augmentation_steps": args.steps,
        "net_shift_points": int(sum(item["shift_points"] for item in trace)),
        "cumulative_gain": float(np.prod([item["gain"] for item in trace])),
        "max_noise_std": float(max(item["noise_std"] for item in trace)),
        "operation_trace": json.dumps(trace, ensure_ascii=False, separators=(",", ":")),
    }


def validate_arguments(args: argparse.Namespace) -> None:
    ratios = (args.train_ratio, args.val_ratio, args.test_ratio)
    if any(not math.isfinite(value) or value <= 0 for value in ratios):
        raise ValueError("拆分比例必须是大于0的有限数")
    if not math.isclose(sum(ratios), 1, rel_tol=0, abs_tol=1e-9):
        raise ValueError("train/val/test比例之和必须为1")
    if args.copies < 0 or args.steps < 1 or args.shift_points < 0:
        raise ValueError("copies、shift-points必须>=0，steps必须>=1")
    if not 0 < args.gain_min <= args.gain_max:
        raise ValueError("强度倍率必须满足0 < gain-min <= gain-max")
    if not math.isfinite(args.noise_ratio) or args.noise_ratio <= 0:
        raise ValueError("noise-ratio必须是大于0的有限数")
    for name in ("shift_prob", "gain_prob", "noise_prob"):
        value = getattr(args, name)
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name}必须在[0,1]内")
    for name in ("split_seed", "augmentation_seed"):
        if not 0 <= getattr(args, name) < 2**32:
            raise ValueError(f"{name}必须在[0,2**32)内")


def load_originals(source_dir: Path) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, List[Path]]:
    spectra_parts: List[np.ndarray] = []
    metadata_parts: List[pd.DataFrame] = []
    paths: List[Path] = []
    axis: Optional[np.ndarray] = None
    for role, filename in RAW_SOURCES:
        path = source_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"原始数据不存在：{path}")
        spectra, current_axis, metadata, _ = read_raw_csv(path, axis)
        if axis is None:
            axis = current_axis.copy()
        ids = metadata["sample_id"].astype(str).str.strip()
        if ids.eq("").any() or ids.str.lower().eq("nan").any() or ids.duplicated().any():
            raise ValueError(f"{path} 的原始编号必须非空且在文件内唯一")
        metadata = metadata.rename(
            columns={
                "sample_id": "original_sample_id",
                "row_in_file": "source_row",
                "csv_line": "source_csv_line",
            }
        )
        metadata["experiment_id"] = path.stem + ":" + ids
        metadata["source_file"] = path.name
        metadata["role"] = role
        metadata["class_id"] = ROLE_CLASS_IDS[role]
        spectra_parts.append(spectra)
        metadata_parts.append(metadata)
        paths.append(path)
    assert axis is not None
    metadata = pd.concat(metadata_parts, ignore_index=True)
    if metadata["experiment_id"].duplicated().any():
        raise ValueError("文件名＋原始编号没有形成唯一实验ID")
    return np.concatenate(spectra_parts), axis, metadata, paths


def assign_splits(metadata: pd.DataFrame, args: argparse.Namespace) -> pd.Series:
    strata = metadata["source_file"].astype(str) + "|" + metadata["concentration_group"].astype(str)
    indices = np.arange(len(metadata))
    try:
        train_idx, held_idx = train_test_split(
            indices,
            test_size=args.val_ratio + args.test_ratio,
            random_state=args.split_seed,
            shuffle=True,
            stratify=strata,
        )
        val_idx, test_idx = train_test_split(
            held_idx,
            test_size=args.test_ratio / (args.val_ratio + args.test_ratio),
            random_state=args.split_seed + 1,
            shuffle=True,
            stratify=strata.iloc[held_idx],
        )
    except ValueError as exc:
        counts = strata.value_counts().sort_values().head().to_dict()
        raise ValueError(f"联合分层数据不足，最小分层为{counts}：{exc}") from exc
    assignment = pd.Series("", index=metadata.index, dtype="object")
    assignment.iloc[train_idx] = "train"
    assignment.iloc[val_idx] = "validation"
    assignment.iloc[test_idx] = "test"
    return assignment


def stable_rng(seed: int, experiment_id: str, index: int, purpose: str) -> np.random.Generator:
    text = f"{seed}|{experiment_id}|{index}|{purpose}".encode("utf-8")
    derived = int.from_bytes(hashlib.sha256(text).digest()[:8], "little")
    return np.random.default_rng(derived)


def base_record(row: pd.Series, split: str, index: int) -> Dict[str, Any]:
    experiment_id = str(row["experiment_id"])
    return {
        "sample_id": f"{experiment_id}:{split}:{index}",
        "experiment_id": experiment_id,
        "source_file": row["source_file"],
        "source_row": int(row["source_row"]),
        "source_csv_line": int(row["source_csv_line"]),
        "original_sample_id": row["original_sample_id"],
        "role": row["role"],
        "class_id": int(row["class_id"]),
        "concentration_text": row["concentration_text"],
        "concentration_group": row["concentration_group"],
        "split": split,
        "augmentation_index": index,
        "is_augmented": index > 0,
        "repaired_points": int(row["repaired_points"]),
    }


def build_outputs(
    spectra: np.ndarray,
    metadata: pd.DataFrame,
    args: argparse.Namespace,
) -> Tuple[Dict[str, Tuple[List[Dict[str, Any]], List[np.ndarray]]], List[Dict[str, Any]]]:
    output_names = dict(CORE_SPLIT_FILES)
    if args.copies > 0:
        output_names["robustness"] = SPLIT_FILES["robustness"]
    outputs = {name: ([], []) for name in output_names}
    manifest: List[Dict[str, Any]] = []
    empty_summary = {
        "augmentation_steps": 0,
        "net_shift_points": 0,
        "cumulative_gain": 1.0,
        "max_noise_std": 0.0,
        "operation_trace": "[]",
    }
    for spectrum, row in zip(spectra, metadata.itertuples(index=False)):
        record = pd.Series(row._asdict())
        split = str(record["split"])
        raw = base_record(record, split, 0)
        outputs[split][0].append(raw)
        outputs[split][1].append(spectrum.copy())
        manifest.append({**raw, **empty_summary})

        purpose = (
            "train" if split == "train"
            else "robustness" if split == "test" and args.copies > 0
            else None
        )
        if purpose is None:
            continue
        for index in range(1, args.copies + 1):
            rng = stable_rng(args.augmentation_seed, str(record["experiment_id"]), index, purpose)
            augmented, summary = augment_one(spectrum, rng, args)
            destination = "train" if purpose == "train" else "robustness"
            generated = base_record(record, destination, index)
            outputs[destination][0].append(generated)
            outputs[destination][1].append(augmented)
            manifest.append({**generated, **summary})
    return outputs, manifest


def write_spectral_csv(
    path: Path,
    records: List[Dict[str, Any]],
    spectra: List[np.ndarray],
    axis: np.ndarray,
) -> None:
    metadata = pd.DataFrame(records, columns=PREPARED_METADATA)
    intensities = pd.DataFrame(np.stack(spectra), columns=[f"{value:.10g}" for value in axis])
    pd.concat([metadata, intensities], axis=1).to_csv(
        path, index=False, encoding="utf-8-sig", float_format="%.10g"
    )


def duplicate_warnings(spectra: np.ndarray, metadata: pd.DataFrame) -> List[Dict[str, Any]]:
    groups: Dict[str, List[int]] = {}
    for index, spectrum in enumerate(spectra):
        digest = hashlib.sha256(np.ascontiguousarray(spectrum).tobytes()).hexdigest()
        groups.setdefault(digest, []).append(index)
    warnings_found = []
    for digest, indices in groups.items():
        splits = set(metadata.iloc[indices]["split"])
        if len(indices) > 1 and len(splits) > 1:
            warnings_found.append(
                {
                    "sha256": digest,
                    "experiment_ids": metadata.iloc[indices]["experiment_id"].tolist(),
                    "splits": sorted(splits),
                }
            )
    return warnings_found


def build_parser() -> argparse.ArgumentParser:
    parser = ChineseParser(description="分层拆分原始实验并生成离线增强训练CSV")
    parser.add_argument("--source-dir", default=str(RAW_DATA_DIR), help="原始CSV目录")
    parser.add_argument("--output-dir", default=str(PREPARED_DATA_DIR), help="准备数据目录")
    parser.add_argument("--train-ratio", type=float, default=.70)
    parser.add_argument("--val-ratio", type=float, default=.15)
    parser.add_argument("--test-ratio", type=float, default=.15)
    parser.add_argument("--split-seed", type=int, default=66)
    parser.add_argument("--augmentation-seed", type=int, default=66)
    parser.add_argument("--copies", type=int, default=4)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--shift-prob", type=float, default=.5)
    parser.add_argument("--gain-prob", type=float, default=.5)
    parser.add_argument("--noise-prob", type=float, default=.5)
    parser.add_argument("--gain-min", type=float, default=.98)
    parser.add_argument("--gain-max", type=float, default=1.02)
    parser.add_argument("--shift-points", type=int, default=1)
    parser.add_argument("--noise-ratio", type=float, default=.005)
    parser.add_argument("--overwrite", action="store_true", help="替换已有准备数据目录")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    setup_console()
    args = build_parser().parse_args(argv)
    validate_arguments(args)
    source_dir = Path(args.source_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if source_dir == output_dir:
        raise ValueError("原始数据目录和输出目录不能相同")
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"输出目录已存在：{output_dir}；如需替换请添加--overwrite")

    spectra, axis, metadata, source_paths = load_originals(source_dir)
    metadata["split"] = assign_splits(metadata, args)
    outputs, augmentation_manifest = build_outputs(spectra, metadata, args)
    duplicates = duplicate_warnings(spectra, metadata)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}_", dir=output_dir.parent))
    try:
        for name in outputs:
            filename = SPLIT_FILES[name]
            write_spectral_csv(temporary / filename, *outputs[name], axis)
        split_columns = [
            "experiment_id", "source_file", "source_row", "source_csv_line",
            "original_sample_id", "role", "class_id", "concentration_text",
            "concentration_group", "split", "repaired_points",
        ]
        metadata[split_columns].to_csv(
            temporary / "split_manifest.csv", index=False, encoding="utf-8-sig"
        )
        pd.DataFrame(augmentation_manifest).to_csv(
            temporary / "augmentation_manifest.csv", index=False, encoding="utf-8-sig"
        )
        distribution = (
            metadata.groupby(["split", "role", "concentration_group"], sort=True)
            .size().rename("count").reset_index().to_dict("records")
        )
        augmentation = {
            "enabled": args.copies > 0,
            "copies": args.copies,
        }
        if args.copies > 0:
            augmentation.update({
                "operation_order": ["horizontal_shift", "gain", "gaussian_noise"],
                "seed": args.augmentation_seed,
                "steps": args.steps,
                "shift_prob": args.shift_prob,
                "gain_prob": args.gain_prob,
                "noise_prob": args.noise_prob,
                "gain_min": args.gain_min,
                "gain_max": args.gain_max,
                "shift_points": args.shift_points,
                "noise_ratio": args.noise_ratio,
            })
        config = {
            "format_version": PREPARATION_FORMAT_VERSION,
            "method": (
                "split_originals_then_offline_augment_training"
                if args.copies > 0 else "split_originals_without_augmentation"
            ),
            "split": {
                "ratios": {
                    "train": args.train_ratio,
                    "validation": args.val_ratio,
                    "test": args.test_ratio,
                },
                "stratify_by": ["source_file", "concentration_group"],
                "seed": args.split_seed,
                "counts": metadata["split"].value_counts().to_dict(),
            },
            "augmentation": augmentation,
            "source_files": data_fingerprints(source_paths),
            "distribution": distribution,
            "cross_split_exact_duplicate_warnings": duplicates,
        }
        config_path = temporary / "preparation_config.json"
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        load_prepared_data(temporary)
        output_names = (
            *(SPLIT_FILES[name] for name in outputs),
            "split_manifest.csv",
            "augmentation_manifest.csv",
        )
        config["output_files"] = [
            {"name": name, "sha256": sha256_file(temporary / name)} for name in output_names
        ]
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

        if output_dir.exists():
            shutil.rmtree(output_dir)
        temporary.replace(output_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)

    counts = {name: len(records) for name, (records, _) in outputs.items()}
    count_text = (
        f"train {counts['train']} | validation {counts['validation']} | "
        f"test {counts['test']}"
    )
    if "robustness" in counts:
        count_text += f" | robustness {counts['robustness']}"
    print(f"完成 | 独立实验 {len(metadata)} | {count_text}")
    if duplicates:
        print(f"警告：发现 {len(duplicates)} 组完全相同原谱跨集合，详见preparation_config.json")
    print(output_dir)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError, argparse.ArgumentTypeError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        raise SystemExit(2)
