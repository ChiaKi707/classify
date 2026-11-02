# -*- coding: utf-8 -*-
"""先为每个原始CSV生成完整增强文件，再按父实验分组拆分三集合。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    AUGMENTED_DATA_DIR, ChineseParser, PREPARED_DATA_DIR, RAW_DATA_DIR,
    RAW_SOURCES, ROLE_CLASS_IDS, setup_console,
)
from data import (
    PREPARED_METADATA, SPLIT_FILES, data_fingerprints, load_prepared_data,
    read_raw_csv, sha256_file,
)


AUGMENTATION_FORMAT_VERSION = 1
PREPARATION_FORMAT_VERSION = 3
AUGMENTED_METADATA = tuple(
    column for column in PREPARED_METADATA if column != "split"
)
TRACE_COLUMNS = (
    "augmentation_steps", "net_shift_points", "cumulative_gain",
    "max_noise_std", "operation_trace",
)


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


def _row_description(value: Any) -> str:
    text = "" if pd.isna(value) else re.sub(r"\s+", " ", str(value).strip())
    if not text or text.lower() == "nan":
        raise ValueError("原始CSV的行描述不能为空")
    return text.replace(":", "_")


def load_originals(
    source_dir: Path,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, List[Path]]:
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
        if ids.eq("").any() or ids.str.lower().eq("nan").any():
            raise ValueError(f"{path} 的实验序号不能为空")
        metadata = metadata.rename(
            columns={
                "sample_id": "original_sample_id",
                "row_in_file": "source_row",
                "csv_line": "source_csv_line",
            }
        )
        metadata["row_description"] = metadata["concentration_text"].map(
            _row_description
        )
        metadata["experiment_id"] = [
            f"{path.stem}:{description}:{sample_id}"
            for description, sample_id in zip(metadata["row_description"], ids)
        ]
        metadata["source_file"] = path.name
        metadata["role"] = role
        metadata["class_id"] = ROLE_CLASS_IDS[role]
        spectra_parts.append(spectra)
        metadata_parts.append(metadata)
        paths.append(path)
    assert axis is not None
    metadata = pd.concat(metadata_parts, ignore_index=True)
    if metadata["experiment_id"].duplicated().any():
        duplicated = metadata.loc[
            metadata["experiment_id"].duplicated(False), "experiment_id"
        ].head().tolist()
        raise ValueError(
            "文件名＋行描述＋实验序号没有形成唯一父实验ID：" + "、".join(duplicated)
        )
    return np.concatenate(spectra_parts), axis, metadata, paths


def stable_rng(seed: int, experiment_id: str, index: int) -> np.random.Generator:
    text = f"{seed}|{experiment_id}|{index}|offline_full_dataset".encode("utf-8")
    derived = int.from_bytes(hashlib.sha256(text).digest()[:8], "little")
    return np.random.default_rng(derived)


def base_record(row: pd.Series, index: int) -> Dict[str, Any]:
    experiment_id = str(row["experiment_id"])
    return {
        "sample_id": f"{experiment_id}:aug_{index}",
        "experiment_id": experiment_id,
        "source_file": row["source_file"],
        "source_row": int(row["source_row"]),
        "source_csv_line": int(row["source_csv_line"]),
        "original_sample_id": row["original_sample_id"],
        "row_description": row["row_description"],
        "role": row["role"],
        "class_id": int(row["class_id"]),
        "concentration_text": row["concentration_text"],
        "concentration_group": row["concentration_group"],
        "augmentation_index": index,
        "is_augmented": index > 0,
        "repaired_points": int(row["repaired_points"]),
    }


def build_augmented_files(
    spectra: np.ndarray,
    metadata: pd.DataFrame,
    args: argparse.Namespace,
) -> Tuple[
    Dict[str, Tuple[List[Dict[str, Any]], List[np.ndarray]]],
    List[Dict[str, Any]],
]:
    outputs = {filename: ([], []) for _, filename in RAW_SOURCES}
    manifest: List[Dict[str, Any]] = []
    original_summary = {
        "augmentation_steps": 0,
        "net_shift_points": 0,
        "cumulative_gain": 1.0,
        "max_noise_std": 0.0,
        "operation_trace": "[]",
    }
    for file_index, (_, source_file) in enumerate(RAW_SOURCES, 1):
        positions = np.flatnonzero(
            metadata["source_file"].to_numpy(dtype=str) == source_file
        )
        for position in positions:
            spectrum = spectra[position]
            row = metadata.iloc[position]
            original = base_record(row, 0)
            outputs[source_file][0].append(original)
            outputs[source_file][1].append(spectrum.copy())
            manifest.append({**original, **original_summary})
            for index in range(1, args.copies + 1):
                rng = stable_rng(
                    args.augmentation_seed, str(row["experiment_id"]), index
                )
                augmented, summary = augment_one(spectrum, rng, args)
                generated = base_record(row, index)
                outputs[source_file][0].append(generated)
                outputs[source_file][1].append(augmented)
                manifest.append({**generated, **summary})
        print(
            f"  [{file_index}/{len(RAW_SOURCES)}] {source_file} | "
            f"父实验 {len(positions)} → 数据行 {len(positions) * (args.copies + 1)}",
            flush=True,
        )
    return outputs, manifest


def write_spectral_csv(
    path: Path,
    records: List[Dict[str, Any]],
    spectra: Sequence[np.ndarray] | np.ndarray,
    axis: np.ndarray,
    metadata_columns: Sequence[str],
) -> None:
    metadata = pd.DataFrame(records, columns=metadata_columns)
    matrix = np.asarray(spectra, dtype=np.float32)
    intensities = pd.DataFrame(matrix, columns=[f"{value:.10g}" for value in axis])
    pd.concat([metadata.reset_index(drop=True), intensities], axis=1).to_csv(
        path, index=False, encoding="utf-8-sig", float_format="%.10g"
    )


def _bool_series(values: pd.Series, path: Path) -> pd.Series:
    if values.dtype == bool:
        return values
    normalized = values.astype(str).str.lower()
    if not normalized.isin(("true", "false")).all():
        raise ValueError(f"{path} 的is_augmented列必须是布尔值")
    return normalized == "true"


def read_augmented_directory(
    directory: Path,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, Dict[str, Any]]:
    config_path = directory / "augmentation_config.json"
    manifest_path = directory / "augmentation_manifest.csv"
    if not config_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"增强目录不完整：{directory}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    try:
        copies = int(config["augmentation"]["copies"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("augmentation_config.json缺少有效的copies") from exc

    frames: List[pd.DataFrame] = []
    spectra_parts: List[np.ndarray] = []
    axis: Optional[np.ndarray] = None
    for _, filename in RAW_SOURCES:
        path = directory / filename
        if not path.is_file():
            raise FileNotFoundError(f"增强目录缺少对应文件：{path}")
        frame = pd.read_csv(path)
        missing = [column for column in AUGMENTED_METADATA if column not in frame]
        if missing:
            raise ValueError(f"{path} 缺少字段：{'、'.join(missing)}")
        spectral_columns = [column for column in frame if column not in AUGMENTED_METADATA]
        try:
            current_axis = np.asarray(
                [float(column) for column in spectral_columns], dtype=np.float64
            )
        except ValueError as exc:
            raise ValueError(f"{path} 包含无法解析为波数的列名") from exc
        if len(current_axis) < 4 or not np.all(np.diff(current_axis) > 0):
            raise ValueError(f"{path} 的波数列必须至少4列且严格递增")
        if axis is None:
            axis = current_axis
        elif len(axis) != len(current_axis) or not np.allclose(
            axis, current_axis, rtol=1e-6, atol=1e-5
        ):
            raise ValueError(f"{path} 的波数轴与其他增强文件不一致")
        matrix = frame[spectral_columns].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(np.float32)
        if not np.isfinite(matrix).all():
            raise ValueError(f"{path} 含非数值或非有限光谱强度")
        metadata = frame[list(AUGMENTED_METADATA)].copy()
        metadata["augmentation_index"] = pd.to_numeric(
            metadata["augmentation_index"], errors="raise"
        ).astype(np.int64)
        metadata["class_id"] = pd.to_numeric(
            metadata["class_id"], errors="raise"
        ).astype(np.int64)
        metadata["is_augmented"] = _bool_series(metadata["is_augmented"], path)
        if set(metadata["source_file"].astype(str)) != {filename}:
            raise ValueError(f"{path} 的source_file与文件名不一致")
        frames.append(metadata)
        spectra_parts.append(matrix)

    assert axis is not None
    metadata = pd.concat(frames, ignore_index=True)
    spectra = np.concatenate(spectra_parts)
    if metadata["sample_id"].duplicated().any():
        raise ValueError("data_augment中的sample_id必须全局唯一")
    expected_indices = set(range(copies + 1))
    groups = metadata.groupby("experiment_id")["augmentation_index"]
    if not all(
        len(values) == copies + 1 and set(values) == expected_indices
        for _, values in groups
    ):
        raise ValueError("每个父实验必须包含一条原谱和完整增强副本")
    if not (
        (metadata["augmentation_index"] == 0) == ~metadata["is_augmented"]
    ).all():
        raise ValueError("data_augment中的增强编号与标记不一致")
    expected_targets = metadata["role"].map(ROLE_CLASS_IDS)
    if expected_targets.isna().any() or not np.array_equal(
        expected_targets.to_numpy(np.int64), metadata["class_id"].to_numpy(np.int64)
    ):
        raise ValueError("data_augment中的role与class_id不一致")

    manifest = pd.read_csv(manifest_path, usecols=["sample_id"])
    if manifest["sample_id"].duplicated().any() or set(
        manifest["sample_id"].astype(str)
    ) != set(metadata["sample_id"].astype(str)):
        raise ValueError("augmentation_manifest.csv与增强文件数据行不一致")
    for item in config.get("output_files", []):
        path = directory / str(item.get("name", ""))
        if not path.is_file() or sha256_file(path) != item.get("sha256"):
            raise ValueError(f"增强文件指纹不一致：{path.name}")
    return spectra, axis, metadata, config


def assign_parent_splits(parents: pd.DataFrame, args: argparse.Namespace) -> pd.Series:
    strata = (
        parents["source_file"].astype(str)
        + "|"
        + parents["concentration_group"].astype(str)
    )
    indices = np.arange(len(parents))
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
        raise ValueError(f"联合分层父实验不足，最小分层为{counts}：{exc}") from exc
    assignment = pd.Series("", index=parents.index, dtype="object")
    assignment.iloc[train_idx] = "train"
    assignment.iloc[val_idx] = "validation"
    assignment.iloc[test_idx] = "test"
    return assignment


def duplicate_warnings(
    spectra: np.ndarray,
    metadata: pd.DataFrame,
    assignments: pd.Series,
) -> List[Dict[str, Any]]:
    original_mask = metadata["augmentation_index"].to_numpy(np.int64) == 0
    original_spectra = spectra[original_mask]
    parents = metadata.loc[original_mask].reset_index(drop=True)
    groups: Dict[str, List[int]] = {}
    for index, spectrum in enumerate(original_spectra):
        digest = hashlib.sha256(np.ascontiguousarray(spectrum).tobytes()).hexdigest()
        groups.setdefault(digest, []).append(index)
    warnings_found = []
    for digest, indices in groups.items():
        splits = set(assignments.iloc[indices])
        if len(indices) > 1 and len(splits) > 1:
            warnings_found.append(
                {
                    "sha256": digest,
                    "experiment_ids": parents.iloc[indices]["experiment_id"].tolist(),
                    "splits": sorted(splits),
                }
            )
    return warnings_found


def augmentation_settings(args: argparse.Namespace) -> Dict[str, Any]:
    settings: Dict[str, Any] = {
        "enabled": args.copies > 0,
        "copies": args.copies,
        "seed": args.augmentation_seed,
    }
    if args.copies > 0:
        settings.update(
            {
                "operation_order": ["horizontal_shift", "gain", "gaussian_noise"],
                "steps": args.steps,
                "shift_prob": args.shift_prob,
                "gain_prob": args.gain_prob,
                "noise_prob": args.noise_prob,
                "gain_min": args.gain_min,
                "gain_max": args.gain_max,
                "shift_points": args.shift_points,
                "noise_ratio": args.noise_ratio,
            }
        )
    return settings


def _temporary_directory(target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f".{target.name}_", dir=target.parent))


def _replace_directory(temporary: Path, target: Path) -> None:
    if target.exists():
        shutil.rmtree(target)
    temporary.replace(target)


def write_augmented_directory(
    target: Path,
    outputs: Dict[str, Tuple[List[Dict[str, Any]], List[np.ndarray]]],
    manifest: List[Dict[str, Any]],
    axis: np.ndarray,
    source_paths: Sequence[Path],
    args: argparse.Namespace,
) -> None:
    temporary = _temporary_directory(target)
    try:
        for filename, (records, spectra) in outputs.items():
            write_spectral_csv(
                temporary / filename, records, spectra, axis, AUGMENTED_METADATA
            )
        pd.DataFrame(
            manifest, columns=(*AUGMENTED_METADATA, *TRACE_COLUMNS)
        ).to_csv(
            temporary / "augmentation_manifest.csv",
            index=False,
            encoding="utf-8-sig",
        )
        output_names = [
            *(filename for _, filename in RAW_SOURCES),
            "augmentation_manifest.csv",
        ]
        config = {
            "format_version": AUGMENTATION_FORMAT_VERSION,
            "method": "offline_augment_each_source_before_group_split",
            "parent_id_fields": [
                "source_file", "row_description", "original_sample_id"
            ],
            "augmentation": augmentation_settings(args),
            "source_files": data_fingerprints(source_paths),
            "source_parent_count": len(manifest) // (args.copies + 1),
            "output_files": [
                {"name": name, "sha256": sha256_file(temporary / name)}
                for name in output_names
            ],
        }
        (temporary / "augmentation_config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        read_augmented_directory(temporary)
        _replace_directory(temporary, target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def write_prepared_directory(
    augmented_dir: Path,
    target: Path,
    spectra: np.ndarray,
    axis: np.ndarray,
    metadata: pd.DataFrame,
    augmentation_config: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, int]:
    original_mask = metadata["augmentation_index"].to_numpy(np.int64) == 0
    parents = metadata.loc[original_mask].reset_index(drop=True).copy()
    parents["split"] = assign_parent_splits(parents, args)
    split_by_parent = dict(
        zip(parents["experiment_id"].astype(str), parents["split"].astype(str))
    )
    metadata = metadata.copy()
    metadata["split"] = metadata["experiment_id"].astype(str).map(split_by_parent)
    if metadata["split"].isna().any():
        raise RuntimeError("增强数据存在未分组的父实验")

    duplicates = duplicate_warnings(spectra, metadata, parents["split"])
    temporary = _temporary_directory(target)
    try:
        row_counts: Dict[str, int] = {}
        for split, filename in SPLIT_FILES.items():
            mask = metadata["split"].to_numpy() == split
            records = metadata.loc[mask, list(PREPARED_METADATA)].to_dict("records")
            write_spectral_csv(
                temporary / filename,
                records,
                spectra[mask],
                axis,
                PREPARED_METADATA,
            )
            row_counts[split] = int(mask.sum())

        split_columns = [
            "experiment_id", "source_file", "source_row", "source_csv_line",
            "original_sample_id", "row_description", "role", "class_id",
            "concentration_text", "concentration_group", "split", "repaired_points",
        ]
        parents[split_columns].to_csv(
            temporary / "split_manifest.csv", index=False, encoding="utf-8-sig"
        )
        parent_distribution = (
            parents.groupby(
                ["split", "role", "concentration_group"], sort=True
            ).size().rename("parent_count").reset_index().to_dict("records")
        )
        row_distribution = (
            metadata.groupby(
                ["split", "role", "concentration_group"], sort=True
            ).size().rename("row_count").reset_index().to_dict("records")
        )
        augmented_names = [
            *(filename for _, filename in RAW_SOURCES),
            "augmentation_manifest.csv",
            "augmentation_config.json",
        ]
        config = {
            "format_version": PREPARATION_FORMAT_VERSION,
            "method": "augment_all_then_group_split",
            "parent_id_fields": [
                "source_file", "row_description", "original_sample_id"
            ],
            "split": {
                "unit": "experiment_id_parent_group",
                "ratios": {
                    "train": args.train_ratio,
                    "validation": args.val_ratio,
                    "test": args.test_ratio,
                },
                "stratify_by": ["source_file", "concentration_group"],
                "seed": args.split_seed,
                "parent_counts": parents["split"].value_counts().to_dict(),
                "row_counts": row_counts,
            },
            "augmentation": augmentation_config["augmentation"],
            "augmented_source": {
                "directory": str(augmented_dir),
                "config_sha256": sha256_file(
                    augmented_dir / "augmentation_config.json"
                ),
                "files": data_fingerprints(
                    [augmented_dir / name for name in augmented_names]
                ),
            },
            "source_files": augmentation_config["source_files"],
            "parent_distribution": parent_distribution,
            "row_distribution": row_distribution,
            "cross_split_exact_duplicate_warnings": duplicates,
        }
        config_path = temporary / "preparation_config.json"
        config_path.write_text(
            json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        load_prepared_data(temporary)
        output_names = [*SPLIT_FILES.values(), "split_manifest.csv"]
        config["output_files"] = [
            {"name": name, "sha256": sha256_file(temporary / name)}
            for name in output_names
        ]
        config_path.write_text(
            json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        load_prepared_data(temporary)
        _replace_directory(temporary, target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return row_counts


def build_parser() -> argparse.ArgumentParser:
    parser = ChineseParser(
        description="先逐文件离线增强全部原始实验，再按父实验分组拆分三集合"
    )
    parser.add_argument("--source-dir", default=str(RAW_DATA_DIR), help="原始CSV目录")
    parser.add_argument(
        "--augmented-dir", default=str(AUGMENTED_DATA_DIR),
        help="逐文件增强结果目录",
    )
    parser.add_argument(
        "--prepared-dir", default=str(PREPARED_DATA_DIR),
        help="训练、验证和测试CSV目录",
    )
    parser.add_argument("--train-ratio", type=float, default=.70)
    parser.add_argument("--val-ratio", type=float, default=.15)
    parser.add_argument("--test-ratio", type=float, default=.15)
    parser.add_argument("--split-seed", type=int, default=66)
    parser.add_argument("--augmentation-seed", type=int, default=66)
    parser.add_argument("--copies", type=int, default=4)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--shift-prob", type=float, default=.5)
    parser.add_argument("--gain-prob", type=float, default=.5)
    parser.add_argument("--noise-prob", type=float, default=.5)
    parser.add_argument("--gain-min", type=float, default=.97)
    parser.add_argument("--gain-max", type=float, default=1.03)
    parser.add_argument("--shift-points", type=int, default=2)
    parser.add_argument("--noise-ratio", type=float, default=.010)
    parser.add_argument(
        "--overwrite", action="store_true",
        help="替换已有增强目录和准备数据目录",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    setup_console()
    args = build_parser().parse_args(argv)
    validate_arguments(args)
    source_dir = Path(args.source_dir).expanduser().resolve()
    augmented_dir = Path(args.augmented_dir).expanduser().resolve()
    prepared_dir = Path(args.prepared_dir).expanduser().resolve()
    directories = (source_dir, augmented_dir, prepared_dir)
    if len(set(directories)) != 3:
        raise ValueError("原始、增强和准备数据目录必须互不相同")
    if any(
        left in right.parents or right in left.parents
        for index, left in enumerate(directories)
        for right in directories[index + 1:]
    ):
        raise ValueError("原始、增强和准备数据目录不能互相嵌套")
    existing = [path for path in (augmented_dir, prepared_dir) if path.exists()]
    if existing and not args.overwrite:
        names = "、".join(str(path) for path in existing)
        raise FileExistsError(f"输出目录已存在：{names}；如需替换请添加--overwrite")

    print("[1/3] 读取并校验原始CSV", flush=True)
    spectra, axis, metadata, source_paths = load_originals(source_dir)
    print(f"  完成 | 父实验 {len(metadata)} | 光谱点 {len(axis)}", flush=True)
    print(
        f"[2/3] 逐文件增强 | 每条生成 {args.copies} 份 | 每份 {args.steps} 轮",
        flush=True,
    )
    outputs, manifest = build_augmented_files(spectra, metadata, args)
    write_augmented_directory(
        augmented_dir, outputs, manifest, axis, source_paths, args
    )
    print(f"  已写入 {augmented_dir}", flush=True)
    print("[3/3] 按父实验整体分组并写出70/15/15集合", flush=True)
    augmented_spectra, augmented_axis, augmented_metadata, augmentation_config = (
        read_augmented_directory(augmented_dir)
    )
    counts = write_prepared_directory(
        augmented_dir,
        prepared_dir,
        augmented_spectra,
        augmented_axis,
        augmented_metadata,
        augmentation_config,
        args,
    )

    print(
        f"完成 | 父实验 {len(metadata)} | 增强后 {len(augmented_metadata)} 行 | "
        f"train {counts['train']} | validation {counts['validation']} | "
        f"test {counts['test']}"
    )
    print(f"逐文件增强：{augmented_dir}")
    print(f"分组结果：{prepared_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError, argparse.ArgumentTypeError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        raise SystemExit(2)
