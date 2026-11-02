# -*- coding: utf-8 -*-
"""原始光谱读取、显式数据集合读取和光谱通道预处理。"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
import warnings
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import CHANNEL_SETS, PREPARED_DATA_DIR, ROLE_CLASS_IDS


SPLIT_FILES = {
    "train": "train.csv",
    "validation": "validation.csv",
    "test": "test.csv",
}
PREPARATION_SUPPORT_FILES = (
    "split_manifest.csv",
    "preparation_config.json",
)
PREPARED_METADATA = (
    "sample_id", "experiment_id", "source_file", "source_row",
    "source_csv_line", "original_sample_id", "row_description", "role",
    "class_id", "concentration_text", "concentration_group", "split",
    "augmentation_index", "is_augmented", "repaired_points",
)


@dataclass(frozen=True)
class PreparedSplit:
    spectra: np.ndarray
    targets: np.ndarray
    metadata: pd.DataFrame


@dataclass(frozen=True)
class PreparedDataset:
    directory: Path
    axis: np.ndarray
    splits: Mapping[str, PreparedSplit]

    def __getitem__(self, name: str) -> PreparedSplit:
        return self.splits[name]


def concentration_group(value: Any) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    match = re.search(r"10\s*\^?\s*([+\-]?\d+)", text)
    if match:
        return f"10^{int(match.group(1))}"
    normalized = re.sub(r"\s+", "", text.lower())
    return normalized or "unknown"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def data_fingerprints(paths: Sequence[Path]) -> list[Dict[str, str]]:
    return [
        {"name": path.name, "path": str(path.resolve()), "sha256": sha256_file(path)}
        for path in paths
    ]


def read_preparation_config(directory: str | Path) -> Dict[str, Any]:
    root = Path(directory).expanduser().resolve()
    path = root / "preparation_config.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"准备数据目录 {root} 缺少：preparation_config.json。"
            "请先运行 python augment_dataset.py"
        )
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("method") != "augment_all_then_group_split":
        raise ValueError("preparation_config.json不是当前的先增强后分组协议")
    try:
        copies = int(config["augmentation"]["copies"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("preparation_config.json缺少有效的augmentation.copies") from exc
    if copies < 0:
        raise ValueError("preparation_config.json中的增强副本数必须>=0")
    enabled = bool(config["augmentation"].get("enabled", copies > 0))
    if enabled != (copies > 0):
        raise ValueError("preparation_config.json中的augmentation.enabled与copies不一致")
    return config


def preparation_file_names(directory: str | Path) -> Tuple[str, ...]:
    read_preparation_config(directory)
    return (*SPLIT_FILES.values(), *PREPARATION_SUPPORT_FILES)


def preparation_file_paths(directory: str | Path) -> list[Path]:
    root = Path(directory).expanduser().resolve()
    return [root / name for name in preparation_file_names(root)]


def prepared_paths(directory: str | Path = PREPARED_DATA_DIR) -> Dict[str, Path]:
    root = Path(directory).expanduser().resolve()
    read_preparation_config(root)
    return {name: root / filename for name, filename in SPLIT_FILES.items()}


def require_prepared_data(directory: str | Path = PREPARED_DATA_DIR) -> Path:
    root = Path(directory).expanduser().resolve()
    names = preparation_file_names(root)
    missing = [name for name in names if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"准备数据目录 {root} 缺少：{'、'.join(missing)}。"
            "请先运行 python augment_dataset.py"
        )
    return root


def read_raw_csv(
    path: Path, target_axis: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, bool]:
    """读取原始/未知光谱CSV；前两列是编号和浓度，其余列是波数。"""
    # 先按对象读取，避免实验序号1被自动转成1.0；波数和强度随后单独转数值。
    raw = pd.read_csv(path, header=None, dtype=object)
    if raw.shape[0] < 2 or raw.shape[1] < 6:
        raise ValueError(f"{path} 格式错误：需要标题行、数据行和至少4个波数点")

    axis = pd.to_numeric(raw.iloc[0, 2:], errors="coerce").to_numpy(np.float64)
    if not np.isfinite(axis).all():
        raise ValueError(f"{path} 的波数轴包含非数值")

    spectra = raw.iloc[1:, 2:].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
    repaired = np.zeros(len(spectra), dtype=np.int64)
    for row, spectrum in enumerate(spectra):
        bad = ~np.isfinite(spectrum)
        if not bad.any():
            continue
        positions = np.flatnonzero(bad)
        limit = max(3, int(round(len(spectrum) * .01)))
        if len(positions) > limit or positions[0] == 0 or positions[-1] == len(spectrum) - 1:
            raise ValueError(f"{path} 第{row + 2}行含无法安全插值的缺失值")
        good = np.flatnonzero(~bad)
        spectrum[bad] = np.interp(positions, good, spectrum[good])
        repaired[row] = len(positions)
    if repaired.sum():
        warnings.warn(
            f"{path} 修复了 {int(repaired.sum())} 个内部缺失点",
            stacklevel=2,
        )

    if np.all(np.diff(axis) < 0):
        axis, spectra = axis[::-1].copy(), spectra[:, ::-1].copy()
    elif not np.all(np.diff(axis) > 0):
        raise ValueError(f"{path} 的波数轴必须严格单调")

    interpolated = False
    if target_axis is not None and not (
        len(axis) == len(target_axis)
        and np.allclose(axis, target_axis, rtol=1e-6, atol=1e-5)
    ):
        tolerance = max(float(np.ptp(target_axis)), 1.0) * 1e-7
        if target_axis[0] < axis[0] - tolerance or target_axis[-1] > axis[-1] + tolerance:
            raise ValueError(f"{path} 的波数范围不能覆盖参考轴")
        spectra = np.stack([np.interp(target_axis, axis, row) for row in spectra])
        axis = np.asarray(target_axis, dtype=np.float64).copy()
        interpolated = True

    metadata = pd.DataFrame(
        {
            "sample_id": raw.iloc[1:, 0].astype(str).str.strip().to_numpy(),
            "concentration_text": raw.iloc[1:, 1].astype(str).to_numpy(),
            "row_in_file": np.arange(1, len(raw), dtype=np.int64),
            "csv_line": np.arange(2, len(raw) + 1, dtype=np.int64),
            "repaired_points": repaired,
        }
    )
    metadata["concentration_group"] = metadata["concentration_text"].map(
        concentration_group
    )
    return spectra.astype(np.float32), axis, metadata, interpolated


def _bool_series(values: pd.Series, path: Path) -> pd.Series:
    if values.dtype == bool:
        return values
    normalized = values.astype(str).str.lower()
    if not normalized.isin(("true", "false")).all():
        raise ValueError(f"{path} 的is_augmented列必须是布尔值")
    return normalized == "true"


def read_prepared_split(
    path: Path,
    expected_split: str,
    target_axis: Optional[np.ndarray] = None,
) -> Tuple[PreparedSplit, np.ndarray]:
    frame = pd.read_csv(path)
    missing = [column for column in PREPARED_METADATA if column not in frame]
    if missing:
        raise ValueError(f"{path} 缺少字段：{'、'.join(missing)}")
    spectral_columns = [column for column in frame if column not in PREPARED_METADATA]
    if len(spectral_columns) < 4:
        raise ValueError(f"{path} 缺少光谱强度列")
    try:
        axis = np.asarray([float(column) for column in spectral_columns], dtype=np.float64)
    except ValueError as exc:
        raise ValueError(f"{path} 包含无法解析为波数的列名") from exc
    if not np.all(np.diff(axis) > 0):
        raise ValueError(f"{path} 的波数列必须严格递增")
    if target_axis is not None and not (
        len(axis) == len(target_axis)
        and np.allclose(axis, target_axis, rtol=1e-6, atol=1e-5)
    ):
        raise ValueError(f"{path} 的波数轴与其他集合不一致")

    spectra = frame[spectral_columns].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    if not np.isfinite(spectra).all():
        raise ValueError(f"{path} 含非数值或非有限光谱强度")
    metadata = frame[list(PREPARED_METADATA)].copy()
    metadata["is_augmented"] = _bool_series(metadata["is_augmented"], path)
    metadata["augmentation_index"] = pd.to_numeric(
        metadata["augmentation_index"], errors="raise"
    ).astype(np.int64)
    metadata["class_id"] = pd.to_numeric(metadata["class_id"], errors="raise").astype(np.int64)
    if set(metadata["split"].astype(str)) != {expected_split}:
        raise ValueError(f"{path} 的split列必须全部为{expected_split}")
    expected_targets = metadata["role"].map(ROLE_CLASS_IDS)
    if expected_targets.isna().any() or not np.array_equal(
        expected_targets.to_numpy(np.int64), metadata["class_id"].to_numpy(np.int64)
    ):
        raise ValueError(f"{path} 的role与class_id不一致")
    if metadata["sample_id"].duplicated().any():
        raise ValueError(f"{path} 存在重复sample_id")
    return PreparedSplit(spectra, metadata["class_id"].to_numpy(np.int64), metadata), axis


def load_prepared_data(directory: str | Path = PREPARED_DATA_DIR) -> PreparedDataset:
    root = require_prepared_data(directory)
    config = read_preparation_config(root)
    copies = int(config["augmentation"]["copies"])
    splits: Dict[str, PreparedSplit] = {}
    axis: Optional[np.ndarray] = None
    for name, path in prepared_paths(root).items():
        split, current_axis = read_prepared_split(path, name, axis)
        if axis is None:
            axis = current_axis
        splits[name] = split
    assert axis is not None

    train, validation, test = (splits[name] for name in SPLIT_FILES)
    for name, split in splits.items():
        if not (
            (split.metadata["augmentation_index"] == 0)
            == ~split.metadata["is_augmented"]
        ).all():
            raise ValueError(f"{name}.csv的增强编号与标记不一致")
        if copies == 0 and split.metadata["is_augmented"].any():
            raise ValueError(f"无增强数据的{name}.csv不能包含增强谱")

    experiment_sets = {
        name: set(split.metadata["experiment_id"].astype(str))
        for name, split in (("train", train), ("validation", validation), ("test", test))
    }
    if (
        experiment_sets["train"] & experiment_sets["validation"]
        or experiment_sets["train"] & experiment_sets["test"]
        or experiment_sets["validation"] & experiment_sets["test"]
    ):
        raise ValueError("train、validation、test存在实验ID交叉")
    for name in ("train", "validation", "test"):
        if set(splits[name].targets.tolist()) != set(ROLE_CLASS_IDS.values()):
            raise ValueError(f"{name}.csv未同时包含三个类别")

    expected_indices = set(range(copies + 1))
    for name, split in splits.items():
        groups = split.metadata.groupby("experiment_id")["augmentation_index"]
        if not all(
            len(values) == copies + 1 and set(values) == expected_indices
            for _, values in groups
        ):
            raise ValueError(
                f"{name}.csv中每个父实验必须包含一条原谱和配置数量的增强谱"
            )

    all_metadata = pd.concat(
        [split.metadata for split in splits.values()], ignore_index=True
    )
    if all_metadata["sample_id"].duplicated().any():
        raise ValueError("数据CSV之间存在重复sample_id")
    split_manifest = pd.read_csv(root / "split_manifest.csv")
    required_manifest = {"experiment_id", "split"}
    if not required_manifest.issubset(split_manifest) or split_manifest["experiment_id"].duplicated().any():
        raise ValueError("split_manifest.csv必须每个独立实验一行")
    clean_rows = all_metadata[all_metadata["augmentation_index"] == 0]
    expected_assignment = dict(zip(clean_rows["experiment_id"].astype(str), clean_rows["split"].astype(str)))
    actual_assignment = dict(zip(split_manifest["experiment_id"].astype(str), split_manifest["split"].astype(str)))
    if actual_assignment != expected_assignment:
        raise ValueError("split_manifest.csv与数据CSV的实验拆分不一致")
    for item in config.get("output_files", []):
        path = root / str(item.get("name", ""))
        if not path.is_file() or sha256_file(path) != item.get("sha256"):
            raise ValueError(f"准备数据文件指纹不一致：{path.name}")
    return PreparedDataset(root, axis.astype(np.float32), splits)


class SpectralPreprocessor:
    def __init__(self, clip: float = 10.0, eps: float = 1e-6, channel_set: str = "raw_snv_d1"):
        if channel_set not in CHANNEL_SETS:
            raise ValueError(f"未知通道组合：{channel_set}")
        self.clip, self.eps, self.channel_set = float(clip), float(eps), channel_set
        self.center: Optional[np.ndarray] = None
        self.scale: Optional[np.ndarray] = None
        self.axis: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray, axis: np.ndarray) -> "SpectralPreprocessor":
        raw = np.asarray(x, dtype=np.float64)
        self.center = np.mean(raw, axis=0).astype(np.float32)
        scale = np.std(raw, axis=0)
        self.scale = np.where(scale > self.eps, scale, 1.0).astype(np.float32)
        self.axis = np.asarray(axis, dtype=np.float32).copy()
        return self

    def _snv(self, x: np.ndarray) -> np.ndarray:
        return (x - np.mean(x, axis=1, keepdims=True)) / np.maximum(
            np.std(x, axis=1, keepdims=True), self.eps
        )

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.center is None or self.scale is None or self.axis is None:
            raise RuntimeError("预处理器尚未fit")
        raw = np.asarray(x, dtype=np.float32)
        channels = [np.clip((raw - self.center) / self.scale, -self.clip, self.clip)]
        if self.channel_set != "raw":
            shape = self._snv(raw)
            channels.append(shape)
        if self.channel_set in ("raw_snv_d1", "raw_snv_d1_d2"):
            first = self._snv(np.gradient(shape, self.axis, axis=1))
            channels.append(first)
        if self.channel_set == "raw_snv_d1_d2":
            channels.append(self._snv(np.gradient(first, self.axis, axis=1)))
        return np.nan_to_num(
            np.stack(channels, axis=1), nan=0.0, posinf=self.clip, neginf=-self.clip
        ).astype(np.float32)

    def state_dict(self) -> Dict[str, Any]:
        if self.center is None or self.scale is None or self.axis is None:
            raise RuntimeError("预处理器尚未fit")
        return {
            "clip": self.clip,
            "eps": self.eps,
            "channel_set": self.channel_set,
            "channel_names": CHANNEL_SETS[self.channel_set],
            "center": torch.from_numpy(self.center.copy()),
            "scale": torch.from_numpy(self.scale.copy()),
            "axis": torch.from_numpy(self.axis.copy()),
        }

    @classmethod
    def from_state_dict(cls, state: Dict[str, Any]) -> "SpectralPreprocessor":
        obj = cls(float(state["clip"]), float(state["eps"]), str(state["channel_set"]))
        obj.center = state["center"].cpu().numpy().astype(np.float32)
        obj.scale = state["scale"].cpu().numpy().astype(np.float32)
        obj.axis = state["axis"].cpu().numpy().astype(np.float32)
        return obj


class SpectraDataset(Dataset):
    def __init__(self, x: np.ndarray, y: Optional[np.ndarray] = None):
        self.x = torch.from_numpy(np.asarray(x, dtype=np.float32))
        self.y = None if y is None else torch.from_numpy(np.asarray(y, dtype=np.int64))

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int):
        return self.x[index] if self.y is None else (self.x[index], self.y[index])
