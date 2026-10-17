# -*- coding: utf-8 -*-
"""扩充CSV读取、来源分组拆分、缺失点处理与光谱通道预处理。"""

from __future__ import annotations

import argparse
import hashlib
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import CHANNEL_SETS, DATA_DIR, DEFAULT_DATA, ROLE_ALIASES, ROLE_CLASS_IDS


def resolve_data_specs(specs: Optional[Sequence[str]]) -> List[str]:
    """省略--data时读取data_augmented中的四个离线扩充文件。"""
    if specs:
        return list(specs)
    missing = [name for _, name in DEFAULT_DATA if not (DATA_DIR / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"默认训练数据目录 {DATA_DIR} 缺少：{'、'.join(missing)}。"
            "请先运行 python augment_dataset.py 生成离线扩充CSV；"
            "也可以用 --data mg=路径 等参数指定文件，详见README.md。"
        )
    return [f"{role}={DATA_DIR / name}" for role, name in DEFAULT_DATA]


def data_fingerprints(specs: Sequence[Tuple[str, Path]]) -> List[Dict[str, str]]:
    """记录数据内容指纹，避免换了数据却继续复用旧实验结果。"""
    records = []
    for role, path in specs:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        records.append({"role": role, "path": str(path.resolve()), "sha256": digest.hexdigest()})
    return records


def parse_data_spec(spec: str) -> Tuple[str, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(
            f"数据参数必须是 role=path，收到：{spec!r}"
        )
    raw_role, raw_path = spec.split("=", 1)
    role_key = re.sub(r"[\s_:\-]", "", raw_role.lower())
    role = ROLE_ALIASES.get(role_key)
    if role is None:
        valid = "mg/mg1、r6g、both/mix/mix11/mix12"
        raise argparse.ArgumentTypeError(f"未知角色 {raw_role!r}；可用：{valid}")
    # 统一绝对路径，避免同一文件用相对/绝对写法时产生不同的拆分清单。
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"数据文件不存在：{path}")
    return role, path


def concentration_group(value: Any) -> str:
    """把诸如 'R6G 10^-4' 变成稳定的分层键。"""
    text = "" if pd.isna(value) else str(value).strip()
    match = re.search(r"10\s*\^?\s*([+\-]?\d+)", text)
    if match:
        return f"10^{int(match.group(1))}"
    normalized = re.sub(r"\s+", "", text.lower())
    return normalized or "unknown"


def read_spectral_csv(
    path: Path, target_axis: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, bool]:
    raw = pd.read_csv(path, header=None)
    if raw.shape[0] < 2 or raw.shape[1] < 6:
        raise ValueError(f"{path} 不是预期格式，需要标题行、至少一条光谱和4个波数点，当前形状 {raw.shape}")

    axis = pd.to_numeric(raw.iloc[0, 2:], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(axis).all():
        bad = np.flatnonzero(~np.isfinite(axis))[:5].tolist()
        raise ValueError(f"{path} 的拉曼位移包含非数值，位置：{bad}")

    intensity_frame = raw.iloc[1:, 2:].apply(pd.to_numeric, errors="coerce")
    spectra = intensity_frame.to_numpy(dtype=np.float64)
    repaired_points = np.zeros(len(spectra), dtype=np.int64)
    if not np.isfinite(spectra).all():
        # 少量内部缺失点用同一条光谱左右相邻点线性插值。
        # 不能直接填 0，否则一、二阶导数会把这个位置变成很强的假峰。
        for row_index, spectrum in enumerate(spectra):
            bad_mask = ~np.isfinite(spectrum)
            bad_count = int(bad_mask.sum())
            if bad_count == 0:
                continue
            max_repair = max(3, int(round(len(spectrum) * 0.01)))
            bad_positions = np.flatnonzero(bad_mask)
            if (
                bad_count > max_repair
                or bad_positions[0] == 0
                or bad_positions[-1] == len(spectrum) - 1
            ):
                raise ValueError(
                    f"{path} 第 {row_index + 2} 行含 {bad_count} 个无法安全插值的缺失/非数值"
                )
            good_positions = np.flatnonzero(~bad_mask)
            spectrum[bad_mask] = np.interp(
                bad_positions, good_positions, spectrum[good_positions]
            )
            repaired_points[row_index] = bad_count
        warnings.warn(
            f"{path} 共发现 {int(repaired_points.sum())} 个内部缺失点，已按相邻波数线性插值；"
            "具体行可在数据拆分清单的“修复点数”列查看。",
            stacklevel=2,
        )
    spectra = spectra.astype(np.float32)

    if np.all(np.diff(axis) < 0):
        axis = axis[::-1].copy()
        spectra = spectra[:, ::-1].copy()
    elif not np.all(np.diff(axis) > 0):
        raise ValueError(f"{path} 的拉曼位移必须严格单调")

    interpolated = False
    if target_axis is not None:
        same_axis = len(axis) == len(target_axis) and np.allclose(
            axis, target_axis, rtol=1e-6, atol=1e-5
        )
        if not same_axis:
            tolerance = max(float(np.ptp(target_axis)), 1.0) * 1e-7
            if target_axis[0] < axis[0] - tolerance or target_axis[-1] > axis[-1] + tolerance:
                raise ValueError(
                    f"{path} 的波数范围 [{axis[0]:.3f}, {axis[-1]:.3f}] "
                    f"不能覆盖参考范围 [{target_axis[0]:.3f}, {target_axis[-1]:.3f}]"
                )
            spectra = np.stack(
                [np.interp(target_axis, axis, row) for row in spectra], axis=0
            ).astype(np.float32)
            axis = target_axis.copy()
            interpolated = True

    sample_ids = raw.iloc[1:, 0].astype(str).to_numpy()
    # 离线扩充编号示例：00023_0为原始版本，00023_1～N为增强版本。
    encoded = pd.Series(sample_ids).str.extract(r"^(\d{5,})_(\d+)$")
    encoded_rows = encoded.notna().all(axis=1).to_numpy()
    if encoded_rows.any() and not encoded_rows.all():
        raise ValueError(f"{path} 的扩充样本编号格式不一致")
    if encoded_rows.all():
        parent_source_row = encoded.iloc[:, 0].astype(np.int64).to_numpy()
        augmentation_index = encoded.iloc[:, 1].astype(np.int64).to_numpy()
        lineage = pd.DataFrame(
            {"parent": parent_source_row, "augmentation": augmentation_index}
        )
        if lineage.duplicated().any():
            raise ValueError(f"{path} 存在重复的原始光谱/扩充编号")
    else:
        parent_source_row = np.arange(1, len(raw), dtype=np.int64)
        augmentation_index = np.zeros(len(parent_source_row), dtype=np.int64)

    metadata = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "concentration_text": raw.iloc[1:, 1].astype(str).to_numpy(),
            "row_in_file": np.arange(1, len(raw), dtype=np.int64),
            # CSV第一行是波数标题，因此实际文件行号从2开始。
            "csv_line": np.arange(2, len(raw) + 1, dtype=np.int64),
            "repaired_points": repaired_points,
            "parent_source_row": parent_source_row,
            "augmentation_index": augmentation_index,
            "is_augmented": augmentation_index > 0,
        }
    )
    metadata["concentration_group"] = metadata["concentration_text"].map(
        concentration_group
    )
    return spectra, axis, metadata, interpolated


def load_training_data(
    specs: Sequence[Tuple[str, Path]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    if not specs:
        raise ValueError("至少需要一个 --data role=path")

    all_x: List[np.ndarray] = []
    all_y: List[np.ndarray] = []
    all_meta: List[pd.DataFrame] = []
    reference_axis: Optional[np.ndarray] = None
    seen_paths: set[Path] = set()

    # 第一份文件的波数轴作为参考，其他文件必要时插值到相同位置。
    for source_number, (role, path) in enumerate(specs):
        resolved = path.resolve()
        if resolved in seen_paths:
            raise ValueError(f"同一文件被重复传入：{path}")
        seen_paths.add(resolved)

        x, axis, meta, interpolated = read_spectral_csv(path, reference_axis)
        if reference_axis is None:
            reference_axis = axis.copy()
        y = np.full(len(x), ROLE_CLASS_IDS[role], dtype=np.int64)
        meta["role"] = role
        meta["source_file"] = str(path)
        meta["source_id"] = f"source_{source_number:02d}_{role}_{path.stem}"
        meta["interpolated"] = interpolated
        all_x.append(x)
        all_y.append(y)
        all_meta.append(meta)

    assert reference_axis is not None
    x_all = np.concatenate(all_x, axis=0).astype(np.float32)
    y_all = np.concatenate(all_y, axis=0).astype(np.int64)
    meta_all = pd.concat(all_meta, ignore_index=True)
    return x_all, y_all, reference_axis, meta_all


def validate_targets(y: np.ndarray, roles: Sequence[str]) -> None:
    values = set(np.asarray(y, dtype=np.int64).tolist())
    if values != set(ROLE_CLASS_IDS.values()):
        raise ValueError(
            "三分类训练必须同时包含mg、r6g、both三类，"
            f"当前角色为{sorted(set(roles))}、类别编号为{sorted(values)}。"
        )


def grouped_stratified_split(
    metadata: pd.DataFrame,
    val_fraction: float,
    test_fraction: float,
    group_size: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """按原始光谱来源绑定拆分，拆分后每个版本均作为一行独立样本。"""
    if val_fraction <= 0 or test_fraction <= 0 or val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction、test_fraction 必须 > 0，且两者之和 < 1")
    if group_size < 1:
        raise ValueError("group_size 必须 >= 1")

    # 固定随机种子后，四个通道模型会得到完全相同的数据拆分。
    rng = np.random.default_rng(seed)
    # assignments记录每行所属原始光谱应进入哪个集合。扩充版本必须跟随原始谱，
    # 防止同一条原始光谱的近似版本跨集合造成数据泄漏。
    assignments = np.full(len(metadata), -1, dtype=np.int8)  # 0 train, 1 val, 2 test
    notes: List[str] = []

    strata = metadata.groupby(
        ["source_id", "concentration_group"], sort=False, dropna=False
    ).indices
    for (source_id, conc), raw_indices in strata.items():
        indices = np.sort(np.asarray(raw_indices, dtype=np.int64))
        parent_rows = metadata.iloc[indices]["parent_source_row"].to_numpy(dtype=np.int64)
        unique_parents = np.unique(parent_rows)
        parent_rank = {int(parent): rank for rank, parent in enumerate(unique_parents)}
        blocks = np.asarray(
            [parent_rank[int(parent)] // group_size for parent in parent_rows],
            dtype=np.int64,
        )
        unique_blocks = np.unique(blocks)

        if len(unique_blocks) >= 3:
            shuffled = rng.permutation(unique_blocks)
            n_test = max(1, int(round(len(unique_blocks) * test_fraction)))
            n_val = max(1, int(round(len(unique_blocks) * val_fraction)))
            while n_test + n_val >= len(unique_blocks):
                if n_test >= n_val and n_test > 1:
                    n_test -= 1
                elif n_val > 1:
                    n_val -= 1
                else:
                    break
            test_blocks = set(shuffled[:n_test].tolist())
            val_blocks = set(shuffled[n_test : n_test + n_val].tolist())
            for idx, block in zip(indices, blocks):
                assignments[idx] = 2 if block in test_blocks else 1 if block in val_blocks else 0
        elif len(unique_parents) >= 3:
            notes.append(
                f"{source_id}/{conc} 只有 {len(unique_blocks)} 个采集块，"
                "退化为逐原始光谱拆分"
            )
            shuffled_parents = rng.permutation(unique_parents)
            n_test = max(1, int(round(len(unique_parents) * test_fraction)))
            n_val = max(1, int(round(len(unique_parents) * val_fraction)))
            if n_test + n_val >= len(unique_parents):
                n_test, n_val = 1, 1
            test_parents = set(shuffled_parents[:n_test].tolist())
            val_parents = set(shuffled_parents[n_test : n_test + n_val].tolist())
            for idx, parent in zip(indices, parent_rows):
                assignments[idx] = (
                    2 if parent in test_parents else 1 if parent in val_parents else 0
                )
        else:
            raise ValueError(
                f"{source_id}/{conc} 只有 {len(unique_parents)} 条原始光谱，无法三路拆分"
            )

    if np.any(assignments < 0):
        raise RuntimeError("内部错误：存在未分配的数据行")
    result = (
        np.flatnonzero(assignments == 0),
        np.flatnonzero(assignments == 1),
        np.flatnonzero(assignments == 2),
    )
    if any(len(part) == 0 for part in result):
        raise ValueError("拆分后有空集合，请增加数据量或调整拆分比例")
    notes.append(
        "来源绑定仅用于集合拆分；拆分完成后，各集合内的原始行和扩充行均逐行参与计算"
    )
    return result[0], result[1], result[2], notes


class SpectralPreprocessor:
    """原脚本 StandardScaler 强度通道 + 逐谱 SNV/导数通道。"""

    def __init__(
        self,
        clip: float = 10.0,
        eps: float = 1e-6,
        channel_set: str = "raw_snv_d1",
    ):
        if channel_set not in CHANNEL_SETS:
            raise ValueError(f"未知通道组合：{channel_set}")
        self.clip = float(clip)
        self.eps = float(eps)
        self.channel_set = channel_set
        self.center: Optional[np.ndarray] = None
        self.scale: Optional[np.ndarray] = None
        self.axis: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray, axis: np.ndarray) -> "SpectralPreprocessor":
        # 只允许传入训练集：测试集绝不能参与均值和标准差计算。
        raw = np.asarray(x, dtype=np.float64)
        self.center = np.mean(raw, axis=0).astype(np.float32)
        scale = np.std(raw, axis=0)
        scale = np.where(scale > self.eps, scale, 1.0)
        self.scale = scale.astype(np.float32)
        self.axis = np.asarray(axis, dtype=np.float32).copy()
        return self

    def _snv(self, x: np.ndarray) -> np.ndarray:
        center = np.mean(x, axis=1, keepdims=True)
        scale = np.std(x, axis=1, keepdims=True)
        return (x - center) / np.maximum(scale, self.eps)

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.center is None or self.scale is None or self.axis is None:
            raise RuntimeError("预处理器尚未 fit")
        raw = np.asarray(x, dtype=np.float32)
        # 第 1 通道：与原项目一致的逐波数点 StandardScaler。
        standardized = np.clip((raw - self.center) / self.scale, -self.clip, self.clip)
        channels = [standardized]
        if self.channel_set == "raw":
            return np.stack(channels, axis=1).astype(np.float32)

        # 第 2 通道：每条光谱单独做 SNV，突出相对峰形。
        shape = self._snv(raw)
        channels.append(shape)
        if self.channel_set == "raw_snv":
            return np.stack(channels, axis=1).astype(np.float32)

        # 第 3 通道：一阶导数，强调峰的上升沿和下降沿。
        first = self._snv(np.gradient(shape, self.axis, axis=1))
        channels.append(first)
        if self.channel_set == "raw_snv_d1":
            return np.nan_to_num(
                np.stack(channels, axis=1),
                nan=0.0,
                posinf=self.clip,
                neginf=-self.clip,
            ).astype(np.float32)

        # 第 4 通道：二阶导数，强调峰曲率，但也可能放大噪声。
        second = self._snv(np.gradient(first, self.axis, axis=1))
        channels.append(second)
        return np.nan_to_num(
            np.stack(channels, axis=1),
            nan=0.0,
            posinf=self.clip,
            neginf=-self.clip,
        ).astype(np.float32)

    def state_dict(self) -> Dict[str, Any]:
        if self.center is None or self.scale is None or self.axis is None:
            raise RuntimeError("预处理器尚未 fit")
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
        obj = cls(
            float(state["clip"]),
            float(state["eps"]),
            str(state.get("channel_set", "raw_snv_d1_d2")),
        )
        obj.center = state["center"].cpu().numpy().astype(np.float32)
        obj.scale = state["scale"].cpu().numpy().astype(np.float32)
        obj.axis = state["axis"].cpu().numpy().astype(np.float32)
        return obj


class SpectraDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        y: Optional[np.ndarray] = None,
    ):
        self.x = torch.from_numpy(np.asarray(x, dtype=np.float32))
        self.y = None if y is None else torch.from_numpy(np.asarray(y, dtype=np.int64))

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int):
        spectrum = self.x[index]
        if self.y is None:
            return spectrum
        return spectrum, self.y[index]
