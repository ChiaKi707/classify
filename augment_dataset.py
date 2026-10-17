# -*- coding: utf-8 -*-
"""把原始拉曼光谱离线扩充为可追溯CSV，供后续训练统一读取。

默认规则：
1. 每条原始光谱保留为编号“原始行号_0”，例如00023_0。
2. 每条原始光谱另外生成4份增强结果，编号后缀依次为1～4。
3. 每份增强结果都从原始光谱重新开始，连续执行5轮增强链。
4. 每轮依次判断是否执行波数轴平移、强度变化和加性高斯噪声；
   三项独立抽签，实际随机参数写入augmentation_manifest.csv。
5. 原始CSV只读，全部生成数据写入单独的data_augmented目录。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from config import ChineseParser, DEFAULT_DATA, RAW_DATA_DIR, ROOT, setup_console
from data import data_fingerprints, read_spectral_csv


def shift_with_edge_values(spectrum: np.ndarray, shift: int) -> np.ndarray:
    """沿波数轴平移，并用最近边界值填充，不进行首尾循环。"""
    result = np.roll(spectrum, shift).copy()
    if shift > 0:
        result[:shift] = result[shift]
    elif shift < 0:
        result[shift:] = result[shift - 1]
    return result


def selected(rng: np.random.Generator, probability: float) -> bool:
    """按给定概率独立决定某项增强是否在本轮执行。"""
    return probability >= 1.0 or (
        probability > 0.0 and bool(rng.random() < probability)
    )


def augment_one(
    spectrum: np.ndarray,
    rng: np.random.Generator,
    steps: int,
    shift_prob: float,
    gain_prob: float,
    noise_prob: float,
    gain_min: float,
    gain_max: float,
    shift_points: int,
    noise_ratio: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """重复执行“平移→强度变化→高斯噪声”，生成一份增强光谱。"""
    augmented = np.asarray(spectrum, dtype=np.float64).copy()
    operation_trace: List[Dict[str, Any]] = []
    shift_count = 0
    gain_count = 0
    noise_count = 0

    for step in range(1, steps + 1):
        # 每轮的三种增强分别独立抽签，但执行顺序始终固定。
        shift_applied = shift_points > 0 and selected(rng, shift_prob)
        shift = 0
        if shift_applied:
            shift_count += 1
            shift = int(rng.integers(-shift_points, shift_points + 1))
            if shift:
                augmented = shift_with_edge_values(augmented, shift)

        gain_applied = selected(rng, gain_prob)
        gain = 1.0
        if gain_applied:
            gain_count += 1
            gain = float(rng.uniform(gain_min, gain_max))
            augmented = augmented * gain

        noise_applied = selected(rng, noise_prob)
        noise_std = 0.0
        if noise_applied:
            noise_count += 1
            # 相对噪声随当前光谱尺度变化，避免对强弱光谱使用同一绝对幅度。
            noise_std = float(noise_ratio * np.std(augmented))
            if noise_std > 0:
                augmented = augmented + rng.normal(
                    0.0, noise_std, size=augmented.shape
                )

        operation_trace.append(
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

    summary = {
        "augmentation_steps": steps,
        "shift_applied_count": shift_count,
        "gain_applied_count": gain_count,
        "noise_applied_count": noise_count,
        "operation_trace": json.dumps(
            operation_trace, ensure_ascii=False, separators=(",", ":")
        ),
    }
    return augmented.astype(np.float32), summary


def validate_arguments(args: argparse.Namespace) -> None:
    if args.copies < 1:
        raise ValueError("--copies 必须>=1")
    if args.steps < 1:
        raise ValueError("--steps 必须>=1")
    if args.shift_points < 0:
        raise ValueError("--shift-points 必须>=0")
    for name in ("gain_min", "gain_max", "noise_ratio"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"--{name.replace('_', '-')} 必须是大于等于0的有限数")
    if args.gain_min <= 0 or args.gain_max <= 0:
        raise ValueError("强度倍率必须大于0")
    if args.gain_min > args.gain_max:
        raise ValueError("--gain-min 不能大于 --gain-max")
    if args.noise_ratio <= 0:
        raise ValueError("--noise-ratio 必须大于0")
    for name in ("shift_prob", "gain_prob", "noise_prob"):
        probability = getattr(args, name)
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError(f"--{name.replace('_', '-')} 必须在[0,1]内")
    if args.shift_prob == args.gain_prob == args.noise_prob == 0:
        raise ValueError("三种增强的执行概率不能同时为0")
    if not 0 <= args.seed < 2**32:
        raise ValueError("--seed 必须在0到2**32-1之间")


def expand_file(
    source: Path,
    destination: Path,
    role: str,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> Tuple[int, int, List[Dict[str, Any]]]:
    spectra, axis, metadata, _ = read_spectral_csv(source)
    raw_header = pd.read_csv(source, header=None, nrows=1)
    raw_id_header = raw_header.iloc[0, 0]
    raw_concentration_header = raw_header.iloc[0, 1]
    id_header = (
        "编号" if pd.isna(raw_id_header) or not str(raw_id_header).strip()
        else str(raw_id_header)
    )
    concentration_header = (
        "浓度" if pd.isna(raw_concentration_header)
        or not str(raw_concentration_header).strip()
        else str(raw_concentration_header)
    )
    if concentration_header == id_header:
        concentration_header = "浓度"

    output_spectra: List[np.ndarray] = []
    output_ids: List[str] = []
    output_concentrations: List[str] = []
    manifest: List[Dict[str, Any]] = []

    for row_index, spectrum in enumerate(spectra, start=1):
        original_id = str(metadata.iloc[row_index - 1]["sample_id"])
        concentration = str(metadata.iloc[row_index - 1]["concentration_text"])
        repaired_points = int(metadata.iloc[row_index - 1]["repaired_points"])
        for augmentation_index in range(args.copies + 1):
            if augmentation_index == 0:
                generated = spectrum.astype(np.float32, copy=True)
                summary = {
                    "augmentation_steps": 0,
                    "shift_applied_count": 0,
                    "gain_applied_count": 0,
                    "noise_applied_count": 0,
                    "operation_trace": "[]",
                }
            else:
                # 四份增强结果都从同一原始光谱开始，各自使用新的随机序列。
                generated, summary = augment_one(
                    spectrum,
                    rng,
                    args.steps,
                    args.shift_prob,
                    args.gain_prob,
                    args.noise_prob,
                    args.gain_min,
                    args.gain_max,
                    args.shift_points,
                    args.noise_ratio,
                )
            encoded_id = f"{row_index:05d}_{augmentation_index}"
            output_spectra.append(generated)
            output_ids.append(encoded_id)
            output_concentrations.append(concentration)
            output_row = len(output_spectra)
            manifest.append(
                {
                    "role": role,
                    "source_file": str(source.resolve()),
                    "source_row": row_index,
                    "source_csv_line": row_index + 1,
                    "original_sample_id": original_id,
                    "repaired_points": repaired_points,
                    "output_file": str(destination.resolve()),
                    "output_row": output_row,
                    "output_csv_line": output_row + 1,
                    "augmented_sample_id": encoded_id,
                    "augmentation_index": augmentation_index,
                    "is_augmented": augmentation_index > 0,
                    **summary,
                }
            )

    axis_columns = [f"{value:.10g}" for value in axis]
    frame = pd.DataFrame(np.stack(output_spectra), columns=axis_columns)
    frame.insert(0, concentration_header, output_concentrations)
    frame.insert(0, id_header, output_ids)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False, encoding="utf-8-sig", float_format="%.10g")
    return len(spectra), len(output_spectra), manifest


def build_parser() -> argparse.ArgumentParser:
    parser = ChineseParser(description="离线扩充MG1/R6G拉曼光谱并保存为新CSV")
    parser.add_argument("--source-dir", default=str(RAW_DATA_DIR), help="原始CSV目录")
    parser.add_argument(
        "--output-dir", default=str(ROOT / "data_augmented"), help="扩充CSV和清单目录"
    )
    parser.add_argument("--copies", type=int, default=4, help="每条原始光谱生成的扩充份数")
    parser.add_argument("--steps", type=int, default=5, help="每份增强光谱重复增强链的轮数")
    parser.add_argument("--shift-prob", type=float, default=0.8, help="每轮执行平移的概率")
    parser.add_argument("--gain-prob", type=float, default=0.8, help="每轮执行强度变化的概率")
    parser.add_argument("--noise-prob", type=float, default=0.8, help="每轮执行高斯噪声的概率")
    parser.add_argument("--gain-min", type=float, default=0.90, help="随机强度倍率下限")
    parser.add_argument("--gain-max", type=float, default=1.10, help="随机强度倍率上限")
    parser.add_argument(
        "--shift-points", type=int, default=2,
        help="左右平移从[-该值, 该值]整数范围均匀抽取",
    )
    parser.add_argument(
        "--noise-ratio", type=float, default=0.01,
        help="高斯噪声标准差占当前光谱标准差的比例",
    )
    parser.add_argument("--seed", type=int, default=66, help="离线扩充随机种子")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    setup_console()
    args = build_parser().parse_args(argv)
    validate_arguments(args)
    source_dir = Path(args.source_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if source_dir == output_dir:
        raise ValueError("原始数据目录和扩充数据目录不能相同")

    specs = [(role, source_dir / filename) for role, filename in DEFAULT_DATA]
    missing = [path.name for _, path in specs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"原始数据目录缺少：{'、'.join(missing)}")

    rng = np.random.default_rng(args.seed)
    all_manifest: List[Dict[str, Any]] = []
    input_count = 0
    output_count = 0
    for role, source in specs:
        destination = output_dir / source.name
        original_rows, expanded_rows, manifest = expand_file(
            source, destination, role, args, rng
        )
        input_count += original_rows
        output_count += expanded_rows
        all_manifest.extend(manifest)
        print(f"{source.name}: {original_rows} -> {expanded_rows}")

    manifest_frame = pd.DataFrame(all_manifest)
    manifest_path = output_dir / "augmentation_manifest.csv"
    manifest_frame.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    config = {
        "method": "offline_iterative_spectral_augmentation",
        "operation_order": ["horizontal_shift", "gain", "gaussian_noise"],
        "parameters": {
            "copies": args.copies,
            "steps": args.steps,
            "shift_prob": args.shift_prob,
            "gain_prob": args.gain_prob,
            "noise_prob": args.noise_prob,
            "gain_min": args.gain_min,
            "gain_max": args.gain_max,
            "shift_points": args.shift_points,
            "noise_ratio": args.noise_ratio,
            "seed": args.seed,
        },
        "independent_original_spectra": input_count,
        "generated_augmented_spectra": output_count - input_count,
        "total_rows": output_count,
        "source_files": data_fingerprints(specs),
        "output_dir": str(output_dir),
    }
    config_path = output_dir / "augmentation_config.json"
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)

    print(f"扩充完成：{input_count}条原始光谱 + {output_count - input_count}条扩充光谱 = {output_count}条")
    print(f"扩充数据目录：{output_dir}")
    print(f"扩充清单：{manifest_path}")
    print(f"参数记录：{config_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError, argparse.ArgumentTypeError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        raise SystemExit(2)
