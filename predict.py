# -*- coding: utf-8 -*-
"""三分类预测入口：复用权重中保存的波数轴和标准化参数。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import (
    CHECKPOINT_FORMAT_VERSION, CLASS_NAMES, ChineseParser, OUTPUT_DIR, PT_DIR,
    TABLE_COLUMNS_CN, TASK_NAME, choose_device, setup_console,
)
from data import SpectralPreprocessor, SpectraDataset, read_spectral_csv
from metrics import make_prediction_frame, predict_loader
from models import ReagentClassMSNet


def load_checkpoint(path: Path, device: torch.device) -> Dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # 兼容旧版 PyTorch
        checkpoint = torch.load(path, map_location=device)
    if checkpoint.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            "模型文件不是当前三分类格式；旧的二输出模型不能直接使用，请重新训练"
        )
    if checkpoint.get("task") != TASK_NAME:
        raise ValueError("模型文件的任务定义与当前三分类工程不一致")
    return checkpoint


def command_predict(args: argparse.Namespace) -> None:
    device = choose_device(args.device)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"模型文件不存在：{checkpoint_path}")
    checkpoint = load_checkpoint(checkpoint_path, device)
    reference_axis = checkpoint["reference_axis"].cpu().numpy().astype(np.float64)
    spectra, _, metadata, interpolated = read_spectral_csv(Path(args.input), reference_axis)
    preprocessor = SpectralPreprocessor.from_state_dict(checkpoint["preprocessor"])
    channels = preprocessor.transform(spectra)
    dataset = SpectraDataset(channels)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = ReagentClassMSNet(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    probabilities, _, _ = predict_loader(model, loader, device)
    metadata["source_file"] = str(args.input)
    metadata["interpolated"] = interpolated
    result = make_prediction_frame(metadata, probabilities)
    output_path = (
        Path(args.output)
        if args.output
        else OUTPUT_DIR / "predict" / (Path(args.input).stem + "_预测.csv")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.rename(columns=TABLE_COLUMNS_CN).to_csv(output_path, index=False, encoding="utf-8-sig")
    print("各类别预测数量：")
    print(result["预测类别"].value_counts().reindex(CLASS_NAMES, fill_value=0).to_string())
    print(f"预测结果已保存：{output_path.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    parser = ChineseParser(description="把未知光谱判为仅MG1、仅R6G或两者都有")
    parser.add_argument("--input", required=True, help="与训练数据同格式的未知光谱CSV")
    parser.add_argument("--checkpoint", default=str(PT_DIR / "model3" / "best.pt"),
                        help="当前三分类格式的模型文件")
    parser.add_argument("--output", help="预测CSV路径；默认保存到output/predict")
    parser.add_argument("--batch-size", type=int, default=128, help="每批预测光谱数")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="计算设备")
    parser.add_argument("--num-workers", type=int, default=0, help="数据加载进程数")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    setup_console()
    args = build_parser().parse_args(argv)
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("--batch-size必须>=1，--num-workers必须>=0")
    command_predict(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        raise SystemExit(2)
