# -*- coding: utf-8 -*-
"""工程自检：只用临时合成光谱验证准备、训练、汇总和预测流程。"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

# 自检不写字节码，并限制CPU线程，避免污染工程或占用过多资源。
sys.dont_write_bytecode = True
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import numpy as np
import pandas as pd
import torch

from config import CHANNEL_SETS, RAW_SOURCES, ROOT, setup_console
from data import SpectralPreprocessor, load_prepared_data, read_raw_csv
from metrics import compute_metrics
from models import available_architectures, build_model
from predict import load_checkpoint
from run_experiments import (
    build_train_arguments,
    can_skip,
    load_experiment_config,
    parsed_train_arguments,
)
from training_control import TRAINING_POLICY_VERSION


SOURCE_FILES = (
    "augment_dataset.py", "compare_models.py", "config.py", "data.py", "metrics.py", "predict.py",
    "representations.py", "research_protocol.py", "run_experiments.py", "train.py",
    "training_control.py", "models/__init__.py", "models/blocks.py", "models/net.py",
    "models/cnn1d.py", "models/resnet1d.py", "models/registry.py",
)
REQUIRED_FILES = (
    *SOURCE_FILES, "check.py", "README.md", "EXPERIMENTS.md",
    "EXPERIMENT_DESIGN.md", "DATA_PREPARATION_PLAN.md", "GIT_GITHUB.md", "requirements.txt",
    "MODELS.md", "configs/msnet_single.json", "configs/msnet_multi.json",
    "configs/cnn1d_single.json", "configs/cnn1d_multi.json",
    "configs/resnet1d_single.json", "configs/resnet1d_multi.json",
    "configs/compare_single.json", "configs/compare_multi.json",
    "configs/compare_plain_single.json", "configs/compare_plain_multi.json",
    "tests/test_experiment_interfaces.py", "tests/test_model_comparison.py",
    "tests/test_training_control.py",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def run(script: Path, arguments: list[str], cwd: Path, expected: int = 0) -> str:
    result = subprocess.run(
        [sys.executable, "-B", str(script), *arguments],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode != expected:
        raise RuntimeError(
            f"{script.name}退出码应为{expected}，实际为{result.returncode}：\n"
            f"{result.stdout}\n{result.stderr}"
        )
    return result.stdout + result.stderr


def make_data(directory: Path) -> dict[str, Path]:
    """生成四个小型原始CSV；每一行就是一个独立模拟实验。"""
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(66)
    axis = np.linspace(600, 1800, 64)
    peak_mg = np.exp(-((axis - 920) / 55) ** 2)
    peak_r6g = np.exp(-((axis - 1330) / 70) ** 2)
    components = {"mg": (1, 0), "r6g": (0, 1), "both": (1, 1)}
    paths: dict[str, Path] = {}
    for file_index, (role, filename) in enumerate(RAW_SOURCES):
        mg, r6g = components[role]
        strength = np.repeat([1.0, .5], 30)
        spectra = .15 + strength[:, None] * (mg * peak_mg + r6g * peak_r6g)
        spectra += rng.normal(0, .01, spectra.shape)
        if file_index == 0:
            spectra[5, 12] = np.nan
        frame = pd.DataFrame(spectra, columns=axis)
        frame.insert(0, "浓度", ["10^-2"] * 30 + ["10^-6"] * 30)
        frame.insert(0, "编号", np.arange(1, 61))
        path = directory / filename
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        paths[filename] = path
    return paths


def copy_project(destination: Path) -> None:
    for name in SOURCE_FILES:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    (destination / "configs").mkdir()
    shutil.copy2(
        ROOT / "configs/msnet_single.json",
        destination / "configs/msnet_single.json",
    )


def main() -> int:
    setup_console()
    torch.set_num_threads(1)

    for name in REQUIRED_FILES:
        require((ROOT / name).is_file(), f"工程缺少文件：{name}")
    for path in ROOT.rglob("*.py"):
        if not {".venv", "venv", "__pycache__"}.intersection(path.parts):
            ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    print("1/5 文件与语法检查通过。", flush=True)

    for architecture in available_architectures():
        for channels, channel_set in enumerate(CHANNEL_SETS, 1):
            require(len(CHANNEL_SETS[channel_set]) == channels, "四种输入必须逐次增加一个通道")
            network = build_model(
                architecture, in_channels=channels, dropout=.15, num_classes=3
            ).eval()
            with torch.no_grad():
                output = network(torch.zeros(2, channels, 64))
            require(
                tuple(output.shape) == (2, 3),
                f"{architecture}/{channel_set}前向传播形状错误",
            )
    regression = subprocess.run(
        [
            sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests",
            "-p", "test_*.py",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    require(
        regression.returncode == 0,
        f"单元和回归测试失败：\n{regression.stdout}\n{regression.stderr}",
    )
    print("2/5 四通道前向传播、最低损失权重和接口回归通过。", flush=True)

    with tempfile.TemporaryDirectory(prefix="spectra_check_") as temporary:
        temp = Path(temporary)
        project = temp / "独立 工程"
        project.mkdir()
        copy_project(project)

        message = run(project / "train.py", [], temp, expected=2)
        require("请先运行 python augment_dataset.py" in message, "缺少准备数据时提示不清晰")
        for entry in (
            "augment_dataset.py", "train.py", "run_experiments.py",
            "compare_models.py", "predict.py",
        ):
            run(project / entry, ["--help"], temp)
        run(project / "train.py", ["--batch-size", "0"], temp, expected=2)
        run(project / "augment_dataset.py", ["--copies", "-1"], temp, expected=2)
        run(
            project / "augment_dataset.py",
            ["--train-ratio", ".8", "--val-ratio", ".2", "--test-ratio", "0"],
            temp,
            expected=2,
        )

        raw_paths = make_data(project / "data")
        spectra, axis, metadata, _ = read_raw_csv(raw_paths["MG1.csv"])
        require(len(spectra) == 60 and int(metadata["repaired_points"].sum()) == 1,
                "原始CSV读取或内部缺失点修复失败")
        _, _, _, interpolated = read_raw_csv(
            raw_paths["R6g1.csv"], np.linspace(610, 1790, 50)
        )
        require(interpolated, "未知光谱波数轴插值未执行")
        for channels, channel_set in enumerate(CHANNEL_SETS, 1):
            processor = SpectralPreprocessor(channel_set=channel_set).fit(spectra, axis)
            transformed = processor.transform(spectra[:2])
            require(
                transformed.shape == (2, channels, 64) and np.isfinite(transformed).all(),
                f"{channel_set}预处理结果异常",
            )
            restored = SpectralPreprocessor.from_state_dict(processor.state_dict())
            np.testing.assert_array_equal(restored.transform(spectra[:2]), transformed)

        run(project / "augment_dataset.py", [], temp)
        prepared = load_prepared_data(project / "data_prepared")
        expected_counts = {"train": 840, "validation": 180, "test": 180}
        require(
            {name: len(prepared[name].targets) for name in expected_counts} == expected_counts,
            "全量增强后的70/15/15父实验分组数量错误",
        )
        train_metadata = prepared["train"].metadata
        require((train_metadata["augmentation_index"] == 0).sum() == 168,
                "训练集原始实验数量错误")
        require(train_metadata["is_augmented"].sum() == 672,
                "训练集离线增强数量错误")
        split_manifest = pd.read_csv(project / "data_prepared/split_manifest.csv")
        require(len(split_manifest) == 240 and split_manifest["experiment_id"].is_unique,
                "拆分清单必须一行对应一个独立实验")
        augmentation_manifest = pd.read_csv(project / "data_augment/augmentation_manifest.csv")
        augmented = augmentation_manifest[augmentation_manifest["is_augmented"]]
        traces = [json.loads(value) for value in augmented["operation_trace"]]
        require(all(len(trace) == 5 for trace in traces), "每份增强谱必须保留五轮操作记录")
        require(
            all(
                {"shift_applied", "gain_applied", "noise_applied"}.issubset(step)
                for trace in traces for step in trace
            ),
            "三种保守增强的追踪信息不完整",
        )
        require(
            all((project / "data_augment" / filename).is_file() for _, filename in RAW_SOURCES),
            "data_augment必须为每个原始CSV保留同名增强结果",
        )
        all_metadata = pd.concat(
            [split.metadata for split in prepared.splits.values()], ignore_index=True
        )
        require(
            all_metadata.groupby("experiment_id")["split"].nunique().max() == 1,
            "同一父实验的原谱与增强副本不能跨集合",
        )
        require(
            all_metadata["experiment_id"].str.count(":").min() >= 2,
            "父实验ID必须包含文件名、行描述和实验序号",
        )
        run(
            project / "augment_dataset.py",
            [
                "--copies", "0",
                "--augmented-dir", str(project / "data_plain_augment"),
                "--prepared-dir", str(project / "data_plain"),
            ],
            temp,
        )
        plain = load_prepared_data(project / "data_plain")
        require(set(plain.splits) == {"train", "validation", "test"},
                "无增强数据只能包含训练、验证和测试集")
        require(
            {name: len(plain[name].targets) for name in plain.splits}
            == {"train": 168, "validation": 36, "test": 36},
            "无增强数据数量错误",
        )
        require(not plain["train"].metadata["is_augmented"].any(),
                "无增强训练集不应包含增强谱")
        plain_manifest = pd.read_csv(project / "data_plain/split_manifest.csv")
        require(plain_manifest.equals(split_manifest),
                "相同拆分参数下增强与无增强目录的原始实验划分必须一致")
        perfect = np.eye(3) * .98 + (1 - np.eye(3)) * .01
        require(compute_metrics(np.arange(3), perfect)["exact_match_accuracy"] == 1,
                "三分类指标计算错误")
        print("3/5 逐文件增强、父实验分组、数据隔离和预处理检查通过。", flush=True)

        output_root = project / "output"
        pt_root = project / "pt"
        matrix_args = [
            "--config", str(project / "configs/msnet_single.json"),
            "--prepared-dir", str(project / "data_prepared"),
            "--output-dir", str(output_root), "--pt-dir", str(pt_root),
            "--epochs", "1", "--device", "cpu",
        ]
        run(project / "run_experiments.py", matrix_args, temp)
        run_root = output_root / "msnet_single"
        results = pd.read_csv(run_root / "results.csv")
        require(len(results) == 4, "四种光谱表示的结果汇总不完整")
        for channels, channel_set in enumerate(CHANNEL_SETS, 1):
            result_dir = run_root / "msnet" / channel_set / "seed_66"
            weight = pt_root / "msnet_single" / "msnet" / channel_set / "seed_66/best.pt"
            record = json.loads((result_dir / "metrics.json").read_text(encoding="utf-8"))
            checkpoint = load_checkpoint(weight, torch.device("cpu"))
            require(checkpoint["model_config"]["in_channels"] == channels,
                    f"{channel_set}权重通道数错误")
            require(record["test"]["n_samples"] == 180, "测试集增强后数量错误")
            require(
                record["test"]["sample_counts"]["test_original"] == 36
                and record["test"]["sample_counts"]["test_augmented"] == 144,
                "测试集父实验与增强行计数错误",
            )
            require(record["training_policy_version"] == TRAINING_POLICY_VERSION,
                    "训练结果缺少当前权重选择协议")
            require(
                {path.name for path in result_dir.iterdir()}
                == {"metrics.json", "history.csv", "diagnostics.png"},
                "单次训练结果应只保留三个必要文件",
            )
        root_files = {path.name for path in run_root.iterdir() if path.is_file()}
        require(
            root_files == {
                "config.json", "split.csv", "results.csv", "summary.csv",
                "selected_model.json", "comparison.png", "training_curves.png",
            },
            "实验矩阵顶层结果文件不符合精简协议",
        )
        reused = run(
            project / "run_experiments.py", [*matrix_args, "--skip-existing"], temp
        )
        require(reused.count("复用完整结果") == 4, "严格结果复用检查异常")
        print("4/5 四种通道短训练、汇总、选模和结果复用检查通过。", flush=True)

        config = load_experiment_config(project / "configs/msnet_single.json")
        arguments = build_train_arguments(
            config, config["architectures"][0], "raw_snv_d1", 66,
            run_id="msnet_single", prepared_dir=project / "data_prepared",
            pt_dir=pt_root, output_dir=output_root, device="cpu", epochs=1,
        )
        parsed = parsed_train_arguments(arguments)
        result_dir = run_root / "msnet/raw_snv_d1/seed_66"
        checkpoint_path = pt_root / "msnet_single/msnet/raw_snv_d1/seed_66/best.pt"
        require(can_skip(parsed, project / "data_prepared", result_dir, checkpoint_path),
                "完整且协议一致的结果应可复用")
        parsed.epochs = 2
        require(not can_skip(parsed, project / "data_prepared", result_dir, checkpoint_path),
                "修改训练参数后不应复用旧结果")

        prediction_path = project / "predictions.csv"
        run(
            project / "predict.py",
            [
                "--input", str(raw_paths["R6g1.csv"]),
                "--selection", str(run_root / "selected_model.json"),
                "--output", str(prediction_path), "--device", "cpu",
            ],
            temp,
        )
        predictions = pd.read_csv(prediction_path)
        require(
            len(predictions) == 60
            and {"仅MG1类别概率", "仅R6G类别概率", "两者都有类别概率", "预测类别"}
            .issubset(predictions),
            "未知样本预测表不完整",
        )
        run(
            project / "predict.py",
            ["--input", str(raw_paths["R6g1.csv"]), "--batch-size", "0"],
            temp,
            expected=2,
        )
        print("5/5 选定模型预测、非法参数和旧结果误用检查通过。", flush=True)

    print("全部自检通过；临时数据和结果已清理，真实data/data_augment/data_prepared/pt/output未改动。")
    print("自检只验证软件流程，不代表真实实验准确率。")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError, AssertionError) as exc:
        print(f"自检失败：{exc}", file=sys.stderr)
        raise SystemExit(2)
