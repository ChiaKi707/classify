# -*- coding: utf-8 -*-
"""工程自检：使用临时合成光谱验证四模型训练和预测，不读取用户实验数据。"""

from __future__ import annotations

import ast
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

# 只影响本次检查进程及其子进程，避免检查时产生大量线程和缓存。
sys.dont_write_bytecode = True
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import numpy as np
import pandas as pd
import torch

from config import CHANNEL_SETS, DEFAULT_DATA, MODELS, ROOT, setup_console
from data import (
    SpectralPreprocessor, data_fingerprints, grouped_stratified_split,
    load_training_data, read_spectral_csv,
)
from metrics import compute_metrics
from models import ReagentClassMSNet
from predict import load_checkpoint
from run_all import build_parser, can_skip
from training_control import TRAINING_POLICY_VERSION

SOURCE_FILES = (
    "config.py", "data.py", "augment_dataset.py", "metrics.py", "train.py",
    "predict.py", "run_all.py", "training_control.py",
    "models/__init__.py", "models/blocks.py", "models/net.py",
)
REQUIRED_FILES = (*SOURCE_FILES, "check.py", "README.md", "EXPERIMENTS.md", "requirements.txt",
                  "tests/test_training_control.py")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def run(script: Path, arguments: list[str], cwd: Path, expected: int = 0) -> str:
    """从其他工作目录调用独立副本，检查导入和相对路径问题。"""
    result = subprocess.run(
        [sys.executable, "-B", str(script), *arguments],
        cwd=cwd, capture_output=True, text=True, encoding="utf-8",
    )
    if result.returncode != expected:
        raise RuntimeError(f"{script.name}运行失败：\n{result.stdout}\n{result.stderr}")
    return result.stdout + result.stderr


def make_data(directory: Path) -> list[tuple[str, Path]]:
    """只为软件检查生成带标签的模拟光谱，不能用来报告真实检测性能。"""
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(66)
    axis = np.linspace(600, 1800, 64)
    peak_mg = np.exp(-((axis - 920) / 55) ** 2)
    peak_r6g = np.exp(-((axis - 1330) / 70) ** 2)
    roles = {"mg": (1, 0), "r6g": (0, 1), "both": (1, 1)}
    specs = []
    for number, (role, name) in enumerate(DEFAULT_DATA):
        mg, r6g = roles[role]
        strength = np.repeat([1.0, 0.5], 30)
        spectra = 0.15 + strength[:, None] * (mg * peak_mg + r6g * peak_r6g)
        spectra = spectra + rng.normal(0, 0.01, (60, len(axis)))
        if number == 0:
            spectra[5, 12] = np.nan
        frame = pd.DataFrame(spectra, columns=axis)
        frame.insert(0, "浓度", ["10^-2"] * 30 + ["10^-6"] * 30)
        frame.insert(0, "编号", np.arange(1, 61))
        path = directory / name
        frame.to_csv(path, index=False, encoding="utf-8")
        specs.append((role, path))
    return specs


def main() -> int:
    setup_console()
    torch.set_num_threads(1)
    for name in REQUIRED_FILES:
        require((ROOT / name).is_file(), f"工程缺少文件：{name}")
    for path in ROOT.rglob("*.py"):
        if ".venv" not in path.parts and "venv" not in path.parts:
            ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    print("1/5 文件与语法检查通过。", flush=True)

    counts = (72088, 72256, 72424, 72592)
    for number in MODELS:
        network = ReagentClassMSNet(in_channels=number).eval()
        with torch.no_grad():
            output = network(torch.zeros(2, number, 524))
        require(tuple(output.shape) == (2, 3), "网络输出形状错误")
        require(sum(p.numel() for p in network.parameters()) == counts[number - 1],
                f"模型{number}参数量与本版本记录不一致；若主动修改结构，请同步更新自检")
    regression = subprocess.run(
        [sys.executable, "-B", "-m", "unittest", "discover", "-s", str(ROOT / "tests"),
         "-p", "test_training_control.py"],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
    )
    require(regression.returncode == 0,
            f"最佳权重与早停回归检查失败：\n{regression.stdout}\n{regression.stderr}")
    print("2/5 网络前向传播、最低损失权重与早停回归检查通过。", flush=True)

    with tempfile.TemporaryDirectory(prefix="spectra_check_") as temporary:
        temp = Path(temporary)
        project = temp / "独立 工程"
        project.mkdir()
        # 只复制少量源代码，不复制用户数据、权重和历史输出。
        for name in SOURCE_FILES:
            target = project / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
        message = run(project / "train.py", [], temp, expected=2)
        require("请先运行 python augment_dataset.py" in message, "缺失扩充数据提示不清晰")
        for entry in ("augment_dataset.py", "train.py", "run_all.py", "predict.py"):
            run(project / entry, ["--help"], temp)
        run(project / "train.py", ["--batch-size", "0"], temp, expected=2)
        run(project / "augment_dataset.py", ["--copies", "0"], temp, expected=2)
        run(project / "augment_dataset.py", ["--steps", "0"], temp, expected=2)

        raw_specs = make_data(project / "data")
        x, y, axis, metadata = load_training_data(raw_specs)
        require(len(x) == 240 and int(metadata["repaired_points"].sum()) == 1,
                "合成数据读取或缺失值修复失败")
        require(metadata.iloc[0]["csv_line"] == 2, "CSV实际行号应从2开始")
        split1 = grouped_stratified_split(metadata, .15, .20, 10, 66)
        split2 = grouped_stratified_split(metadata, .15, .20, 10, 66)
        for first, second in zip(split1[:3], split2[:3]):
            np.testing.assert_array_equal(first, second)
        require(set(split1[0]).isdisjoint(split1[1]) and
                set(split1[0]).isdisjoint(split1[2]), "训练集与其他集合重叠")
        for number, channel_set in enumerate(CHANNEL_SETS, start=1):
            processor = SpectralPreprocessor(channel_set=channel_set).fit(x[split1[0]], axis)
            channels = processor.transform(x[:2])
            require(channels.shape == (2, number, 64) and np.isfinite(channels).all(),
                    "预处理通道形状或数值异常")
            restored = SpectralPreprocessor.from_state_dict(processor.state_dict())
            np.testing.assert_array_equal(restored.transform(x[:2]), channels)
        _, _, _, interpolated = read_spectral_csv(
            raw_specs[1][1], np.linspace(610, 1790, 50)
        )
        require(interpolated, "波数对齐插值未执行")
        all_labels = np.array([0, 1, 2])
        scores = np.full((3, 3), .01)
        np.fill_diagonal(scores, .98)
        require(compute_metrics(all_labels, scores)["macro_f1_three_class"] == 1,
                "三类别评价错误")
        run(project / "augment_dataset.py", [], temp)
        augmented_specs = [
            (role, project / "data_augmented" / filename)
            for role, filename in DEFAULT_DATA
        ]
        expanded_x, _, _, expanded_metadata = load_training_data(augmented_specs)
        require(len(expanded_x) == 1200, "离线扩充后应为原数据的5倍")
        require(int(expanded_metadata["is_augmented"].sum()) == 960,
                "离线扩充标记数量错误")
        augmentation_manifest = pd.read_csv(
            project / "data_augmented" / "augmentation_manifest.csv"
        )
        require(
            expanded_metadata["sample_id"].str.match(r"^\d{5,}_\d+$").all(),
            "离线扩充样本编号格式错误",
        )
        original_manifest = augmentation_manifest[~augmentation_manifest["is_augmented"]]
        augmented_manifest = augmentation_manifest[augmentation_manifest["is_augmented"]]
        require((original_manifest["augmentation_steps"] == 0).all()
                and (augmented_manifest["augmentation_steps"] == 5).all(),
                "离线增强链轮数错误")
        require("noise_applied_count" in augmented_manifest
                and "blur_applied_count" not in augmented_manifest,
                "第三种增强应为高斯噪声")
        traces = [json.loads(value) for value in augmented_manifest["operation_trace"]]
        require(all(len(trace) == 5 for trace in traces), "增强操作记录不完整")
        require(all("noise_applied" in step and "noise_std" in step
                    and "blur_applied" not in step
                    for trace in traces for step in trace),
                "高斯噪声操作记录不完整")
        shifts = {
            int(step["shift_points"])
            for trace in traces
            for step in trace
            if step["shift_applied"]
        }
        require(
            shifts.issubset({-2, -1, 0, 1, 2}),
            "离线扩充平移范围错误",
        )
        expanded_split = grouped_stratified_split(expanded_metadata, .15, .20, 10, 66)
        require(tuple(map(len, expanded_split[:3])) == (400, 400, 400),
                "离线扩充数据的防泄漏拆分数量错误")
        for indices in expanded_split[:3]:
            versions_per_parent = (
                expanded_metadata.iloc[indices]
                .groupby(["source_id", "parent_source_row"]).size()
            )
            require((versions_per_parent == 5).all(),
                    "同一集合内没有逐行保留父光谱的全部5个版本")
        train_parent_keys = set(zip(
            expanded_metadata.iloc[expanded_split[0]]["source_id"],
            expanded_metadata.iloc[expanded_split[0]]["parent_source_row"],
        ))
        held_out_parent_keys = set(zip(
            expanded_metadata.iloc[np.concatenate(expanded_split[1:3])]["source_id"],
            expanded_metadata.iloc[np.concatenate(expanded_split[1:3])]["parent_source_row"],
        ))
        require(train_parent_keys.isdisjoint(held_out_parent_keys),
                "同一原始光谱的扩充版本跨越训练与留出集合")
        print("3/5 数据修复、离线扩充、防泄漏拆分、预处理和指标检查通过。", flush=True)

        args = ["--epochs", "1", "--batch-size", "16", "--device", "cpu"]
        run(project / "run_all.py", args, temp)
        summary = pd.read_csv(project / "output/compare/tables/四模型对比结果.csv")
        require(len(summary) == 4 and "最佳验证损失" in summary, "四模型汇总不完整")
        for number in MODELS:
            checkpoint = load_checkpoint(project / "pt" / f"model{number}/best.pt",
                                         torch.device("cpu"))
            require(checkpoint["model_config"]["in_channels"] == number, "权重通道配置错误")
            require(checkpoint["model_config"]["num_classes"] == 3, "权重类别配置错误")
            result_dir = project / "output" / f"model{number}"
            history = pd.read_csv(result_dir / "tables/training_history.csv")
            metrics = json.loads((result_dir / "test_metrics.json").read_text(encoding="utf-8"))
            require(math.isclose(metrics["best_validation_loss"], history["val_loss"].min(),
                                 rel_tol=1e-12, abs_tol=1e-15),
                    "保存权重对应的验证损失不是训练历史最低值")
            require(metrics["best_epoch"] == int(history.loc[history["val_loss"].idxmin(), "epoch"]),
                    "最佳轮次与训练历史不一致")
            require(checkpoint["training_policy_version"] == TRAINING_POLICY_VERSION,
                    "权重缺少当前训练策略版本")
            require((result_dir / "figures/training_history.png").stat().st_size > 0,
                    "训练曲线缺失")
            require((result_dir / "figures/test_confusion_matrix.png").stat().st_size > 0,
                    "混淆矩阵缺失")
        reused = run(project / "run_all.py", [*args, "--skip-existing"], temp)
        require(reused.count("复用完整结果") == 4, "跳过已完成模型功能异常")
        print("4/5 独立目录四模型训练、权重保存、图表汇总与结果复用通过。", flush=True)

        options = build_parser().parse_args([
            *args, "--pt-dir", str(project / "pt"), "--output-dir", str(project / "output"),
        ])
        fingerprints = data_fingerprints(augmented_specs)
        require(can_skip(options, 3, fingerprints), "完整模型应当可复用")
        options.epochs = 2
        require(not can_skip(options, 3, fingerprints), "修改训练参数后不应复用旧结果")
        options.epochs = 1
        changed = [dict(item) for item in fingerprints]
        changed[0]["sha256"] = "changed"
        require(not can_skip(options, 3, changed), "更换数据后不应复用旧结果")
        for filename in ("training_config.json", "test_metrics.json"):
            path = project / "output/model3" / filename
            content = path.read_text(encoding="utf-8")
            old_policy = json.loads(content)
            old_policy.pop("training_policy_version")
            try:
                path.write_text(json.dumps(old_policy, ensure_ascii=False), encoding="utf-8")
                require(not can_skip(options, 3, fingerprints),
                        f"{filename}使用旧权重选择规则时不应复用")
            finally:
                path.write_text(content, encoding="utf-8")

        run(project / "predict.py", ["--input", str(raw_specs[1][1]), "--device", "cpu"], temp)
        predictions = pd.read_csv(project / "output/predict/R6g1_预测.csv")
        require(len(predictions) == 60 and {
            "仅MG1类别概率", "仅R6G类别概率", "两者都有类别概率",
            "MG1存在概率", "R6G存在概率", "预测类别",
        }.issubset(predictions),
                "预测表格不完整")
        run(project / "predict.py", [
            "--input", str(raw_specs[1][1]), "--device", "cpu", "--batch-size", "0",
        ], temp, expected=2)
        print("5/5 默认三分类模型预测、非法参数和旧结果误用检查通过。", flush=True)

    print("全部自检通过。临时合成数据与测试输出已清理，未改动工程data/pt/output。")
    print("自检只证明软件流程可运行，不代表真实光谱检测准确率。")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError, AssertionError) as exc:
        print(f"自检失败：{exc}", file=sys.stderr)
        raise SystemExit(2)
