# -*- coding: utf-8 -*-
"""通用配置：项目路径、四种通道版本、标签及可重复实验设置。"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

# 默认目录以当前工程为基准，复制整个工程后无需更改路径。
ROOT = Path(__file__).resolve().parent
RAW_DATA_DIR = ROOT / "data"
DATA_DIR = ROOT / "data_augmented"
PT_DIR = ROOT / "pt"
OUTPUT_DIR = ROOT / "output"

# 版本2开始把业务任务定义为三个互斥类别：仅MG1、仅R6G、两者都有。
# 版本3改为可追溯的离线扩充；训练期间不再动态改变光谱。
# 版本4采用“平移→强度变化→高斯模糊”的多轮概率增强链及简化编号。
# 版本5把第三项修正为加性高斯噪声，其他离线扩充规则保持不变。
# 版本6保留来源绑定拆分，但各集合内所有版本均逐行参与训练或评价。
# 待测样品保证至少含一种目标试剂，因此不建模“二者都无”。
CHECKPOINT_FORMAT_VERSION = 6
TASK_NAME = "three_class_reagent_presence"

# 四个版本共用同一网络；区别仅是输入通道数与对应的光谱预处理。
MODELS = {
    1: {"folder": "model1", "channel_set": "raw", "name": "模型1：原始光谱"},
    2: {"folder": "model2", "channel_set": "raw_snv", "name": "模型2：原始光谱+SNV"},
    3: {"folder": "model3", "channel_set": "raw_snv_d1", "name": "模型3：再加一阶导数"},
    4: {"folder": "model4", "channel_set": "raw_snv_d1_d2", "name": "模型4：再加二阶导数"},
}
DEFAULT_DATA = (
    ("mg", "MG1.csv"),
    ("r6g", "R6g1.csv"),
    ("both", "R6g1_MG1.csv"),
    ("both", "R6g1_MG2.csv"),
)

ANALYTE_NAMES = ("MG1", "R6G")
# 三个类别的顺序同时决定训练标签、Softmax输出和混淆矩阵顺序。
CLASS_NAMES = ("仅MG1", "仅R6G", "两者都有")
# 每个三分类结果对应的成分存在状态，用于派生成分召回率和误报率。
CLASS_ANALYTE_TARGETS = (
    (1, 0),
    (0, 1),
    (1, 1),
)
# 四个消融实验是“累积增加”通道，便于逐步判断每个通道的贡献。
CHANNEL_SETS = {
    "raw": ("standardized_raw",),
    "raw_snv": ("standardized_raw", "snv"),
    "raw_snv_d1": ("standardized_raw", "snv", "first_derivative"),
    "raw_snv_d1_d2": (
        "standardized_raw",
        "snv",
        "first_derivative",
        "second_derivative",
    ),
}
CHANNEL_NAMES_CN = {
    "standardized_raw": "标准化原始光谱",
    "snv": "SNV光谱",
    "first_derivative": "一阶导数",
    "second_derivative": "二阶导数",
}
TABLE_COLUMNS_CN = {
    "sample_id": "样本编号",
    "parent_source_row": "原始光谱行序号",
    "augmentation_index": "扩充编号",
    "is_augmented": "是否扩充光谱",
    "concentration_text": "浓度文本",
    "row_in_file": "样本行序号",
    "csv_line": "CSV文件行号",
    "repaired_points": "修复点数",
    "concentration_group": "浓度分组",
    "role": "数据类别",
    "source_file": "源文件",
    "source_id": "数据源编号",
    "interpolated": "是否插值",
    "split": "数据拆分",
    "group": "分组",
    "n_samples": "样本数",
    "exact_match_accuracy": "完全正确率",
    "macro_f1_three_class": "三类别平均F1",
    "macro_f1_observed_class": "实际类别平均F1",
    "macro_f1_analyte": "成分平均F1",
    "hamming_loss": "标签错误率",
    "MG1_positive_support": "MG1阳性数",
    "MG1_negative_support": "MG1阴性数",
    "MG1_recall": "MG1召回率",
    "MG1_false_positive_rate": "MG1误报率",
    "MG1_f1": "MG1的F1",
    "R6G_positive_support": "R6G阳性数",
    "R6G_negative_support": "R6G阴性数",
    "R6G_recall": "R6G召回率",
    "R6G_false_positive_rate": "R6G误报率",
    "R6G_f1": "R6G的F1",
    "epoch": "训练轮次",
    "learning_rate": "学习率",
    "train_loss": "训练损失",
    "val_loss": "验证损失",
    "val_exact_accuracy": "验证集完全正确率",
    "val_macro_f1_three_class": "验证集三类别平均F1",
    "val_macro_f1_analyte": "验证集成分平均F1",
    "checkpoint_updated": "是否更新最低损失权重",
    "early_stopping_improved": "是否达到早停有效改善",
    "best_validation_loss": "历史最低验证损失",
    "early_stopping_reference_loss": "早停参考验证损失",
    "epochs_without_significant_improvement": "连续无有效改善轮数",
}
ROLE_CLASS_IDS = {
    "mg": 0,
    "r6g": 1,
    "both": 2,
}
ROLE_ALIASES = {
    "mg": "mg",
    "mg1": "mg",
    "r6g": "r6g",
    "both": "both",
    "mix": "both",
    "mixture": "both",
    "mix11": "both",
    "mix12": "both",
}


class ChineseHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    def _get_help_string(self, action):
        return super()._get_help_string(action).replace(
            "(default: %(default)s)", "（默认：%(default)s）"
        )


class ChineseParser(argparse.ArgumentParser):
    """共用中文帮助与错误提示，三个入口使用同一套命令风格。"""

    def __init__(self, *args, **kwargs):
        kwargs["add_help"] = False
        kwargs.setdefault("formatter_class", ChineseHelpFormatter)
        super().__init__(*args, **kwargs)
        self._positionals.title = "位置参数"
        self._optionals.title = "选项"
        self.add_argument("-h", "--help", action="help", help="显示使用说明并退出")

    def format_usage(self):
        return super().format_usage().replace("usage: ", "用法：", 1)

    def format_help(self):
        return super().format_help().replace("usage: ", "用法：", 1)

    def error(self, message):
        self.print_usage(sys.stderr)
        self.exit(2, f"参数错误：{message}\n")


def setup_console() -> None:
    """统一中文输出编码，兼容Windows终端和子进程。"""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了 CUDA，但当前 PyTorch 无法使用 CUDA。")
    return torch.device(name)
