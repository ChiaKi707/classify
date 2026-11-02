# -*- coding: utf-8 -*-
"""项目路径、任务标签、输入表示和可重复实验设置。"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

from representations import available_representations, get_representation

# 默认目录以当前工程为基准，复制整个工程后无需更改路径。
ROOT = Path(__file__).resolve().parent
RAW_DATA_DIR = ROOT / "data"
PREPARED_DATA_DIR = ROOT / "data_prepared"
PT_DIR = ROOT / "pt"
OUTPUT_DIR = ROOT / "output"

# 版本7使用显式的train/validation/test/robustness离线数据协议。
CHECKPOINT_FORMAT_VERSION = 7
TASK_NAME = "three_class_reagent_presence"

RAW_SOURCES = (
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
    channel_set: get_representation(channel_set).channels
    for channel_set in available_representations()
}
CHANNEL_NAMES_CN = {
    "standardized_raw": "标准化原始光谱",
    "snv": "SNV光谱",
    "first_derivative": "一阶导数",
    "second_derivative": "二阶导数",
}
TABLE_COLUMNS_CN = {
    "sample_id": "样本编号",
    "concentration_text": "浓度文本",
    "row_in_file": "样本行序号",
    "csv_line": "CSV文件行号",
    "repaired_points": "修复点数",
    "concentration_group": "浓度分组",
    "source_file": "源文件",
    "interpolated": "是否插值",
}
ROLE_CLASS_IDS = {
    "mg": 0,
    "r6g": 1,
    "both": 2,
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
