# -*- coding: utf-8 -*-
"""模型包公开入口：三个一维结构与可扩展注册表。"""

from .cnn1d import CNN1D
from .net import ReagentClassMSNet
from .resnet1d import ResNet1D, ResidualBlock1D
from .registry import (
    ArchitectureSpec, available_architectures, build_model, get_architecture,
    register_architecture,
)

__all__ = [
    "ArchitectureSpec", "CNN1D", "ReagentClassMSNet", "ResNet1D",
    "ResidualBlock1D", "available_architectures", "build_model",
    "get_architecture", "register_architecture",
]
