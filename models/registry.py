# -*- coding: utf-8 -*-
"""神经网络结构注册表；新增结构只需登记构建函数。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Tuple

import torch.nn as nn

from .cnn1d import CNN1D
from .net import ReagentClassMSNet
from .resnet1d import ResNet1D


Builder = Callable[[int, float, int, Mapping[str, Any]], nn.Module]


@dataclass(frozen=True)
class ArchitectureSpec:
    name: str
    display_name: str
    backend: str
    builder: Builder


_ARCHITECTURES: Dict[str, ArchitectureSpec] = {}


def register_architecture(spec: ArchitectureSpec) -> None:
    if spec.name in _ARCHITECTURES:
        raise ValueError(f"模型结构重复注册：{spec.name}")
    _ARCHITECTURES[spec.name] = spec


def get_architecture(name: str) -> ArchitectureSpec:
    try:
        return _ARCHITECTURES[name]
    except KeyError as exc:
        raise ValueError(
            f"未知模型结构 {name!r}；可用：{', '.join(available_architectures())}"
        ) from exc


def available_architectures() -> Tuple[str, ...]:
    return tuple(_ARCHITECTURES)


def build_model(
    architecture: str,
    *,
    in_channels: int,
    dropout: float,
    num_classes: int,
    model_options: Mapping[str, Any] | None = None,
) -> nn.Module:
    spec = get_architecture(architecture)
    return spec.builder(
        int(in_channels), float(dropout), int(num_classes), dict(model_options or {})
    )


def _build_msnet(
    in_channels: int,
    dropout: float,
    num_classes: int,
    model_options: Mapping[str, Any],
) -> nn.Module:
    if model_options:
        raise ValueError(
            "当前msnet尚无额外结构参数，收到："
            + ", ".join(sorted(model_options))
        )
    return ReagentClassMSNet(
        in_channels=in_channels, dropout=dropout, num_classes=num_classes
    )


def _fixed_builder(model: type[nn.Module], display_name: str) -> Builder:
    def build(
        in_channels: int,
        dropout: float,
        num_classes: int,
        model_options: Mapping[str, Any],
    ) -> nn.Module:
        if model_options:
            raise ValueError(
                f"当前{display_name}没有额外结构参数，收到："
                + ", ".join(sorted(model_options))
            )
        return model(in_channels, dropout, num_classes)

    return build


for _spec in (
    ArchitectureSpec(
        name="msnet",
        display_name="多尺度残差一维卷积网络",
        backend="pytorch",
        builder=_build_msnet,
    ),
    ArchitectureSpec(
        name="cnn1d",
        display_name="经典全卷积一维CNN",
        backend="pytorch",
        builder=_fixed_builder(CNN1D, "cnn1d"),
    ),
    ArchitectureSpec(
        name="resnet1d",
        display_name="经典时间序列一维ResNet",
        backend="pytorch",
        builder=_fixed_builder(ResNet1D, "resnet1d"),
    ),
):
    register_architecture(_spec)

