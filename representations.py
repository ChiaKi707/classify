# -*- coding: utf-8 -*-
"""光谱表示注册表：把输入通道组合与网络结构彻底分开。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class RepresentationSpec:
    name: str
    display_name: str
    channels: Tuple[str, ...]


_REPRESENTATIONS: Dict[str, RepresentationSpec] = {}


def register_representation(spec: RepresentationSpec) -> None:
    if spec.name in _REPRESENTATIONS:
        raise ValueError(f"光谱表示重复注册：{spec.name}")
    if not spec.channels:
        raise ValueError("光谱表示至少需要一个输入通道")
    _REPRESENTATIONS[spec.name] = spec


def get_representation(name: str) -> RepresentationSpec:
    try:
        return _REPRESENTATIONS[name]
    except KeyError as exc:
        raise ValueError(
            f"未知光谱表示 {name!r}；可用：{', '.join(available_representations())}"
        ) from exc


def available_representations() -> Tuple[str, ...]:
    return tuple(_REPRESENTATIONS)


for _spec in (
    RepresentationSpec("raw", "标准化原始光谱", ("standardized_raw",)),
    RepresentationSpec(
        "raw_snv", "原始光谱+SNV", ("standardized_raw", "snv")
    ),
    RepresentationSpec(
        "raw_snv_d1",
        "原始光谱+SNV+一阶导数",
        ("standardized_raw", "snv", "first_derivative"),
    ),
    RepresentationSpec(
        "raw_snv_d1_d2",
        "原始光谱+SNV+一阶导数+二阶导数",
        ("standardized_raw", "snv", "first_derivative", "second_derivative"),
    ),
):
    register_representation(_spec)



