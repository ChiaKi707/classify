# -*- coding: utf-8 -*-
"""实验协议版本、数据准备溯源、代码指纹和环境快照。"""

from __future__ import annotations

import hashlib
import json
import platform
import re
from pathlib import Path
from typing import Any, Dict

import matplotlib
import numpy as np
import pandas as pd
import sklearn
import torch


EXPERIMENT_PROTOCOL_VERSION = 4
CODE_FILES = (
    "config.py", "representations.py", "data.py", "metrics.py", "train.py",
    "training_control.py", "research_protocol.py", "models/__init__.py",
    "models/blocks.py", "models/net.py", "models/cnn1d.py",
    "models/resnet1d.py", "models/registry.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def code_fingerprints(root: Path) -> Dict[str, str]:
    return {relative: sha256_file(root / relative) for relative in CODE_FILES}


def preparation_provenance(directory: Path) -> Dict[str, Any]:
    config_path = directory / "preparation_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return {
        "directory": str(directory.resolve()),
        "config_sha256": sha256_file(config_path),
        "config": config,
        "split_manifest_sha256": sha256_file(directory / "split_manifest.csv"),
        "augmentation_manifest_sha256": sha256_file(directory / "augmentation_manifest.csv"),
    }


def environment_snapshot() -> Dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "matplotlib": matplotlib.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def normalize_run_id(value: str) -> str:
    run_id = value.strip()
    if not run_id or not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
        raise ValueError("--run-id只能包含字母、数字、下划线、点和连字符")
    return run_id
