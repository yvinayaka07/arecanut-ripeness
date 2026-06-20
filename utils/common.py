"""
utils/common.py
===============
Shared utilities: config loading, logging, device resolution, seeding.
"""

from __future__ import annotations

import logging
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class Config:
    """Dot-notation access wrapper around a YAML dictionary."""

    def __init__(self, data: dict):
        for key, value in data.items():
            setattr(self, key, Config(value) if isinstance(value, dict) else value)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


def load_config(config_path: str | Path = "config.yaml") -> Config:
    """Load YAML config and return a Config object."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)
    return Config(data)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logger(name: str = "arecanut", level: str = "INFO") -> logging.Logger:
    """Return a simple console logger."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                                           datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def get_device(preference: str = "auto") -> torch.device:
    """Resolve the best available compute device."""
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    # Auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def seed_everything(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
