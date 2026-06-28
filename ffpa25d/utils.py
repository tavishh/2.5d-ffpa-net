"""
Utility helpers for the 2.5D FFPA-Net:
  * config loading with a single level of `_base_` inheritance,
  * deterministic seeding,
  * device resolution (cuda -> cpu fallback).

Losses live in ffpa25d/losses/loss.py; metrics in ffpa25d/metrics.py.
"""
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into `base` (override wins)."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path) -> dict:
    """Load a YAML config, resolving a single level of `_base_` inheritance."""
    config_path = Path(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if "_base_" in config:
        base_path = (config_path.parent / config["_base_"]).resolve()
        with open(base_path, "r", encoding="utf-8") as f:
            base_config = yaml.safe_load(f)
        config = deep_merge(base_config, config)
        del config["_base_"]
    return config


def set_seed(seed: int = 42):
    """Seed python / numpy / torch (and CUDA if present) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str = "cuda") -> str:
    """Return `requested` unless cuda was asked for but is unavailable."""
    if requested == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        return "cpu"
    return requested