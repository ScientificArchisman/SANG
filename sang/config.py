"""Config loading helpers."""
from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    cfg = yaml.safe_load(open(path))
    cfg["tv"] = (cfg["frames"] - 1) // 4 + 1
    cfg["spatial"] = cfg["res"] // 8
    cfg.setdefault("ta", 9)
    return cfg
