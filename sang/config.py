"""Config loading helpers."""
from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    cfg = yaml.safe_load(open(path))
    cfg["tv"] = (cfg["frames"] - 1) // 4 + 1
    cfg["spatial"] = cfg["res"] // 8
    # Audio ticks per window, derived so the sanity checks exercise the real shapes.
    if "ta" not in cfg:
        n = round((cfg["frames"] - 1) / cfg["fps"] * 16000)   # samples, matching data.audio_samples
        if str(cfg.get("audio_encoder", "")).startswith("wavlm"):
            cfg["ta"] = max(1, (n - 400) // 320 + 1)          # WavLM: 25 ms window, 20 ms hop
        else:
            cfg["ta"] = max(1, round(n / 16000 * 12.5))       # Mimi: 12.5 Hz codes
    return cfg
