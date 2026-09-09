"""Config loading helpers."""
from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    cfg = yaml.safe_load(open(path))
    cfg["tv"] = (cfg["frames"] - 1) // 4 + 1
    cfg["spatial"] = cfg["res"] // 8
    # Audio ticks per window, derived rather than guessed. The old `setdefault("ta", 9)` made
    # scripts/sanity.py validate tv=5, Ta=9 while training actually ran tv=7, Ta=33 -- the checks
    # passed on a configuration that never existed. See docs/v3_improvement_plan.md Part VI, D4.
    if "ta" not in cfg:
        n = round((cfg["frames"] - 1) / cfg["fps"] * 16000)   # samples, matching data.audio_samples
        if str(cfg.get("audio_encoder", "")).startswith("wavlm"):
            cfg["ta"] = max(1, (n - 400) // 320 + 1)          # WavLM: 25 ms window, 20 ms hop
        else:
            cfg["ta"] = max(1, round(n / 16000 * 12.5))       # Mimi: 12.5 Hz codes
    return cfg
