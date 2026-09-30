#!/usr/bin/env python3
"""LSE-C / LSE-D (joonson/syncnet_python via sang.bench.lse) for talking-head mp4s with audio,
real and generated through ONE pipeline -- the only way the numbers mean anything.

    sbatch bash_scripts/job.sh scripts/eval_lse.py results/extras/demo_a results/extras/demo_b   # folders or files

Grouping: *_real.mp4 = real crops (from scripts/demo_motion.py); every other file is grouped by its
folder, so render each setting into its own --out folder and pass *_gen.mp4 from each.
Prints per-file (offset, LSE-D, LSE-C) and, per group, mean LSE-C / LSE-D and the AV-offset
histogram. Read generated LSE-C against the real row: above real means exaggerated mouths
(SyncNet rewards them), not better sync. Clips need > 100 frames with a detected face (> 4 s).
"""
import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from sang.bench import lse


def group_of(p: Path) -> str:
    """'real' for *_real.mp4, else the demo folder (one --out folder per setting)."""
    return "real" if p.stem.endswith("_real") else p.parent.name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--out", default=None, help="json file for all rows")
    args = ap.parse_args()

    files = []
    for a in map(Path, args.videos):                  # a folder = its *_real.mp4 + *_gen.mp4
        files += sorted(a.glob("*_real.mp4")) + sorted(a.glob("*_gen.mp4")) if a.is_dir() else [a]
    seen, uniq = set(), []
    for f in files:                                   # the same real clip may come from several folders
        key = f.name if f.stem.endswith("_real") else str(f)
        if key not in seen:
            seen.add(key)
            uniq.append(f)
    rows, t0 = [], time.time()
    for v in uniq:
        off, d, c = lse(v)
        rows.append({"file": v.name, "group": group_of(v), "offset": off, "lse_d": d, "lse_c": c})
        print(f"{v.name[:60]:<60} off {off:>5.0f}  LSE-D {d:6.3f}  LSE-C {c:6.3f}", flush=True)

    by = defaultdict(list)
    for r in rows:
        by[r["group"]].append(r)
    print(f"\n{'group':<34}{'n':>4}{'ok':>4}{'LSE-C':>9}{'LSE-D':>9}  offsets", flush=True)
    for g in sorted(by, key=lambda k: (k != "real", k)):
        ok = [r for r in by[g] if np.isfinite(r["lse_c"])]
        c = np.mean([r["lse_c"] for r in ok]) if ok else float("nan")
        d = np.mean([r["lse_d"] for r in ok]) if ok else float("nan")
        offs = dict(sorted(Counter(int(r["offset"]) for r in ok).items()))
        print(f"{g:<34}{len(by[g]):>4}{len(ok):>4}{c:>9.3f}{d:>9.3f}  {offs}", flush=True)
    print(f"\n{len(rows)} files, {time.time() - t0:.0f} s", flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
