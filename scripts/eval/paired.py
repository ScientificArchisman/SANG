#!/usr/bin/env python3
"""Paired comparison of eval_hdtf.py runs on the same clips and seed (the first run is the reference).

    python scripts/eval/paired.py jem_ctrl_s0 jem_own_s0 jem_rand_s0 jem_orc_s0

Per run: FVD, FID, CSIM and LSE-C (rendered runs), mean lip CCC, head range and speed as % of real, blinks, median
copy_ratio, guide diagnostics, and the clips whose head std error |rot_std - real_rot_std| is smaller than the reference's (37 of 57 is a two-sided sign
test at p = 0.033). With footage (--demo) it also prints whether the footage predicts the segment: the leave-one-out
R^2 of each real statistic from the footage's own, over the population mean (reports/SANG joint embedding
matching.md, stage 0a).
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def load(run: str, root: Path) -> dict:
    d = root / run / "clips" / "sang"
    if not d.exists():
        sys.exit(f"{d}: no sang clips")
    return {p.stem: json.loads(p.read_text()) for p in d.glob("*.json")}


def loo_r2(x: np.ndarray, y: np.ndarray) -> float:
    """Leave-one-out R^2 of y from a line in x, against the leave-one-out mean of y (0 = no better than the mean)."""
    err = base = 0.0
    for i in range(len(x)):
        o = np.arange(len(x)) != i
        b, a = np.polyfit(x[o], y[o], 1)
        err += (y[i] - a - b * x[i]) ** 2
        base += (y[i] - y[o].mean()) ** 2
    return float(1 - err / base)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--root", default=str(REPO / "results" / "hdtf"))
    args = ap.parse_args()
    warnings.simplefilter("ignore", RuntimeWarning)
    runs = {r: load(r, Path(args.root)) for r in args.runs}
    clips = sorted(set.intersection(*(set(v) for v in runs.values())))
    print(f"{len(clips)} clips in all {len(runs)} runs")
    col = lambda r, k: np.array([runs[r][c].get(k, np.nan) for c in clips], float)
    ref = args.runs[0]
    err0 = np.abs(col(ref, "rot_std") - col(ref, "real_rot_std"))
    for r in args.runs:
        real = col(r, "real_rot_std")
        err = np.abs(col(r, "rot_std") - real)
        out = {"lip_ccc": np.nanmean(col(r, "lip_ccc")),
               "head %": 100 * np.nanmean(col(r, "rot_std")) / np.nanmean(real),
               "speed %": 100 * np.nanmean(col(r, "rot_vel")) / np.nanmean(col(r, "real_rot_vel")),
               "blinks": np.nanmean(col(r, "blinks_per_min")), "copy_ratio": np.nanmedian(col(r, "copy_ratio")),
               "head err": np.nanmean(err), "guide_d": np.nanmean(col(r, "guide_d")),
               "mu_spread": np.nanmean(col(r, "mu_spread")), "t_motion": np.nanmedian(col(r, "t_motion"))}
        sj = Path(args.root) / r / "summary.json"
        vid = json.loads(sj.read_text())["rows"].get("sang", {}).get("m", {}) if sj.exists() else {}
        out = {k: vid[k] for k in ("FVD", "FID", "CSIM", "LSE-C") if k in vid} | out
        wins = "" if r == ref else f"  head err better on {int((err < err0).sum())}/{len(clips)}"
        print(f"{r:<24} " + "  ".join(f"{k} {v:.3f}" for k, v in out.items() if np.isfinite(v)) + wins)
    for k in ("rot_std", "rot_vel", "blinks_per_min"):
        x, y = col(ref, f"demo_{k}"), col(ref, f"real_{k}")
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() > 3 and x[ok].std() > 0 and y[ok].std() > 0:
            print(f"footage -> segment {k:<15} r {np.corrcoef(x[ok], y[ok])[0, 1]:.3f}  "
                  f"LOO R^2 {loo_r2(x[ok], y[ok]):.3f}  (n {ok.sum()})")


if __name__ == "__main__":
    main()
