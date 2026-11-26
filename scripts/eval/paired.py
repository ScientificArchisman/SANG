#!/usr/bin/env python3
"""Paired comparison of eval_hdtf.py runs on the same clips and seed (the first run is the reference).

    python scripts/eval/paired.py r4_s60_d30_s0 other_run_s0
    python scripts/eval/paired.py spc_s0:sang spc_s0:sang+own spc_s0:sang+swap   # rows of one run (default row: sang)

Per run: FVD, FID, CSIM and LSE-C (rendered runs), dFVD against the reference with a 95 % bootstrap interval over
clips (each resample re-pools the I3D features of the same clips for both runs), mean lip CCC, head range and speed as % of real, blinks, median
copy_ratio, and the clips whose head std error |rot_std - real_rot_std| is smaller than the reference's (37 of 57 is a two-sided sign
test at p = 0.033). With footage (--demo) it also prints whether the footage predicts the segment: the leave-one-out
R^2 of each real statistic from the footage's own, over the population mean.
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def clip_dir(run: str, root: Path) -> Path:
    name, _, row = run.partition(":")
    return root / name / "clips" / (row or "sang")


def load(run: str, root: Path) -> dict:
    d = clip_dir(run, root)
    if not d.exists():
        sys.exit(f"{d}: no clips")
    return {p.stem: json.loads(p.read_text()) for p in d.glob("*.json")}


def dfvd(ref: Path, run: Path, clips: list, n: int = 200, seed: int = 0) -> tuple[float, float, float]:
    """FVD(run) - FVD(ref) on the shared clips, and its 95 % bootstrap interval over clips."""
    from sang.eval.metrics import frechet
    z = {d: [np.load(d / f"{c}.npz") for c in clips] for d in (ref, run)}
    f = {d: ([v["i3d_gt"] for v in z[d]], [v["i3d_gen"] for v in z[d]]) for d in z}
    fvd = lambda d, i: frechet(np.concatenate([f[d][0][k] for k in i]), np.concatenate([f[d][1][k] for k in i]))
    delta = lambda i: fvd(run, i) - fvd(ref, i)
    rng = np.random.default_rng(seed)
    boot = [delta(rng.integers(0, len(clips), len(clips))) for _ in range(n)]
    return delta(range(len(clips))), *np.percentile(boot, [2.5, 97.5])


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
               "head err": np.nanmean(err), "t_motion": np.nanmedian(col(r, "t_motion")),
               "jump rot x": np.nanmean(col(r, "jump_rot")), "real jump rot x": np.nanmean(col(r, "real_jump_rot")),
               "bg dDrag px": np.nanmean(col(r, "bg_drag") - col(r, "bg_real_drag")),
               "bg dNear px": np.nanmean(col(r, "bg_drag_near") - col(r, "bg_real_drag_near")),
               "bg real px": np.nanmean(col(r, "bg_real_drag")), "bg def px": np.nanmean(col(r, "bg_def")),
               "bg reject": np.nanmean(col(r, "bg_reject")),
               "head px %": 100 * np.nanmean(col(r, "bg_rot_std_px")) / np.nanmean(real),
               "bg n/a": float(np.sum(np.isfinite(col(r, "bg_px")) & ~np.isfinite(col(r, "bg_drag")))) or np.nan}
        x, y = col(r, "bg_def"), col(r, "bg_drag")
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() > 3:  # phase 0a gate: the renderer's own drag must rank clips as the flow does (rho >= 0.7)
            rank = lambda v: np.argsort(np.argsort(v))
            out["rho def~drag"] = float(np.corrcoef(rank(x[ok]), rank(y[ok]))[0, 1])
        name, _, row = r.partition(":")
        sj = Path(args.root) / name / "summary.json"
        vid = json.loads(sj.read_text())["rows"].get(row or "sang", {}).get("m", {}) if sj.exists() else {}
        out = {k: vid[k] for k in ("FVD", "FID", "CSIM", "LSE-C") if k in vid} | out
        wins = "" if r == ref or not np.isfinite(err).any() else f"  head err better on {int((err < err0).sum())}/{len(clips)}"
        d0, d1 = clip_dir(ref, Path(args.root)), clip_dir(r, Path(args.root))
        if r != ref and all((d / f"{c}.npz").exists() for d in (d0, d1) for c in clips):
            wins += "  dFVD {:.2f} [{:.2f}, {:.2f}]".format(*dfvd(d0, d1, clips))
        print(f"{r:<24} " + "  ".join(f"{k} {v:.3f}" for k, v in out.items() if np.isfinite(v)) + wins)
    for k in ("rot_std", "rot_vel", "blinks_per_min"):
        x, y = col(ref, f"demo_{k}"), col(ref, f"real_{k}")
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() > 3 and x[ok].std() > 0 and y[ok].std() > 0:
            print(f"footage -> segment {k:<15} r {np.corrcoef(x[ok], y[ok])[0, 1]:.3f}  "
                  f"LOO R^2 {loo_r2(x[ok], y[ok]):.3f}  (n {ok.sum()})")


if __name__ == "__main__":
    main()
