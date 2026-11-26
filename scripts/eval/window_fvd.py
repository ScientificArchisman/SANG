#!/usr/bin/env python3
"""FVD per chained window of finished eval_hdtf.py runs (reports/SANG second stage hard examples.md, diagnostic D3).

    python scripts/eval/window_fvd.py mbes_r4 mbes_r4:gt_motion w2_flat dg0_d30

flow.generate starts a window every 64 frames, so the I3D features of the non-overlapping 16-frame segments split
cleanly: segments 0-3 are window 1 (from the photo), 4-7 window 2, 8-11 window 3, 12-14 window 4 (each from a
generated prefix). Every window keeps the same 3 segments per clip (FVD is biased by the sample count), and
dFVD(w) = FVD(window w) - FVD(window 1) gets a 95 % bootstrap interval over clips. The gt_motion row has no seams, so its
dFVD is what position alone does; a seam fix can only win the generated row's dFVD minus that. Every run after the first
also prints its dFVD minus the first run's, on the same bootstrap resamples of the shared clips (paired).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.eval.paired import clip_dir  # noqa: E402

WINDOW, SEG, PER = 64, 16, 3


def windows(z: dict) -> dict:
    """{window: (gt [3, D], gen [3, D])} of one clip's features, the first PER segments of each window."""
    k = WINDOW // SEG
    n = min(len(z["i3d_gt"]), len(z["i3d_gen"]))
    return {w: (z["i3d_gt"][w * k:w * k + PER], z["i3d_gen"][w * k:w * k + PER])
            for w in range(n // k + 1) if w * k + PER <= n}


def window_fvd(clips: list, n_boot: int = 200, seed: int = 0) -> dict:
    """clips: per clip {window: (gt, gen)} -> {window: (FVD, dFVD vs window 1, low, high, bootstrap dFVDs)} over the
    windows every clip has; the same seed and clip count give the same resamples, so runs pair index by index."""
    from sang.eval.metrics import frechet
    ws = sorted(set.intersection(*(set(c) for c in clips)))
    fvd = lambda w, idx: frechet(np.concatenate([clips[i][w][0] for i in idx]),
                                 np.concatenate([clips[i][w][1] for i in idx]))
    rng = np.random.default_rng(seed)
    boots = [rng.integers(0, len(clips), len(clips)) for _ in range(n_boot)]
    base = [fvd(ws[0], b) for b in boots]
    out = {}
    for w in ws:
        d = [fvd(w, b) - b0 for b, b0 in zip(boots, base)]
        out[w + 1] = (fvd(w, range(len(clips))), fvd(w, range(len(clips))) - fvd(ws[0], range(len(clips))),
                      *np.percentile(d, [2.5, 97.5]), np.array(d))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run or run:row (default row sang)")
    ap.add_argument("--root", default=str(REPO / "results" / "hdtf"))
    args = ap.parse_args()
    dirs = {r: clip_dir(r, Path(args.root)) for r in args.runs}
    for r in [r for r, d in dirs.items() if not any(d.glob("*.npz"))]:
        print(f"{r}: no features in {dirs.pop(r)}")
    if not dirs:
        return
    stems = sorted(set.intersection(*({p.stem for p in d.glob("*.npz")} for d in dirs.values())))
    ref = None
    for r, d in dirs.items():
        res = window_fvd([windows(np.load(d / f"{c}.npz")) for c in stems])
        print(f"{r:<24} n {len(stems)}  " + "  ".join(
            f"w{w} {f:.2f}" + ("" if w == 1 else f" ({df:+.2f} [{lo:+.2f}, {hi:+.2f}])")
            for w, (f, df, lo, hi, _) in res.items()))
        if ref is None:
            ref, name = res, r
            continue
        print(f"{'  minus ' + name:<24} " + "  ".join(
            f"w{w} {res[w][1] - ref[w][1]:+.2f} [{np.percentile(res[w][4] - ref[w][4], 2.5):+.2f}, "
            f"{np.percentile(res[w][4] - ref[w][4], 97.5):+.2f}]" for w in res if w > 1 and w in ref))


def self_check() -> None:
    """Windows 2+ of a run whose late segments are shifted score worse than window 1, an unshifted run does not."""
    rng = np.random.default_rng(0)
    clip = lambda shift: {"i3d_gt": rng.normal(size=(15, 8)),
                          "i3d_gen": rng.normal(size=(15, 8)) + np.r_[np.zeros(4), np.full(11, shift)][:, None]}
    bad = window_fvd([windows(clip(1.0)) for _ in range(40)], n_boot=20)
    same = window_fvd([windows(clip(0.0)) for _ in range(40)], n_boot=20)
    assert set(bad) == {1, 2, 3, 4} and min(bad[w][1] for w in (2, 3, 4)) > 4, bad
    assert max(abs(same[w][1]) for w in (2, 3, 4)) < 2, same
    assert min(np.percentile(bad[w][4] - same[w][4], 2.5) for w in (2, 3, 4)) > 0  # paired excess over a clean run
    print("scripts/eval/window_fvd.py self-check ok")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        self_check()
    else:
        main()
