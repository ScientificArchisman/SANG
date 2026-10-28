#!/usr/bin/env python3
"""Training and validation loss of a training run on one plot, read from its log.

    python scripts/plot_curves.py slurm_logs/sang_motion_176962.out --name b1_lognorm03
    python scripts/plot_curves.py results/train/train_motion_<job>.txt --name <run> [--batches]

Every curve is the flow-matching loss including the velocity term (scripts/train_motion.py); lower is better.
  validation        EVAL lines: the EMA model on fixed windows of the val clips (unseen speakers), uniform
                    t, no condition dropout, the same t and noise at every eval
  training          EVAL_TRAIN lines (train_eval: true): the same measurement on as many fixed TRAINING
                    clips, so the gap between the two curves is generalisation to new people
  training batches  'step' lines: the loss the optimiser saw on one batch every log_every steps (raw
                    weights, the training t distribution, condition dropout). A different measurement from
                    validation, so its gap to validation is not a generalisation gap. Drawn with a running
                    mean when the log has no EVAL_TRAIN lines, or with --batches
Writes <out>/<name>.png and .pdf and prints the evaluation table.
"""
import argparse
import re
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def pairs(tokens: list[str]) -> dict:
    """'k1 v1 k2 v2 ...' -> {k: float}, stopping at the first value that is not a number."""
    out = {}
    for k, v in zip(tokens[::2], tokens[1::2]):
        try:
            out[k] = float(v)
        except ValueError:
            break
    return out


def parse(text: str) -> dict:
    """Training log -> {'batches': [(step, loss)], 'val': [(step, {..})], 'train': [(step, {..})], 'best': line}."""
    res = {"batches": [], "val": [], "train": [], "best": None}
    for line in text.splitlines():
        if m := re.match(r"step\s+(\d+) lr \S+ (.*?) gnorm", line):
            res["batches"].append((int(m[1]), pairs(m[2].split())["loss"]))
        elif m := re.match(r"EVAL step (\d+) (.*)", line):
            res["val"].append((int(m[1]), pairs(m[2].split())))
        elif m := re.match(r"EVAL_TRAIN step (\d+) (.*)", line):
            res["train"].append((int(m[1]), pairs(m[2].split())))
        elif line.startswith("done. best val_loss"):
            res["best"] = line
    return res


def running_mean(y: np.ndarray, w: int) -> np.ndarray:
    """Centred mean over w points (fewer at the ends), so the curve does not lag behind the steps."""
    h = w // 2
    return np.array([y[max(0, i - h): i + h + 1].mean() for i in range(len(y))])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", help="training log: slurm_logs/sang_motion_<job>.out or results/train/*.txt")
    ap.add_argument("--name", default=None, help="run name for the title and file (default: the log's name)")
    ap.add_argument("--batches", action="store_true", help="also draw the per-batch training loss")
    ap.add_argument("--window", type=int, default=10, help="running mean over this many logged batches")
    ap.add_argument("--out", default=str(REPO / "results/curves"))
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r = parse(Path(args.log).read_text(errors="replace"))
    if not r["val"]:
        raise SystemExit(f"{args.log}: no EVAL lines")
    name = args.name or Path(args.log).stem
    vs = np.array([s for s, _ in r["val"]])
    vl = np.array([d["val_loss"] for _, d in r["val"]])
    tr = dict((s, d["loss"]) for s, d in r["train"])
    i_best = int(vl.argmin())

    print(f"{name}: {len(r['batches'])} logged batches, {len(vs)} evaluations"
          + ("" if tr else " (no EVAL_TRAIN lines: training curve = per-batch optimiser loss)"))
    print(f"{'step':>7} {'val':>8} {'train':>8} {'gap':>8}")
    for s, v in zip(vs, vl):
        t = tr.get(int(s))
        print(f"{s:>7} {v:>8.4f} " + (f"{t:>8.4f} {v - t:>8.4f}" if t is not None else f"{'':>8} {'':>8}")
              + ("   <- best (best.pt)" if s == vs[i_best] else ""))

    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=150)
    ys = list(vl) + list(tr.values())
    if r["batches"] and (args.batches or not tr):
        bs = np.array([s for s, _ in r["batches"]])
        bl = np.array([l for _, l in r["batches"]])
        ax.plot(bs / 1e3, bl, color="#9ecae1", lw=0.6, alpha=0.7)
        rm = running_mean(bl, args.window)
        ax.plot(bs / 1e3, rm, color="#3182bd", lw=1.6,
                label=f"training batches (optimiser loss, centred mean of {args.window * (bs[1] - bs[0]) if len(bs) > 1 else 0:,} steps)")
        ys += list(rm[bs >= vs[0]])
    if tr:
        ts = np.array(sorted(tr))
        ax.plot(ts / 1e3, [tr[s] for s in ts], "o-", color="#08519c", lw=1.8, ms=4,
                label="training (fixed train clips, scored like validation)")
    ax.plot(vs / 1e3, vl, "o-", color="#d94801", lw=1.8, ms=4, label="validation (unseen speakers)")
    ax.axvline(vs[i_best] / 1e3, color="grey", ls=":", lw=1)
    at_end = vs[i_best] > 0.85 * vs[-1]
    ax.annotate(f"best val {vl[i_best]:.3f}\n(step {vs[i_best]:,})", (vs[i_best] / 1e3, vl[i_best]),
                xytext=(-6 if at_end else 0, 18), textcoords="offset points", ha="right" if at_end else "center",
                fontsize=8, color="grey")
    lo, hi = min(ys), max(ys)
    ax.set_ylim(lo - 0.08 * (hi - lo), hi + 0.08 * (hi - lo))
    ax.set_xlim(0, vs[-1] / 1e3 * 1.02)
    ax.set_xlabel("training step (thousands)")
    ax.set_ylabel("flow-matching loss")
    ax.set_title(f"{name}: training and validation loss", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, frameon=False)
    if not tr:
        fig.text(0.01, 0.01, "Training curve = loss of each training batch (training noise levels, condition dropout, "
                             "raw weights); validation = EMA weights, uniform noise levels, no dropout.\n"
                             "The gap between them is partly the different measurement, not only overfitting.",
                 fontsize=6.5, color="grey", va="bottom")
        fig.subplots_adjust(bottom=0.2)
    fig.tight_layout(rect=(0, 0.06 if not tr else 0, 1, 1))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight")
    print(f"wrote {out / name}.png and .pdf")


if __name__ == "__main__":
    main()

