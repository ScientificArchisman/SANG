#!/usr/bin/env python3
"""Fit the gaze-head automaton (sang.model.automaton; reports/SANG neurosymbolic jitter and gaze.md) on real TalkVid
motion through the generator's latent, with its ablations, and the same family on SANG's own samples.

    sbatch bash_scripts/run.sh scripts/fit_automaton.py --gen runs/mbes/round4.pt --out runs/automaton

Writes <out>/gha.pt (the full automaton), gha_1state.pt (one random-walk state per chain: a learned Kalman smoother)
and gha_novor.pt (G = 0). Each is {"real": automaton, "sang": automaton fitted on SANG's shielded samples or None,
"stats": its G0 numbers}; only gha.pt carries "sang" (the particle ratio). Windows are the training loader's
(training speakers, prefix + window frames, relative to the photo); GATE G1 uses the unseen val speakers.
GATE G0  fit sanity: median fixation >= 3 frames, VOR share >= 0.2 (full model).
GATE G1  the shield leaves real motion alone: head range within 2 %, >= 95 % of head beats and of the deepest lid
         dips survive, RMS change <= 1.5 x the learned noise sd.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from sang.eval.naturalness import head_beats
from sang.model.automaton import COLS, Shield, fit
from sang.model.flow import build, sample
from train_motion import setup_data

VARIANTS = (("gha", True, True), ("gha_1state", False, True), ("gha_novor", True, False))


def windows(d, ds, n: int, args, dev, fn, shuffle: bool) -> torch.Tensor:
    """Up to n windows of a dataset, each batch through fn(encoded batch) -> wide [B, P + L, 69], on the CPU."""
    out, got = [], 0
    for b in DataLoader(ds, batch_size=args.batch, shuffle=shuffle, num_workers=args.workers):
        w = fn(d.encode({k: v.to(dev) for k, v in b.items()}))
        out.append(w.float().cpu())
        got += len(w)
        if got >= n:
            break
    return torch.cat(out)[:n]


def gate_g0(name: str, st: dict, full: bool) -> None:
    ok = st["median_fix"] >= 3 and (st["vor_share"] >= 0.2 or not full)
    print(f"GATE G0 {name}: median fixation {st['median_fix']:.0f} frames (>= 3), VOR share {st['vor_share']:.2f}"
          f"{' (>= 0.2)' if full else ''}, saccades/min {st['sacc_per_min']:.0f}, noise sd head "
          f"{np.round(st['r_head'], 4).tolist()} eye {np.round(st['r_eye'], 5).tolist()} -> {'PASS' if ok else 'FAIL'}",
          flush=True)


def gate_g1(name: str, gha, w: torch.Tensor) -> None:
    """The shield's change on unseen real windows w [N, T, 69]."""
    o = w[..., COLS]
    dl = gha.correct(w)
    c = o + dl
    rng = lambda x: (x.amax(1) - x.amin(1)).sum(0)
    head_range = (rng(c[..., :3]) / rng(o[..., :3])).tolist()
    hit = tot = 0
    for a, b in zip(o[..., :3].cpu().numpy(), c[..., :3].cpu().numpy()):
        h0, h1 = head_beats(a), head_beats(b)
        tot += len(h0)
        hit += sum(bool(len(h1)) and int(np.abs(h1 - t).min()) <= 1 for t in h0)
    beats = hit / max(tot, 1)
    dip = lambda x: x.median(1).values - x.amin(1)
    d0, d1 = dip(o[..., 7]), dip(c[..., 7])
    top = d0 >= d0.quantile(0.9)
    lids = float(d1[top].sum() / d0[top].sum())
    rms = lambda x: x.pow(2).mean((0, 1)).sqrt()
    rh = (rms(dl[..., :3]) / gha.head.r.sqrt()).tolist()
    rg = (rms(dl[..., 3:5]) / gha.eye.r[:2].sqrt()).tolist()
    jit = lambda x: x.diff(2, dim=1).pow(2).mean()
    jh, jg = float(jit(c[..., :3]) / jit(o[..., :3])), float(jit(c[..., 3:5]) / jit(o[..., 3:5]))
    ok = (all(abs(r - 1) <= 0.02 for r in head_range) and beats >= 0.95 and lids >= 0.95
          and max(rh + rg) <= 1.5)
    f = lambda v: np.round(v, 3).tolist()
    print(f"GATE G1 {name}: head range x{f(head_range)} (within 2 %), head beats kept {beats:.3f}, lid dips kept "
          f"{lids:.3f} (>= 0.95), rms change / noise sd head {f(rh)} gaze {f(rg)} (<= 1.5) | second-difference power "
          f"x head {jh:.3f} gaze {jg:.3f} -> {'PASS' if ok else 'FAIL'}", flush=True)


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", default=str(REPO / "runs/mbes/round4.pt"), help="generator checkpoint (latent, data cfg)")
    ap.add_argument("--out", default=str(REPO / "runs/automaton"))
    ap.add_argument("--n-train", type=int, default=8000, help="real training windows for the fit")
    ap.add_argument("--n-val", type=int, default=1000, help="real val-speaker windows for GATE G1")
    ap.add_argument("--sang-windows", type=int, default=2000, help="SANG samples for the particle ratio's grammar")
    ap.add_argument("--no-sang", action="store_true", help="skip SANG's grammar (no particle ratio)")
    ap.add_argument("--rounds", type=int, default=8, help="hard-EM rounds")
    ap.add_argument("--shield-steps", type=int, default=2, help="shield steps of the sampler SANG's grammar is fit on")
    ap.add_argument("--g", type=float, default=None, help="guidance (default: the run's cfg_audio)")
    ap.add_argument("--steps", type=int, default=None, help="Euler steps (default: the run's sample_steps)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ck = torch.load(args.gen, map_location="cpu", weights_only=False)
    cfg = {**ck["cfg"], "preload": False}
    d = setup_data(cfg, dev)
    real = lambda b: d.decode(torch.cat([b["prefix"], b["target"]], 1))
    t0 = time.time()
    w_tr = windows(d, d.train_ds, args.n_train, args, dev, real, shuffle=True)
    w_va = windows(d, d.val_ds, args.n_val, args, dev, real, shuffle=False)
    print(f"{len(w_tr)} train / {len(w_va)} val windows of {w_tr.shape[1]} frames ({time.time() - t0:.0f} s)",
          flush=True)
    fits = {}
    for name, states, vor in VARIANTS:
        t0 = time.time()
        print(f"fit {name} (states {states}, VOR {vor})", flush=True)
        gha, st = fit(w_tr.to(dev), states=states, vor=vor, rounds=args.rounds)
        print(f"  {time.time() - t0:.0f} s, G {np.round(st['G'], 5).tolist()}", flush=True)
        gate_g0(name, st, states and vor)
        gate_g1(name, gha, w_va.to(dev))
        fits[name] = (gha, st)
    sang = None
    if not args.no_sang:
        model = build(ck["cfg"]).to(dev).eval()
        model.load_state_dict(ck["ema"])
        g = float(args.g or ck["cfg"]["cfg_audio"])
        steps = int(args.steps or ck["cfg"]["sample_steps"])
        shield = Shield(fits["gha"][0], d.latent, d.k, n=args.shield_steps)
        gen = torch.Generator(device=dev).manual_seed(args.seed)

        def fake(b):
            demo = model.encode_demo(b)
            keep = b["demo_keep"].bool() if demo is not None else None
            y = sample(model, b["audio"], b["ref"], b["target"].shape[1], prefix=b["prefix"], steps=steps,
                       cfg_audio=g, generator=gen, demo=demo, demo_keep=keep, shield=shield)
            return d.decode(torch.cat([b["prefix"], y], 1))

        t0 = time.time()
        w_sang = windows(d, d.train_ds, args.sang_windows, args, dev, fake, shuffle=True)
        print(f"{len(w_sang)} SANG windows (g {g:g}, {steps} steps, shield {args.shield_steps}; "
              f"{time.time() - t0:.0f} s); fit its grammar", flush=True)
        sang, st = fit(w_sang.to(dev), rounds=args.rounds)
        gate_g0("sang", st, False)
    for name, (gha, st) in fits.items():
        p = out / f"{name}.pt"
        torch.save({"real": gha.state(), "sang": sang.state() if sang is not None and name == "gha" else None,
                    "stats": st}, p)
        print(f"saved {p}", flush=True)


if __name__ == "__main__":
    main()
