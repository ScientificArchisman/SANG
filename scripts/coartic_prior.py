#!/usr/bin/env python3
"""C1 go/no-go (no training): does a Cohen-Massaro coarticulation prior predict the real lips beyond SANG's
mean prediction? Definitions and the reasoning behind the controls in sang/coartic.py. Motion space only,
~15 min on one GPU.

    sbatch bash_scripts/job.sh scripts/coartic_prior.py --ckpt runs/b1_lognorm03/last.pt \\
        --sync-offsets <cache>/sync_offsets.json

1. fit the prior on --train-clips training clips (real lips + phoneme spikes; nothing generated)
2. on the val clips that naturalness.py and error_split.py score: --samples SANG samples per clip (seeds
   --seed ..), their mean as SANG's prediction; partial(B | A) checks that the mean is pinned down;
   partial(prior | mean) is the gate statistic (95 % intervals over videos); held-out blends
3. the same by group: language, person overlap, speaking-rate third, real lip-movement third
GO if partial(prior | mean) >= --gate (0.1): the prior carries lip information SANG lacks, so C1 goes on
(the prior as an input feature, a sampling guide or a training target). NO-GO: SANG already captures
what the prior knows; go to C2 (lip-reading expert loss). INCONCLUSIVE: SANG's samples still disagree
too much (partial(B | A) >= --max-baseline); rerun with more --samples.
"""
import argparse
import importlib.util
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from sang.coartic import (CLASSES, blend_gain, cluster_ci, events, fit_prior, partial_corr, prior_trace, verdict,
                          video_of)
from sang.diagnose import corr
from sang.error_split import is_phone, speech_rate
from sang.motion_model import Norm, build, ema_weights, parse_spec, sampler_kwargs
from sang.naturalness import PhonemeRecognizer, load_readouts
from sang.paths import motion_cache
from sang.sync import clip_key

PARTIALS = {"partial_baseline": "half B | half A (noise check)", "partial_prior_half": "prior | half A",
            "partial_prior": "prior | SANG mean (the gate)"}
BLENDS = {"half B on half A": (("gen_a",), "gen_b"), "prior on half A": (("gen_a",), "prior"),
          "prior on SANG mean": (("gen",), "prior")}


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def summarise(recs: list[dict]) -> dict:
    """Val clip records (traces lip / gen / gen_a / gen_b / prior, scores r_* and partial_*) -> means,
    partial correlations with 95 % intervals over videos, and the held-out blends."""
    if len(recs) < 4:
        return {"clips": len(recs)}
    m = lambda k: float(np.nanmean([r[k] for r in recs]))
    vids = [video_of(r["clip"]) for r in recs]
    out = {"clips": len(recs), "videos": len(set(vids)), "r_sample": m("r_sample"), "r_sang": m("r_sang"),
           "r_prior": m("r_prior"), "r_prior_sang": m("r_prior_sang"),
           "prior_beats_sang_pct": float(100 * np.mean([r["r_prior"] > r["r_sang"] for r in recs]))}
    for k in PARTIALS:
        x = np.array([r[k] for r in recs], np.float64)
        mean, ci = cluster_ci(x, vids)
        out[k] = {"mean": mean, "ci": ci, "median": float(np.nanmedian(x)),
                  "pos_pct": float(100 * np.mean(x[np.isfinite(x)] > 0))}
    out["blend"] = {name: blend_gain(recs, base, other) for name, (base, other) in BLENDS.items()}
    return out


def groups_of_recs(recs: list[dict], group_names) -> dict:
    """Clip groups: language / person overlap, then thirds of speaking rate and of real lip movement."""
    out = {g: [r for r in recs if g in r["groups"]] for g in group_names}
    for key, names in (("rate", ("slow speech", "middle speech", "fast speech")),
                       ("sy", ("least lip movement", "middle lip movement", "most lip movement"))):
        v = np.array([r[key] for r in recs])
        e = np.nanquantile(v, [1 / 3, 2 / 3])
        for name, lo, hi in zip(names, (-np.inf, e[0], e[1]), (e[0], e[1], np.inf)):
            out[f"{name} third"] = [r for r in recs if lo <= r[key] < hi]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(REPO / "runs/b1_lognorm03/last.pt"))
    ap.add_argument("--sampler", default="g=2,mouth=1.25,tau=0.5,avg=4,steps=6")
    ap.add_argument("--sync-offsets", default=None, help="<cache>/sync_offsets.json (as for naturalness.py)")
    ap.add_argument("--n", type=int, default=300, help="val clips")
    ap.add_argument("--train-clips", type=int, default=600, help="training clips the prior is fitted on")
    ap.add_argument("--samples", type=int, default=8, help="SANG samples per val clip (even); their mean is SANG's prediction")
    ap.add_argument("--max-frames", type=int, default=250)
    ap.add_argument("--min-frames", type=int, default=125)
    ap.add_argument("--gate", type=float, default=0.1, help="GO if partial(prior | SANG mean) reaches this")
    ap.add_argument("--max-baseline", type=float, default=0.05,
                    help="INCONCLUSIVE if partial(half B | half A) reaches this: SANG's mean is not pinned down")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="")
    ap.add_argument("--out", default=str(REPO / "results/coartic"))
    args = ap.parse_args()
    if args.samples < 2 or args.samples % 2:
        raise SystemExit("--samples must be even and >= 2")

    es = load_script("error_split")
    dev = "cuda"
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    cfg = ck["cfg"]
    model = build(cfg).to(dev).eval()
    model.load_state_dict(ema_weights(ck, None))
    norm = Norm(**ck["norm"]).to(dev)
    samp = sampler_kwargs(parse_spec(args.sampler), cfg, None)
    cache = motion_cache(cfg["cache_dir"])
    read = load_readouts(cache / "openness.json")
    train, val, so, groups_of = es.aligned_splits(cfg, cache, args.sync_offsets, None, args.min_frames)
    val = val[: args.n]
    train = random.Random(args.seed).sample(train, min(args.train_clips, len(train)))
    print(f"{len(train)} training clips for the prior, {len(val)} val clips x {args.samples} SANG samples; "
          f"{args.ckpt} (step {ck['step']}), sampler {args.sampler}", flush=True)
    phon = PhonemeRecognizer(dev)
    skipped = sorted(t for t in phon.tokens.values() if not is_phone(t))
    print(f"phoneme vocabulary: {sum(map(is_phone, phon.tokens.values()))} phonemes; not phonemes: {skipped}", flush=True)
    t0 = time.time()

    # ---------------- 1. fit the prior on training clips (no generation)
    fit_clips = []
    for i, r in enumerate(train, 1):
        try:
            c = es.load_clip(r, so, read, phon, args.max_frames)
            if c["lip"].std() > 1e-6 and c["spikes"]:
                fit_clips.append({"lip": c["lip"], "events": events(c["spikes"], c["pause"])})
        except Exception as e:                                   # one bad clip never kills the run
            print(f"  skip {Path(r['clip']).name}: {type(e).__name__}: {e}", flush=True)
        if i % 100 == 0:
            print(f"  train {i}/{len(train)} clips, {time.time() - t0:.0f} s", flush=True)
    print(f"\n1. FIT on {len(fit_clips)} training clips", flush=True)
    fit = fit_prior(fit_clips, log=lambda s: print(s, flush=True))
    tg = fit["targets"]
    print(f"   lag {fit['lag']:+d} frames (> 0: the lips follow the phoneme spike); "
          f"train corr(prior, real) {fit['train_corr']:.3f}")
    print(f"   {'class':<18}{'target (z)':>11}{'alpha':>8}{'theta':>8}{'reach (frames)':>16}")
    for k in CLASSES:
        print(f"   {k:<18}{tg.get(k, float('nan')):>+11.2f}{fit['alpha'][k]:>8.2f}{fit['theta'][k]:>8.2f}"
              f"{1 / fit['theta'][k]:>16.1f}")
    toks = sorted((t for t in tg if t not in CLASSES and t != "<pause>"), key=lambda t: tg[t])
    print(f"   most closed: {', '.join(f'{t} {tg[t]:+.2f}' for t in toks[:6])}")
    print(f"   most open:   {', '.join(f'{t} {tg[t]:+.2f}' for t in toks[-6:])}")

    # ---------------- 2. val: SANG's mean vs the prior
    recs, h = [], args.samples // 2
    for i, r in enumerate(val, 1):
        try:
            c = es.load_clip(r, so, read, phon, args.max_frames)
            gs = [es.generate_lips(model, norm, cfg, read, c, 0, samp, args.seed + j, dev) for j in range(args.samples)]
            g, ga, gb = np.mean(gs, 0), np.mean(gs[:h], 0), np.mean(gs[h:], 0)
            p = prior_trace(events(c["spikes"], c["pause"]), c["n"], tg, fit["alpha"], fit["theta"], fit["lag"])
            real = c["lip"]
            if min(real.std(), ga.std(), gb.std(), p.std()) < 1e-6:
                continue
            recs.append({"clip": clip_key(r), "groups": sorted(groups_of(r)), "lip": real, "gen": g, "gen_a": ga,
                         "gen_b": gb, "prior": p, "r_sample": corr(gs[0], real), "r_sang": corr(g, real),
                         "r_prior": corr(p, real), "r_prior_sang": corr(p, g),
                         "partial_baseline": partial_corr(real, gb, ga), "partial_prior_half": partial_corr(real, p, ga),
                         "partial_prior": partial_corr(real, p, g),
                         "rate": speech_rate(c["spikes"], c["pause"], c["n"]), "sy": float(real.std())})
        except Exception as e:
            print(f"  skip {Path(r['clip']).name}: {type(e).__name__}: {e}", flush=True)
        if i % 50 == 0:
            print(f"  val {i}/{len(val)} clips, {time.time() - t0:.0f} s", flush=True)

    groups = {k: summarise(v) for k, v in groups_of_recs(recs, load_script("naturalness").GROUPS).items()}
    a = groups["all"]
    v = verdict(a["partial_prior"]["mean"], a["partial_baseline"]["mean"], args.gate, args.max_baseline)
    print(f"\n2. VAL ({a['clips']} clips, {a['videos']} videos, {args.samples} SANG samples each)")
    print(f"   lip corr with the real lips: one SANG sample {a['r_sample']:.3f}, SANG mean {a['r_sang']:.3f}, prior "
          f"{a['r_prior']:.3f} (prior beats the SANG mean on {a['prior_beats_sang_pct']:.0f} % of clips); "
          f"prior vs SANG mean {a['r_prior_sang']:.3f}")
    print("   partial correlation with the real lips (mean ± 95 % interval over videos; median; % of clips > 0):")
    for k, label in PARTIALS.items():
        x = a[k]
        print(f"     {label:<32}{x['mean']:+.3f} ± {x['ci']:.3f}   median {x['median']:+.3f}   {x['pos_pct']:.0f} %")
    print("   held-out blends z(base) + w z(other), w fitted on the other half of the videos:")
    for name, b in a["blend"].items():
        if b:
            print(f"     {name:<20} lip corr {b['base']:.3f} -> {b['blend']:.3f} ({b['gain']:+.3f}; better on "
                  f"{b['better_clips_pct']:.0f} % of clips; w {b['weights']})")
    print(f"   VERDICT: {v}  (noise check partial(B | A) {a['partial_baseline']['mean']:+.3f} vs < {args.max_baseline}; "
          f"partial(prior | SANG mean) {a['partial_prior']['mean']:+.3f} vs >= {args.gate})")

    print("\n3. BY GROUP (partials: means; gain: held-out blend 'prior on SANG mean')")
    w = max(len(k) for k in groups) + 2
    print(f"   {'group':<{w}}{'clips':>6}{'SANG':>7}{'prior':>7}{'B|A':>8}{'p|A':>8}{'p|mean':>8}{'gain':>8}")
    for k, x in groups.items():
        if x["clips"] >= 4:
            gain = x["blend"]["prior on SANG mean"].get("gain", float("nan"))
            print(f"   {k:<{w}}{x['clips']:>6}{x['r_sang']:>7.3f}{x['r_prior']:>7.3f}{x['partial_baseline']['mean']:>+8.3f}"
                  f"{x['partial_prior_half']['mean']:>+8.3f}{x['partial_prior']['mean']:>+8.3f}{gain:>+8.3f}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    name = f"{Path(args.ckpt).parent.name}_step{ck['step']}" + (f"_{args.name}" if args.name else "") + ".json"
    keep = ("clip", "groups", "r_sample", "r_sang", "r_prior", "r_prior_sang", *PARTIALS, "rate", "sy")
    (out / name).write_text(json.dumps({"ckpt": args.ckpt, "step": ck["step"], "sampler": args.sampler,
                                        "samples": args.samples, "verdict": v, "gate": args.gate,
                                        "max_baseline": args.max_baseline, "fit": fit, "groups": groups,
                                        "clips": [{k: r[k] for k in keep} for r in recs]}, indent=1, default=float))
    print(f"\nwrote {out / name} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
