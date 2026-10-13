#!/usr/bin/env python3
"""C0: split the remaining lip-sync error of one checkpoint, no training. Motion space only (no
rendering): every val clip is generated once with the given sampler and its lip opening compared with
the real clip's, on the same aligned val clips as scripts/naturalness.py.

    sbatch bash_scripts/job.sh scripts/error_split.py --ckpt runs/b1_lognorm03/last.pt \\
        --sync-offsets <cache>/sync_offsets.json

Reports (definitions in sang/error_split.py):
  1. the error split into bias / amplitude / timing (a constant shift fixes it) / shape, and the lags
  2. by sound: closures p/b/m, f/v, open vowels, rounded vowels, other vowels, other consonants, pauses
  3. fast transitions: generated vs real lip speed where the real lips move fastest
  4. by group: all, English, non-English, no person overlap; and by speaking-rate third
Reading: timing large -> data alignment (C3 audit); shape concentrated in fast transitions / sluggish
lips -> coarticulation (C1) or a lip-reading loss (C2); one language far worse -> C1 / C3.
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from sang.error_split import (PARTS, SOUNDS, frame_errors, frame_sounds, phone_spikes, pool_sounds, speech_rate,
                              split_clip, summarise)
from sang.motion import to_target
from sang.motion_model import Norm, build, ema_weights, parse_spec, sampler_kwargs
from sang.naturalness import PhonemeRecognizer, energy_db, guided_generate, load_readouts, load_wav, pauses
from sang.paths import motion_cache
from sang.sync import SyncOffsets, clip_key, shift_ticks, shift_wav
from sang.talkvid import clip_languages, filter_langs, parse_langs

DEFAULT_SAMPLER = "g=2,mouth=1.25,tau=0.5,avg=4,steps=6"


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(REPO / "runs/b1_lognorm03/last.pt"))
    ap.add_argument("--sampler", default=DEFAULT_SAMPLER)
    ap.add_argument("--sync-offsets", default=None, help="<cache>/sync_offsets.json (as for naturalness.py)")
    ap.add_argument("--lang", nargs="+", default=["all"], help="val languages (default all: the split reports both)")
    ap.add_argument("--n", type=int, default=300, help="val clips")
    ap.add_argument("--max-frames", type=int, default=250)
    ap.add_argument("--min-frames", type=int, default=125)
    ap.add_argument("--max-lag", type=int, default=4, help="frames; the clip's best constant shift is searched in +/- this")
    ap.add_argument("--reach", type=int, default=3, help="frames from a phoneme spike that still carry its label")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="")
    ap.add_argument("--out", default=str(REPO / "results/error_split"))
    args = ap.parse_args()

    dev = "cuda"
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    cfg = ck["cfg"]
    model = build(cfg).to(dev).eval()
    model.load_state_dict(ema_weights(ck, None))
    norm = Norm(**ck["norm"]).to(dev)
    samp = sampler_kwargs(parse_spec(args.sampler), cfg, None)
    cache = motion_cache(cfg["cache_dir"])
    read = load_readouts(cache / "openness.json")

    # the same val clips as naturalness.py
    tm, nat = load_script("train_motion"), load_script("naturalness")
    train_rows, val = tm.split_by_speaker(tm.load_index(cache), cfg["val_frac"], cfg["seed"])
    lang_of = clip_languages(cache)
    meta = cache / "talkvid_meta.json"
    person_of = {k: v.get("person") for k, v in json.loads(meta.read_text()).items()} if meta.exists() else {}
    train_rows, _ = filter_langs(train_rows, parse_langs(cfg.get("lang")), lang_of)
    train_persons = {person_of.get(clip_key(r)) for r in train_rows} - {None}
    langs = parse_langs(args.lang)
    if langs is not None:
        val, _ = filter_langs(val, langs, lang_of)
    so = SyncOffsets.load(args.sync_offsets) if args.sync_offsets else None
    if so is not None:
        val, why = so.filter(val)
        print(f"sync offsets {args.sync_offsets}: dropped {why}", flush=True)
    val = [r for r in sorted(val, key=lambda r: r["clip"]) if r["n"] >= args.min_frames][: args.n]
    print(f"{len(val)} val clips from {len({Path(r['path']).parent.name for r in val})} videos; "
          f"{args.ckpt} (step {ck['step']}), sampler {args.sampler}", flush=True)

    phon = PhonemeRecognizer(dev)
    recs, fes, t0 = [], [], time.time()
    for i, r in enumerate(val, 1):
        try:
            d = torch.load(r["path"], map_location="cpu", weights_only=True)
            n = min(args.max_frames, int(d["n"]))
            s = so.shift(r) if so is not None else 0
            wav = shift_wav(load_wav(r["clip"], r.get("start", 0), n), s)
            audio = shift_ticks(d["audio"][: 2 * n].float(), s)
            y_real = to_target(d["m"][:n].float())
            ref = norm.ref(d["kp"][:1].float().to(dev), y_real[:1].to(dev))
            y, _ = guided_generate(model, audio[None].to(dev), ref, n, cfg, norm, wav, None, "none", args.seed, sampler=samp)
            g = read["lip"](norm.untarget(y[0]).cpu()).numpy()
            real = read["lip"](y_real).numpy()
            pause = pauses(energy_db(wav, n))
            spikes = phone_spikes(phon.posteriors(wav).numpy(), phon.tokens, n)
            rec = split_clip(g, real, args.max_lag)
            if not rec:
                continue
            rec.update(clip=clip_key(r), rate=speech_rate(spikes, pause, n),
                       groups=sorted(nat.clip_groups(r, lang_of, person_of, train_persons)))
            recs.append(rec)
            fes.append(frame_errors(g, real, frame_sounds(spikes, pause, n, args.reach)))
        except Exception as e:                                   # one bad clip never kills the run
            print(f"  skip {Path(r['clip']).name}: {type(e).__name__}: {e}", flush=True)
        if i % 50 == 0:
            print(f"  {i}/{len(val)} clips, {time.time() - t0:.0f} s", flush=True)

    allc = summarise(recs)
    gap = allc["gap"]
    print(f"\n1. ERROR SPLIT ({allc['clips']} clips)")
    print(f"   lip corr {allc['r']:.3f} -> {allc['r_best']:.3f} at each clip's best shift (|lag| <= {args.max_lag}); "
          f"CCC {allc['ccc']:.3f}; amplitude {allc['std_r']:.3f} (median {allc['std_r_median']:.3f}); "
          f"slope {allc['slope']:.3f}; bias {allc['bias']:+.3f} real std (median |bias| {allc['abs_bias_median']:.3f})")
    print(f"   correlation gap {gap['total']:.3f} = timing {gap['timing']:.3f} ({100 * gap['timing'] / gap['total']:.0f} %) "
          f"+ shape {gap['shape']:.3f} ({100 * gap['shape'] / gap['total']:.0f} %)")
    print(f"   amplitude < 0.8 in {allc['std_r_low_pct']:.0f} % of clips, > 1.25 in {allc['std_r_high_pct']:.0f} %; "
          f"|bias| > 1 real std in {allc['abs_bias_gt1_pct']:.0f} %")
    print(f"   {'part':<10}{'pooled':>9}{'per clip':>10}   (pooled = share of all squared error in lip-opening units; "
          f"per clip = mean of per-clip shares, inflated by clips whose real lips barely move)")
    for p in PARTS:
        print(f"   {p:<10}{allc['pooled_pct'][p]:>8.1f}%{allc['parts_pct'][p]:>9.1f}%")
    print(f"   best lag per clip (> 0 = generated later): {allc['lag_hist']}; |lag| >= 2 in {allc['abs_lag_ge2_pct']:.0f} % of clips")

    snd = pool_sounds(fes, allc["sy_mean"])
    print("\n2. BY SOUND (frames labelled by the nearest phoneme; signed error in pooled real std, > 0 = lips too open)")
    print(f"   {'sound':<18}{'% frames':>10}{'% error':>10}{'ratio':>8}{'signed':>9}")
    for c in SOUNDS:
        if c in snd:
            v = snd[c]
            print(f"   {c:<18}{v['frames_pct']:>10.1f}{v['error_pct']:>10.1f}{v['ratio']:>8.2f}{v['signed']:>+9.3f}")

    print(f"\n3. LIP SPEED, generated / real (median over clips): at each trace's median speed {allc['speed_q50_r']:.3f}, "
          f"at its 90th percentile {allc['speed_q90_r']:.3f}; mean speed {allc['speed_r']:.3f}. On the real lips' fastest "
          f"20 % of frames {allc['speed_r_fast']:.3f} (biased low for any imperfect model: compare models, not against 1)")

    groups = {g: summarise([r for r in recs if g in r["groups"]]) for g in nat.GROUPS}
    rates = np.array([r["rate"] for r in recs])
    edges = np.nanquantile(rates, [1 / 3, 2 / 3])
    for name, lo, hi in (("slow third", -np.inf, edges[0]), ("middle third", edges[0], edges[1]), ("fast third", edges[1], np.inf)):
        sel = [r for r in recs if lo <= r["rate"] < hi]
        if sel:
            groups[f"{name} ({min(r['rate'] for r in sel):.1f}-{max(r['rate'] for r in sel):.1f} ph/s)"] = summarise(sel)
    stds = np.array([r["sy"] for r in recs])
    edges = np.quantile(stds, [1 / 3, 2 / 3])
    for name, lo, hi in (("least lip movement", -np.inf, edges[0]), ("middle lip movement", edges[0], edges[1]),
                         ("most lip movement", edges[1], np.inf)):
        groups[f"{name} third"] = summarise([r for r in recs if lo <= r["sy"] < hi])
    print("\n4. BY GROUP (parts: pooled shares)")
    w = max(len(k) for k in groups) + 2
    print(f"   {'group':<{w}}{'clips':>6}{'corr':>7}{'best':>7}{'CCC':>7}{'amp':>7}{'amp50':>7}{'v90':>7}" + "".join(f"{p:>10}" for p in PARTS))
    for k, v in groups.items():
        if v["clips"]:
            print(f"   {k:<{w}}{v['clips']:>6}{v['r']:>7.3f}{v['r_best']:>7.3f}{v['ccc']:>7.3f}{v['std_r']:>7.3f}"
                  f"{v['std_r_median']:>7.3f}{v['speed_q90_r']:>7.3f}" + "".join(f"{v['pooled_pct'][p]:>9.1f}%" for p in PARTS))
    print("   amp = mean amplitude ratio, amp50 = its median, v90 = 90th-percentile lip speed ratio (median over clips)")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    name = f"{Path(args.ckpt).parent.name}_step{ck['step']}" + (f"_{args.name}" if args.name else "") + ".json"
    (out / name).write_text(json.dumps({"ckpt": args.ckpt, "step": ck["step"], "sampler": args.sampler,
                                        "summary": allc, "sounds": snd, "groups": groups, "clips": recs},
                                       indent=1, default=float))
    print(f"\nwrote {out / name} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
