#!/usr/bin/env python3
"""Naturalness metrics, REAL vs GENERATED, on unseen validation speakers. Motion space only: no rendering,
so ~300 clips take minutes. This is where the TalkVid lip-sync numbers (lip corr / CCC) come from.

    sbatch bash_scripts/run.sh scripts/naturalness.py --ckpt runs/l42_gft/last.pt \\
        --sync-offsets <cache>/sync_offsets.json --variants g=2,steps=6 --name l42_gft_plain

Needs <cache>/openness.json from scripts/calibrate_openness.py. Every variant (a sampler spec) generates every
clip once with the same seed, so the columns differ only by the sampler.

Metrics (real is the reference column; aim to match it, not to maximise):
    closure_depth    median lowest lip openness around /p b m/           lower = more closed
    closure_viol     share of /p b m/ with the lips open (threshold = real's 90th pct, so real = 0.10)
    lip_corr         per-clip correlation of lip openness with the real clip  (lip-sync proxy)
    lip_std_r        lip openness std / real's                              ~1
    lip_ccc          concordance with the real lip trajectory (timing AND amplitude; 1 = identical)
    blinks_per_min, ibi_median_s, ibi_cv, blink_ms, blink_at_pause (vs pause_cover = chance)
    beat_align       head-stroke / loudness-onset alignment (Bailando-style), vs beat_chance
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from sang.data.audio import load_wav
from sang.data.dataset import split_by_speaker
from sang.model.flow import Norm, build, generate, parse_spec, sampler_kwargs
from sang.model.latent import MotionLatent
from sang.model.target import to_target
from sang.eval.naturalness import (FPS, PhonemeRecognizer, audio_onsets, beat_alignment, blink_events, closure_minima,
                                   closure_offset, energy_db, head_beats, load_readouts, pause_near, pauses)
from sang.data.paths import load_index
from sang.data.sync import SyncOffsets, shift_ticks, shift_wav


def measure(y_raw: torch.Tensor, c: dict, read: dict, st: dict, real=None) -> dict:
    lip = read["lip"](y_raw).cpu().numpy()
    eye = read["eye"](y_raw).cpu().numpy()
    near = pause_near(c["pause"])
    bl = blink_events(eye)
    hb, ab = head_beats(y_raw[:, :3].cpu().numpy()), audio_onsets(c["db"])
    bas, chance = beat_alignment(hb, ab, len(lip))
    out = {"n": len(lip), "lip": lip, "eye": eye,
           "minima": closure_minima(lip, c["events"], st["closure_offset"]) if st else np.array([]),
           "blinks": bl, "blink_at_pause": [bool(near[b]) for _, b, _ in bl],
           "pause_cover": float(near.mean()), "bas": bas, "chance": chance}
    if real is not None:
        out["lip_corr"] = float(np.corrcoef(lip, real["lip"])[0, 1]) if lip.std() > 1e-8 else float("nan")
        out["lip_std_r"] = float(lip.std() / max(real["lip"].std(), 1e-8))
        out["eye_std_r"] = float(eye.std() / max(real["eye"].std(), 1e-8))
        cov = float(np.mean((lip - lip.mean()) * (real["lip"] - real["lip"].mean())))
        out["lip_ccc"] = 2 * cov / (lip.var() + real["lip"].var() + (lip.mean() - real["lip"].mean()) ** 2 + 1e-12)
    return out


def summarise(ms: list[dict], st: dict) -> dict:
    minutes = sum(m["n"] for m in ms) / FPS / 60
    ibis = [(b2[1] - b1[1]) / FPS for m in ms for b1, b2 in zip(m["blinks"], m["blinks"][1:])]
    durs = [(e - s) * 1000 / FPS for m in ms for s, _, e in m["blinks"]]
    mins = np.concatenate([m["minima"] for m in ms]) if ms else np.array([])
    at = [a for m in ms for a in m["blink_at_pause"]]
    s = {"clips": len(ms), "minutes": round(minutes, 2),
         "closure_depth": float(np.median(mins)) if len(mins) else float("nan"),
         "closure_viol": float((mins > st["tau_violate"]).mean()) if len(mins) else float("nan"),
         "blinks_per_min": sum(len(m["blinks"]) for m in ms) / max(minutes, 1e-9),
         "ibi_median_s": float(np.median(ibis)) if ibis else float("nan"),
         "ibi_cv": float(np.std(ibis) / np.mean(ibis)) if len(ibis) > 1 else float("nan"),
         "blink_ms": float(np.median(durs)) if durs else float("nan"),
         "blink_at_pause": float(np.mean(at)) if at else float("nan"),
         "pause_cover": float(np.mean([m["pause_cover"] for m in ms])),
         "beat_align": float(np.nanmean([m["bas"] for m in ms])),
         "beat_chance": float(np.nanmean([m["chance"] for m in ms]))}
    for k in ("lip_corr", "lip_std_r", "lip_ccc", "eye_std_r"):
        if ms and k in ms[0]:
            s[k] = float(np.nanmean([m[k] for m in ms]))
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=300, help="val clips")
    ap.add_argument("--max-frames", type=int, default=250)
    ap.add_argument("--min-frames", type=int, default=125)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variants", nargs="+", default=[""],
                    help="sampler specs, e.g. 'g=2,steps=6' (sang.model.flow.SAMPLER_KEYS); "
                         "default: the run's guidance and steps")
    ap.add_argument("--sync-offsets", required=True,
                    help="<cache>/sync_offsets.json: drop no-sync val clips and move each clip's audio (features and "
                         "waveform) by its SyncNet offset, so lip metrics compare against aligned ground truth")
    ap.add_argument("--name", default="", help="suffix for the results file, so parallel sweeps do not overwrite")
    ap.add_argument("--out", default=str(REPO / "results/naturalness"))
    args = ap.parse_args()

    dev = "cuda"
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    cfg = ck["cfg"]
    model = build(cfg).to(dev).eval()
    model.load_state_dict(ck["ema"])
    norm = Norm(**ck["norm"]).to(dev)
    latent = MotionLatent.of(ck).to(dev)
    specs = [(v, parse_spec(v)) for v in args.variants]
    cache = Path(cfg["cache_dir"])
    read = load_readouts(cache / "openness.json")
    for k, r in read.items():
        print(f"readout {k}: held-out R^2 {r.r2:.3f}", flush=True)

    _, val = split_by_speaker(load_index(cache), cfg["val_frac"], cfg["seed"])
    so = SyncOffsets.load(args.sync_offsets)
    val, why = so.filter(val)
    print(f"sync offsets {args.sync_offsets}: dropped {why}", flush=True)
    val = [r for r in sorted(val, key=lambda r: r["clip"]) if r["n"] >= args.min_frames][: args.n]
    print(f"{len(val)} val clips from {len({Path(r['path']).parent.name for r in val})} unseen speakers", flush=True)

    phon = PhonemeRecognizer(dev)
    clips, t0 = [], time.time()
    for r in val:
        try:
            d = torch.load(r["path"], map_location="cpu", weights_only=True)
            n = min(args.max_frames, int(d["n"]))
            s = so.shift(r)
            wav = shift_wav(load_wav(Path(r["clip"]).with_suffix(".m4a"), r.get("start", 0), n), s)
            db = energy_db(wav, n)
            m = d["m"][:n].float()
            y_real = to_target(m)
            clips.append({"row": r, "n": n, "y": y_real, "kp0": d["kp"][0].float(), "m_ref": m[:1],
                          "audio": shift_ticks(d["audio"][: 2 * n].float(), s), "db": db,
                          "pause": pauses(db), "events": phon.bilabials(wav, n)})
        except Exception as e:
            print(f"  skip {Path(r['clip']).name}: {type(e).__name__}: {e}", flush=True)
    print(f"audio pass: {len(clips)} clips, {sum(len(c['events']) for c in clips)} bilabials, "
          f"{time.time() - t0:.0f} s", flush=True)

    offs = [o for c in clips for o in closure_offset(read["lip"](c["y"]).numpy(), c["events"])]
    st = {"closure_offset": int(Counter(offs).most_common(1)[0][0]) if offs else 0}
    real = [measure(c["y"], c, read, st) for c in clips]
    mins = np.concatenate([m["minima"] for m in real])
    st["tau_violate"] = float(np.percentile(mins, 90))
    print(f"closure offset {st['closure_offset']:+d} frames, closure threshold {st['tau_violate']:.3f}", flush=True)

    table = {"real": summarise(real, st)}
    for vname, spec in specs:
        samp = sampler_kwargs(spec, cfg)
        col = vname or "model"
        ms, t0 = [], time.time()
        for c, rm in zip(clips, real):
            m_ref = c["m_ref"].to(dev)
            ref = norm.ref(c["kp0"][None].to(dev), m_ref)
            g = torch.Generator(device=dev).manual_seed(args.seed)
            z = generate(model, c["audio"][None].to(dev), ref, c["n"], window=cfg["frames"], n_prefix=cfg["prefix"],
                         generator=g, **samp)[0]
            ms.append(measure(to_target(latent.to_motion(z, m_ref)).cpu(), c, read, st, real=rm))
        table[col] = summarise(ms, st)
        table[col]["gen_s"] = time.time() - t0
        print(f"{col}: {time.time() - t0:.0f} s", flush=True)

    cols = [k for k in ("lip_corr", "lip_ccc", "lip_std_r", "closure_viol", "closure_depth", "eye_std_r",
                        "blinks_per_min", "blink_at_pause", "beat_align", "beat_chance", "gen_s")]
    w = max(len(c) for c in table) + 2
    print("\n" + f"{'setting':<{w}}" + "".join(f"{k:>15}" for k in cols))
    for c, s in table.items():
        print(f"{c:<{w}}" + "".join(f"{s.get(k, float('nan')):>15.3f}" for k in cols))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    name = f"{Path(args.ckpt).parent.name}_step{ck['step']}_sync{f'_{args.name}' if args.name else ''}.json"
    (out / name).write_text(json.dumps({"ckpt": args.ckpt, "step": ck["step"], "stats": st, "table": table}, indent=1))
    print(f"\nwrote {out / name}", flush=True)


if __name__ == "__main__":
    main()

