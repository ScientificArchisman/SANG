#!/usr/bin/env python3
"""Rule-based naturalness metrics, REAL vs GENERATED, on unseen validation speakers. Motion space
only: no rendering, so ~400 clips take minutes.

    sbatch bash_scripts/job.sh scripts/naturalness.py --ckpt runs/motion_12k_anneal/best.pt

Needs <cache>/openness.json from scripts/calibrate_openness.py. Pass 1 measures the REAL clips and
writes <cache>/naturalness_stats.json (closure timing and depth, blink intervals and durations),
which the constraints are built from. Pass 2 generates every clip once per --modes entry with the
same seed, so the columns differ only by the constraints:

    none    the model as trained
    lips    bilabial closure enforced at sampling (sang/naturalness.py, no retraining)
    blinks  blinks added in the model's long gaps, snapped to pauses
    both

Metrics (real is the reference column; aim to match it, not to maximise):
    closure_depth    median lowest lip openness around /p b m/           lower = more closed
    closure_viol     share of /p b m/ with the lips open (threshold = real's 90th pct, so real = 0.10)
    lip_corr         per-clip correlation of lip openness with the real clip  (lip-sync proxy)
    lip_std_r        lip openness std / real's                              ~1
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

from sang.motion import to_target
from sang.motion_model import Norm, build
from sang.naturalness import (FPS, Guide, PhonemeRecognizer, audio_onsets, beat_alignment, blink_events,
                              closure_minima, closure_offset, energy_db, guided_generate, head_beats,
                              load_readouts, load_wav, pauses)


def pause_near(pause: np.ndarray, reach: int = 5) -> np.ndarray:
    """Frames within `reach` of a pause frame."""
    k = np.ones(2 * reach + 1)
    return np.convolve(pause.astype(float), k, mode="same") > 0


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
    for k in ("lip_corr", "lip_std_r", "eye_std_r"):
        if ms and k in ms[0]:
            s[k] = float(np.nanmean([m[k] for m in ms]))
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=300, help="val clips")
    ap.add_argument("--max-frames", type=int, default=250)
    ap.add_argument("--min-frames", type=int, default=125)
    ap.add_argument("--modes", nargs="+", default=["none", "lips", "blinks", "both"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(REPO / "results/naturalness"))
    args = ap.parse_args()

    dev = "cuda"
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    cfg = ck["cfg"]
    model = build(cfg).to(dev).eval()
    model.load_state_dict(ck["ema"])
    norm = Norm(**ck["norm"]).to(dev)
    cache = Path(cfg["cache_dir"])
    read = load_readouts(cache / "openness.json")
    for k, r in read.items():
        print(f"readout {k}: held-out R^2 {r.r2:.3f}", flush=True)

    import importlib.util
    spec = importlib.util.spec_from_file_location("tm", REPO / "scripts" / "train_motion.py")
    tm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tm)
    _, val = tm.split_by_speaker(tm.load_index(cache), cfg["val_frac"], cfg["seed"])
    val = [r for r in sorted(val, key=lambda r: r["clip"]) if r["n"] >= args.min_frames][: args.n]
    print(f"{len(val)} val clips from {len({Path(r['path']).parent.name for r in val})} unseen speakers", flush=True)

    # ---------------- pass 1: real clips, audio events
    phon = PhonemeRecognizer(dev)
    clips, t0 = [], time.time()
    for r in val:
        try:
            d = torch.load(r["path"], map_location="cpu", weights_only=True)
            n = min(args.max_frames, int(d["n"]))
            wav = load_wav(r["clip"], r.get("start", 0), n)
            db = energy_db(wav, n)
            clips.append({"row": r, "n": n, "y": to_target(d["m"][:n].float()), "kp0": d["kp"][0].float(),
                          "audio": d["audio"][: 2 * n].float(), "wav": wav, "db": db, "pause": pauses(db),
                          "events": phon.bilabials(wav, n)})
        except Exception as e:
            print(f"  skip {Path(r['clip']).name}: {type(e).__name__}: {e}", flush=True)
    print(f"audio pass: {len(clips)} clips, {sum(len(c['events']) for c in clips)} bilabials, "
          f"{time.time() - t0:.0f} s", flush=True)

    offs = [o for c in clips for o in closure_offset(read["lip"](c["y"]).numpy(), c["events"])]
    off = Counter(offs).most_common(1)[0][0] if offs else 0
    st = {"closure_offset": int(off)}
    real = [measure(c["y"], c, read, st) for c in clips]
    mins = np.concatenate([m["minima"] for m in real])
    st.update(tau_close=float(np.median(mins)), tau_violate=float(np.percentile(mins, 90)))
    ibis = [(b2[1] - b1[1]) / FPS for m in real for b1, b2 in zip(m["blinks"], m["blinks"][1:])]
    st.update(ibi_s=[round(x, 3) for x in ibis],
              blink_frames=[int(e - s) for m in real for s, _, e in m["blinks"]] or [5],
              eye_closed=float(np.median([m["eye"][b] for m in real for _, b, _ in m["blinks"]] or [np.nan])),
              offset_hist={int(k): v for k, v in sorted(Counter(offs).items())})
    if not ibis:
        raise SystemExit("no blinks found in real motion: the eye readout cannot see them; check openness.json R^2")
    (cache / "naturalness_stats.json").write_text(json.dumps(st))
    print(f"closure offset {off:+d} frames (hist {st['offset_hist']}), tau_close {st['tau_close']:.3f}, "
          f"eye_closed {st['eye_closed']:.3f}, {len(ibis)} real inter-blink intervals", flush=True)

    # ---------------- pass 2: generate per mode
    guide = Guide(read, st, phon)
    table = {"real": summarise(real, st)}
    extra = {}
    for mode in args.modes:
        ms, infos, t0 = [], Counter(), time.time()
        for c, rm in zip(clips, real):
            ref = norm.ref(c["kp0"][None].to(dev), c["y"][:1].to(dev))
            y, info = guided_generate(model, c["audio"][None].to(dev), ref, c["n"], cfg, norm, c["wav"],
                                      guide, mode, args.seed, events=c["events"])
            infos.update(info)
            ms.append(measure(norm.untarget(y[0]).cpu(), c, read, st, real=rm))
        table[mode] = summarise(ms, st)
        extra[mode] = dict(infos)
        print(f"mode {mode}: {time.time() - t0:.0f} s  {dict(infos)}", flush=True)

    keys = [k for k in table["none" if "none" in table else "real"] if k not in ("clips", "minutes")]
    print("\n" + f"{'metric':<16}" + "".join(f"{c:>10}" for c in table))
    for k in keys:
        print(f"{k:<16}" + "".join(f"{table[c].get(k, float('nan')):>10.3f}" for c in table))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    name = f"{Path(args.ckpt).parent.name}_step{ck['step']}.json"
    (out / name).write_text(json.dumps({"ckpt": args.ckpt, "step": ck["step"], "stats": {k: v for k, v in st.items() if k != "ibi_s"},
                                        "table": table, "constraints": extra}, indent=1))
    print(f"\nwrote {out / name}", flush=True)


if __name__ == "__main__":
    main()
