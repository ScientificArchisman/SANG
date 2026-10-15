#!/usr/bin/env python3
"""End-to-end speed of SANG, stage by stage (report 'SANG latency reduction' 1.4, E0), and the quality gate
for the exact speed-ups (gate B: same motion, fast vs original pipeline frames).

    sbatch --gres=gpu:h100:1 bash_scripts/job.sh scripts/latency.py --data /beegfs/work/$USER/HDTF \\
        --list configs/hdtf_test.csv [--ckpt2 runs/<other>/last.pt --sampler2 g=2,mouth=1.25,steps=3]

Configurations, run in turn on each clip (first frame = source photo, the clip's audio, --seconds long):
  baseline  --ckpt with --sampler and the original pipeline (two CFG calls per step, per-frame FP32
            render, CPU paste-back, synchronous mp4 encode)
  fast      the same model and sampler with every pipeline speed-up (sang.pipeline.Opts.fast) and the
            encoder on a background thread fed batch by batch
  model2    --ckpt2 with --sampler2 and the fast pipeline (optional: a second model or sampler)
Stages: source (face detection + source motion), audio (WavLM), motion (the DiT), render, paste, encode
(the time the mp4 still needs after the last frame). total = wall clock from the photo to a finished mp4
with audio. RTF = total / clip length; fps = frames / total.
Gate B (baseline vs fast, same motion up to float rounding): PSNR of the full frame and of the mouth box.
"""
import argparse
import importlib.util
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "third_party"))

from sang.bench import write_mp4
from sang.fastpath import AsyncWriter, Timer
from sang.motion import decode_clip
from sang.pipeline import FPS, Opts, Talker
from sang.voice import save_wav


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean())
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def mouth_box(lmk: np.ndarray, H: int, W: int) -> tuple[slice, slice]:
    """Generous box around the mouth from LivePortrait's 203 landmarks (corners 48 / 66, lip gap 90 / 102)."""
    pts = lmk[[48, 66, 90, 102]]
    c = pts.mean(0)
    w = abs(lmk[66, 0] - lmk[48, 0])
    x0, x1 = int(max(0, c[0] - 0.9 * w)), int(min(W, c[0] + 0.9 * w))
    y0, y1 = int(max(0, c[1] - 0.7 * w)), int(min(H, c[1] + 0.8 * w))
    return slice(y0, y1), slice(x0, x1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=str(REPO / "runs/b1_lognorm03/last.pt"))
    ap.add_argument("--sampler", default="g=2,mouth=1.25,tau=0.5,avg=4,steps=6")
    ap.add_argument("--ckpt2", default=None, help="a second checkpoint, run with the fast pipeline")
    ap.add_argument("--sampler2", default=None, help="its sampler spec (default: --sampler)")
    ap.add_argument("--data", required=True, help="HDTF root (contains crops512/)")
    ap.add_argument("--list", default=str(REPO / "configs/hdtf_test.csv"))
    ap.add_argument("--n", type=int, default=20, help="clips measured (after the warm-up clips)")
    ap.add_argument("--warmup", type=int, default=2, help="clips run first and not counted (CUDA / cuDNN warm-up)")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--start", default="source", choices=["null", "source"])
    ap.add_argument("--name", default="")
    ap.add_argument("--out", default=str(REPO / "results/latency"))
    args = ap.parse_args()

    ev = load_script("eval_hdtf")
    clips = ev.read_list(args.list, Path(args.data), Path(args.data) / "crops512")[: args.warmup + args.n]
    base = Talker(args.ckpt, args.sampler)
    talkers = {"baseline": (base, Opts()), "fast": (base, Opts.fast())}
    if args.ckpt2:
        talkers["model2"] = (Talker(args.ckpt2, args.sampler2 or args.sampler, codec=base.codec, wavlm=base.wavlm),
                             Opts.fast())
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"{gpu}; {len(clips)} clips ({args.warmup} warm-up) x {args.seconds:g} s; configs "
          + "; ".join(f"{k}: {Path(t.cfg['out_dir']).name} step {t.step}, {o}" for k, (t, o) in talkers.items()), flush=True)

    recs = {k: [] for k in talkers}
    gate = []
    tmp = Path(tempfile.mkdtemp(prefix="sang_latency_"))
    for i, clip in enumerate(clips):
        frames = decode_clip(str(clip), int(args.seconds * FPS))
        wav = ev.load_wav(clip, len(frames))
        wav_path = save_wav(wav, tmp / "a.wav")
        src, n = frames[0], len(frames)
        outs = {}
        for name, (talker, opts) in talkers.items():
            torch.cuda.reset_peak_memory_stats()
            timer = Timer()
            mp4 = tmp / f"{name}.mp4"
            t0 = time.perf_counter()
            if opts.gpu_paste:
                w = AsyncWriter(mp4, fps=FPS, audio=wav_path)
                out, _, _ = talker.run(src, wav, n, opts, start=args.start, timer=timer, sink=w.write, keep=name == "fast")
                t1 = time.perf_counter()
                w.close()
            else:
                out, _, _ = talker.run(src, wav, n, opts, start=args.start, timer=timer)
                t1 = time.perf_counter()
                write_mp4(out, mp4, fps=FPS, audio=wav_path)
            total = time.perf_counter() - t0
            rec = {"clip": clip.stem, "frames": n, **timer.t, "encode": time.perf_counter() - t1, "total": total,
                   "fps": n / total, "rtf": total / (n / FPS), "vram_gb": torch.cuda.max_memory_allocated() / 1e9}
            if i >= args.warmup:
                recs[name].append(rec)
            if name in ("baseline", "fast"):
                outs[name] = out
        if "fast" in outs and outs["fast"] is not None:
            a, b = outs["baseline"], outs["fast"]
            ys, xs = mouth_box(base.codec.landmarks(src), *src.shape[:2])
            g = {"clip": clip.stem, "psnr": psnr(a, b), "psnr_mouth": psnr(a[:, ys, xs], b[:, ys, xs]),
                 "max_abs": int(np.abs(a.astype(int) - b.astype(int)).max())}
            if i >= args.warmup:
                gate.append(g)
        line = "  ".join(f"{k} {r[-1]['total']:.2f} s ({r[-1]['fps']:.0f} fps)" for k, r in recs.items() if r and i >= args.warmup)
        print(f"[{i + 1}/{len(clips)}] {clip.stem:<28} {'warm-up' if i < args.warmup else line}"
              + (f"  gate B PSNR {gate[-1]['psnr']:.1f} dB, mouth {gate[-1]['psnr_mouth']:.1f} dB" if gate and i >= args.warmup else ""),
              flush=True)

    stages = ("source", "audio", "motion", "render", "paste", "encode", "total")
    summ = {}
    print(f"\nMEDIAN (p90) seconds per {args.seconds:g} s clip over {args.n} clips, {gpu}")
    print(f"{'config':<10}" + "".join(f"{s:>16}" for s in stages) + f"{'fps':>8}{'RTF':>7}{'speed-up':>10}{'VRAM GB':>9}")
    for name, rs in recs.items():
        if not rs:
            continue
        q = lambda k, p: float(np.percentile([r.get(k, 0.0) for r in rs], p))
        summ[name] = {k: {"median": q(k, 50), "p90": q(k, 90)} for k in stages}
        summ[name].update(fps=q("fps", 50), rtf=q("rtf", 50), vram_gb=max(r["vram_gb"] for r in rs))
    for name, s in summ.items():
        sp = summ["baseline"]["total"]["median"] / s["total"]["median"]
        print(f"{name:<10}" + "".join(f"{s[k]['median']:>9.3f} ({s[k]['p90']:.2f})" for k in stages)
              + f"{s['fps']:>8.1f}{s['rtf']:>7.3f}{sp:>9.2f}x{s['vram_gb']:>9.1f}")
    if gate:
        p = np.array([g["psnr"] for g in gate])
        pm = np.array([g["psnr_mouth"] for g in gate])
        print(f"\nGATE B (fast vs baseline frames, same motion): PSNR median {np.median(p):.1f} dB (min {p.min():.1f}), "
              f"mouth box median {np.median(pm):.1f} dB (min {pm.min():.1f}); max |diff| {max(g['max_abs'] for g in gate)} "
              f"grey levels. Pass: >= 40 dB full frame, >= 38 dB mouth")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    name = (args.name or gpu.replace(" ", "_")) + ".json"
    (out / name).write_text(json.dumps({"gpu": gpu, "args": vars(args), "summary": summ, "gate_b": gate,
                                        "clips": recs}, indent=1, default=float))
    print(f"\nwrote {out / name}", flush=True)


if __name__ == "__main__":
    main()
