#!/usr/bin/env python3
"""Side-by-side demos: REAL | GENERATED, with the clip's own audio, on speakers the model never saw.

    sbatch bash_scripts/job.sh scripts/demo_motion.py --ckpt runs/motion_12k_anneal/best.pt --n 6

For each clip: frame 0 is the source image, the clip's own speech drives the model, and the real
video is cropped with the SAME box as the source, so the two halves line up. Clips come from the
validation speakers of the run's own split, so nothing shown was trained on. Writes
<out>/<k>_<clip>_sbs.mp4 (512x1024: real left, generated right) and <k>_<clip>_gen.mp4.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "third_party"))

from sang.bench import write_mp4
from sang.motion import MotionCodec, decode_clip, from_target, to_target
from sang.motion_model import Norm, build, generate

SR, FPS = 16000, 25


def val_clips(cache: Path, val_frac: float, seed: int, min_frames: int) -> list[str]:
    """Clip paths of the val speakers, by the same hash rule scripts/train_motion.py uses."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("tm", REPO / "scripts" / "train_motion.py")
    tm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tm)
    rows = tm.load_index(cache)
    _, val = tm.split_by_speaker(rows, val_frac, seed)
    seen, out = set(), []
    for r in sorted(val, key=lambda r: r["clip"]):         # one clip per speaker, for variety
        spk = Path(r["path"]).parent.name
        if spk not in seen and r["n"] >= min_frames and r.get("start", 0) == 0:
            seen.add(spk)
            out.append(r["clip"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--out", default=str(REPO / "results/extras/demo"))
    ap.add_argument("--clips", nargs="*", help="explicit clip paths instead of val speakers")
    ap.add_argument("--cfg", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = "cuda"
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    cfg = ck["cfg"]
    model = build(cfg).to(dev).eval()
    model.load_state_dict(ck["ema"])
    norm = Norm(**ck["norm"]).to(dev)
    print(f"checkpoint {args.ckpt}: step {ck['step']}, best val {ck.get('best', float('nan')):.4f}", flush=True)

    n_frames = int(args.seconds * FPS)
    clips = args.clips or val_clips(Path(cfg["cache_dir"]), cfg["val_frac"], cfg["seed"], n_frames)[: args.n]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from decord import AudioReader
    from sang.codec import load_wavlm
    codec = MotionCodec(device=dev)
    wavlm = load_wavlm(cfg.get("audio_encoder", "wavlm-large"), device=dev)

    for k, clip in enumerate(clips):
        try:
            frames = decode_clip(clip, n_frames)
            src = frames[0]
            lmk = codec.landmarks(src)
            if lmk is None:
                raise ValueError("no face in frame 0")
            real, _ = codec.crop_with_box(frames, codec._box(lmk), size=512)   # same box as the source

            wav = torch.from_numpy(AudioReader(str(Path(clip).with_suffix(".m4a")), sample_rate=SR,
                                               mono=True)[:].asnumpy()).float()[:, : len(frames) * SR // FPS]
            with torch.no_grad():
                audio = wavlm.encode(wav[None].to(dev)).float()
            sm = codec.source_motion(src)
            m_src = sm["m"].to(dev)
            ref = norm.ref(sm["kp"].to(dev), to_target(m_src))
            g = torch.Generator(device=dev).manual_seed(args.seed)
            y = generate(model, audio, ref, len(frames), window=cfg["frames"], n_prefix=cfg["prefix"],
                         steps=cfg["sample_steps"], cfg_audio=cfg["cfg_audio"] if args.cfg is None else args.cfg,
                         generator=g)
            m = from_target(norm.untarget(y[0]), m_src)
            gen = codec.render(src, m.cpu(), relative=False, stitch=False)

            n = min(len(real), len(gen))
            m4a = Path(clip).with_suffix(".m4a")     # starts at frame 0 too; the mux cuts it with -shortest
            stem = f"{k:02d}_{Path(clip).stem[:40]}"
            write_mp4(np.concatenate([real[:n], gen[:n]], axis=2), out / f"{stem}_sbs.mp4", fps=FPS, audio=m4a)
            write_mp4(gen[:n], out / f"{stem}_gen.mp4", fps=FPS, audio=m4a)
            print(f"[{k + 1}/{len(clips)}] {stem}_sbs.mp4  ({n / FPS:.1f} s)", flush=True)
        except Exception as e:
            print(f"[{k + 1}/{len(clips)}] SKIP {Path(clip).name}: {type(e).__name__}: {e}", flush=True)
    (out / "demo.json").write_text(json.dumps({"ckpt": args.ckpt, "step": ck["step"], "clips": clips}, indent=1))
    print(f"done -> {out}", flush=True)


if __name__ == "__main__":
    main()
