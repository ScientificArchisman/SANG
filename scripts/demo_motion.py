#!/usr/bin/env python3
"""Side-by-side demos: REAL | GENERATED, with the clip's own audio, on speakers the model never saw.

    sbatch bash_scripts/job.sh scripts/demo_motion.py --ckpt runs/d1_97h/last.pt --n 6 \
        --sampler g=2,mouth=1.25,tau=0.5,avg=4,steps=6

For each clip: frame 0 is the source image, the clip's own speech drives the model (sang.pipeline.Talker),
and the real video is cropped with the SAME box as the source, so the two halves line up. Clips come from
the validation speakers of the run's own split. Writes <out>/<k>_<clip>_sbs.mp4 (512x1024: real left,
generated right), <k>_<clip>_gen.mp4, and <clip>_real.mp4 (the real crop alone, for scripts/eval_lse.py).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from sang.bench import write_mp4
from sang.codec import load_wav
from sang.data import split_by_speaker, val_clips
from sang.motion import decode_clip
from sang.paths import load_index, motion_cache
from sang.pipeline import FPS, Talker


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--out", default=str(REPO / "results/extras/demo"))
    ap.add_argument("--clips", nargs="*", help="explicit clip paths instead of val speakers")
    ap.add_argument("--sampler", default=None, help="sampler spec (sang.motion_model.SAMPLER_KEYS); default: the run's")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start", default="source", choices=["null", "source"])
    ap.add_argument("--stitch", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--voice", default=None,
                    help="also write <stem>_voice-<name>_gen.mp4: the same video speaking in this voice bank's voice")
    ap.add_argument("--voice-k", type=int, default=None)
    args = ap.parse_args()

    dev = "cuda"
    talker = Talker(args.ckpt, args.sampler, dev)
    cfg = talker.cfg
    print(f"checkpoint {args.ckpt}: step {talker.step}", flush=True)
    n_frames = int(args.seconds * FPS)
    if not args.clips:
        _, val = split_by_speaker(load_index(motion_cache(cfg["cache_dir"])), cfg["val_frac"], cfg["seed"])
    clips = args.clips or val_clips(val, n_frames)[: args.n]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    knnvc = bank = None
    if args.voice:
        from sang.voice import SpeakerEncoder, load_knnvc, voice_bank
        knnvc = load_knnvc(dev)
        bank = voice_bank(args.voice, knnvc, None if (Path(args.voice) / "bank.pt").exists() else SpeakerEncoder(dev))

    codec = talker.codec
    for k, clip in enumerate(clips):
        try:
            frames = decode_clip(clip, n_frames)
            lmk = codec.landmarks(frames[0])
            if lmk is None:
                raise ValueError("no face in frame 0")
            real, _ = codec.crop_with_box(frames, codec._box(lmk), size=512)
            m4a = Path(clip).with_suffix(".m4a")
            wav = load_wav(m4a, 0, len(frames))
            gen, _, _ = talker.run(frames[0], wav, len(frames), seed=args.seed, stitch=args.stitch, paste=False,
                                   start=args.start)
            n = min(len(real), len(gen))
            stem = f"{k:02d}_{Path(clip).stem[:40]}" + ("" if args.stitch else "_nostitch")
            write_mp4(np.concatenate([real[:n], gen[:n]], axis=2), out / f"{stem}_sbs.mp4", fps=FPS, audio=m4a)
            write_mp4(gen[:n], out / f"{stem}_gen.mp4", fps=FPS, audio=m4a)
            real_mp4 = out / f"{Path(clip).stem[:40]}_real.mp4"
            if not real_mp4.exists():
                write_mp4(real[:n], real_mp4, fps=FPS, audio=m4a)
            if bank is not None:
                from sang.voice import convert, save_wav
                vw = save_wav(convert(wav, bank, knnvc, k=args.voice_k), out / f"{stem}_voice-{bank.dir.name}.wav")
                write_mp4(gen[:n], out / f"{stem}_voice-{bank.dir.name}_gen.mp4", fps=FPS, audio=vw)
            print(f"[{k + 1}/{len(clips)}] {stem}_sbs.mp4  ({n / FPS:.1f} s)", flush=True)
        except Exception as e:
            print(f"[{k + 1}/{len(clips)}] SKIP {Path(clip).name}: {type(e).__name__}: {e}", flush=True)
    (out / "demo.json").write_text(json.dumps({"ckpt": args.ckpt, "step": talker.step, "clips": clips}, indent=1))
    print(f"done -> {out}", flush=True)


if __name__ == "__main__":
    main()

