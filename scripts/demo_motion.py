#!/usr/bin/env python3
"""Side-by-side demos: REAL | GENERATED, with the clip's own audio, on speakers the model never saw.

    sbatch bash_scripts/job.sh scripts/demo_motion.py --ckpt runs/motion_12k_anneal/best.pt --n 6

For each clip: frame 0 is the source image, the clip's own speech drives the model, and the real
video is cropped with the SAME box as the source, so the two halves line up. Clips come from the
validation speakers of the run's own split, so nothing shown was trained on. Writes
<out>/<k>_<clip>_sbs.mp4 (512x1024: real left, generated right), <k>_<clip>_gen.mp4, and
<clip>_real.mp4 (the real crop alone, for scripts/eval_lse.py).
--guide lips|blinks|both adds the rule constraints (same seed, so compare with the plain run).
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
from sang.motion_model import Norm, build, ema_weights, guidance_vector, load_ema, parse_spec, sampler_kwargs
from sang.naturalness import Guide, guided_generate

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
    ap.add_argument("--cfg-mouth", type=float, default=None,
                    help="audio guidance on the 18 mouth coordinates only (per-region guidance); default = --cfg")
    ap.add_argument("--cfg-rescale", type=float, default=0.0,
                    help="CFG-rescale blend (0 = off, 0.7 = Lin et al.): keeps guidance timing, restores amplitude")
    ap.add_argument("--sampler", default=None,
                    help="sampler spec, e.g. 'g=2,mouth=1.25,avg=4' (sang.motion_model.SAMPLER_KEYS); overrides "
                         "--cfg/--cfg-mouth/--cfg-rescale/--steps")
    ap.add_argument("--ema", default=None, help="extra EMA decay saved by training with ema_extra (e.g. 0.999); default = the main EMA")
    ap.add_argument("--guide-ckpt", default=None, help="autoguidance guide checkpoint (for ag / ag_mouth in --sampler)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--guide", default="none", choices=["none", "lips", "blinks", "both"],
                    help="rule constraints at sampling (sang/naturalness.py); needs openness.json + "
                         "naturalness_stats.json in the cache dir")
    ap.add_argument("--start", default="null", choices=["null", "source"],
                    help="first window: 'null' = the dropped-prefix configuration training used; "
                         "'source' = continue from the source photo's own motion after 0.4 s of silence")
    ap.add_argument("--stitch", action=argparse.BooleanOptionalAction, default=True,
                    help="LivePortrait's stitching module (default ON; --no-stitch to disable): pins shoulders and nearby "
                         "background to the source. Generated demos: background motion 2.41x -> 0.56x real (jobs "
                         "173584/173682); on the 42-d target its cost measured ~0 (mouth corr 0.824 vs 0.809)")
    ap.add_argument("--held", default="camera", choices=["camera", "head"],
                    help="how the 8 non-driven keypoints follow the pose: camera = upstream LivePortrait convention (default); head = rotate them with the head (SANG before 2026-09-29; suspected background-warp cause)")
    ap.add_argument("--voice", default=None,
                    help="also write <stem>_voice-<name>_gen.mp4: the same video speaking in this voice bank's "
                         "voice (voices/<name> from scripts/enroll_voice.py, or a folder of the person's audio)")
    ap.add_argument("--voice-k", type=int, default=None)
    args = ap.parse_args()

    dev = "cuda"
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    cfg = ck["cfg"]
    model = build(cfg).to(dev).eval()
    model.load_state_dict(ema_weights(ck, args.ema))
    norm = Norm(**ck["norm"]).to(dev)
    gamma = guidance_vector(cfg["cfg_audio"] if args.cfg is None else args.cfg, mouth=args.cfg_mouth)
    samp = sampler_kwargs(parse_spec(args.sampler), cfg, load_ema(args.guide_ckpt, dev, norm) if args.guide_ckpt else None
                          ) if args.sampler else None
    print(f"checkpoint {args.ckpt}: step {ck['step']}, best val {ck.get('best', float('nan')):.4f}", flush=True)

    n_frames = int(args.seconds * FPS)
    clips = args.clips or val_clips(Path(cfg["cache_dir"]), cfg["val_frac"], cfg["seed"], n_frames)[: args.n]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from decord import AudioReader
    from sang.codec import load_wavlm
    codec = MotionCodec(device=dev)
    wavlm = load_wavlm(cfg.get("audio_encoder", "wavlm-large"), device=dev)
    guide = Guide.load(Path(cfg["cache_dir"]), dev) if args.guide != "none" else None
    knnvc = bank = None
    if args.voice:
        from sang.voice import SpeakerEncoder, load_knnvc, voice_bank
        knnvc = load_knnvc(dev)
        enrolling = not (Path(args.voice) / "bank.pt").exists()
        bank = voice_bank(args.voice, knnvc, SpeakerEncoder(dev) if enrolling else None)

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
            sm = codec.source_motion(src)
            m_src = sm["m"].to(dev)
            ref = norm.ref(sm["kp"].to(dev), to_target(m_src))
            start, lead = None, 0
            if args.start == "source":                       # the photo, still and silent, for P frames
                lead = cfg["prefix"]
                start = norm.target(to_target(m_src)).unsqueeze(1).expand(1, lead, -1)
            with torch.no_grad():
                audio = wavlm.encode(torch.nn.functional.pad(wav, (lead * SR // FPS, 0))[None].to(dev)).float()
            y, info = guided_generate(model, audio, ref, len(frames), cfg, norm, wav[0], guide, args.guide,
                                      args.seed, cfg_audio=gamma, start=start, cfg_rescale=args.cfg_rescale,
                                      sampler=samp)
            m = from_target(norm.untarget(y[0]), m_src, held=args.held)
            gen = codec.render(src, m.cpu(), relative=False, stitch=args.stitch)

            n = min(len(real), len(gen))
            m4a = Path(clip).with_suffix(".m4a")     # starts at frame 0 too; the mux cuts it with -shortest
            stem = (f"{k:02d}_{Path(clip).stem[:40]}" + ("" if args.guide == "none" else f"_{args.guide}")
                    + ("" if args.start == "null" else f"_start-{args.start}")
                    + ("" if args.held == "camera" else f"_held-{args.held}")
                    + ("" if args.stitch else "_nostitch"))
            write_mp4(np.concatenate([real[:n], gen[:n]], axis=2), out / f"{stem}_sbs.mp4", fps=FPS, audio=m4a)
            write_mp4(gen[:n], out / f"{stem}_gen.mp4", fps=FPS, audio=m4a)
            real_mp4 = out / f"{Path(clip).stem[:40]}_real.mp4"          # one per clip, shared by all settings
            if not real_mp4.exists():                                    # SyncNet needs one face per video
                write_mp4(real[:n], real_mp4, fps=FPS, audio=m4a)
            if bank is not None:                         # face from the ORIGINAL audio; only the track changes
                from sang.voice import convert, save_wav
                vw = save_wav(convert(wav[0], bank, knnvc, k=args.voice_k), out / f"{stem}_voice-{bank.dir.name}.wav")
                write_mp4(gen[:n], out / f"{stem}_voice-{bank.dir.name}_gen.mp4", fps=FPS, audio=vw)
            print(f"[{k + 1}/{len(clips)}] {stem}_sbs.mp4  ({n / FPS:.1f} s) {info or ''}", flush=True)
        except Exception as e:
            print(f"[{k + 1}/{len(clips)}] SKIP {Path(clip).name}: {type(e).__name__}: {e}", flush=True)
    (out / "demo.json").write_text(json.dumps({"ckpt": args.ckpt, "step": ck["step"], "clips": clips}, indent=1))
    print(f"done -> {out}", flush=True)


if __name__ == "__main__":
    main()
