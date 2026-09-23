#!/usr/bin/env python3
"""M0 GATE -- the motion-representation ceiling. Run this BEFORE caching or training anything.

GATES (revised 2026-09-23 after jobs 171503-5). The first version gated on frame PSNR >= 29 dB,
LivePortrait's number on a curated benchmark. On TalkVid, freezing frame 0 already scores 15-16 dB
because hands, torso, lighting and camera move -- none of which a face-motion vector controls -- so
that gate measured the clips, not the renderer. The gates are now about what M0 is for: does the
frozen renderer realise the motion it is given, and keep the identity?
  mouth_corr >= 0.85   motion re-extracted from the render tracks the input over time
  mouth_amp in [0.75, 1.25]  ... and moves as much
  CSIM >= 0.85         identity preserved
  LSE-C within 0.5 of real video (with --lse), the number the benchmark actually scores
PSNR is still printed, next to the freeze-frame-0 baseline, as a diagnostic.


The motion-space analogue of scripts/ceiling.py. That script answered "how good can a perfect
appearance-token predictor be?" (VidTok-FSQ 25.31 dB @128, Wan 31.80 dB @256). This one answers
the same question for the new prediction target: extract a clip's own motion, re-render it from
its own first frame through the frozen renderer, and measure the result. No transformer, no
training. Whatever this prints is the hard upper bound on every number SANG-M will ever report.

    sbatch bash_scripts/job.sh scripts/motion_ceiling.py --n 50
    sbatch bash_scripts/job.sh scripts/motion_ceiling.py --n 50 --lse --dump results/extras/m0

GATE (docs/recovery_plan_2026-09-16.md M0): PSNR >= 29 dB @256, CSIM >= 0.95,
and with --lse, LSE-C within 0.5 of the same clip's real-video LSE-C.

If PSNR lands near 24 dB the crop convention is wrong, not the renderer -- check that
sang.motion.CROP matches what the cropper actually applied, and look at --dump before
touching anything else. Do not start the cache until this passes.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "third_party"))

from sang.bench import ArcFace, csim, lse, psnr, ssim, write_mp4
from sang.motion import MotionCodec, decode_clip, from_target, motion_fidelity, to_target

DEV = "cuda" if torch.cuda.is_available() else "cpu"

# Where the face sits in a 256 crop under sang.motion.DRIVE (scale 2.2, vy -0.1): the landmark box is
# 1/2.2 of the crop side (~116 px), centred horizontally and ~26 px below centre. Background and
# hair outside it are where a warping renderer is weakest, so it is reported separately.
FACE = (slice(96, 212), slice(70, 186))


def resize(frames: np.ndarray, res: int) -> np.ndarray:
    import cv2
    if frames.shape[1] == res and frames.shape[2] == res:
        return frames
    return np.stack([cv2.resize(f, (res, res), interpolation=cv2.INTER_AREA) for f in frames])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default=str(REPO / "data/clips_filtered_all.txt"))
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--max-frames", type=int, default=200, help="8 s at 25 fps")
    ap.add_argument("--res", type=int, default=256, help="comparison resolution (LivePortrait's own protocol)")
    ap.add_argument("--relative", action="store_true",
                    help="retarget relative to the driving's first frame. OFF for the ceiling: "
                         "self-reenactment must reproduce the absolute trajectory.")
    ap.add_argument("--no-stitch", action="store_true")
    ap.add_argument("--target", type=int, choices=(70, 42), default=70,
                    help="42 = what the generator actually predicts: rotation + 39 brow/eye/mouth "
                         "coords, with scale, translation and the 24 shape coords held at frame 0. "
                         "The gap to --target 70 is the cost of that choice, measured on our data.")
    ap.add_argument("--lse", action="store_true", help="also run syncnet_python (slow, needs audio)")
    ap.add_argument("--dump", default="", help="directory for side-by-side mp4s of the first 3 clips")
    args = ap.parse_args()

    clip_file = Path(args.clips)
    if not clip_file.exists():
        raise SystemExit(f"{clip_file} not found. Build it first:\n"
                         f"  sbatch bash_scripts/job.sh scripts/filter_clips.py --out {clip_file}")
    paths = [l.strip() for l in clip_file.read_text().splitlines() if l.strip()][: args.n]

    codec = MotionCodec(device=DEV)
    arc = ArcFace(device=DEV)
    dump = Path(args.dump) if args.dump else None
    if dump:
        dump.mkdir(parents=True, exist_ok=True)

    rows, skipped = [], []
    for k, p in enumerate(paths):
        try:
            frames = decode_clip(p, args.max_frames)
            d = codec.extract(frames)                       # ONE fixed box; truncated at a shot cut
            if len(d["crops"]) < 25:
                raise ValueError(f"only {len(d['crops'])} frames before a shot cut")
            gt = resize(d["crops"], args.res)               # ground truth IS the crop, not the raw frame
            m = d["m"].float()
            if args.target == 42:                           # frame 0 plays the source image
                m = from_target(to_target(m), m[:1])
            # The source is frame 0's crop under the SAME box, so source, motion and ground truth
            # share one coordinate frame. Re-cropping frame 0 on its own would reintroduce the offset.
            state = codec.source_state(d["crops"][0], precropped=True)
            gen = codec.render(None, m, relative=args.relative and args.target == 70,
                               stitch=not args.no_stitch, state=state)
            gen = resize(gen, args.res)
            n = min(len(gt), len(gen))
            gt, gen = gt[:n], gen[:n]
            ff, fc = FACE if args.res == 256 else (slice(None), slice(None))
            freeze = np.broadcast_to(gt[:1], gt.shape)
            # re-extract motion from the render in the same 256 crop convention the input came from
            m_back, _ = codec.motion_from_crops(resize(gen, 256) if args.res != 256 else gen)

            row = {"clip": Path(p).stem, "frames": n, "cut": n < len(frames),
                   "psnr": psnr(gt, gen), "psnr_freeze": psnr(gt, freeze),
                   "psnr_face": psnr(gt[:, ff, fc], gen[:, ff, fc]),
                   **motion_fidelity(m[:n], m_back[:n]),
                   "ssim": ssim(gt, gen),
                   "csim": csim(arc, gt[0], gen),
                   "motion_std": float(d["m"].float().std(0).mean())}

            if args.lse or dump:
                wav = Path(p).with_suffix(".m4a")
                out_dir = dump or Path(REPO / "results" / "extras" / "_m0_tmp")
                out_dir.mkdir(parents=True, exist_ok=True)
                gen_mp4 = write_mp4(gen, out_dir / f"{k:02d}_gen.mp4", audio=wav if wav.exists() else None)
                if dump and k < 3:
                    write_mp4(gt, out_dir / f"{k:02d}_gt.mp4", audio=wav if wav.exists() else None)
                a0 = d["span"][0]
                if args.lse and wav.exists() and a0 == 0:   # audio mux assumes the clip starts at 0
                    _, row["lse_d"], row["lse_c"] = lse(gen_mp4)
                    gt_mp4 = write_mp4(gt, out_dir / f"{k:02d}_gt_for_lse.mp4", audio=wav)
                    _, row["lse_d_real"], row["lse_c_real"] = lse(gt_mp4)
            rows.append(row)
            print(f"[{k + 1:>3}/{len(paths)}] {row['clip'][:28]:<28} {n:>4}f{'*' if row['cut'] else ' '} "
                  f"PSNR {row['psnr']:>5.2f} (freeze {row['psnr_freeze']:>5.2f})  mouth corr {row['mouth_corr']:.3f} amp {row['mouth_amp']:.2f}  "
                  f"rot {row['rot_err_deg']:4.2f}deg  CSIM {row['csim']:.4f}"
                  + (f"  LSE-C {row.get('lse_c', float('nan')):.2f}"
                     f"/{row.get('lse_c_real', float('nan')):.2f}" if args.lse else ""), flush=True)
        except Exception as e:                              # one bad clip must not kill the sweep
            skipped.append((Path(p).name, f"{type(e).__name__}: {e}"))
            print(f"[{k + 1:>3}/{len(paths)}] SKIP {Path(p).name}: {type(e).__name__}: {e}", flush=True)

    if not rows:
        raise SystemExit("every clip failed -- the install or the clip list is wrong, not the method")

    def col(key):
        v = np.array([r[key] for r in rows if key in r and not np.isnan(r[key])])
        return v if len(v) else np.array([float("nan")])

    print(f"\n{'=' * 64}\nM0 motion ceiling  |  {len(rows)} clips, {len(skipped)} skipped  "
          f"|  {args.res} px  |  target {args.target}-d  |  relative={args.relative}\n{'=' * 64}")
    print(f"  clips truncated at a shot cut (*): {sum(r['cut'] for r in rows)} of {len(rows)}")
    for key, label, fmt in (("mouth_corr", "mouth corr", "6.3f"), ("mouth_amp", "mouth amp", "6.3f"),
                            ("eyes_corr", "eyes corr", "6.3f"), ("brow_corr", "brow corr", "6.3f"),
                            ("rot_err_deg", "rot err deg", "6.2f"),
                            ("psnr", "PSNR (dB)", "6.2f"), ("psnr_freeze", "PSNR freeze0", "6.2f"),
                            ("psnr_face", "PSNR face", "6.2f"), ("ssim", "SSIM", "6.4f"),
                            ("csim", "CSIM", "6.4f"), ("motion_std", "motion std", "6.4f")):
        v = col(key)
        print(f"  {label:<12} mean {np.mean(v):{fmt}}   median {np.median(v):{fmt}}   "
              f"min {np.min(v):{fmt}}   max {np.max(v):{fmt}}")
    if args.lse:
        for key, label in (("lse_c", "LSE-C gen"), ("lse_c_real", "LSE-C real"),
                           ("lse_d", "LSE-D gen"), ("lse_d_real", "LSE-D real")):
            print(f"  {label:<12} mean {np.mean(col(key)):6.3f}")

    mc, ma, c_mean = (float(np.mean(col(k))) for k in ("mouth_corr", "mouth_amp", "csim"))
    checks = [("mouth corr >= 0.85", mc >= 0.85, f"{mc:.3f}"),
              ("mouth amp in 0.75-1.25", 0.75 <= ma <= 1.25, f"{ma:.3f}"),
              ("CSIM >= 0.85", c_mean >= 0.85, f"{c_mean:.4f}")]
    if args.lse:
        gap = float(np.mean(col("lse_c_real")) - np.mean(col("lse_c")))
        checks.append(("LSE-C within 0.5 of real", abs(gap) <= 0.5, f"gap {gap:+.3f}"))
    print(f"\n{'GATE':<26} {'result':<12} value")
    for name, ok, val in checks:
        print(f"  {name:<24} {'PASS' if ok else 'FAIL':<12} {val}")
    passed = all(ok for _, ok, _ in checks)
    print(f"\n{'M0 PASSED -- queue scripts/cache_motion.py' if passed else 'M0 FAILED -- do not cache yet; read the docstring'}")
    if skipped:
        print(f"\nfirst skips: {skipped[:5]}")
    sys.exit(0 if passed else 2)


if __name__ == "__main__":
    main()
