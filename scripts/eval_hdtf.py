#!/usr/bin/env python3
"""HDTF evaluation with the metrics SOTA talking-head papers report: FID, FVD-16, CSIM, LSE-C, LSE-D (plus PSNR / SSIM).

    # evaluate (GPU job; ~1 h for 75 clips)
    sbatch --time=16:00:00 bash_scripts/run.sh scripts/eval_hdtf.py \\
        --ckpt runs/l42_gft/last.pt --data /beegfs/work/$USER/HDTF --rows sang --real-stats --sampler g=2,steps=6 \\
        --name l42_gft
    # extra footage: a demo model sees the 30 s before a test segment that starts at 60 s
    sbatch ... scripts/eval_hdtf.py --ckpt runs/demo/last.pt ... --start 60 --demo 30 --name demo_s60_d30
    # the renderer ceiling when the real motion is squeezed through k codes of the motion latent
    sbatch ... scripts/eval_hdtf.py ... --rows gt_motion --gt-latent runs/latent/pca.pt --gt-k 42 --name lat_pca_k42

Protocol (also written to <out>/protocol.json):
  data      <data>/crops512/*.mp4 (commit 02791a9 scripts/download_hdtf.py): HDTF's official face crop
            (method 1), 512x512, 25 fps, 16 kHz audio.
  test set  --list configs/hdtf_test.csv: 75 clips, one per video (md5 per file, checked); SANG never trained on HDTF.
  per clip  --seconds (default 10 s) from --start (default 0 s); source = the segment's first frame; driving audio =
            the clip's own audio. With --start > 0 a clip too short for the whole segment is skipped.
  footage   --demo N (s, needs a demo model, N <= --start): the N s just before the segment are the person's extra
            footage, read once by the model (sang.render.pipeline.Talker.run demo=). The segment is the same for
            every N, so rows with different N score the same frames.
  rows      real       the ground-truth clip itself (LSE and CSIM only: the ceiling of each metric)
            gt_motion  the clip's real LivePortrait motion (--gt-latent: squeezed through the first --gt-k codes
                       of a motion latent) and re-rendered from frame 0 through the same renderer and
                       paste-back (the renderer ceiling: what perfect audio-to-motion would score)
            sang       SANG-M generated from audio, rendered with stitching, pasted back into
                       frame 0 with LivePortrait's crop mask so the output is in the GT's frame
            sang_fit   the same with the generator first fitted to the footage (--demo; test-time LoRA,
                       --gen-fit-steps / --gen-fit-lr / --gen-fit-kl; sang.model.critic.fit_generator). With the sang row
                       in the same run, both use the same seed and clips: the paired fit vs no-fit test
            ditto      Ditto (antgroup/ditto-talkinghead, ACM MM 2025) videos on the same clips (same source frame,
                       same audio; generated with commit 02791a9's scripts/ditto_hdtf.py), scored from --ditto-dir
                       through the identical metric code
  metrics   FID     Inception-v3 pool3 over all frames, generated vs ground truth, pooled over clips
            FVD     I3D (StyleGAN-V port), non-overlapping 16-frame clips, pooled over clips
            CSIM    ArcFace (buffalo_l) cosine, source frame vs every 5th output frame
            LSE-C/D joonson/syncnet_python, full face-detection pipeline (sang.eval.metrics.lse)
            PSNR / SSIM vs the ground-truth frames: audio-driven output is a different take, so
            these measure closeness to the real performance, not reconstruction quality
  motion    (sang row; ditto row with --real-stats, its motion read back from its video like the real clip's)
            blinks_per_min and lip_std of the generated motion; with --real-stats also the real clip's
            (real_blinks_per_min), lip_ccc / lip_corr against the real lip trajectory, and head rotation std and mean
            |frame-to-frame change| in degrees (mean of the 3 axes): rot_std / rot_vel, real_rot_std / real_rot_vel
  critic    --critic (scripts/train_critic.py; docs/person_critic_plan.md): critic_real / critic_sync, the person
            critic's mean logits on SANG's motion, and with --real-stats real_critic_real / real_critic_sync on the
            real clip's. --sampler dg=..,dgs=.. guides the sampler by it; --fit-steps N > 0 first fits it to the
            footage (LoRA, --fit-rank / --fit-blocks / --fit-lr), 0 = the critic as trained (the ablation).
            With --demo and --real-stats: copy_ratio, how close SANG's motion is to the footage over how close the
            real segment is (sang.eval.naturalness.copy_ratio; << 1 = replaying the footage).
            --motion-only: the sang row's motion statistics and critic scores only, no rendering or video metrics
            (the same seed gives the same motion as a full run, so its FVD carries over).
            --head-check S: real_critic_*_head_{all,slow,fast}, the real motion's critic scores with its head scaled by
            S about the photo's pose: all of it, only the drift slower than one 64-frame window, or only the rest.
--n-clips N stops after the first N clips that score (a quick paired test).
Resumable: per-clip results go to <out>/clips/<row>/<clip>.{json,npz}; a rerun skips them.
"""
import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "third_party"))

from sang.data.audio import load_wav

SR, FPS, HOP = 16000, 25, 640
ROWS = ("real", "gt_motion", "sang", "sang_fit", "ditto")
SANG_ROWS = ("sang", "sang_fit")
MOTION_KEYS = ("blinks_per_min", "real_blinks_per_min", "lip_std", "lip_corr", "lip_ccc", "rot_std", "real_rot_std",
               "rot_vel", "real_rot_vel", "critic_real", "real_critic_real", "critic_sync", "real_critic_sync",
               "copy_ratio", "critic_person", "real_critic_person") + tuple(
    f"real_critic_{k}_head_{p}" for p in ("all", "slow", "fast") for k in ("real", "sync", "person"))
METRICS = ("FID", "FVD", "CSIM", "LSE-C", "LSE-D", "PSNR", "SSIM")


def read_list(path: str, data: Path) -> list[Path]:
    """Clips of the frozen test-set CSV (crop_rel under --data); warns if a file is missing or changed (md5)."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    clips = [data / r["crop_rel"] for r in rows]
    changed = []
    for r, c in zip(rows, clips):
        if c.exists():
            h = hashlib.md5()
            with open(c, "rb") as fh:
                for block in iter(lambda: fh.read(1 << 20), b""):
                    h.update(block)
            if h.hexdigest() != r["md5"]:
                changed.append(c.stem)
    missing = [c.stem for c in clips if not c.exists()]
    if missing or changed:
        print(f"WARNING: test list {path}: {len(missing)} missing {missing[:5]}, {len(changed)} changed since "
              f"it was frozen {changed[:5]}", flush=True)
    return clips


class Evaluator:
    def __init__(self, args):
        from sang.data.dataset import load_cfg
        from sang.eval.metrics import ArcFace, I3DFeats, InceptionFeats
        from sang.eval.naturalness import load_readouts
        from sang.render.liveportrait import MotionCodec
        self.args, self.dev = args, "cuda"
        self.codec = MotionCodec(device=self.dev)
        self.arc = ArcFace(device=self.dev)
        self.incep, self.i3d = InceptionFeats(self.dev), I3DFeats(self.dev)
        self.last_t, self.last_stats, self.read = {}, {}, None
        if set(SANG_ROWS) & set(args.rows) or ("ditto" in args.rows and args.real_stats):
            cache = Path(load_cfg(str(REPO / "configs/train_motion.yaml"), [])["cache_dir"])
            try:
                self.read = load_readouts(cache / "openness.json")
            except OSError as e:
                if args.real_stats:
                    sys.exit(f"--real-stats needs the openness readouts: {e}")
                print(f"no openness readouts ({e}): no blink / lip stats", flush=True)
        if set(SANG_ROWS) & set(args.rows):
            from sang.render.pipeline import Talker
            fit = dict(steps=args.fit_steps, rank=args.fit_rank, blocks=args.fit_blocks, lr=args.fit_lr)
            gen_fit = dict(steps=args.gen_fit_steps, lr=args.gen_fit_lr, kl=args.gen_fit_kl)
            self.talker = Talker(args.ckpt, args.sampler, self.dev, codec=self.codec, critic=args.critic, fit=fit,
                                 gen_fit=gen_fit)
            if "sang_fit" in args.rows:
                print(f"generator person fit {gen_fit}", flush=True)
            if args.critic:
                print(f"person critic {args.critic}, test-time fit {fit if args.fit_steps else 'off'}", flush=True)
            print(f"SANG {args.ckpt} (step {self.talker.step}), sampler {self.talker.sampler}", flush=True)
        if args.gt_latent:
            from sang.model.latent import MotionLatent
            self.gt_latent = MotionLatent.unpack(torch.load(args.gt_latent, map_location="cpu",
                                                            weights_only=False)).to(self.dev)

    def gen_sang(self, frames, wav, demo=None, fit_gen=False):
        """sang.render.pipeline.Talker; the stage times (source, demo, audio, motion, render, paste; CUDA-synchronised)
        go into the clip's json."""
        a = self.args
        real = self.real_motion(frames) if a.real_stats else None
        gen, m, timer = self.talker.run(frames[0], wav, len(frames), seed=a.seed, demo=demo, render=not a.motion_only,
                                        gen_demo=round(a.demo * FPS) if a.critic_demo > a.demo else None,
                                        fit_gen=fit_gen)
        self.last_t = {f"t_{k}": v for k, v in timer.t.items()}
        if demo is not None:
            self.last_t["demo_frames"] = self.talker.demo_frames
        self.last_stats = {**self.motion_stats(m, real), **self.talker.last_critic}
        if real is not None and real["span"][0] == 0:  # the critic needs the real motion aligned with the audio
            self.last_stats.update(self.talker.score_motion(real["m"]))
            for part in ("all", "slow", "fast") if a.head_check is not None else ():
                sc = self.talker.score_motion(real["m"], (a.head_check, part))
                self.last_stats.update({f"{k}_head_{part}": v for k, v in sc.items()})
        if real is not None and self.talker.demo_motion is not None:
            from sang.eval.naturalness import copy_ratio
            from sang.model.target import to_target
            self.last_stats["copy_ratio"] = copy_ratio(to_target(m.float().cpu()), real["y"],
                                                       to_target(self.talker.demo_motion.float().cpu()))
        return gen, frames

    def real_motion(self, frames) -> dict:
        """The real clip's 42-d target, lip and eye openness (from its LivePortrait motion), and the frames
        they cover."""
        from sang.model.target import to_target
        d = self.codec.extract(frames)
        y = to_target(d["m"].float())
        return {"y": y, "m": d["m"], "lip": self.read["lip"](y).cpu().numpy(), "eye": self.read["eye"](y).cpu().numpy(),
                "span": d["span"]}

    def motion_stats(self, m, real=None) -> dict:
        """Blinks and lip opening of the generated motion [n, 70]; against the real clip's when given."""
        if self.read is None:
            return {}
        from sang.eval.naturalness import blink_events, ccc
        from sang.model.target import to_target
        y = to_target(m.float())
        lip, eye = self.read["lip"](y).cpu().numpy(), self.read["eye"](y).cpu().numpy()
        out = {"blinks_per_min": len(blink_events(eye)) * 60.0 * FPS / len(eye), "lip_std": float(lip.std())}
        if real is not None:
            s0, r = real["span"][0], real["lip"]
            g = lip[s0:s0 + len(r)]
            k = len(g)
            r = r[:k]
            rot = lambda v: (float(v.std(0).mean()), float(np.abs(np.diff(v, axis=0)).mean()))
            (s_1, v_1), (s_r, v_r) = rot(y[s0:s0 + k, :3].cpu().numpy()), rot(real["y"][:k, :3].numpy())
            out.update(real_blinks_per_min=len(blink_events(real["eye"])) * 60.0 * FPS / len(real["eye"]),
                       lip_ccc=ccc(g, r), lip_corr=float(np.corrcoef(g, r)[0, 1]),
                       rot_std=s_1, real_rot_std=s_r, rot_vel=v_1, real_rot_vel=v_r)
        return out

    def gen_gt_motion(self, frames, wav):
        from sang.render.pipeline import GpuPaster
        a = self.args
        d = self.codec.extract(frames)
        m = d["m"].float().to(self.dev)
        if a.gt_latent:
            m = self.gt_latent.to_motion(self.gt_latent.from_motion(m, m[:1], a.gt_k), m[:1])
        crops = self.codec.render(self.codec.source_state(d["crops"][0]), m.cpu())
        s0 = d["span"][0]
        gt = frames[s0:s0 + len(crops)]
        return GpuPaster(d["M_c2o"], gt[0], crops.shape[1], device=self.dev)(crops), gt

    def score(self, row: str, clip: Path, out_dir: Path, gen, gt, src, start: int = 0):
        from sang.data.audio import save_wav
        from sang.eval.metrics import csim, lse, psnr, ssim, write_mp4
        n = min(len(gen), len(gt))
        gen, gt = gen[:n], gt[:n]
        res = {"clip": clip.stem, "frames": n, "csim": csim(self.arc, src, gen, stride=5)}
        feats = {}
        if row != "real":
            res["psnr"] = psnr(gt, gen)
            res["ssim"] = ssim(gt[::5], gen[::5])
            feats = {"incep_gen": self.incep(gen), "incep_gt": self.incep(gt), "i3d_gen": self.i3d(gen),
                     "i3d_gt": self.i3d(gt)}
        vid = out_dir / "videos" / row / f"{clip.stem}.mp4"
        vid.parent.mkdir(parents=True, exist_ok=True)
        tmp_wav = vid.with_suffix(".wav")
        save_wav(load_wav(clip, start, n), tmp_wav)  # the segment's own audio (--start)
        write_mp4(gen, vid, fps=FPS, audio=tmp_wav)
        tmp_wav.unlink(missing_ok=True)
        try:
            res["lse_offset"], res["lse_d"], res["lse_c"] = lse(vid)
        except Exception as e:
            print(f"  [lse] {clip.stem} {row}: {type(e).__name__}: {e}", flush=True)
            res["lse_offset"] = res["lse_d"] = res["lse_c"] = float("nan")
        if not self.args.save_videos:
            vid.unlink(missing_ok=True)
        return res, feats


def run(args) -> None:
    out = Path(args.out or REPO / "results" / "hdtf" / args.name)
    (out / "clips").mkdir(parents=True, exist_ok=True)
    clips = read_list(args.list, Path(args.data))
    if not clips:
        sys.exit(f"no clips in {args.list}")
    (out / "test_list.txt").write_text("".join(f"{c.stem}\n" for c in clips))
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True,
                            encoding="utf-8", errors="replace").stdout.strip()
    (out / "protocol.json").write_text(json.dumps({**vars(args), "n_clips": len(clips), "commit": commit}, indent=1, default=str))
    print(f"HDTF eval '{args.name}': {len(clips)} clips x {args.seconds:g} s, rows {args.rows} -> {out}", flush=True)

    ev = Evaluator(args)
    from sang.render.liveportrait import decode_clip
    t0, scored = time.time(), 0
    for i, clip in enumerate(clips, 1):
        if args.n_clips and scored >= args.n_clips:
            break
        todo = [r for r in args.rows if not (out / "clips" / r / f"{clip.stem}.json").exists()]
        if not todo:
            scored += 1
            continue
        try:
            s0, n_test = round(args.start * FPS), round(args.seconds * FPS)
            n_demo = round(max(args.demo, args.critic_demo) * FPS)  # the critic may see more than the generator
            frames = decode_clip(str(clip), n_test, start=s0)
            if s0 and len(frames) < n_test:
                raise ValueError(f"only {len(frames) / FPS:.1f} s after {args.start:g} s")
            wav = load_wav(clip, s0, len(frames))
            demo = None
            if n_demo and set(SANG_ROWS) & set(todo):
                dfr = decode_clip(str(clip), n_demo, start=s0 - n_demo)
                demo = (dfr, load_wav(clip, s0 - n_demo, len(dfr)))
            for row in todo:
                if row == "real":
                    gen, gt = frames, frames
                elif row == "gt_motion":
                    gen, gt = ev.gen_gt_motion(frames, wav)
                elif row == "ditto":
                    vid = Path(args.ditto_dir) / f"{clip.stem}.mp4"
                    if not vid.exists():
                        raise FileNotFoundError(f"{vid}: no Ditto video")
                    gen = decode_clip(str(vid), len(frames))
                    gt = frames[: len(gen)]
                    # Ditto's motion read back from its video, as the real clip's is
                    ev.last_stats = (ev.motion_stats(ev.codec.extract(gen)["m"], ev.real_motion(frames))
                                     if args.real_stats else {})
                else:
                    gen, gt = ev.gen_sang(frames, wav, demo, fit_gen=row == "sang_fit")
                if args.motion_only:
                    res, feats = {"clip": clip.stem, "frames": len(frames)}, {}
                else:
                    res, feats = ev.score(row, clip, out, gen, gt, frames[0], s0)
                if row in SANG_ROWS:
                    res.update(ev.last_t)
                if row in (*SANG_ROWS, "ditto"):
                    res.update(ev.last_stats)
                d = out / "clips" / row
                d.mkdir(parents=True, exist_ok=True)
                if feats:
                    np.savez_compressed(d / f"{clip.stem}.npz", **feats)
                (d / f"{clip.stem}.json").write_text(json.dumps(res))
                msg = "  ".join(f"{k} {v:.3f}" for k, v in res.items() if isinstance(v, float))
                print(f"[{i}/{len(clips)}] {clip.stem:<28} {row:<9} {msg}", flush=True)
            scored += 1
        except Exception as e:
            print(f"[{i}/{len(clips)}] SKIP {clip.stem}: {type(e).__name__}: {e}", flush=True)
    print(f"clips done in {time.time() - t0:.0f} s", flush=True)
    summarise(out, args)


def label(row: str, args) -> str:
    if row in SANG_ROWS:
        fit = f", person fit {args.gen_fit_steps} steps" if row == "sang_fit" else ""
        return f"SANG-M ({args.name}: {args.sampler or 'run defaults'}{fit})"
    if row == "gt_motion":
        return "Real motion → LivePortrait (ceiling)" + (f" through {args.gt_k} codes" if args.gt_latent else "")
    return {"real": "Real video", "ditto": "Ditto"}[row]


def markdown(rows: dict, args) -> str:
    fmt = lambda v: "–" if v is None else f"{v:.3f}"
    lines = ["| Method | n | " + " | ".join(METRICS) + " |", "|---" * (len(METRICS) + 2) + "|"]
    lines += [f"| {label(row, args)} | {r['n_clips']} | " + " | ".join(fmt(r["m"].get(k)) for k in METRICS) + " |"
              for row, r in rows.items()]
    return "\n".join(lines)


def summarise(out: Path, args) -> dict:
    from sang.eval.metrics import frechet
    rows = {}
    for row in ROWS:
        d = out / "clips" / row
        js = [json.loads(p.read_text()) for p in sorted(d.glob("*.json"))] if d.exists() else []
        if not js:
            continue
        mean = lambda k: float(np.nanmean([j[k] for j in js if k in j])) if any(k in j for j in js) else None
        m = {"CSIM": mean("csim"), "LSE-C": mean("lse_c"), "LSE-D": mean("lse_d"), "PSNR": mean("psnr"), "SSIM": mean("ssim")}
        npz = [np.load(p) for p in sorted(d.glob("*.npz"))]
        if npz:
            cat = lambda k: np.concatenate([z[k] for z in npz])
            m["FID"] = frechet(cat("incep_gt"), cat("incep_gen"))
            m["FVD"] = frechet(cat("i3d_gt"), cat("i3d_gen"))
        rows[row] = {"n_clips": len(js), "m": {k: v for k, v in m.items() if v is not None}}
        mot = {k: mean(k) for k in MOTION_KEYS if mean(k) is not None}
        if mot:
            rows[row]["motion"] = mot
        tj = [j for j in js if "t_motion" in j]
        if tj:
            stages = sorted({k for j in tj for k in j if k.startswith("t_")})
            tot = [sum(j[k] for k in stages if k in j) for j in tj]
            rows[row]["timing"] = {"clips": len(tj), **{k: float(np.median([j[k] for j in tj if k in j])) for k in stages},
                                   "total_s": float(np.median(tot)),
                                   "fps": float(np.median([j["frames"] / t for j, t in zip(tj, tot)]))}
    summary = {"name": args.name, "rows": rows, "protocol": json.loads((out / "protocol.json").read_text())}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    md = markdown(rows, args)
    (out / "summary.md").write_text(md + "\n")
    print("\n" + md, flush=True)
    for row, r in rows.items():
        if "motion" in r:
            print(f"\n{row} motion, mean per clip: " + "  ".join(f"{k} {v:.3f}" for k, v in r["motion"].items()), flush=True)
        if "timing" in r:
            t = r["timing"]
            print(f"\n{row} generation, median per {args.seconds:g} s clip over {t['clips']} clips: "
                  + "  ".join(f"{k[2:]} {v:.2f} s" for k, v in t.items() if k.startswith("t_"))
                  + f"  | total {t['total_s']:.2f} s = {t['fps']:.1f} frames/s (rendering and paste-back included; "
                    f"scoring and the mp4 not)", flush=True)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=str(REPO / "runs/l42_gft/last.pt"))
    ap.add_argument("--data", required=True, help="HDTF root (contains crops512/)")
    ap.add_argument("--name", default="sang", help="run name -> results/hdtf/<name>")
    ap.add_argument("--out", default=None)
    ap.add_argument("--list", default=str(REPO / "configs/hdtf_test.csv"), help="the frozen test-set CSV")
    ap.add_argument("--ditto-dir", default=str(REPO / "results" / "hdtf" / "ditto_videos"),
                    help="Ditto's videos for the ditto row")
    ap.add_argument("--seconds", type=float, default=10.0, help="evaluated length per clip")
    ap.add_argument("--start", type=float, default=0.0, help="where the evaluated segment starts (s)")
    ap.add_argument("--demo", type=float, default=0.0,
                    help="seconds of footage just before --start given to a demo model (0 = none)")
    ap.add_argument("--rows", nargs="+", default=["real", "gt_motion", "sang"], choices=ROWS,
                    help="ditto needs --ditto-dir, so it is opt-in")
    ap.add_argument("--sampler", default=None, help="sampler spec, e.g. 'g=2,steps=6' "
                                                    "(sang.model.flow.SAMPLER_KEYS); default: the run's guidance and steps")
    ap.add_argument("--real-stats", action="store_true",
                    help="also extract each clip's real motion for the lip / blink / head-rotation stats (~10 s a clip)")
    ap.add_argument("--gt-latent", default=None, help="ceiling row through a motion latent (scripts/fit_latent.py)")
    ap.add_argument("--gt-k", type=int, default=None, help="--gt-latent: codes kept")
    ap.add_argument("--save-videos", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--critic", default=None, help="person critic (scripts/train_critic.py): scores, dg / dgs guidance")
    ap.add_argument("--critic-demo", type=float, default=None,
                    help="seconds of footage the critic sees (default --demo); the generator still sees --demo")
    ap.add_argument("--fit-steps", type=int, default=0, help="test-time LoRA fit of the critic to the footage "
                                                              "(0 = off, the ablation)")
    ap.add_argument("--fit-rank", type=int, default=4)
    ap.add_argument("--fit-blocks", type=int, default=1, help="last trunk blocks that get LoRA (plus the head)")
    ap.add_argument("--fit-lr", type=float, default=1e-3)
    ap.add_argument("--gen-fit-steps", type=int, default=20, help="sang_fit row: steps of the generator's person fit")
    ap.add_argument("--gen-fit-lr", type=float, default=1e-3)
    ap.add_argument("--gen-fit-kl", type=float, default=1.0, help="sang_fit row: path-KL weight to the unadapted model")
    ap.add_argument("--n-clips", type=int, default=0, help="stop after this many scored clips (0 = all)")
    ap.add_argument("--motion-only", action="store_true", help="sang row motion stats and critic scores only")
    ap.add_argument("--head-check", type=float, default=None,
                    help="also score the real motion with its head scaled by this about the photo's pose: all of it, "
                         "only the drift slower than one critic window, only the rest (does the critic see head range?)")
    args = ap.parse_args()
    if args.head_check is not None and not (args.critic and args.real_stats):
        sys.exit("--head-check needs --critic and --real-stats")
    if args.motion_only and not set(args.rows) <= set(SANG_ROWS):
        sys.exit("--motion-only scores the sang rows only: pass --rows sang (and sang_fit)")
    if "sang_fit" in args.rows and not args.demo:
        sys.exit("the sang_fit row fits the generator to the footage: pass --demo")
    args.critic_demo = args.demo if args.critic_demo is None else args.critic_demo
    if args.critic_demo != args.demo and not args.critic:
        sys.exit("--critic-demo needs --critic")
    if args.fit_steps and not (args.critic and args.critic_demo):
        sys.exit("--fit-steps needs --critic and footage (--demo or --critic-demo): the fit uses the footage")
    if args.gt_latent and not args.gt_k:
        sys.exit("--gt-latent needs --gt-k")
    if max(args.demo, args.critic_demo) > args.start:
        sys.exit(f"--demo / --critic-demo need --start >= the footage length: the footage comes before the segment")
    if args.start and "ditto" in args.rows:
        sys.exit("the Ditto videos cover the first seconds of each clip only: no ditto row with --start")
    run(args)


if __name__ == "__main__":
    main()
