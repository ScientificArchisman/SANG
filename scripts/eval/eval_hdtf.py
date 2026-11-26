#!/usr/bin/env python3
"""HDTF evaluation with the metrics SOTA talking-head papers report: FID, FVD-16, CSIM, LSE-C, LSE-D (plus PSNR / SSIM).

    # evaluate (GPU job; ~1 h for 75 clips)
    sbatch --time=16:00:00 bash_scripts/run.sh scripts/eval/eval_hdtf.py \\
        --ckpt runs/l42_gft/last.pt --data /beegfs/work/$USER/HDTF --rows sang --real-stats --sampler g=2,steps=6 \\
        --name l42_gft
    # extra footage: a demo model sees the 30 s before a test segment that starts at 60 s
    sbatch ... scripts/eval/eval_hdtf.py --ckpt runs/demo/last.pt ... --start 60 --demo 30 --name demo_s60_d30
    # the renderer ceiling when the real motion is squeezed through k codes of the motion latent
    sbatch ... scripts/eval/eval_hdtf.py ... --rows gt_motion --gt-latent runs/latent/pca.pt --gt-k 42 --name lat_pca_k42

Protocol (also written to <out>/protocol.json):
  data      <data>/crops512/*.mp4 (scripts/data/download_hdtf.py): HDTF's official face crop
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
            sang       SANG generated from audio, rendered with stitching, pasted back into
                       frame 0 with LivePortrait's crop mask so the output is in the GT's frame
            ditto      Ditto (antgroup/ditto-talkinghead, ACM MM 2025) videos on the same clips (same source frame,
                       same audio; generated with scripts/eval/ditto_hdtf.py), scored from --ditto-dir
                       through the identical metric code
  metrics   FID     Inception-v3 pool3 over all frames, generated vs ground truth, pooled over clips
            FVD     I3D (StyleGAN-V port), non-overlapping 16-frame clips, pooled over clips
            CSIM    ArcFace (buffalo_l) cosine, source frame vs every 5th output frame
            LSE-C/D joonson/syncnet_python, full face-detection pipeline (sang.data.sync.lse)
            PSNR / SSIM vs the ground-truth frames: audio-driven output is a different take, so
            these measure closeness to the real performance, not reconstruction quality
  motion    (sang row; ditto row with --real-stats, its motion read back from its video like the real clip's)
            blinks_per_min and lip_std of the generated motion; with --real-stats also the real clip's
            (real_blinks_per_min), lip_ccc / lip_corr against the real lip trajectory, and head rotation std and mean
            |frame-to-frame change| in degrees (mean of the 3 axes): rot_std / rot_vel, real_rot_std / real_rot_vel
            With --demo and --real-stats: copy_ratio, how close SANG's motion is to the footage over how close the
            real segment is (sang.eval.naturalness.copy_ratio; << 1 = replaying the footage), and the footage's own
            head rotation std / change and blink rate (demo_*: does the footage predict the segment? scripts/eval/paired.py).
            --motion-only: the sang row's motion statistics only, no rendering or video metrics (the same seed gives
            the same motion as a full run, so its FVD carries over).
  renderer  --calib own gated swap oracle (with --demo): pass 1 fits shape-locked SPADE offsets on LivePortrait's
            decoder per clip (sang.render.calibrate) to <out>/calib/<clip>.pt, own on the footage re-rendered from the
            test photo, oracle on the test segment itself (a diagnostic). The first 5 clips run the pre-flight first
            (noise and blur must raise the loss) and the run stops if it fails. Pass 2 adds rows <row>+<arm> for
            gt_motion and sang: the same motion (the same seed) rendered with the calibrated decoder. gated copies
            +own where the held-out footage loss fell (cal_accept), else the frozen row; swap uses the next clip's
            own offsets (is the gain personal?). Compare rows with scripts/eval/paired.py run:row.
  background --bg (reports/SANG static background control.md, phase 0): the gt_motion and sang rows are rendered
            with LivePortrait's dense motion probed (sang.eval.background): BG-Drag, the background's DIS-flow
            displacement in px from the zero-motion render (real clip: from its first frame) on pixels the paste
            covers, the real clip shows still, no generated head covers and that have texture; bg_real_* the real
            clip on the same pixels (the floor); BG-Def / Leak / Occ the same read off the renderer's deformation,
            slot weights and occlusion map. --bg-arms adds rows <row>+<arm> with the same motion: expr (pose, scale and
            translation held at the photo's), pose (expression held), noscale, notrans (attribution), lock (identity
            flow forced where the real clip is still and no head is: the oracle lock).
--n-clips N stops after the first N clips that score (a quick paired test).
Resumable: per-clip results go to <out>/clips/<row>/<clip>.{json,npz}; a rerun skips them.
"""
import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "third_party"))

from sang.data.audio import load_wav

SR, FPS, HOP = 16000, 25, 640
ROWS = ("real", "gt_motion", "sang", "ditto")
ARMS = ("own", "gated", "swap", "oracle")
BG_ARMS = ("expr", "pose", "noscale", "notrans", "lock")
HOLD = {"expr": slice(0, 7), "pose": slice(7, None), "noscale": slice(0, 1), "notrans": slice(4, 7)}  # held at the photo
BG_KEYS = tuple(f"bg_{k}" for k in ("px", "offset", "drag", "drag95", "drag_near", "jitter", "reject", "real_drag",
                                     "real_drag95", "real_drag_near", "real_jitter", "real_reject", "def", "leak", "occ"))
MOTION_KEYS = ("blinks_per_min", "real_blinks_per_min", "lip_std", "lip_corr", "lip_ccc", "rot_std", "real_rot_std",
               "rot_vel", "real_rot_vel", "copy_ratio", "demo_rot_std", "demo_rot_vel", "demo_blinks_per_min",
               "pre_ok", "cal_accept", "cal_held0", "cal_held", "cal_theta", "cal_s", "gated_own", "jump", "jump_rot",
               "real_jump", "real_jump_rot") + BG_KEYS
METRICS = ("FID", "FVD", "CSIM", "LSE-C", "LSE-D", "PSNR", "SSIM")


def jump(y: np.ndarray, cuts: list) -> float:
    """Mean |y[c] - y[c-1]| at the window starts `cuts` over the mean frame-to-frame step elsewhere: 1 = the chain
    joins windows as smoothly as frames move inside them (the real clip, at the same frames, is the noise floor)."""
    d = np.linalg.norm(np.diff(y, axis=0), axis=1)
    b = np.array([c - 1 for c in cuts if 1 <= c < len(y)], int)
    if not len(b) or len(b) == len(d):
        return float("nan")
    return float(d[b].mean() / np.delete(d, b).mean())


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
        self.last_t, self.last_stats, self.last_bg, self.read = {}, {}, {}, None
        if "sang" in args.rows or ("ditto" in args.rows and args.real_stats):
            cache = Path(load_cfg(str(REPO / "configs/train_motion.yaml"), [])["cache_dir"])
            try:
                self.read = load_readouts(cache / "openness.json")
            except OSError as e:
                if args.real_stats:
                    sys.exit(f"--real-stats needs the openness readouts: {e}")
                print(f"no openness readouts ({e}): no blink / lip stats", flush=True)
        if "sang" in args.rows:
            from sang.render.pipeline import Talker
            self.talker = Talker(args.ckpt, args.sampler, self.dev, codec=self.codec)
            print(f"SANG {args.ckpt} (step {self.talker.step}), sampler {self.talker.sampler}", flush=True)
        if args.gt_latent:
            from sang.model.latent import MotionLatent
            self.gt_latent = MotionLatent.unpack(torch.load(args.gt_latent, map_location="cpu",
                                                            weights_only=False)).to(self.dev)

    def gen_sang(self, frames, wav, demo=None, stats=True, arm=""):
        """sang.render.pipeline.Talker; the stage times (source, demo, audio, motion, render, paste; CUDA-synchronised)
        go into the clip's json. stats=False skips the motion statistics (a calibrated row: the same motion).
        With --bg the motion is rendered by render_bg (the same calls as Talker.run, probed; arm: a BG arm)."""
        a = self.args
        real = self.real_motion(frames) if a.real_stats and stats else None
        gen, m, timer = self.talker.run(frames[0], wav, len(frames), seed=a.seed, demo=demo,
                                        render=not (a.motion_only or a.bg))
        if a.bg and not a.motion_only:
            crop = self.codec.crop_image(np.ascontiguousarray(frames[0]))
            gen = self.render_bg(self.codec.source_state(crop["img_crop_256x256"]), crop["M_c2o"], frames[0], m,
                                 self.codec.source_motion(frames[0], crop=crop)["m"], frames, arm)
        self.last_t = {f"t_{k}": v for k, v in timer.t.items()}
        if demo is not None:
            self.last_t["demo_frames"] = self.talker.demo_frames
        self.last_stats = {}
        if not stats:
            return gen, frames
        self.last_stats = self.motion_stats(m, real)
        if real is not None and self.talker.demo_motion is not None:
            from sang.eval.naturalness import copy_ratio
            from sang.model.target import to_target
            self.last_stats["copy_ratio"] = copy_ratio(to_target(m.float().cpu()), real["y"],
                                                       to_target(self.talker.demo_motion.float().cpu()))
        if self.talker.demo_motion is not None:
            from sang.model.target import to_target
            fy = to_target(self.talker.demo_motion.float().cpu())[:, :3].numpy()
            self.last_stats.update(demo_rot_std=float(fy.std(0).mean()), demo_rot_vel=float(np.abs(np.diff(fy, axis=0)).mean()))
            fs = self.motion_stats(self.talker.demo_motion)
            if "blinks_per_min" in fs:
                self.last_stats["demo_blinks_per_min"] = fs["blinks_per_min"]
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
        W = self.talker.cfg["frames"] if hasattr(self, "talker") else 0
        cuts = list(range(W, len(y), W)) if W else []  # flow.generate starts a window every W frames
        yn = y.cpu().numpy()
        if cuts:
            out.update(jump=jump(yn, cuts), jump_rot=jump(yn[:, :3], cuts))
        if real is not None:
            s0, r = real["span"][0], real["lip"]
            g = lip[s0:s0 + len(r)]
            k = len(g)
            r = r[:k]
            rot = lambda v: (float(v.std(0).mean()), float(np.abs(np.diff(v, axis=0)).mean()))
            (s_1, v_1), (s_r, v_r) = rot(y[s0:s0 + k, :3].cpu().numpy()), rot(real["y"][:k, :3].numpy())
            if cuts:
                ry = real["y"].numpy()
                out.update(real_jump=jump(ry, [c - s0 for c in cuts]), real_jump_rot=jump(ry[:, :3], [c - s0 for c in cuts]))
            out.update(real_blinks_per_min=len(blink_events(real["eye"])) * 60.0 * FPS / len(real["eye"]),
                       lip_ccc=ccc(g, r), lip_corr=float(np.corrcoef(g, r)[0, 1]),
                       rot_std=s_1, real_rot_std=s_r, rot_vel=v_1, real_rot_vel=v_r)
        return out

    def calibrate(self, frames, footage, arms, preflight: bool) -> dict:
        """One clip's decoder offsets (sang.render.calibrate), both fitted from the test photo as SANG renders it: own
        on the footage (every 4th frame, the last 5 s held out), oracle on the test segment (every 2nd frame).
        preflight: return before any fit if the noise / blur check fails."""
        from sang.render import calibrate as cal
        from sang.render.pipeline import crop_mask
        codec, lp = self.codec, self.codec.lp
        G, fx = lp.spade_generator, lp.appearance_feature_extractor
        lp.motion_extractor.requires_grad_(False)
        kw = dict(stage_w=self.args.calib_stages)
        state = codec.source_state(codec.crop_image(np.ascontiguousarray(frames[0]))["img_crop_256x256"])
        with torch.no_grad():
            anchor = codec.warped(state, state["x_s"]).float()
        kp_of = lambda img: cal.keypoints(lp, img)
        d = codec.extract(np.ascontiguousarray(footage))
        a, b = d["span"]
        T, hold = len(d["m"]), 5 * FPS
        if T - hold < 32:
            raise ValueError(f"only {T} footage frames before a shot cut")
        idx = np.arange(0, T, 4)
        src = (codec, state, footage[a:b], d["m"], d["M_c2o"].numpy())
        train, held = cal.pairs(*src, idx[idx < T - hold]), cal.pairs(*src, idx[idx >= T - hold])
        w = torch.from_numpy(crop_mask(train["Y"].shape[-1])).to(self.dev)
        log = cal.check(G, fx, held, w, **kw)
        if preflight and not log["pre_ok"]:
            return {"log": log}
        out = {}
        out["own"], fl = cal.fit(G, fx, kp_of, train, held, anchor, w, **kw)
        log |= fl
        del train, held
        if "oracle" in arms:
            s = codec.extract(np.ascontiguousarray(frames))
            a, b = s["span"]
            seg = cal.pairs(codec, state, frames[a:b], s["m"], s["M_c2o"].numpy(), np.arange(0, len(s["m"]), 2))
            out["oracle"], fl = cal.fit(G, fx, kp_of, seg, None, anchor, w, **kw)
            log |= {f"oracle_{k}": v for k, v in fl.items()}
        return out | {"log": log}

    def gen_gt_motion(self, frames, wav, arm=""):
        from sang.render.pipeline import GpuPaster
        a = self.args
        d = self.codec.extract(frames)
        m = d["m"].float().to(self.dev)
        if a.gt_latent:
            m = self.gt_latent.to_motion(self.gt_latent.from_motion(m, m[:1], a.gt_k), m[:1])
        s0 = d["span"][0]
        gt = frames[s0:s0 + len(m)]
        state = self.codec.source_state(d["crops"][0])
        if a.bg:
            return self.render_bg(state, d["M_c2o"], gt[0], m, m[:1], gt, arm), gt
        crops = self.codec.render(state, m.cpu())
        return GpuPaster(d["M_c2o"], gt[0], crops.shape[1], device=self.dev)(crops), gt

    def render_bg(self, state, M_c2o, photo, m, m0, real, arm="") -> np.ndarray:
        """Absolute motion m [n,70] rendered as Talker.run renders it (FP16, 16 frames a call, pasted into photo with
        M_c2o), with the dense motion probed; sets self.last_bg (sang.eval.background). m0 [1,70], the photo's own
        motion, is the anchor render and what the attribution arms hold (HOLD); lock forces the identity flow, per
        frame, where the real clip is still and no head envelope (x_s and x_d hull) is: the oracle lock."""
        import cv2
        from sang.eval import background as bgm
        from sang.render.liveportrait import as3x3
        from sang.render.pipeline import GpuPaster, crop_mask
        codec, S = self.codec, 512
        m, m0 = m.float().cpu().clone(), m0.float().cpu().reshape(1, -1)
        if arm in HOLD:
            m[:, HOLD[arm]] = m0[:, HOLD[arm]]
        M = as3x3(M_c2o)[:2]
        H, W = photo.shape[:2]
        px = lambda x: ((x[..., :2] + 1) / 2 * (S - 1)).float().cpu().numpy()  # grid coords -> crop px
        x_d = px(codec.driving_keypoints(state, m))
        heads = bgm.head_masks(np.concatenate([np.repeat(px(state["x_s"]), len(x_d), 0), x_d], 1), S)
        still = (crop_mask(S) >= 0.5) & cv2.warpAffine(bgm.still_mask(real).astype(np.uint8), M, (S, S),
                                                       flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP).astype(bool)
        head = heads.any(0)
        B64 = bgm.cells(still & ~head).to(self.dev)
        ref = {}
        paster = GpuPaster(M_c2o, photo, S, device=self.dev)
        anchor = paster(torch.cat(list(codec.render_iter(state, m0, half=True, probe=lambda o, i: ref.update(o)))))[0]
        w, rows = state["f_s"][0].abs().float().mean(0), []
        lock = (bgm.cells(still[None] & ~heads), ref["occlusion_map"]) if arm == "lock" else None
        gen = np.concatenate([paster(b) for b in codec.render_iter(
            state, m, batch=16, half=True, probe=lambda o, i: rows.append(bgm.def_stats(o, ref, B64, w)), lock=lock)])
        foot = np.zeros((H, W), bool)
        if not paster.empty:
            foot[paster.y0:paster.y1, paster.x0:paster.x1] = paster.mask[0, 0].cpu().numpy() >= 0.5
        head_f = cv2.warpAffine(head.astype(np.uint8), M, (W, H), flags=cv2.INTER_NEAREST).astype(bool)
        self.last_bg = bgm.bg_metrics(gen, real, anchor, foot, head_f, self.args.bg_texture)
        if B64.sum() > 0:
            self.last_bg |= {f"bg_{k}": float(torch.cat([r[k] for r in rows]).mean()) for k in rows[0]}
        return gen

    def score(self, row: str, clip: Path, out_dir: Path, gen, gt, src, start: int = 0):
        from sang.data.audio import save_wav
        from sang.data.sync import lse
        from sang.data.video import write_mp4
        from sang.eval.metrics import csim, psnr, ssim
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
        save_wav(load_wav(clip, start, n), tmp_wav)
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
    cal_dir = out / "calib"
    fitted = fit_all(ev, args, clips, cal_dir) if args.calib else []
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
            n_demo = round(args.demo * FPS)
            frames = decode_clip(str(clip), n_test, start=s0)
            if s0 and len(frames) < n_test:
                raise ValueError(f"only {len(frames) / FPS:.1f} s after {args.start:g} s")
            wav = load_wav(clip, s0, len(frames))
            demo = None
            if n_demo and any(r.partition("+")[0] == "sang" for r in todo):
                dfr = decode_clip(str(clip), n_demo, start=s0 - n_demo)
                demo = (dfr, load_wav(clip, s0 - n_demo, len(dfr)))
            calib = None
            if args.calib:
                if clip.stem not in fitted:
                    raise ValueError("no renderer calibration (see its [calib] line)")
                calib = torch.load(cal_dir / f"{clip.stem}.pt")
                calib["swap"] = torch.load(cal_dir / f"{fitted[(fitted.index(clip.stem) + 1) % len(fitted)]}.pt")["own"]
            rows_of_clip(ev, args, clip, out, i, len(clips), todo, frames, wav, demo, s0, calib)
            scored += 1
        except Exception as e:
            print(f"[{i}/{len(clips)}] SKIP {clip.stem}: {type(e).__name__}: {e}", flush=True)
    print(f"clips done in {time.time() - t0:.0f} s", flush=True)
    summarise(out, args)


def fit_all(ev, args, clips, cal_dir) -> list[str]:
    """--calib pass 1: every clip's offsets to <out>/calib/<clip>.pt before any row, so swap can take the next clip's.
    The first 5 clips check the loss before fitting, and the run stops if that pre-flight fails on any of them.
    Returns the clips with an own fit, in list order."""
    from sang.render.liveportrait import decode_clip
    cal_dir.mkdir(exist_ok=True)
    s0, n_test, n_demo = round(args.start * FPS), round(args.seconds * FPS), round(args.demo * FPS)
    todo, fitted = (clips[:args.n_clips + 1] if args.n_clips else clips), []
    for i, clip in enumerate(todo):
        f = cal_dir / f"{clip.stem}.pt"
        try:
            if not f.exists():
                frames = decode_clip(str(clip), n_test, start=s0)
                if len(frames) < n_test:
                    raise ValueError(f"only {len(frames) / FPS:.1f} s after {args.start:g} s")
                footage = decode_clip(str(clip), n_demo, start=s0 - n_demo)
                torch.save(ev.calibrate(frames, footage, args.calib, preflight=i < 5), f)
            cal = torch.load(f)
        except Exception as e:
            print(f"[calib {i + 1}/{len(todo)}] SKIP {clip.stem}: {type(e).__name__}: {e}", flush=True)
            continue
        print(f"[calib {i + 1}/{len(todo)}] {clip.stem:<28} "
              + "  ".join(f"{k} {v:.4g}" for k, v in cal["log"].items()), flush=True)
        if i < 5 and not cal["log"]["pre_ok"]:
            sys.exit(f"pre-flight failed on {clip.stem}: noise or blur lowers the calibration loss. Drop the finest "
                     "stage (--calib-stages 0 0.5 0.25) under a new --name")
        if "own" in cal:
            fitted.append(clip.stem)
    if not fitted:
        sys.exit("--calib: no clip was fitted")
    return fitted


def rows_of_clip(ev, args, clip, out, i, n, todo, frames, wav, demo, s0, calib=None) -> None:
    """Generate, score and save every row still to do for one clip. <row>+<arm> renders <row> with the decoder offsets
    calib[arm]; <row>+gated copies <row>+own where its held-out check passed, else <row>."""
    from sang.render.calibrate import offsets
    from sang.render.liveportrait import decode_clip
    for row in todo:
        base, _, arm = row.partition("+")
        d = out / "clips" / row
        d.mkdir(parents=True, exist_ok=True)
        if arm == "gated":
            ok = bool(calib["log"]["cal_accept"])
            src = out / "clips" / (f"{base}+own" if ok else base)
            shutil.copy(src / f"{clip.stem}.npz", d / f"{clip.stem}.npz")
            res = json.loads((src / f"{clip.stem}.json").read_text()) | {"gated_own": float(ok)}
            (d / f"{clip.stem}.json").write_text(json.dumps(res))
            print(f"[{i}/{n}] {clip.stem:<28} {row:<9} = {src.name}", flush=True)
            continue
        ev.last_bg = {}
        with offsets(ev.codec.lp.spade_generator, calib[arm] if arm in ARMS else None):
            if base == "real":
                gen, gt = frames, frames
            elif base == "gt_motion":
                gen, gt = ev.gen_gt_motion(frames, wav, arm)
            elif base == "ditto":
                vid = Path(args.ditto_dir) / f"{clip.stem}.mp4"
                if not vid.exists():
                    raise FileNotFoundError(f"{vid}: no Ditto video")
                gen = decode_clip(str(vid), len(frames))
                gt = frames[: len(gen)]
                ev.last_stats = (ev.motion_stats(ev.codec.extract(gen)["m"], ev.real_motion(frames))
                                 if args.real_stats else {})
            else:
                gen, gt = ev.gen_sang(frames, wav, demo, stats=not arm, arm=arm)
        if args.motion_only:
            res, feats = {"clip": clip.stem, "frames": len(frames)}, {}
        else:
            res, feats = ev.score(row, clip, out, gen, gt, frames[0], s0)
        if base == "sang":
            res.update(ev.last_t)
        if base in ("sang", "ditto"):
            res.update(ev.last_stats)
        if arm == "own":
            res.update(calib["log"])
        res.update(ev.last_bg)
        if feats:
            np.savez_compressed(d / f"{clip.stem}.npz", **feats)
        (d / f"{clip.stem}.json").write_text(json.dumps(res))
        msg = "  ".join(f"{k} {v:.3f}" for k, v in res.items() if isinstance(v, float))
        print(f"[{i}/{n}] {clip.stem:<28} {row:<9} {msg}", flush=True)


def label(row: str, args) -> str:
    base, _, arm = row.partition("+")
    if base == "sang":
        s = f"SANG-M ({args.name}: {args.sampler or 'run defaults'})"
    elif base == "gt_motion":
        s = "Real motion → LivePortrait (ceiling)" + (f" through {args.gt_k} codes" if args.gt_latent else "")
    else:
        s = {"real": "Real video", "ditto": "Ditto"}[base]
    if arm in BG_ARMS:
        return s + {"expr": ", expression only", "pose": ", pose only", "noscale": ", scale held",
                    "notrans": ", translation held", "lock": ", oracle background lock"}[arm]
    return s + (f", calibrated renderer: {arm}" if arm else "")


def markdown(rows: dict, args) -> str:
    fmt = lambda v: "–" if v is None else f"{v:.3f}"
    lines = ["| Method | n | " + " | ".join(METRICS) + " |", "|---" * (len(METRICS) + 2) + "|"]
    lines += [f"| {label(row, args)} | {r['n_clips']} | " + " | ".join(fmt(r["m"].get(k)) for k in METRICS) + " |"
              for row, r in rows.items()]
    return "\n".join(lines)


def summarise(out: Path, args) -> dict:
    from sang.eval.metrics import frechet
    rows = {}
    for row in [r for b in ROWS for r in (b, *(f"{b}+{a}" for a in ARMS + BG_ARMS))]:
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
    ap.add_argument("--gt-latent", default=None, help="ceiling row through a motion latent (scripts/preprocess/fit_latent.py)")
    ap.add_argument("--gt-k", type=int, default=None, help="--gt-latent: codes kept")
    ap.add_argument("--save-videos", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--calib", nargs="+", default=[], choices=ARMS,
                    help="renderer calibration arms (sang.render.calibrate): gt_motion and sang are also rendered with "
                         "LivePortrait's decoder calibrated per clip; needs --demo")
    ap.add_argument("--calib-stages", nargs=3, type=float, default=[1.0, 0.5, 0.25],
                    help="--calib: loss weights of the appearance extractor's three stages, finest first")
    ap.add_argument("--bg", action="store_true", help="background stillness metrics on the gt_motion and sang rows")
    ap.add_argument("--bg-arms", nargs="+", default=[], choices=BG_ARMS,
                    help="rows <row>+<arm> of gt_motion and sang with the same motion (implies --bg)")
    ap.add_argument("--bg-texture", type=float, default=5e-5,
                    help="--bg: smallest structure-tensor eigenvalue of a measured pixel (sang.eval.background.textured)")
    ap.add_argument("--n-clips", type=int, default=0, help="stop after this many scored clips (0 = all)")
    ap.add_argument("--motion-only", action="store_true", help="sang row motion stats only (no rendering)")
    args = ap.parse_args()
    if args.motion_only and args.rows != ["sang"]:
        sys.exit("--motion-only scores the sang row only: pass --rows sang")
    if args.calib:
        if not args.demo or args.motion_only:
            sys.exit("--calib fits the renderer on the footage and renders with it: pass --demo, not --motion-only")
        if "gated" in args.calib and "own" not in args.calib:
            sys.exit("--calib gated chooses between own and the frozen renderer: add own")
        if not {"gt_motion", "sang"} & set(args.rows):
            sys.exit("--calib renders the gt_motion and sang rows: pass one of them")
        args.rows = [r for b in args.rows for r in (b, *(f"{b}+{a}" for a in ARMS
                                                          if a in args.calib and b in ("gt_motion", "sang")))]
    args.bg = args.bg or bool(args.bg_arms)
    if args.bg and (args.calib or args.motion_only):
        sys.exit("--bg renders the rows itself: not with --calib or --motion-only")
    if args.bg_arms:
        if not {"gt_motion", "sang"} & set(args.rows):
            sys.exit("--bg-arms renders the gt_motion and sang rows: pass one of them")
        args.rows = [r for b in args.rows for r in (b, *(f"{b}+{a}" for a in args.bg_arms
                                                          if b in ("gt_motion", "sang")))]
    if args.gt_latent and not args.gt_k:
        sys.exit("--gt-latent needs --gt-k")
    if args.demo > args.start:
        sys.exit("--demo needs --start >= the footage length: the footage comes before the segment")
    if args.start and "ditto" in args.rows:
        sys.exit("the Ditto videos cover the first seconds of each clip only: no ditto row with --start")
    run(args)


if __name__ == "__main__":
    main()
