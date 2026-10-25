#!/usr/bin/env python3
"""HDTF evaluation with the metrics SOTA talking-head papers report: FID, FVD-16, CSIM, LSE-C,
LSE-D (plus PSNR / SSIM), and a merged table with the published numbers (sang/sota_hdtf.py).

    # evaluate (GPU job; ~2-3 min per clip, so give it time)
    sbatch --time=12:00:00 --cpus-per-task=8 bash_scripts/job.sh scripts/eval_hdtf.py \\
        --ckpt runs/motion_12k_anneal/best.pt --data /beegfs/work/$USER/HDTF --name g2
    # a second setting, SANG row only (real and ceiling rows are shared with the first run)
    sbatch ... scripts/eval_hdtf.py --ckpt ... --data ... --name m125 --cfg-mouth 1.25 --rows sang
    # d1 with the blink rule; d1 with the speaker-style oracle (the real clip's own lip range)
    sbatch ... scripts/eval_hdtf.py ... --name d1_blinks --rows sang --rules blinks
    sbatch ... scripts/eval_hdtf.py ... --name d1_oracle --rows sang --lip-oracle range
    # table only: merge any runs with the published numbers -> docs/hdtf_results.md
    python scripts/eval_hdtf.py --table results/hdtf/*/summary.json --write-doc docs/hdtf_results.md

Protocol (also written to <out>/protocol.json):
  data      <data>/crops512/*.mp4 from scripts/download_hdtf.py: HDTF's official face crop
            (method 1), 512x512, 25 fps, 16 kHz audio.
  test set  one clip per video (its _0 clip), videos ordered by an md5 hash of the name, first --n
            (default 75, the size of SoulX-FlashHead's HDTF test set); saved to test_list.txt so
            every run scores the same clips. SANG never trained on HDTF (TalkVid only).
  per clip  first --seconds (default 10 s); source = frame 0; driving audio = the clip's own audio.
  rows      real       the ground-truth clip itself (LSE and CSIM only: the ceiling of each metric)
            gt_motion  the clip's real LivePortrait motion, reduced to SANG's 42-d target and
                       re-rendered from frame 0 through the same renderer and paste-back
                       (the renderer ceiling: what perfect audio-to-motion would score)
            sang       SANG-M generated from audio, rendered with stitching, pasted back into
                       frame 0 with LivePortrait's crop mask so the output is in the GT's frame
            ditto      Ditto (antgroup/ditto-talkinghead, ACM MM 2025) re-run on the same clips by
                       scripts/ditto_hdtf.py (same source frame, same audio); scored here from
                       --ditto-dir, so both methods go through the identical metric code
  test list --list configs/hdtf_test.csv (scripts/hdtf_testset.py) fixes the clips for every run
  metrics   FID     Inception-v3 pool3 over all frames, generated vs ground truth, pooled over clips
            FVD     I3D (StyleGAN-V port), non-overlapping 16-frame clips, pooled over clips
            CSIM    ArcFace (buffalo_l) cosine, source frame vs every 5th output frame
            LSE-C/D joonson/syncnet_python, full face-detection pipeline (sang.bench.lse)
            PSNR / SSIM vs the ground-truth frames: audio-driven output is a different take, so
            these measure closeness to the real performance, not reconstruction quality
  motion    (sang row, with openness.json in the run's motion cache) blinks_per_min and lip_std of the
            generated motion; with --rules the constraint info (own_blinks, added_blinks); with
            --lip-oracle also the real clip's (real_blinks_per_min) and lip_ccc / lip_corr against the
            real lip trajectory after and before (lip_ccc0) the oracle
Resumable: per-clip results go to <out>/clips/<row>/<clip>.{json,npz}; a rerun skips them.
"""
import argparse
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

from sang.sota_hdtf import METRICS, markdown_table

SR, FPS, HOP = 16000, 25, 640
ROWS = ("real", "gt_motion", "sang", "ditto")
MOTION_KEYS = ("blinks_per_min", "real_blinks_per_min", "own_blinks", "added_blinks", "lip_std", "lip_corr",
               "lip_ccc0", "lip_ccc")


# ---------------------------------------------------------------------- test set
def read_list(path: str, data: Path, crops: Path) -> list[Path]:
    """Clips from scripts/hdtf_testset.py's CSV (crop_rel under --data; md5 checked) or a plain
    file of clip names."""
    if str(path).endswith(".csv"):
        import csv
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        clips = [data / r["crop_rel"] if r.get("crop_rel") else crops / f"{r['clip']}.mp4" for r in rows]
        changed = []
        for r, c in zip(rows, clips):
            if r.get("md5") and c.exists():
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
    return [crops / f"{l.strip()}.mp4" for l in Path(path).read_text().splitlines() if l.strip()]


def test_list(crops: Path, n: int) -> list[Path]:
    """One clip per video (prefer _0), videos in md5-hash order, first n."""
    by_video = {}
    for p in sorted(crops.glob("*.mp4")):
        vid, _, k = p.stem.rpartition("_")
        by_video.setdefault(vid, []).append((int(k) if k.isdigit() else 0, p))
    order = sorted(by_video, key=lambda v: hashlib.md5(v.encode()).hexdigest())
    return [min(by_video[v])[1] for v in order][:n]


def load_wav(path: Path, n_frames: int) -> torch.Tensor:
    from decord import AudioReader
    w = torch.from_numpy(AudioReader(str(path), sample_rate=SR, mono=True)[:].asnumpy()).float()[0]
    w = w[: n_frames * HOP]
    return torch.nn.functional.pad(w, (0, max(0, n_frames * HOP - w.numel())))


# ---------------------------------------------------------------------- the evaluator
class Evaluator:
    def __init__(self, args):
        from sang.bench import ArcFace, I3DFeats, InceptionFeats
        from sang.codec import load_wavlm
        from sang.motion import MotionCodec
        from sang.motion_model import Norm, build, ema_weights, guidance_vector, load_ema, parse_spec, sampler_kwargs
        self.args, self.dev = args, "cuda"
        self.codec = MotionCodec(device=self.dev)
        self.arc = ArcFace(device=self.dev)
        self.incep, self.i3d = InceptionFeats(self.dev), I3DFeats(self.dev)
        self.last_t, self.last_stats, self.read = {}, {}, None
        if "sang" in args.rows:
            from sang.pipeline import Opts, Talker
            cfg = torch.load(args.ckpt, map_location="cpu", weights_only=False)["cfg"]
            gamma = guidance_vector(cfg["cfg_audio"] if args.cfg is None else args.cfg, mouth=args.cfg_mouth)
            self.talker = Talker(args.ckpt, args.sampler, self.dev, codec=self.codec, ema=args.ema, cfg_audio=gamma,
                                 steps=args.steps, guide_ckpt=args.guide_ckpt, rules=args.rules)
            try:
                self.read = self.talker.readouts
            except OSError as e:                             # no openness.json: generation unchanged, no motion stats
                if args.lip_oracle:
                    sys.exit(f"--lip-oracle needs the openness readouts: {e}")
                self.read = None
                print(f"no openness readouts ({e}): no blink / lip stats", flush=True)
            self.opts = Opts.fast() if args.fast else Opts(cfg_batch=args.cfg_batch, render_batch=args.render_batch,
                                                           half=args.half, gpu_paste=args.gpu_paste,
                                                           reuse_crop=args.reuse_crop)
            print(f"SANG {args.ckpt} (step {self.talker.step}); pipeline {self.opts}", flush=True)

    # -- generators: each returns (output frames, matching ground-truth frames)
    def gen_sang(self, frames, wav):
        """sang.pipeline.Talker: with every option off this is the original path, stage for stage; the stage
        times (source, audio, motion, render, paste; CUDA-synchronised) go into the clip's json."""
        from sang.naturalness import lip_style
        a, src = self.args, frames[0]
        real = self.real_motion(frames) if a.lip_oracle else None
        gen, m, timer = self.talker.run(src, wav, len(frames), self.opts, seed=a.seed, held=a.held, stitch=a.stitch,
                                        paste=a.paste_back, start=a.start, restyle=a.lip_oracle or "range",
                                        style=lip_style(real["lip"], a.style_q) if real else None)
        self.last_t = {f"t_{k}": v for k, v in timer.t.items()}
        self.last_stats = self.motion_stats(m, real)
        if not a.paste_back:
            box = self.codec._box(self.codec.landmarks(src))
            return gen, self.codec.crop_with_box(frames, box, size=gen.shape[1])[0]
        return gen, frames

    def real_motion(self, frames) -> dict:
        """The real clip's lip and eye openness (from its LivePortrait motion) and the frames they cover."""
        from sang.motion import to_target
        d = self.codec.extract(frames)
        y = to_target(d["m"].float())
        return {"lip": self.read["lip"](y).cpu().numpy(), "eye": self.read["eye"](y).cpu().numpy(), "span": d["span"]}

    def motion_stats(self, m, real=None) -> dict:
        """Blinks and lip opening of the generated motion [n, 70]; against the real clip's when given."""
        if self.read is None:
            return {}
        from sang.motion import to_target
        from sang.naturalness import blink_events, ccc
        y = to_target(m.float())
        lip, eye = self.read["lip"](y).cpu().numpy(), self.read["eye"](y).cpu().numpy()
        out = {"blinks_per_min": len(blink_events(eye)) * 60.0 * FPS / len(eye), "lip_std": float(lip.std()),
               **{k: float(v) for k, v in self.talker.last.get("info", {}).items()}}
        if real is not None:
            s0, r = real["span"][0], real["lip"]
            g = lip[s0:s0 + len(r)]
            g0 = self.read["lip"](self.talker.last["y"]).cpu().numpy()[s0:s0 + len(r)]
            r = r[:len(g)]
            out.update(real_blinks_per_min=len(blink_events(real["eye"])) * 60.0 * FPS / len(real["eye"]),
                       lip_ccc=ccc(g, r), lip_ccc0=ccc(g0, r), lip_corr=float(np.corrcoef(g, r)[0, 1]))
        return out

    def gen_gt_motion(self, frames, wav):
        from sang.motion import from_target, paste_back, to_target
        a = self.args
        d = self.codec.extract(frames)                      # fixed box; truncated at a shot cut
        m = d["m"].float()
        if a.gt_target in (42, 45):                         # 45: SANG's target with the real head placement
            m = from_target(to_target(m, ref=m[:1] if a.gt_target == 45 else None), m[:1], held=a.held)
        state = self.codec.source_state(d["crops"][0], precropped=True)
        crops = self.codec.render(None, m, relative=False, stitch=a.stitch, state=state)
        s0 = d["span"][0]
        gt = frames[s0:s0 + len(crops)]
        if not a.paste_back:
            return crops, self.codec.crop_with_box(gt, np.array(self.codec._box(self.codec.landmarks(gt[0]))), size=crops.shape[1])[0]
        return paste_back(crops, d["M_c2o"], gt[0]), gt

    # -- metrics for one row of one clip
    def score(self, row: str, clip: Path, out_dir: Path, gen, gt, src):
        from sang.bench import csim, lse, psnr, ssim, write_mp4
        a = self.args
        n = min(len(gen), len(gt))
        gen, gt = gen[:n], gt[:n]
        res = {"clip": clip.stem, "frames": n, "csim": csim(self.arc, src, gen, stride=5)}
        feats = {}
        if row != "real":
            res["psnr"] = psnr(gt, gen)
            res["ssim"] = ssim(gt[::5], gen[::5])
            feats = {"incep_gen": self.incep(gen[::a.fid_stride]), "incep_gt": self.incep(gt[::a.fid_stride]),
                     "i3d_gen": self.i3d(gen), "i3d_gt": self.i3d(gt)}
        if a.lse:
            vid = out_dir / "videos" / row / f"{clip.stem}.mp4"
            vid.parent.mkdir(parents=True, exist_ok=True)
            tmp_wav = vid.with_suffix(".wav")
            from sang.voice import save_wav
            save_wav(load_wav(clip, n), tmp_wav)
            write_mp4(gen, vid, fps=FPS, audio=tmp_wav)
            tmp_wav.unlink(missing_ok=True)
            try:                                             # a SyncNet failure must not cost the clip its other metrics
                res["lse_offset"], res["lse_d"], res["lse_c"] = lse(vid)
            except Exception as e:
                print(f"  [lse] {clip.stem} {row}: {type(e).__name__}: {e}", flush=True)
                res["lse_offset"] = res["lse_d"] = res["lse_c"] = float("nan")
            if not a.save_videos:
                vid.unlink(missing_ok=True)
        return res, feats


def run(args) -> None:
    out = Path(args.out or REPO / "results" / "hdtf" / args.name)
    (out / "clips").mkdir(parents=True, exist_ok=True)
    crops = Path(args.data) / "crops512"
    if args.list:
        clips = read_list(args.list, Path(args.data), crops)
    else:
        clips = test_list(crops, args.n)
    if not clips:
        sys.exit(f"no clips under {crops}; run scripts/download_hdtf.py --out {args.data}")
    (out / "test_list.txt").write_text("".join(f"{c.stem}\n" for c in clips))
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True,
                            encoding="utf-8", errors="replace").stdout.strip()
    (out / "protocol.json").write_text(json.dumps({**vars(args), "n_clips": len(clips), "commit": commit}, indent=1, default=str))
    print(f"HDTF eval '{args.name}': {len(clips)} clips x {args.seconds:g} s, rows {args.rows} -> {out}", flush=True)

    ev = Evaluator(args)
    from sang.motion import decode_clip
    t0 = time.time()
    for i, clip in enumerate(clips, 1):
        todo = [r for r in args.rows if not (out / "clips" / r / f"{clip.stem}.json").exists()]
        if not todo:
            continue
        try:
            frames = decode_clip(str(clip), int(args.seconds * FPS))
            wav = load_wav(clip, len(frames))
            for row in todo:
                if row == "real":
                    gen, gt = frames, frames
                elif row == "gt_motion":
                    gen, gt = ev.gen_gt_motion(frames, wav)
                elif row == "ditto":
                    vid = Path(args.ditto_dir) / f"{clip.stem}.mp4"
                    if not vid.exists():
                        raise FileNotFoundError(f"{vid}: run scripts/ditto_hdtf.py first")
                    gen = decode_clip(str(vid), len(frames))
                    gt = frames[: len(gen)]
                else:
                    gen, gt = ev.gen_sang(frames, wav)
                res, feats = ev.score(row, clip, out, gen, gt, frames[0])
                if row == "sang":
                    res.update(ev.last_t)
                    res.update(ev.last_stats)
                d = out / "clips" / row
                d.mkdir(parents=True, exist_ok=True)
                if feats:
                    np.savez_compressed(d / f"{clip.stem}.npz", **feats)
                (d / f"{clip.stem}.json").write_text(json.dumps(res))
                msg = "  ".join(f"{k} {v:.3f}" for k, v in res.items() if isinstance(v, float))
                print(f"[{i}/{len(clips)}] {clip.stem:<28} {row:<9} {msg}", flush=True)
        except Exception as e:                               # one bad clip never kills the run
            print(f"[{i}/{len(clips)}] SKIP {clip.stem}: {type(e).__name__}: {e}", flush=True)
    print(f"clips done in {time.time() - t0:.0f} s", flush=True)
    summarise(out, args)


def summarise(out: Path, args) -> dict:
    from sang.bench import frechet
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
        if tj:                                       # generation speed: median over clips (the first warms up)
            stages = sorted({k for j in tj for k in j if k.startswith("t_")})
            tot = [sum(j[k] for k in stages if k in j) for j in tj]
            rows[row]["timing"] = {"clips": len(tj), **{k: float(np.median([j[k] for j in tj if k in j])) for k in stages},
                                   "total_s": float(np.median(tot)),
                                   "fps": float(np.median([j["frames"] / t for j, t in zip(tj, tot)]))}
    summary = {"name": args.name, "rows": rows, "protocol": json.loads((out / "protocol.json").read_text())}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    md = markdown_table(ours_rows([summary]))
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


LABELS = {"real": ("Real video", "-"), "gt_motion": ("Real motion → LivePortrait (ceiling)", "renderer ceiling"),
          "sang": ("SANG-M", "flow-matching DiT, 42-d"),
          "ditto": ("Ditto (re-run, our protocol)", "LivePortrait motion, diffusion")}


def ours_rows(summaries: list[dict]) -> list[dict]:
    out, seen = [], set()
    for s in summaries:
        p = s.get("protocol", {})
        for row in ROWS:
            if row not in s["rows"]:
                continue
            label, fam = LABELS[row]
            if row == "sang":
                knobs = ([p["sampler"]] if p.get("sampler") else
                         [f"γ {p.get('cfg') or 2:g}"] + ([f"mouth γ {p['cfg_mouth']:g}"] if p.get("cfg_mouth") else []))
                knobs += [f"rules {p['rules']}"] if p.get("rules", "none") != "none" else []
                knobs += [f"lip oracle {p['lip_oracle']}"] if p.get("lip_oracle") else []
                label = f"SANG-M ({s['name']}: {', '.join(knobs)})"
            elif row in seen:                                # real / ceiling rows: once is enough
                continue
            seen.add(row)
            out.append({"source": f"Ours (n={s['rows'][row]['n_clips']}, {p.get('seconds', '?'):g} s)", "method": label,
                        "family": fam, "params": {"sang": "53.4 M +LP", "ditto": "n/r +LP"}.get(row),
                        "m": s["rows"][row]["m"]})
    return out


def table(paths: list[str], write_doc: str | None) -> None:
    summaries = [json.loads(Path(p).read_text()) for p in paths]
    md = markdown_table(ours_rows(summaries))
    print(md)
    if write_doc:
        head = ("# HDTF results: SANG-M and published methods\n\n"
                "Generated by `python scripts/eval_hdtf.py --table ... --write-doc ...`. Our rows (top) share one\n"
                "protocol (see each run's `protocol.json`); published rows are copied from the named table and only\n"
                "compare within the same source (sang/sota_hdtf.py). LSE-C = Sync-C, LSE-D = Sync-D. n/r = not reported;\n"
                "+LP = also runs the frozen LivePortrait renderer (~130 M). E-FID needs a 3DMM fitter and is not computed.\n\n")
        Path(write_doc).write_text(head + md + "\n")
        print(f"\nwrote {write_doc}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=str(REPO / "runs/motion_12k_anneal/best.pt"))
    ap.add_argument("--data", help="HDTF root from scripts/download_hdtf.py (contains crops512/)")
    ap.add_argument("--name", default="sang", help="run name -> results/hdtf/<name>")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n", type=int, default=75, help="test clips (one per video)")
    ap.add_argument("--list", default=None, help="configs/hdtf_test.csv (scripts/hdtf_testset.py) or a file of clip names; overrides --n")
    ap.add_argument("--ditto-dir", default=str(REPO / "results" / "hdtf" / "ditto_videos"),
                    help="Ditto's videos for the ditto row (scripts/ditto_hdtf.py --out)")
    ap.add_argument("--seconds", type=float, default=10.0, help="evaluated length per clip")
    ap.add_argument("--rows", nargs="+", default=["real", "gt_motion", "sang"], choices=ROWS,
                    help="ditto needs --ditto-dir from scripts/ditto_hdtf.py, so it is opt-in")
    ap.add_argument("--start", default="source", choices=["null", "source"])
    ap.add_argument("--cfg", type=float, default=None, help="audio guidance (default: the run's, 2.0)")
    ap.add_argument("--cfg-mouth", type=float, default=None, help="audio guidance on the mouth only")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--sampler", default=None, help="sampler spec, e.g. 'g=2,mouth=1.25,avg=4'; overrides --cfg/--cfg-mouth/--steps")
    ap.add_argument("--ema", default=None, help="extra EMA decay saved by training with ema_extra (e.g. 0.999); default = the main EMA")
    ap.add_argument("--guide-ckpt", default=None, help="autoguidance guide checkpoint (for ag / ag_mouth in --sampler)")
    ap.add_argument("--rules", default="none", choices=["none", "lips", "blinks", "both"],
                    help="rule constraints at sampling (sang/naturalness.py): bilabial closures, blinks at pauses")
    ap.add_argument("--lip-oracle", default=None, choices=["level", "range"],
                    help="speaker-style oracle: move SANG's lip opening to the REAL clip's median (level) or "
                         "robust range (range) before rendering; timing untouched. An upper bound, not a method")
    ap.add_argument("--style-q", type=float, default=5.0, help="--lip-oracle range: the q / 100-q percentiles")
    ap.add_argument("--held", default="camera", choices=["camera", "head"])
    ap.add_argument("--fast", action="store_true", help="every speed option below (sang.pipeline.Opts.fast)")
    ap.add_argument("--cfg-batch", action="store_true", help="CFG pair of each Euler step as one batch")
    ap.add_argument("--render-batch", type=int, default=1, help="frames per renderer call (1 = per-frame loop)")
    ap.add_argument("--half", action="store_true", help="FP16 warping module and decoder")
    ap.add_argument("--gpu-paste", action="store_true", help="paste-back on the GPU inside the face's box")
    ap.add_argument("--reuse-crop", action="store_true", help="detect the source face once, not three times")
    ap.add_argument("--gt-target", type=int, default=42, choices=[42, 45, 70],
                    help="ceiling row: SANG's 42-d target, 45 = with the real head placement (C4), or full 70-d")
    ap.add_argument("--stitch", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--paste-back", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--lse", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--save-videos", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--fid-stride", type=int, default=1, help="use every k-th frame for FID")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--table", nargs="+", default=None, help="summary.json files: print/write the merged table and exit")
    ap.add_argument("--write-doc", default=None, help="with --table: write the markdown here (e.g. docs/hdtf_results.md)")
    args = ap.parse_args()
    if args.table is not None:
        return table(args.table, args.write_doc)
    if not args.data:
        sys.exit("--data is required (the --out folder of scripts/download_hdtf.py)")
    run(args)


if __name__ == "__main__":
    main()
