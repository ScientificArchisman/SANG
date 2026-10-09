#!/usr/bin/env python3
"""Generate Ditto videos for the frozen HDTF test set, so eval_hdtf.py can score Ditto with exactly
the metric code, clips, source frames and audio that SANG gets.

Ditto = antgroup/ditto-talkinghead (ACM MM 2025, arXiv 2411.19509, Apache-2.0): audio -> HuBERT ->
diffusion over LivePortrait motion -> LivePortrait renderer, i.e. the closest published system to
SANG. (github.com/megagonlabs/ditto is an unrelated entity-matching model.)

Runs in Ditto's own conda env (bash_scripts/install_ditto.sh), on a GPU node:

    SANG_ENV=ditto sbatch --time=08:00:00 --cpus-per-task=8 bash_scripts/job.sh scripts/ditto_hdtf.py \\
        --data /beegfs/work/$USER/HDTF --list configs/hdtf_test.csv
    # then score it (avcodec env, like every other row):
    sbatch --time=08:00:00 --cpus-per-task=8 bash_scripts/job.sh scripts/eval_hdtf.py \\
        --data /beegfs/work/$USER/HDTF --list configs/hdtf_test.csv --name ditto --rows ditto

Per clip: source = the clip's frame 0 (as SANG), audio = the clip's first --seconds (16 kHz mono),
Ditto's offline pipeline with its default settings (50 diffusion steps unless --steps), seeded.
Output: <out>/<clip>.mp4 (512x512, 25 fps, with audio) + <out>/timings.json (wall-clock per clip,
model loading excluded). Resumable: finished clips are skipped.
"""
import argparse
import csv
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


def ffmpeg_on_path(bin_dir: Path) -> str:
    """Ditto muxes with os.system("ffmpeg ..."): put imageio-ffmpeg's static binary first on PATH as
    `ffmpeg`, so the mux does not depend on whatever ffmpeg the node or env has (none, in the ditto env)."""
    exe = ffmpeg_exe()
    if exe == "ffmpeg":
        return exe
    bin_dir.mkdir(parents=True, exist_ok=True)
    link = bin_dir / "ffmpeg"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(exe)
    os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"
    return str(link)


def mux(video: Path, wav: Path, dst: Path) -> None:
    """Ditto's own mux, run by us when its os.system call left no output."""
    r = subprocess.run([ffmpeg_exe(), "-loglevel", "error", "-y", "-i", str(video), "-i", str(wav), "-map", "0:v",
                        "-map", "1:a", "-c:v", "copy", "-c:a", "aac", str(dst)], capture_output=True, text=True)
    if r.returncode != 0 or not dst.exists():
        raise RuntimeError(f"mux failed: {r.stderr.strip()[-300:]}")


def clip_paths(list_path: str, data: Path) -> list[Path]:
    if list_path.endswith(".csv"):
        with open(list_path, newline="", encoding="utf-8") as f:
            return [data / r["crop_rel"] for r in csv.DictReader(f)]
    return [data / "crops512" / f"{l.strip()}.mp4" for l in Path(list_path).read_text().splitlines() if l.strip()]


def seed_everything(seed: int) -> None:
    import torch
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="HDTF root (download_hdtf.py --out)")
    ap.add_argument("--list", default=str(REPO / "configs" / "hdtf_test.csv"))
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--out", default=str(REPO / "results" / "hdtf" / "ditto_videos"))
    ap.add_argument("--ditto-root", default=str(REPO / "third_party" / "ditto-talkinghead"))
    ap.add_argument("--model-dir", default=None, help="default <ditto-root>/checkpoints/ditto_pytorch")
    ap.add_argument("--cfg-pkl", default=None, help="default <ditto-root>/checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl")
    ap.add_argument("--steps", type=int, default=50, help="Ditto's diffusion steps (its default 50)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.ditto_root).resolve()
    model_dir = Path(args.model_dir or root / "checkpoints" / "ditto_pytorch")
    cfg_pkl = Path(args.cfg_pkl or root / "checkpoints" / "ditto_cfg" / "v0.4_hubert_cfg_pytorch.pkl")
    for p in (root / "inference.py", model_dir, cfg_pkl):
        if not p.exists():
            sys.exit(f"missing {p}: run bash_scripts/install_ditto.sh on the login node")
    sys.path.insert(0, str(root))
    from inference import run                        # Ditto's own offline entry point
    from stream_pipeline_offline import StreamSDK

    import cv2
    clips = clip_paths(args.list, Path(args.data))
    out = Path(args.out)
    tmp = out / "_inputs"
    tmp.mkdir(parents=True, exist_ok=True)
    tpath = out / "timings.json"
    timings = json.loads(tpath.read_text()) if tpath.exists() else {}
    todo = [c for c in clips if not (out / f"{c.stem}.mp4").exists()]
    print(f"Ditto on {len(clips)} clips ({len(todo)} to do), {args.seconds:g} s each, {args.steps} steps -> {out}", flush=True)
    print(f"ffmpeg for Ditto's mux: {ffmpeg_on_path(out / '_bin')}", flush=True)

    t0 = time.time()
    sdk = StreamSDK(str(cfg_pkl), str(model_dir))
    print(f"model loaded in {time.time() - t0:.0f} s", flush=True)
    for i, clip in enumerate(todo, 1):
        dst = out / f"{clip.stem}.mp4"
        src, wav = tmp / f"{clip.stem}.png", tmp / f"{clip.stem}.wav"
        try:
            cap = cv2.VideoCapture(str(clip))
            ok, frame0 = cap.read()
            cap.release()
            if not ok:
                raise RuntimeError("cannot read frame 0")
            cv2.imwrite(str(src), frame0)
            subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(clip), "-t", f"{args.seconds:.3f}",
                            "-vn", "-ac", "1", "-ar", "16000", str(wav)], check=True)
            seed_everything(args.seed)
            t1 = time.time()
            try:
                run(sdk, str(wav), str(src), str(dst), {"setup_kwargs": {"sampling_timesteps": args.steps}})
            except Exception:
                sdk = StreamSDK(str(cfg_pkl), str(model_dir))     # a fresh pipeline, once, then retry
                run(sdk, str(wav), str(src), str(dst), {"setup_kwargs": {"sampling_timesteps": args.steps}})
            secs = time.time() - t1
            tmp_video = Path(str(dst) + ".tmp.mp4")
            if not dst.exists():
                if not tmp_video.exists():
                    raise RuntimeError("Ditto rendered no video")
                mux(tmp_video, wav, dst)
            n = int(round(args.seconds * 25))
            timings[clip.stem] = {"seconds": round(secs, 3), "frames": n, "fps": round(n / secs, 2)}
            tpath.write_text(json.dumps(timings, indent=1))
            print(f"[{i}/{len(todo)}] {clip.stem:<28} {secs:6.1f} s  ({n / secs:.1f} frames/s)", flush=True)
        except Exception as e:
            print(f"[{i}/{len(todo)}] SKIP {clip.stem}: {type(e).__name__}: {e}", flush=True)
        finally:
            if dst.exists():                                  # keep the silent render if the mux failed
                Path(str(dst) + ".tmp.mp4").unlink(missing_ok=True)
    if timings:
        fps = [v["fps"] for v in timings.values()]
        print(f"done: {len(timings)} clips; Ditto speed {np.median(fps):.1f} frames/s median "
              f"(offline pipeline incl. face detection and mux, {args.steps} steps)", flush=True)


if __name__ == "__main__":
    main()
