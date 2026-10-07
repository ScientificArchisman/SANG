#!/usr/bin/env python3
"""Download and process HDTF (Zhang et al., CVPR 2021; CC BY 4.0) the way its authors describe.

    pip install yt-dlp          # once (bash_scripts/install_eval.sh does it)
    python scripts/download_hdtf.py --out /beegfs/work/$USER/HDTF --workers 4
    # rerun the same line to resume; failures are listed in <out>/failed.tsv

Needs internet (YouTube), so run it on the LOGIN node, not in a GPU job. Steps, per the official
README (github.com/MRzzm/HDTF):
  1. annotations   the 15 xx_*.txt files from the HDTF repo  -> <out>/annotations/
  2. raw videos    yt-dlp at the reference resolution in xx_resolution.txt  -> <out>/raw/<name>.mp4
  3. clips         cut by xx_annotion_time.txt, de-interlaced, 25 fps, 16 kHz mono audio
                   -> <out>/clips/<name>_<k>.mp4   (named as HDTF does: Radio11_0, Radio11_1, ...)
  4. 512 crops     the fixed face window of xx_crop_wh.txt (official method 1), scaled if the
                   download is not at the reference height, resized to 512x512
                   -> <out>/crops512/<name>_<k>.mp4   <- what scripts/eval_hdtf.py reads
Many of the 370 videos have been removed from YouTube since 2021; expect a few dozen failures.
YouTube may also ask for a login: pass --cookies cookies.txt (exported from a browser).
"""
import argparse
import json
import re
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SUBSETS = ("RD", "WDA", "WRA")
KINDS = ("video_url", "resolution", "annotion_time", "crop_wh", "crop_ratio")
RAW_URL = "https://raw.githubusercontent.com/MRzzm/HDTF/main/HDTF_dataset/{subset}_{kind}.txt"


# ---------------------------------------------------------------------- annotations
def fetch_annotations(dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for s in SUBSETS:
        for k in KINDS:
            f = dst / f"{s}_{k}.txt"
            if not f.exists():
                urllib.request.urlretrieve(RAW_URL.format(subset=s, kind=k), f)


def _lines(path: Path) -> list[list[str]]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8-sig").replace("\ufeff", "")       # WDA_video_url.txt starts with a BOM
    return [l.split() for l in text.splitlines() if l.strip()]


def to_seconds(t: str) -> float:
    """'01:35' or '1:02:03' -> seconds."""
    parts = [float(p) for p in t.split(":")]
    sec = 0.0
    for p in parts:
        sec = sec * 60 + p
    return sec


def parse_annotations(ann: Path, subsets=SUBSETS) -> list[dict]:
    """One dict per source video: name, url, ref_height, clips [(start_s, end_s)], crops {k: (x, w, y, h)}."""
    videos = []
    for s in subsets:
        url = {r[0].removesuffix(".mp4"): r[1] for r in _lines(ann / f"{s}_video_url.txt") if len(r) >= 2}
        res = {r[0].removesuffix(".mp4"): int(r[1]) for r in _lines(ann / f"{s}_resolution.txt") if len(r) >= 2}
        times = {r[0].removesuffix(".mp4"): r[1:] for r in _lines(ann / f"{s}_annotion_time.txt")}
        crops = {}
        for r in _lines(ann / f"{s}_crop_wh.txt"):
            if len(r) >= 5:
                crops[r[0].removesuffix(".mp4")] = tuple(int(float(v)) for v in r[1:5])   # min_w, w, min_h, h
        for name in sorted(url):                                # Radio24 has a URL but no time stamps: skipped
            clips = []
            for span in times.get(name, []):
                m = re.match(r"^([\d:.]+)-([\d:.]+)$", span)
                if m:
                    a, b = to_seconds(m.group(1)), to_seconds(m.group(2))
                    if b > a:
                        clips.append((a, b))
            videos.append({"subset": s, "name": name, "url": url[name], "ref_height": res.get(name, 720),
                           "clips": clips,
                           "crops": {k: crops[f"{name}_{k}"] for k in range(len(clips)) if f"{name}_{k}" in crops}})
    return videos


def crop_filter(box: tuple[int, int, int, int], ref_h: int, real_w: int, real_h: int, size: int = 512) -> str:
    """ffmpeg crop+scale for an HDTF crop_wh box (min_w, w, min_h, h) given at the reference height,
    rescaled to the downloaded frame size and clamped inside it."""
    x, w, y, h = box
    s = real_h / float(ref_h)
    x, w, y, h = round(x * s), round(w * s), round(y * s), round(h * s)
    w, h = min(w, real_w), min(h, real_h)
    x, y = max(0, min(x, real_w - w)), max(0, min(y, real_h - h))
    w, h = w - w % 2, h - h % 2
    return f"crop={w}:{h}:{x}:{y},scale={size}:{size}:flags=lanczos"


# ---------------------------------------------------------------------- media helpers
def ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


def probe_size(path: Path) -> tuple[int, int]:
    """(width, height) of the first video stream, via ffmpeg's banner (no ffprobe needed)."""
    out = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", str(path)], capture_output=True,
                         encoding="utf-8", errors="replace").stderr       # the banner carries YouTube titles
    m = re.search(r"Video:.*?(\d{2,5})x(\d{2,5})", out)
    if not m:
        raise RuntimeError(f"cannot read the frame size of {path}")
    return int(m.group(1)), int(m.group(2))


def download(v: dict, raw: Path, cookies: str | None, retries: int = 2) -> Path:
    dst = raw / f"{v['name']}.mp4"
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    import yt_dlp
    h = v["ref_height"]
    opts = {
        "format": f"bv*[height<={h}][ext=mp4]+ba[ext=m4a]/bv*[height<={h}]+ba/b[height<={h}]/b",
        "merge_output_format": "mp4",
        "outtmpl": str(raw / f"{v['name']}.%(ext)s"),
        "ffmpeg_location": ffmpeg_exe(),
        "quiet": True, "no_warnings": True, "noprogress": True,
        "retries": 3, "fragment_retries": 3,
    }
    if cookies:
        opts["cookiefile"] = cookies
    err = None
    for _ in range(retries + 1):
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([v["url"]])
            if dst.exists():
                return dst
            hits = sorted(raw.glob(f"{v['name']}.*"))          # merged into another container
            if hits:
                subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(hits[0]), "-c", "copy", str(dst)], check=True)
                return dst
        except Exception as e:                                  # removed / private / rate-limited
            err = e
            time.sleep(5)
    raise RuntimeError(f"download failed: {err}")


def cut_and_crop(v: dict, src: Path, clips_dir: Path, crops_dir: Path, size: int, do_crop: bool) -> list[str]:
    done = []
    W, H = probe_size(src)
    for k, (a, b) in enumerate(v["clips"]):
        clip = clips_dir / f"{v['name']}_{k}.mp4"
        if not clip.exists():
            subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-ss", f"{a:.3f}", "-to", f"{b:.3f}", "-i", str(src),
                            "-vf", "yadif=deint=interlaced,fps=25", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p",
                            "-c:a", "aac", "-ar", "16000", "-ac", "1", str(clip)], check=True)
        if do_crop and k in v["crops"]:
            crop = crops_dir / f"{v['name']}_{k}.mp4"
            if not crop.exists():
                subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(clip),
                                "-vf", crop_filter(v["crops"][k], v["ref_height"], W, H, size),
                                "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", "-c:a", "copy", str(crop)], check=True)
        done.append(clip.stem)
    return done


# ---------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="dataset root: annotations/, raw/, clips/, crops512/ go here")
    ap.add_argument("--subsets", nargs="+", default=list(SUBSETS), choices=SUBSETS)
    ap.add_argument("--workers", type=int, default=4, help="parallel downloads")
    ap.add_argument("--max-videos", type=int, default=0, help="0 = all (370); small numbers for a dry run")
    ap.add_argument("--cookies", default=None, help="Netscape cookies.txt if YouTube asks to sign in")
    ap.add_argument("--size", type=int, default=512, help="output size of the face crops")
    ap.add_argument("--no-crop", action="store_true", help="only cut clips, skip the 512 face crops")
    ap.add_argument("--keep-raw", action="store_true", help="keep the full downloaded videos (default: delete after cutting)")
    args = ap.parse_args()

    try:
        import yt_dlp  # noqa: F401
    except ImportError:
        sys.exit("yt-dlp is not installed: pip install yt-dlp")

    out = Path(args.out).expanduser().resolve()
    ann, raw, clips, crops = out / "annotations", out / "raw", out / "clips", out / f"crops{args.size}"
    for d in (raw, clips, crops):
        d.mkdir(parents=True, exist_ok=True)
    fetch_annotations(ann)
    videos = [v for v in parse_annotations(ann, args.subsets) if v["clips"]]
    if args.max_videos:
        videos = videos[: args.max_videos]
    n_clips = sum(len(v["clips"]) for v in videos)
    print(f"HDTF: {len(videos)} videos, {n_clips} clips -> {out}", flush=True)

    ok, failed, t0 = [], [], time.time()

    def work(v):
        if all((clips / f"{v['name']}_{k}.mp4").exists() for k in range(len(v["clips"]))) and \
                (args.no_crop or all((crops / f"{v['name']}_{k}.mp4").exists() for k in v["crops"])):
            return v, [f"{v['name']}_{k}" for k in range(len(v["clips"]))]
        src = download(v, raw, args.cookies)
        names = cut_and_crop(v, src, clips, crops, args.size, not args.no_crop)
        if not args.keep_raw:
            src.unlink(missing_ok=True)
        return v, names

    with ThreadPoolExecutor(max(1, args.workers)) as ex:
        futs = {ex.submit(work, v): v for v in videos}
        for i, f in enumerate(as_completed(futs), 1):
            v = futs[f]
            try:
                _, names = f.result()
                ok += names
                status = f"ok ({len(names)} clips)"
            except Exception as e:
                failed.append((v["name"], v["url"], f"{type(e).__name__}: {str(e)[:200]}"))
                status = f"FAILED {type(e).__name__}"
            print(f"[{i}/{len(videos)}] {v['name']:<32} {status}  ({time.time() - t0:.0f} s)", flush=True)

    (out / "failed.tsv").write_text("".join(f"{n}\t{u}\t{e}\n" for n, u, e in failed))
    manifest = {"clips": sorted(ok), "failed_videos": [n for n, _, _ in failed], "size": args.size,
                "crops_dir": crops.name, "source": "github.com/MRzzm/HDTF (CC BY 4.0)"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    n_crops = len(list(crops.glob("*.mp4")))
    print(f"\ndone: {len(ok)} clips ({n_crops} 512-px crops) from {len(videos) - len(failed)}/{len(videos)} videos; "
          f"{len(failed)} failed -> {out / 'failed.tsv'}", flush=True)


if __name__ == "__main__":
    main()
