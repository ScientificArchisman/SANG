"""Audio-video offset correction for the TalkVid motion cache.

The lip-sync diagnosis (scripts/diagnose_lips.py, job 176657) found that 41 % of real val clips are
off by >= 2 frames according to SyncNet, that the offset is a property of the SOURCE VIDEO (one
video: 13 clips, all -2), and that lip correlation falls with it: 0.66 at offsets 0/-1, 0.57 at -2,
0.28 at -3/-4, ~0 where SyncNet finds no sync. scripts/sync_offsets.py measures 2 clips per video
and writes <cache>/sync_offsets.json; training and evaluation then shift the cached audio.

Convention. `offset` is SyncNet's "AV offset" in 25 fps frames (joonson/syncnet_python): NEGATIVE
when the audio is LATE. (Checked on job 176657: the model's own loss-minimising audio shift has the
same sign, corr 0.65, and generated lips lag the real ones on negative-offset clips.) The
correction moves the audio by `offset` frames:

    out[t] = audio[t - offset]        offset -2 -> out[t] = audio[t + 2]: late audio pulled forward

WavLM features tick at 50 Hz (2 per frame), so the shift is round(2 * offset) ticks, which lets a
video's mean offset (e.g. -0.5 from clips at 0 and -1) be applied at half-frame resolution.
"""
import json
import math
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import torch

FPS, TPF, SR = 25, 2, 16000
TICK = SR // (FPS * TPF)            # 320 samples per WavLM tick (20 ms)


# ---------------------------------------------------------------------- keys
def clip_key(row: dict) -> str:
    """'<speaker>/<clip>' from a cache-index row: stable across machines and cache moves."""
    p = Path(row["path"])
    return f"{p.parent.name}/{p.stem}"


def video_of(row: dict) -> str:
    """TalkVid's speaker directory is the source YouTube video, the unit offsets are constant in."""
    return Path(row["path"]).parent.name


def pick_clips(rows: list[dict], per_video: int, min_frames: int = 0) -> list[dict]:
    """Up to `per_video` clips per video, spread over the video (first, last, then in between),
    deterministic. Clips shorter than min_frames are skipped (SyncNet needs > 100 frames)."""
    by = defaultdict(list)
    for r in rows:
        if r["n"] >= min_frames:
            by[video_of(r)].append(r)
    out = []
    for v in sorted(by):
        rs = sorted(by[v], key=lambda r: r["path"])
        if len(rs) <= per_video:
            out += rs
            continue
        idx = sorted({round(i * (len(rs) - 1) / max(1, per_video - 1)) for i in range(per_video)})
        out += [rs[i] for i in idx]
    return out


# ---------------------------------------------------------------------- shifting
def shift_ticks(x: torch.Tensor, s: int) -> torch.Tensor:
    """[T, ...] -> same length, out[t] = x[t - s], edge-padded."""
    if not s:
        return x
    T = x.shape[0]
    idx = (torch.arange(T, device=x.device) - int(s)).clamp(0, T - 1)
    return x[idx]


def shift_wav(wav: torch.Tensor, s: int, tick: int = TICK) -> torch.Tensor:
    """The same physical shift on a 16 kHz waveform [N]: out[i] = wav[i - s * tick], zero-padded."""
    if not s:
        return wav
    k, n = int(s) * tick, len(wav)
    out = torch.zeros_like(wav)
    if abs(k) >= n:
        return out
    if k > 0:
        out[k:] = wav[:n - k]
    else:
        out[:n + k] = wav[-k:]
    return out


# ---------------------------------------------------------------------- measuring
def real_clip_mp4(row: dict, n_frames: int, out: Path, ffmpeg: str | None = None) -> Path:
    """Frames [start, start + n) of a TalkVid clip with its separate .m4a audio, as one mp4 that
    syncnet_python can read: the exact segment the cache holds."""
    if ffmpeg is None:
        try:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            ffmpeg = "ffmpeg"
    s0 = row.get("start", 0) / FPS
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-ss", f"{s0:.3f}", "-i", str(row["clip"]),
                    "-ss", f"{s0:.3f}", "-i", str(Path(row["clip"]).with_suffix(".m4a")),
                    "-t", f"{n_frames / FPS:.3f}", "-map", "0:v:0", "-map", "1:a:0", "-r", str(FPS),
                    "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", str(out)], check=True)
    return out


def _finite(x) -> bool:
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def merge(measured: list[dict], rows: list[dict], min_conf: float = 3.0, max_abs: float = 10.0,
          max_spread: float = 1.0) -> dict:
    """SyncNet measurements (dicts with key, video, offset, conf) + every cache row -> one record per
    clip: {keep, offset, shift_ticks, source, ...}.

    A measurement is usable when conf >= min_conf and |offset| < max_abs (syncnet searches +/-15, so a
    value at the edge means it found no sync). A video's offset is the mean of its usable clips; it
    is 'consistent' when they agree within max_spread frames. Then per clip:
      measured, unusable           -> drop (no sync / low confidence: off-screen or dubbed speaker)
      video consistent             -> keep, the video's mean offset
      measured, video inconsistent -> keep, the clip's own offset
      otherwise                    -> drop (video unmeasured, unusable, or inconsistent)"""
    meas = {m["key"]: m for m in measured}
    by_video = defaultdict(list)
    for m in measured:
        m["ok"] = _finite(m.get("offset")) and _finite(m.get("conf")) and m["conf"] >= min_conf \
            and abs(m["offset"]) < max_abs
        by_video[m["video"]].append(m)
    vids = {}
    for v, ms in by_video.items():
        good = [m["offset"] for m in ms if m["ok"]]
        vids[v] = None if not good else {"offset": sum(good) / len(good), "n": len(good),
                                         "spread": max(good) - min(good)}
    clips = {}
    for r in rows:
        k, v = clip_key(r), video_of(r)
        m, ve = meas.get(k), vids.get(v)
        if m is not None and not m["ok"]:
            no_sync = not _finite(m.get("offset")) or abs(m["offset"]) >= max_abs
            rec = {"keep": False, "why": "no sync" if no_sync else "low confidence", "source": "clip",
                   "offset": m.get("offset"), "conf": m.get("conf")}
        elif ve is not None and ve["spread"] <= max_spread:
            rec = {"keep": True, "offset": ve["offset"], "source": "video", "n_video": ve["n"]}
            if m is not None:
                rec.update(clip_offset=m["offset"], conf=m["conf"])
        elif m is not None:
            rec = {"keep": True, "offset": m["offset"], "conf": m["conf"], "source": "clip"}
        elif ve is not None:
            rec = {"keep": False, "why": "video offsets disagree", "source": "video"}
        elif v in by_video:
            rec = {"keep": False, "why": "no sync in video", "source": "video"}
        else:
            rec = {"keep": False, "why": "not measured", "source": "none"}
        rec["shift_ticks"] = int(round(TPF * rec["offset"])) if rec["keep"] else 0
        clips[k] = rec

    kept = [c for c in clips.values() if c["keep"]]
    hist = Counter(round(2 * c["offset"]) / 2 for c in kept)
    meta = {"clips": len(clips), "kept": len(kept), "videos": len({video_of(r) for r in rows}),
            "videos_measured": len(by_video), "videos_usable": sum(v is not None for v in vids.values()),
            "videos_inconsistent": sum(v is not None and v["spread"] > max_spread for v in vids.values()),
            "dropped": dict(Counter(c["why"] for c in clips.values() if not c["keep"])),
            "offset_hist_kept": {str(k): v for k, v in sorted(hist.items())},
            "frac_kept_abs_ge2": sum(abs(c["offset"]) >= 2 for c in kept) / max(1, len(kept)),
            "frac_kept_abs_ge1": sum(abs(c["offset"]) >= 1 for c in kept) / max(1, len(kept)),
            "min_conf": min_conf, "max_abs": max_abs, "max_spread": max_spread,
            "convention": "SyncNet AV offset in frames, negative = audio late; audio shifted by "
                          "shift_ticks = round(2 * offset): out[t] = audio[t - shift_ticks]"}
    return {"meta": meta, "clips": clips}


# ---------------------------------------------------------------------- applying
class SyncOffsets:
    """<cache>/sync_offsets.json at load time: which clips to keep and how far to move their audio."""

    def __init__(self, data: dict, path: str = ""):
        self.clips, self.meta, self.path = data["clips"], data.get("meta", {}), path

    @classmethod
    def load(cls, path) -> "SyncOffsets":
        return cls(json.loads(Path(path).read_text()), str(path))

    def get(self, row: dict) -> dict | None:
        return self.clips.get(clip_key(row))

    def shift(self, row: dict) -> int:
        rec = self.get(row)
        return int(rec["shift_ticks"]) if rec and rec.get("keep") else 0

    def filter(self, rows: list[dict], keep_unknown: bool = False) -> tuple[list[dict], dict]:
        """Rows to keep, plus counts of what was dropped and why."""
        out, why = [], Counter()
        for r in rows:
            rec = self.get(r)
            if rec is None:
                if keep_unknown:
                    out.append(r)
                else:
                    why["not in offsets file"] += 1
            elif rec.get("keep"):
                out.append(r)
            else:
                why[rec.get("why", "dropped")] += 1
        return out, dict(why)
