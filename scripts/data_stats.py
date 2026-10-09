#!/usr/bin/env python3
"""Statistics of SANG's training and validation data (the motion cache as training uses it), for
reports and slides: clips, hours, source videos (= speakers), clips per video, clip lengths,
spoken language, audio-video offsets, and TalkVid's own labels (language, person, gender, ethnicity, age).

    sbatch --time=02:00:00 --cpus-per-task=8 bash_scripts/job.sh scripts/data_stats.py
    # -> results/data_stats/data_stats.json (+ a printed summary)
    python scripts/data_stats.py      # login node is enough once <cache>/lang_lid.json covers every clip

Two language labels per clip: TalkVid's own (metadata/filtered_video_clips.json, matched by video id and
start/end time, sang/talkvid.py; also written to <cache>/talkvid_meta.json for later filtering) and
Whisper's (below). Percentages are printed for both, by clips and by hours.

Splits are exactly training's: speaker-disjoint hash split (val_frac, seed from the config), then the
SyncNet offset filter (<cache>/sync_offsets.json) that drops no-sync clips. TalkVid's own
metadata is not in the cache, so the language of each clip is identified here with Whisper-large-v3
(first <= 10 s of the clip's audio, the language token's probability after <|startoftranscript|>),
cached per clip in <cache>/lang_lid.json so a rerun only labels new clips. A clip whose top language
has probability < --min-prob is 'unk'. A "speaker" is a TalkVid source video (its cache folder): the
same person can appear in several videos, so this is an upper bound on distinct people.
"""
import argparse
import importlib.util
import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from sang.paths import DATA_ROOT, motion_cache
from sang.talkvid import WHISPER_NAMES, load_lookup, match
from sang.sync import SyncOffsets, clip_key, video_of

FPS = 25
CLIPS_PER_VIDEO_BINS = [(1, 1), (2, 2), (3, 5), (6, 10), (11, 20), (21, 50), (51, 10 ** 9)]
SECONDS_BINS = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 10 ** 9)]


def load_train_module():
    spec = importlib.util.spec_from_file_location("tm", REPO / "scripts" / "train_motion.py")
    tm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tm)
    return tm


def bin_counts(values, bins) -> list[dict]:
    out = []
    for lo, hi in bins:
        label = f"{lo}" if lo == hi else (f">{lo - 1}" if hi >= 10 ** 9 else f"{lo}-{hi}")
        out.append({"bin": label, "count": int(sum(lo <= v <= hi for v in values))})
    return out


def breakdown(rows: list[dict], label_of) -> list[dict]:
    """Rows grouped by label_of(row) -> [{label, clips, pct_clips, hours, pct_hours, videos}], largest first."""
    clips, hours, vids = Counter(), Counter(), {}
    for r in rows:
        k = label_of(r)
        clips[k] += 1
        hours[k] += r["n"] / FPS / 3600
        vids.setdefault(k, set()).add(video_of(r))
    n, h = len(rows), sum(hours.values())
    return [{"label": k, "clips": c, "pct_clips": round(100 * c / n, 2), "hours": round(hours[k], 2),
             "pct_hours": round(100 * hours[k] / h, 2) if h else 0.0, "videos": len(vids[k])}
            for k, c in clips.most_common()]


def talkvid_stats(rows: list[dict], langs: dict, meta: dict) -> dict:
    """TalkVid-label breakdowns of one split, and how often Whisper's language agrees with TalkVid's."""
    def field(f):
        return lambda r: (meta.get(clip_key(r)) or {}).get(f, "unmatched")
    out = {"matched": sum(meta.get(clip_key(r)) is not None for r in rows),
           "persons": len({meta[clip_key(r)].get("person") for r in rows if meta.get(clip_key(r))}),
           **{f: breakdown(rows, field(f)) for f in ("language", "gender", "ethnicity", "age", "category")}}
    agree = [WHISPER_NAMES.get(langs[k]["lang"]) == meta[k].get("language")
             for k in (clip_key(r) for r in rows) if meta.get(k) and langs.get(k, {}).get("lang", "unk") != "unk"]
    out["whisper_agrees"] = round(float(np.mean(agree)), 4) if agree else None
    return out


def split_stats(rows: list[dict], langs: dict, offsets: SyncOffsets | None, meta: dict | None = None) -> dict:
    frames = [int(r["n"]) for r in rows]
    per_video = Counter(video_of(r) for r in rows)
    hours_video = Counter()
    for r in rows:
        hours_video[video_of(r)] += r["n"] / FPS / 3600
    lang_clips, lang_hours = Counter(), Counter()
    for r in rows:
        code = langs.get(clip_key(r), {}).get("lang", "unk")
        lang_clips[code] += 1
        lang_hours[code] += r["n"] / FPS / 3600
    votes = {}
    for r in rows:                                          # a video's language = its clips' majority
        votes.setdefault(video_of(r), Counter())[langs.get(clip_key(r), {}).get("lang", "unk")] += 1
    lang_videos = Counter(c.most_common(1)[0][0] for c in votes.values())
    counts = sorted(per_video.values(), reverse=True)
    top10 = max(1, len(counts) // 10)
    out = {
        "clips": len(rows), "frames": int(sum(frames)), "hours": round(sum(frames) / FPS / 3600, 2),
        "videos": len(per_video),
        "clip_seconds": {"mean": round(float(np.mean(frames)) / FPS, 2), "median": round(float(np.median(frames)) / FPS, 2),
                         "min": round(min(frames) / FPS, 2), "max": round(max(frames) / FPS, 2),
                         "hist": bin_counts([f / FPS for f in frames], SECONDS_BINS)},
        "clips_per_video": {"mean": round(float(np.mean(counts)), 2), "median": float(np.median(counts)),
                            "max": counts[0], "hist": bin_counts(counts, CLIPS_PER_VIDEO_BINS),
                            "share_top10pct_videos": round(sum(counts[:top10]) / len(rows), 3)},
        "hours_per_video": {"mean": round(float(np.mean(list(hours_video.values()))) * 60, 2),
                            "median": round(float(np.median(list(hours_video.values()))) * 60, 2), "unit": "minutes"},
        "language": {"clips": dict(lang_clips.most_common()), "hours": {k: round(v, 2) for k, v in lang_hours.most_common()},
                     "videos": dict(lang_videos.most_common())},
    }
    code = lambda r: langs.get(clip_key(r), {}).get("lang", "unk")
    out["language_whisper"] = breakdown(rows, lambda r: WHISPER_NAMES.get(code(r), code(r)))
    if meta is not None:
        out["talkvid"] = talkvid_stats(rows, langs, meta)
    if offsets is not None:
        offs = [abs(offsets.get(r)["offset"]) for r in rows if offsets.get(r) and offsets.get(r).get("keep")]
        out["av_offset_abs_frames"] = {"corrected_clips": int(sum(o > 0 for o in offs)),
                                       "hist": bin_counts([round(o) for o in offs], [(0, 0), (1, 1), (2, 2), (3, 10 ** 9)])}
    return out


class WhisperLID:
    """Batched spoken-language ID from Whisper's first decoder step."""

    def __init__(self, name: str, dev: str):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        from transformers.models.whisper.tokenization_whisper import LANGUAGES
        self.p = WhisperProcessor.from_pretrained(name, local_files_only=True)
        self.m = WhisperForConditionalGeneration.from_pretrained(name, local_files_only=True,
                                                                 torch_dtype=torch.float16).to(dev).eval()
        tok, self.dev = self.p.tokenizer, dev
        ids = {c: tok.convert_tokens_to_ids(f"<|{c}|>") for c in LANGUAGES}
        ids = {c: i for c, i in ids.items() if i is not None and i != tok.unk_token_id}
        self.codes, self.ids = list(ids), torch.tensor(list(ids.values()), device=dev)
        self.sot = tok.convert_tokens_to_ids("<|startoftranscript|>")

    @torch.no_grad()
    def __call__(self, wavs: list[np.ndarray]) -> list[tuple[str, float]]:
        x = self.p.feature_extractor(wavs, sampling_rate=16000, return_tensors="pt").input_features
        x = x.to(self.dev, torch.float16)
        dec = torch.full((len(wavs), 1), self.sot, device=self.dev)
        p = self.m(input_features=x, decoder_input_ids=dec).logits[:, -1].float()[:, self.ids].softmax(-1)
        best = p.argmax(-1)
        return [(self.codes[int(b)], float(p[i, b])) for i, b in enumerate(best)]


def label_languages(rows: list[dict], cache: Path, whisper: str, dev: str, batch: int, workers: int) -> dict:
    from sang.naturalness import load_wav
    path = cache / "lang_lid.json"
    langs = json.loads(path.read_text()) if path.exists() else {}
    todo = [r for r in rows if clip_key(r) not in langs]
    print(f"language ID: {len(langs)} cached, {len(todo)} to label", flush=True)
    if not todo:
        return langs
    lid = WhisperLID(whisper, dev)

    def load(r):
        try:
            return load_wav(r["clip"], r.get("start", 0), min(int(r["n"]), 10 * FPS)).numpy()
        except Exception:
            return None

    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        for i in range(0, len(todo), batch):
            chunk = todo[i:i + batch]
            wavs = list(ex.map(load, chunk))
            ok = [(r, w) for r, w in zip(chunk, wavs) if w is not None and len(w) > 1600]
            for r, w in zip(chunk, wavs):
                if w is None or len(w) <= 1600:
                    langs[clip_key(r)] = {"lang": "unk", "p": 0.0, "error": "audio"}
            if ok:
                for (r, _), (code, p) in zip(ok, lid([w for _, w in ok])):
                    langs[clip_key(r)] = {"lang": code, "p": round(p, 3)}
            if (i // batch) % 20 == 0 or i + batch >= len(todo):
                path.write_text(json.dumps(langs))
                done = min(i + batch, len(todo))
                print(f"  {done}/{len(todo)} clips labelled ({time.time() - t0:.0f} s)", flush=True)
    path.write_text(json.dumps(langs))
    return langs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO / "configs" / "train_motion.yaml"))
    ap.add_argument("--sync-offsets", default=None, help="default <cache>/sync_offsets.json if it exists; 'none' = off")
    ap.add_argument("--whisper", default="openai/whisper-large-v3")
    ap.add_argument("--min-prob", type=float, default=0.5)
    ap.add_argument("--no-lid", action="store_true", help="skip language ID (languages become 'unk')")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--talkvid-meta", default=str(DATA_ROOT / "talkvid" / "metadata" / "filtered_video_clips.json"),
                    help="TalkVid's metadata json; 'none' = skip")
    ap.add_argument("--out", default=str(REPO / "results" / "data_stats" / "data_stats.json"))
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    cache = motion_cache(cfg["cache_dir"])
    tm = load_train_module()
    rows = tm.load_index(cache)
    train, val = tm.split_by_speaker(rows, cfg["val_frac"], cfg["seed"])
    so_path = None if args.sync_offsets == "none" else Path(args.sync_offsets or cache / "sync_offsets.json")
    so = SyncOffsets.load(so_path) if so_path and so_path.exists() else None
    if so is not None:
        train_used, drop_t = so.filter(train)
        val_used, drop_v = so.filter(val)
    else:
        train_used, val_used, drop_t, drop_v = train, val, {}, {}
    min_frames = cfg["frames"] + cfg["prefix"]                # MotionWindows needs one full window
    train_used = [r for r in train_used if r["n"] >= min_frames]
    val_used = [r for r in val_used if r["n"] >= min_frames]

    langs = {} if args.no_lid else label_languages(train_used + val_used, cache, args.whisper,
                                                   "cuda" if torch.cuda.is_available() else "cpu",
                                                   args.batch, args.workers)
    langs = {k: (v if v.get("p", 0) >= args.min_prob else {**v, "lang": "unk"}) for k, v in langs.items()}

    meta = None
    if args.talkvid_meta != "none" and Path(args.talkvid_meta).exists():
        lookup = load_lookup(args.talkvid_meta)
        meta = {clip_key(r): match(lookup, video_of(r), Path(r["clip"]).stem) for r in rows}
        meta = {k: v for k, v in meta.items() if v}
        (cache / "talkvid_meta.json").write_text(json.dumps(meta))
        print(f"TalkVid metadata: {len(meta)}/{len(rows)} cached clips matched -> {cache / 'talkvid_meta.json'}", flush=True)

    res = {"source": "TalkVid motion cache", "cache": str(cache), "fps": FPS,
           "split": {"rule": "hash of source-video name", "val_frac": cfg["val_frac"], "seed": cfg["seed"]},
           "sync_offsets": str(so_path) if so is not None else None,
           "cache_total": {"clips": len(rows), "videos": len({video_of(r) for r in rows}),
                           "hours": round(sum(r["n"] for r in rows) / FPS / 3600, 2)},
           "dropped": {"train": drop_t, "val": drop_v},
           "language_id": None if args.no_lid else {"model": args.whisper, "min_prob": args.min_prob,
                                                     "audio": "first <= 10 s of each clip"},
           "talkvid_meta": args.talkvid_meta if meta is not None else None,
           "train": split_stats(train_used, langs, so, meta), "val": split_stats(val_used, langs, so, meta)}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))

    for name in ("train", "val"):
        s = res[name]
        top = ", ".join(f"{k} {v}" for k, v in list(s["language"]["clips"].items())[:8])
        print(f"\n{name}: {s['clips']} clips, {s['hours']} h, {s['videos']} source videos; "
              f"clip {s['clip_seconds']['median']} s median; {s['clips_per_video']['median']:g} clips/video median "
              f"(max {s['clips_per_video']['max']}, top 10% of videos hold {s['clips_per_video']['share_top10pct_videos']:.0%} of clips)")
        print(f"  languages (clips): {top}")
        tables = [("Whisper", s["language_whisper"])]
        if "talkvid" in s:
            t = s["talkvid"]
            tables.insert(0, (f"TalkVid labels, {t['matched']}/{s['clips']} clips matched", t["language"]))
        for title, rows_ in tables:
            print(f"  language ({title}):            clips        hours")
            for b in rows_:
                print(f"    {b['label']:<14} {b['clips']:>7} {b['pct_clips']:>7.2f} %  {b['hours']:>7.2f} {b['pct_hours']:>7.2f} %")
        if "talkvid" in s:
            print(f"  Whisper agrees with TalkVid's language on {100 * (t['whisper_agrees'] or 0):.1f} % of matched clips; "
                  f"{t['persons']} distinct TalkVid persons")
    if meta is not None:
        def persons(rs):
            return {meta[clip_key(r)].get("person") for r in rs if meta.get(clip_key(r))}
        both = persons(train_used) & persons(val_used)
        n_val = sum(meta.get(clip_key(r), {}).get("person") in both for r in val_used)
        print(f"\nTalkVid persons in both train and val: {len(both)} ({n_val}/{len(val_used)} val clips); "
              f"the split is by source video, so one person with several videos can land on both sides")
    print(f"\ncache total {res['cache_total']}; dropped by the offset filter: train {drop_t}, val {drop_v}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
