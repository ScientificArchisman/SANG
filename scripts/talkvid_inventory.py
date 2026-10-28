#!/usr/bin/env python3
"""What TalkVid contains, how much of it SANG has, and which clips to download next.

    python scripts/talkvid_inventory.py                                          # report only
    python scripts/talkvid_inventory.py --target-hours 100 --select <list.json>  # + a download list
    # -> results/talkvid_inventory/inventory.json (+ a printed report)

Inputs (defaults from sang/paths.py):
  --meta   TalkVid's full metadata, one entry per clip: YouTube link, start/end time, size, fps, quality
           scores and labels (<DATA_ROOT>/talkvid/metadata/filtered_video_clips.json)
  --clips  the clips on disk, <video id>/<video id>_<n>_<start>_<end>.mp4 + .m4a (TalkVid's downloader layout)
  --cache  the motion cache index (index_*.jsonl): what training actually sees

Report:
  1. TalkVid as published: clips, hours, videos, persons; clip length, resolution, fps, quality scores;
     language, gender, ethnicity, age, category; hours per video and per person
  2. what we have: clips and hours on disk, how many match a metadata entry, disk per hour, the resolution
     actually downloaded (a sample), and the cache: hours cached, downloaded clips that are not cached,
     and the length of cached clips the cache cut off (shot cuts, --max-frames)
  3. coverage of TalkVid by language
  4. the training split, and what TalkVid has that we could still add
  5. why downloaded clips are not cached: never queued (not in --queued), rejected by
     scripts/filter_clips.py (--filtered, reasons from --filter-report), failed in scripts/cache_motion.py
     ('skip' lines in --cache-logs), or passed the filter and were never cached. With --cache-list: the
     usable ones (never queued, or passed and never cached), train-split videos only and no val person,
     one path per line: run scripts/filter_clips.py on it, then scripts/cache_motion.py. Long clips: the
     hours cache_motion.py --chunks K would add for several K, and with --chunk-list the cached train-side
     clips that have footage left after their cached part
  6. with --target-hours: a download list in TalkVid's own entry format, so TalkVid's
     data_pipeline/0_video_download/download_clips.py takes it as --input. It adds the missing hours
     (target - hours cached now), counted up to --max-seconds per clip, times --overshoot for removed
     videos and clips that fail filtering or caching:
       - only videos that scripts/train_motion.py's split puts in TRAIN (same hash, val_frac, seed), and
         no person from our val set: the val set and every earlier val number stay as they are;
       - only persons we do not have yet (--known-persons also allows the training persons we have);
       - diversity first: rounds of one clip per person, persons in a seeded random order, a person's
         videos interleaved, until the hours are reached.
"""
import argparse
import glob
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from itertools import zip_longest
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from sang.data import split_by_speaker
from sang.paths import DATA_ROOT, MOTION_CACHE, TALKVID_CLIPS, load_index
from sang.talkvid import clip_times, video_id

FPS = 25
LENGTH_BINS = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 30), (30, 10 ** 9)]
SIDE_BINS = [(0, 720, "< 720p"), (720, 1080, "720p"), (1080, 1440, "1080p"), (1440, 2160, "1440p"), (2160, 10 ** 9, "4K")]
LABELS = {"language": "Language", "gender": "Gender", "ethnicity": "Ethnicity", "age": "Age Group",
          "category": "Video Category"}


def num(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def person_of(info: dict, vid: str) -> str:
    """TalkVid's Person ID ('420.0' and '420' are the same person); the video when it has none."""
    p = str(info.get("Person ID") or "").strip()
    if re.fullmatch(r"\d+\.0", p):
        p = p[:-2]
    return p or f"video:{vid}"


def records(entries: list) -> list[dict]:
    """Metadata entries -> flat records, one per clip with a video id and end > start."""
    out = []
    for i, e in enumerate(entries):
        info = e.get("info") or {}
        vid = video_id(str(info.get("Video Link") or ""))
        try:
            s, t = float(e["start-time"]), float(e["end-time"])
        except (KeyError, TypeError, ValueError):
            continue
        if not vid or t <= s:
            continue
        h, w = num(e.get("height")), num(e.get("width"))
        out.append({"i": i, "vid": vid, "start": s, "end": t, "dur": t - s, "person": person_of(info, vid),
                    "side": min(h, w), "portrait": h > w, "fps": num(e.get("fps")),
                    "dover": num(e["dover_scores"]) if e.get("dover_scores") is not None else None,
                    "cotracker": num(e["cotracker_ratio"]) if e.get("cotracker_ratio") is not None else None,
                    **{k: str(info.get(v) or "missing") for k, v in LABELS.items()}})
    return out


def scan_clips(root: Path) -> list[dict]:
    """Clips on disk: one record per <video>/<clip>.mp4, with the .m4a's size added to its bytes."""
    out = []
    if not root.exists():
        return out
    for d in os.scandir(root):
        if not d.is_dir():
            continue
        files = {f.name: f.stat().st_size for f in os.scandir(d.path) if f.is_file()}
        for name, size in files.items():
            if not name.endswith(".mp4"):
                continue
            stem = name[:-4]
            t = clip_times(stem)
            out.append({"vid": d.name, "stem": stem, "path": os.path.join(d.path, name),
                        "start": t[0] if t else None, "dur": (t[1] - t[0]) if t else None,
                        "bytes": size + files.get(stem + ".m4a", 0)})
    return out


def match_clips(clips: list[dict], recs: list[dict], tol: float = 0.05) -> None:
    """clip['rec'] = index of the metadata record with the same video and start/end within tol s, else None."""
    by_vid = defaultdict(list)
    for j, r in enumerate(recs):
        by_vid[r["vid"]].append(j)
    for c in clips:
        c["rec"] = None
        if c["start"] is None or c["vid"] not in by_vid:
            continue
        e = c["start"] + c["dur"]
        j = min(by_vid[c["vid"]], key=lambda j: abs(recs[j]["start"] - c["start"]) + abs(recs[j]["end"] - e))
        if abs(recs[j]["start"] - c["start"]) <= tol and abs(recs[j]["end"] - e) <= tol:
            c["rec"] = j


def breakdown(recs: list[dict], key, top: int = 0) -> list[dict]:
    clips, secs, persons = Counter(), Counter(), defaultdict(set)
    for r in recs:
        k = key(r)
        clips[k] += 1
        secs[k] += r["dur"]
        persons[k].add(r["person"])
    tot = sum(secs.values()) or 1.0
    rows = [{"label": k, "clips": clips[k], "hours": secs[k] / 3600, "pct_hours": 100 * secs[k] / tot,
             "persons": len(persons[k])} for k, _ in secs.most_common()]
    if top and len(rows) > top:
        rest = rows[top:]
        rows = rows[:top] + [{"label": f"other ({len(rest)})", "clips": sum(r["clips"] for r in rest),
                              "hours": sum(r["hours"] for r in rest), "pct_hours": sum(r["pct_hours"] for r in rest),
                              "persons": sum(r["persons"] for r in rest)}]
    return rows


def bin_label(v: float, bins) -> str:
    for b in bins:
        if b[0] <= v < b[1]:
            return b[2] if len(b) > 2 else (f"{b[0]}-{b[1]} s" if b[1] < 10 ** 9 else f">= {b[0]} s")
    return "unknown"


def quant(xs, qs=(0, 10, 50, 90, 100)) -> dict:
    xs = [x for x in xs if x is not None]
    return {f"p{q}": float(np.percentile(xs, q)) for q in qs} if xs else {}


def summary(recs: list[dict]) -> dict:
    per_vid, per_person = Counter(), Counter()
    for r in recs:
        per_vid[r["vid"]] += r["dur"]
        per_person[r["person"]] += r["dur"]
    return {"clips": len(recs), "hours": sum(r["dur"] for r in recs) / 3600, "videos": len(per_vid),
            "persons": len(per_person), "clip_seconds": quant([r["dur"] for r in recs]),
            "minutes_per_video": quant([v / 60 for v in per_vid.values()]),
            "minutes_per_person": quant([v / 60 for v in per_person.values()])}


def print_table(title: str, rows: list[dict], cols=("clips", "hours", "pct_hours", "persons")) -> None:
    print(f"\n{title}")
    w = max([len(str(r["label"])) for r in rows] + [10])
    print(f"  {'':<{w}}" + "".join(f"{c.replace('pct_hours', '% hours'):>11}" for c in cols))
    for r in rows:
        print(f"  {str(r['label']):<{w}}" + "".join(
            f"{r[c]:>11.1f}" if isinstance(r[c], float) else f"{r[c]:>11,}" for c in cols))


def fmt_q(q: dict, unit: str = "") -> str:
    return " / ".join(f"{q[k]:.1f}" for k in ("p0", "p10", "p50", "p90", "p100")) + f" {unit} (min / p10 / median / p90 / max)"


def probe(paths: list[str]) -> list[dict]:
    """Width, height and fps of a few downloaded clips (OpenCV); [] without OpenCV."""
    try:
        import cv2
    except ImportError:
        return []
    out = []
    for p in paths:
        cap = cv2.VideoCapture(p)
        if cap.isOpened():
            out.append({"w": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "h": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                        "fps": round(cap.get(cv2.CAP_PROP_FPS), 2)})
        cap.release()
    return out


def read_names(path) -> set[str] | None:
    """Clip file names in a list of paths (paths may predate a data move); None if the list is missing."""
    p = Path(path)
    return {Path(l.strip()).name for l in p.read_text().splitlines() if l.strip()} if p.exists() else None


def cache_failures(pattern: str) -> dict[str, str]:
    """'  skip <clip>.mp4: <Error>: <message>' lines of cache_motion.py logs -> {clip name: reason}."""
    out = {}
    for f in glob.glob(pattern):
        with open(f, errors="replace") as fh:
            for line in fh:
                m = re.match(r"\s+skip (\S+\.mp4): (.*)", line)
                if m:
                    out[m[1]] = re.sub(r"\d+(\.\d+)?", "N", m[2].strip())[:60]
    return out


def select(recs: list[dict], eligible: list[int], need_hours: float, max_seconds: float, seed: int) -> list[int]:
    """Diversity first: rounds of one clip per person (seeded person order, a person's videos interleaved)
    until the clips add up to need_hours, each clip counted up to max_seconds."""
    rng = random.Random(seed)
    by_person = defaultdict(lambda: defaultdict(list))
    for j in eligible:
        by_person[recs[j]["person"]][recs[j]["vid"]].append(j)
    order = sorted(by_person)
    rng.shuffle(order)
    queue = {}
    for p in order:
        vids = sorted(by_person[p])
        rng.shuffle(vids)
        lists = []
        for v in vids:
            js = sorted(by_person[p][v])
            rng.shuffle(js)
            lists.append(js)
        queue[p] = [j for grp in zip_longest(*lists) for j in grp if j is not None]
    picked, secs, rnd = [], 0.0, 0
    while secs < need_hours * 3600:
        more = False
        for p in order:
            if rnd < len(queue[p]):
                j = queue[p][rnd]
                picked.append(j)
                secs += min(recs[j]["dur"], max_seconds)
                more = True
                if secs >= need_hours * 3600:
                    break
        if not more:
            break
        rnd += 1
    return picked


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meta", default=str(DATA_ROOT / "talkvid" / "metadata" / "filtered_video_clips.json"))
    ap.add_argument("--clips", default=str(TALKVID_CLIPS))
    ap.add_argument("--cache", default=str(MOTION_CACHE))
    ap.add_argument("--config", default=str(REPO / "configs/train_motion.yaml"), help="val_frac and seed of the split")
    ap.add_argument("--probe", type=int, default=100, help="downloaded clips whose resolution is read (0 = none)")
    ap.add_argument("--target-hours", type=float, default=0.0, help="total cached hours wanted; 0 = report only")
    ap.add_argument("--max-seconds", type=float, default=20.0,
                    help="a clip counts up to this long (cache_motion.py keeps --max-frames / 25 s: 500 -> 20 s)")
    ap.add_argument("--overshoot", type=float, default=1.3,
                    help="download this many times the missing hours (removed videos, clips failing filter/cache)")
    ap.add_argument("--min-seconds", type=float, default=3.0, help="shorter clips are not selected (74 frames ~ 3 s)")
    ap.add_argument("--lang", nargs="+", default=None, help="only select these TalkVid languages, e.g. English")
    ap.add_argument("--known-persons", action="store_true", help="also select training persons we already have")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--select", default=None, help="write the download list here (TalkVid entry format)")
    ap.add_argument("--queued", default=str(REPO / "data/clips_all.txt"), help="clip list given to filter_clips.py")
    ap.add_argument("--filtered", default=str(REPO / "data/clips_filtered_all.txt"), help="clips it kept")
    ap.add_argument("--filter-report", default=str(REPO / "data/clips_filtered_all.report.jsonl"))
    ap.add_argument("--cache-logs", default=str(REPO / "slurm_logs/*.out"), help="glob of cache_motion.py logs")
    ap.add_argument("--cache-list", default=None, help="write the usable not-cached clips here, one path per line")
    ap.add_argument("--chunk-list", default=None,
                    help="write the cached train-side clips with footage left after their cached part (for --chunks)")
    ap.add_argument("--out", default=str(REPO / "results/talkvid_inventory/inventory.json"))
    args = ap.parse_args()
    if args.target_hours and not args.select:
        ap.error("--target-hours needs --select <list.json>")
    rep = {}

    t0 = time.time()
    entries = json.loads(Path(args.meta).read_text())
    recs = records(entries)
    print(f"{args.meta}: {len(entries):,} entries, {len(recs):,} usable, read in {time.time() - t0:.0f} s")
    rep["talkvid"] = s = summary(recs)
    print(f"\n1. TALKVID: {s['clips']:,} clips, {s['hours']:,.1f} h, {s['videos']:,} videos, {s['persons']:,} persons")
    print(f"  clip length      {fmt_q(s['clip_seconds'], 's')}")
    print(f"  per video        {fmt_q(s['minutes_per_video'], 'min')}")
    print(f"  per person       {fmt_q(s['minutes_per_person'], 'min')}")
    for name, key in (("clip length", lambda r: bin_label(r["dur"], LENGTH_BINS)),
                      ("resolution (short side)", lambda r: bin_label(r["side"], SIDE_BINS)),
                      ("orientation", lambda r: "portrait" if r["portrait"] else "landscape"),
                      ("fps", lambda r: f"{r['fps']:.0f}" if r["fps"] else "missing")):
        rows = breakdown(recs, key, top=8)
        rep["talkvid"][name] = rows
        print_table(f"  by {name}", rows)
    rep["talkvid"]["dover"] = quant([r["dover"] for r in recs])
    rep["talkvid"]["cotracker"] = quant([r["cotracker"] for r in recs])
    if rep["talkvid"]["dover"]:
        print(f"\n  DOVER quality    {fmt_q(rep['talkvid']['dover'])}")
    if rep["talkvid"]["cotracker"]:
        print(f"  CoTracker ratio  {fmt_q(rep['talkvid']['cotracker'])}")
    for k in LABELS:
        rows = breakdown(recs, lambda r: r[k], top=15)
        rep["talkvid"][k] = rows
        print_table(f"  by {k}", rows)

    clips = scan_clips(Path(args.clips))
    match_clips(clips, recs)
    have = [c for c in clips if c["rec"] is not None]
    for c in have:
        recs[c["rec"]]["have"] = True
    disk_h = sum(c["dur"] or 0 for c in clips) / 3600
    gb = sum(c["bytes"] for c in clips) / 1e9
    rows = load_index(Path(args.cache)) if any(Path(args.cache).glob("index_*.jsonl")) else []
    cached, cached_end, cached_chunks = Counter(), {}, Counter()
    for r in rows:
        k = (Path(r["clip"]).parent.name, Path(r["clip"]).stem)
        cached[k] += r["n"] / FPS
        cached_end[k] = max(cached_end.get(k, 0.0), (r.get("start", 0) + r["n"]) / FPS)
        cached_chunks[k] += 1
    cached_h = sum(cached.values()) / 3600
    on_disk = {(c["vid"], c["stem"]): c for c in clips}
    not_cached = [c for k, c in on_disk.items() if k not in cached]
    cut_off = sum(max(0.0, (on_disk[k]["dur"] or 0) - v) for k, v in cached.items() if k in on_disk) / 3600
    sample = probe([c["path"] for c in random.Random(0).sample(clips, min(args.probe, len(clips)))])
    rep["ours"] = {"clips_on_disk": len(clips), "videos_on_disk": len({c["vid"] for c in clips}), "hours_on_disk": disk_h,
                   "matched": len(have), "gb_on_disk": gb, "gb_per_hour": gb / disk_h if disk_h else None,
                   "cached_clips": len(cached), "cached_hours": cached_h,
                   "cached_not_on_disk": sum(k not in on_disk for k in cached),
                   "on_disk_not_cached": len(not_cached), "on_disk_not_cached_hours": sum(c["dur"] or 0 for c in not_cached) / 3600,
                   "cut_off_hours": cut_off, "probe": sample}
    o = rep["ours"]
    print(f"\n2. OURS on disk ({args.clips}): {o['clips_on_disk']:,} clips, {o['videos_on_disk']:,} videos, "
          f"{disk_h:.2f} h, {gb:.1f} GB ({o['gb_per_hour'] or 0:.1f} GB/h)")
    print(f"  matched to a TalkVid entry: {len(have):,} ({100 * len(have) / max(1, len(clips)):.1f} %)")
    if sample:
        sides = Counter(bin_label(min(p["w"], p["h"]), SIDE_BINS) for p in sample)
        fpss = Counter(f"{p['fps']:g}" for p in sample)
        print(f"  downloaded resolution ({len(sample)} clips): " + ", ".join(f"{k} {v}" for k, v in sides.most_common())
              + "; fps " + ", ".join(f"{k} {v}" for k, v in fpss.most_common(5)))
    print(f"  cache ({args.cache}): {len(cached):,} clips, {cached_h:.2f} h")
    print(f"  downloaded but not cached (filtered out, failed, or never queued): {o['on_disk_not_cached']:,} clips, "
          f"{o['on_disk_not_cached_hours']:.2f} h")
    print(f"  cached clips' length the cache cut off (shot cuts, --max-frames): {cut_off:.2f} h")
    mine = [recs[c["rec"]] for c in have]
    if mine:
        rep["ours"]["summary"] = summary(mine)
        print(f"  matched clips: {fmt_q(rep['ours']['summary']['clip_seconds'], 's')}")

    all_l = {r["label"]: r for r in breakdown(recs, lambda r: r["language"])}
    our_l = {r["label"]: r for r in breakdown(mine, lambda r: r["language"])}
    rep["coverage"] = []
    print("\n3. COVERAGE by language (our matched clips on disk / TalkVid)")
    print(f"  {'language':<16}{'TalkVid h':>11}{'ours h':>9}{'%':>7}{'persons':>17}")
    for lang, r in list(all_l.items())[:16]:
        u = our_l.get(lang, {"hours": 0.0, "persons": 0})
        rep["coverage"].append({"language": lang, "talkvid_hours": r["hours"], "our_hours": u["hours"],
                                "talkvid_persons": r["persons"], "our_persons": u["persons"]})
        print(f"  {lang:<16}{r['hours']:>11.1f}{u['hours']:>9.2f}{100 * u['hours'] / r['hours']:>7.1f}"
              f"{u['persons']:>8,} / {r['persons']:<8,}")
    tot_ours = sum(r["dur"] for r in mine) / 3600
    print(f"  {'all':<16}{s['hours']:>11.1f}{tot_ours:>9.2f}{100 * tot_ours / s['hours']:>7.1f}")

    cfg = yaml.safe_load(Path(args.config).read_text())
    split = lambda vids: split_by_speaker([{"path": f"/x/{v}/c.pt"} for v in vids], cfg["val_frac"], cfg["seed"])
    tr, va = split(sorted({r["vid"] for r in recs} | {c["vid"] for c in clips}))
    val_vids = {Path(r["path"]).parent.name for r in va}
    person_of_clip = {(c["vid"], c["stem"]): recs[c["rec"]]["person"] for c in have}
    val_persons = {p for (v, _), p in person_of_clip.items() if v in val_vids}
    known = set(person_of_clip.values())
    langs = set(args.lang) if args.lang else None
    eligible = [j for j, r in enumerate(recs)
                if not r.get("have") and r["vid"] not in val_vids and r["person"] not in val_persons
                and (args.known_persons or r["person"] not in known)
                and r["dur"] >= args.min_seconds and r["start"] >= 0.001
                and (langs is None or r["language"] in langs)]
    pool = [recs[j] for j in eligible]
    rep["split"] = {"val_frac": cfg["val_frac"], "seed": cfg["seed"], "val_videos_ours": len(val_vids & {c["vid"] for c in clips}),
                    "val_persons_ours": len(val_persons), "known_persons": len(known), "pool": summary(pool) if pool else {}}
    print(f"\n4. SPLIT (val_frac {cfg['val_frac']}, seed {cfg['seed']}): our val = {rep['split']['val_videos_ours']} videos, "
          f"{len(val_persons)} persons; we have {len(known):,} persons")
    if pool:
        p = rep["split"]["pool"]
        print(f"  can add ({'all languages' if langs is None else ', '.join(sorted(langs))}; train videos only; no val "
              f"person{'' if args.known_persons else '; new persons only'}): {p['clips']:,} clips, {p['hours']:,.1f} h, "
              f"{p['videos']:,} videos, {p['persons']:,} persons")
        print_table("  pool by language", breakdown(pool, lambda r: r["language"], top=10))

    queued, kept = read_names(args.queued), read_names(args.filtered)
    rejected = {}
    if Path(args.filter_report).exists():
        for l in Path(args.filter_report).read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                if r.get("reject"):
                    rejected[Path(r["clip"]).name] = r["reject"]
    failed = cache_failures(args.cache_logs)
    rec_of = {(c["vid"], c["stem"]): c["rec"] for c in clips}
    classes, usable = defaultdict(list), []
    for c in not_cached:
        name = c["stem"] + ".mp4"
        j = rec_of[(c["vid"], c["stem"])]
        person = recs[j]["person"] if j is not None else f"video:{c['vid']}"
        if c["vid"] in val_vids:
            why = "left out: val-split video"
        elif person in val_persons:
            why = "left out: person in our val set"
        elif queued is not None and name not in queued:
            why = "never queued (not filtered yet)"
        elif name in rejected:
            why = f"filter rejected: {rejected[name]}"
        elif kept is not None and name not in kept:
            why = "filter rejected: decode / no face"
        elif name in failed:
            why = f"cache failed: {failed[name]}"
        else:
            why = "passed the filter, never cached" if kept is not None else "unknown (no filter lists)"
        classes[why].append(c)
        if why.startswith(("never queued", "passed the filter", "unknown")):
            usable.append(c)
    hrs = lambda cs: sum(c["dur"] or 0 for c in cs) / 3600
    capped = lambda cs: sum(min(c["dur"] or 0, args.max_seconds) for c in cs) / 3600
    rep["not_cached"] = {k: {"clips": len(v), "hours": hrs(v), "hours_counted": capped(v)}
                         for k, v in sorted(classes.items(), key=lambda kv: -hrs(kv[1]))}
    rep["not_cached_usable"] = {"clips": len(usable), "hours": hrs(usable), "hours_counted": capped(usable),
                                "videos": len({c["vid"] for c in usable}), "file": args.cache_list}
    print(f"\n5. NOT CACHED: {len(not_cached):,} downloaded clips, {hrs(not_cached):.2f} h "
          f"(lists: queued {'yes' if queued is not None else 'MISSING'}, filtered {'yes' if kept is not None else 'MISSING'}, "
          f"report {len(rejected):,} rejections, cache logs {len(failed):,} failures)")
    print(f"  {'':<52}{'clips':>9}{'hours':>9}{'counted':>9}")
    for k, v in rep["not_cached"].items():
        print(f"  {k[:52]:<52}{v['clips']:>9,}{v['hours']:>9.2f}{v['hours_counted']:>9.2f}")
    u = rep["not_cached_usable"]
    print(f"  usable (never queued + passed and never cached, train side): {u['clips']:,} clips, {u['hours']:.2f} h, "
          f"{u['hours_counted']:.2f} h counted (up to {args.max_seconds:g} s per clip), {u['videos']:,} videos")
    if args.cache_list:
        out = Path(args.cache_list)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(sorted(c["path"] for c in usable)) + "\n")
        print(f"  wrote {out} ({len(usable):,} clips)")

    long_ = []
    for k, end in cached_end.items():
        c = on_disk.get(k)
        if c is None or c["dur"] is None or c["vid"] in val_vids:
            continue
        j = c["rec"]
        if (recs[j]["person"] if j is not None else f"video:{c['vid']}") in val_persons:
            continue
        if c["dur"] - end >= args.min_seconds:
            long_.append((c, end, cached_chunks[k]))
    rep["chunks"] = []
    print(f"\n  LONG CLIPS: {len(long_):,} cached train-side clips have >= {args.min_seconds:g} s after their cached part "
          f"({sum(c['dur'] - e for c, e, _ in long_) / 3600:.2f} h). With cache_motion.py --chunks K "
          f"(segments of up to {args.max_seconds:g} s; an upper bound, shot cuts take some):")
    print(f"  {'K':>4}{'from cached clips':>19}{'from the usable list':>22}{'cache total':>13}{'new frames':>12}")
    for K in (1, 2, 3, 4, 6, 10):
        more = sum(min(c["dur"] - e, max(0, K - n) * args.max_seconds) for c, e, n in long_) / 3600
        new_h = sum(min(c["dur"] or 0, K * args.max_seconds) for c in usable) / 3600
        rep["chunks"].append({"K": K, "from_cached_h": more, "from_usable_h": new_h, "total_h": cached_h + more + new_h})
        print(f"  {K:>4}{more:>17.1f} h{new_h:>20.1f} h{cached_h + more + new_h:>11.1f} h{(more + new_h) * 3600 * FPS / 1e6:>10.1f} M")
    if args.chunk_list:
        out = Path(args.chunk_list)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(sorted(c["path"] for c, _, _ in long_)) + "\n")
        print(f"  wrote {out} ({len(long_):,} clips; run cache_motion.py --chunks K on it, chunk 0 is skipped)")

    if args.target_hours:
        missing = args.target_hours - cached_h
        need = max(0.0, missing) * args.overshoot
        picked = select(recs, eligible, need, args.max_seconds, args.seed)
        sel = [recs[j] for j in picked]
        raw_h = sum(r["dur"] for r in sel) / 3600
        cap_h = sum(min(r["dur"], args.max_seconds) for r in sel) / 3600
        per_person = Counter(r["person"] for r in sel)
        rep["selection"] = {"target_hours": args.target_hours, "cached_hours": cached_h, "missing_hours": missing,
                            "overshoot": args.overshoot, "max_seconds": args.max_seconds, "clips": len(sel),
                            "hours": raw_h, "hours_counted": cap_h, "videos": len({r["vid"] for r in sel}),
                            "persons": len(per_person), "clips_per_person": quant(list(per_person.values())),
                            "est_gb": raw_h * (o["gb_per_hour"] or 0), "language": breakdown(sel, lambda r: r["language"], top=10),
                            "file": args.select}
        q = rep["selection"]
        print(f"\n6. DOWNLOAD LIST for {args.target_hours:g} h: missing {missing:.1f} h x {args.overshoot:g} = {need:.1f} h counted "
              f"(clips up to {args.max_seconds:g} s)")
        print(f"  {q['clips']:,} clips, {raw_h:.1f} h ({cap_h:.1f} h counted), {q['videos']:,} videos (one yt-dlp call each), "
              f"{q['persons']:,} persons, clips per person {fmt_q(q['clips_per_person'])}")
        if o["gb_per_hour"]:
            print(f"  disk at our current {o['gb_per_hour']:.1f} GB/h: ~{q['est_gb']:.0f} GB")
        if cap_h < need - 1e-6:
            print(f"  ! the pool has only {cap_h:.1f} of the {need:.1f} h: allow more (--known-persons, all languages)")
        print_table("  selection by language", q["language"])
        out = Path(args.select)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps([entries[r["i"]] for r in sorted(sel, key=lambda r: (r["vid"], r["start"]))]))
        print(f"  wrote {out} ({len(sel):,} entries)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1, default=float))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

