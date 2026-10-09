"""TalkVid's own per-clip metadata (<DATA_ROOT>/talkvid/metadata/filtered_video_clips.json): person ID,
language, gender, ethnicity, age group and video category, as labelled by the TalkVid authors.

Our clips are named <video_id>_<tag>_<start>_<end>.mp4 (e.g. -04ZSRBGcsk_NA_1150.624_1155.846) and live in
a folder named after the YouTube video id; metadata entries carry the YouTube link and start/end times
in seconds. A clip matches the entry of the same video whose start and end are both within `tol` seconds.
"""
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

FIELDS = {"Person ID": "person", "Language": "language", "Gender": "gender", "Ethnicity": "ethnicity",
          "Age Group": "age", "Video Category": "category"}

# Whisper language codes -> TalkVid's language names, for comparing the two labels
WHISPER_NAMES = {"en": "English", "zh": "Chinese", "ko": "Korean", "es": "Spanish", "hi": "Hindi", "ja": "Japanese",
                 "ms": "Malay", "cy": "Welsh", "sw": "Swahili", "ur": "Urdu", "fr": "French", "de": "German",
                 "ar": "Arabic", "ru": "Russian", "pt": "Portuguese", "it": "Italian", "pl": "Polish", "tr": "Turkish",
                 "vi": "Vietnamese", "th": "Thai", "id": "Indonesian", "nl": "Dutch", "yue": "Cantonese"}


def video_id(link: str) -> str:
    u = urlparse(link)
    v = parse_qs(u.query).get("v")
    return v[0] if v else u.path.rstrip("/").split("/")[-1]


def clip_times(stem: str) -> tuple[float, float] | None:
    """'-04ZSRBGcsk_NA_1150.624_1155.846' -> (1150.624, 1155.846); None if the name carries no times."""
    parts = stem.rsplit("_", 2)
    try:
        return float(parts[1]), float(parts[2])
    except (IndexError, ValueError):
        return None


def build_lookup(entries: list[dict]) -> dict[str, list[tuple[float, float, dict]]]:
    """Metadata entries -> {video id: [(start, end, {person, language, ...}), ...]}."""
    out = {}
    for e in entries:
        info = e.get("info", {})
        vid = video_id(info.get("Video Link", ""))
        if not vid:
            continue
        labels = {short: str(info[k]) for k, short in FIELDS.items() if k in info}
        out.setdefault(vid, []).append((float(e["start-time"]), float(e["end-time"]), labels))
    return out


def load_lookup(path) -> dict[str, list[tuple[float, float, dict]]]:
    return build_lookup(json.loads(Path(path).read_text()))


def match(lookup: dict, vid: str, stem: str, tol: float = 0.05) -> dict | None:
    """TalkVid labels of one of our clips (folder = video id, stem = clip name), or None."""
    t = clip_times(stem)
    if t is None or vid not in lookup:
        return None
    s, e = t
    best = min(lookup[vid], key=lambda x: abs(x[0] - s) + abs(x[1] - e))
    return best[2] if abs(best[0] - s) <= tol and abs(best[1] - e) <= tol else None
