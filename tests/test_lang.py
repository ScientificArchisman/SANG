"""--lang: train and validate on a subset of languages (sang/data/talkvid.py, scripts/train_motion.py,
scripts/naturalness.py). CPU only."""
import json
import subprocess
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from sang.data import talkvid


def test_parse_langs_accepts_names_codes_commas_and_all():
    assert talkvid.parse_langs(None) is None and talkvid.parse_langs(["all"]) is None and talkvid.parse_langs("all") is None
    assert talkvid.parse_langs(["English"]) == ["English"]
    assert talkvid.parse_langs(["en", "ZH"]) == ["Chinese", "English"]
    assert talkvid.parse_langs(["english,chinese"]) == ["Chinese", "English"]
    assert talkvid.parse_langs(["English", "all"]) is None


def test_clip_language_is_talkvid_else_whisper(tmp_path):
    (tmp_path / "talkvid_meta.json").write_text(json.dumps({
        "v/a": {"language": "English"}, "v/b": {"language": "Unknown"}, "v/c": {"language": "Chinese"}}))
    (tmp_path / "lang_lid.json").write_text(json.dumps({
        "v/a": {"lang": "zh", "p": 0.9}, "v/b": {"lang": "en", "p": 0.8}, "v/d": {"lang": "ko", "p": 0.3},
        "v/e": {"lang": "es", "p": 0.7}}))
    lang = talkvid.clip_languages(tmp_path)
    assert lang == {"v/a": "English", "v/b": "English", "v/c": "Chinese", "v/e": "Spanish"}
    rows = [{"path": f"/c/v/{k}.pt"} for k in "abcdef"]
    keep, dropped = talkvid.filter_langs(rows, ["English"], lang)
    assert [Path(r["path"]).stem for r in keep] == ["a", "b"]
    assert dropped == {"no label": 2, "Chinese": 1, "Spanish": 1}
    assert talkvid.filter_langs(rows, None, lang) == (rows, {})


def test_training_with_lang_filters_train_and_val_and_records_it(tmp_path):
    from sang.data.dataset import split_by_speaker
    from sang.data.paths import load_index
    cache = tmp_path / "cache"
    rows, g = [], torch.Generator().manual_seed(0)
    for s in range(6):
        for c in range(2):
            p = cache / f"spk{s}" / f"c{c}.pt"
            p.parent.mkdir(parents=True, exist_ok=True)
            T = 90
            m = torch.randn(T, 70, generator=g) * 0.05
            m[:, 0] = 1.2
            torch.save({"m": m.half(), "kp": torch.randn(T, 63, generator=g).half(),
                        "audio": torch.randn(2 * T, 1024, generator=g).half(), "n": T}, p)
            rows.append({"clip": f"/x/spk{s}/c{c}.mp4", "path": str(p), "n": T})
    (cache / "index_000.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    train, val = split_by_speaker(load_index(cache), 0.4, 0)
    assert train and val
    (cache / "talkvid_meta.json").write_text(json.dumps(
        {f"spk{s}/c{c}": {"language": "English" if c == 0 else "Chinese", "person": str(s)}
         for s in range(6) for c in range(2)}))
    out = tmp_path / "run"
    over = [f"cache_dir={cache}", f"out_dir={out}", "sync_offsets=", "dim=32", "layers=1", "heads=2", "max_steps=2", "warmup_steps=1",
            "eval_every=2", "log_every=1", "batch_size=2", "workers=0", "eval_batches=1", "sample_steps=2",
            "val_frac=0.4", "windows_per_clip=2"]
    log = subprocess.run([sys.executable, str(REPO / "scripts" / "train_motion.py"), "--lang", "en", *over],
                         capture_output=True, encoding="utf-8", cwd=REPO)
    assert log.returncode == 0, log.stderr[-2000:]
    assert f"train {len(train)} -> {len(train) // 2} clips, val {len(val)} -> {len(val) // 2} clips" in log.stdout
    ck = torch.load(out / "last.pt", weights_only=False)
    assert ck["cfg"]["lang"] == ["English"]

    bad = subprocess.run([sys.executable, str(REPO / "scripts" / "train_motion.py"), "--lang", "Klingon", *over],
                         capture_output=True, encoding="utf-8", cwd=REPO)
    assert bad.returncode != 0 and "Klingon" in bad.stdout + bad.stderr


def test_naturalness_groups_val_clips_by_language_and_person_overlap():
    import importlib.util
    spec = importlib.util.spec_from_file_location("nat", REPO / "scripts" / "naturalness.py")
    nat = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nat)
    row = {"path": "/c/vidA/clip1.pt"}
    assert nat.clip_groups(row, {"vidA/clip1": "Chinese"}, {"vidA/clip1": "7"}, {"3"}) == {"all", "non-English", "no person overlap"}
    assert nat.clip_groups(row, {"vidA/clip1": "English"}, {"vidA/clip1": "7"}, {"7"}) == {"all", "English"}
    assert nat.clip_groups(row, {}, {}, set()) == {"all"}
    s = nat.lip_summary([{"lip_corr": 0.5, "lip_ccc": 0.4}, {"lip_corr": float("nan"), "lip_ccc": 0.6}])
    assert s == {"clips": 2, "lip_corr": 0.5, "lip_ccc": 0.5}

