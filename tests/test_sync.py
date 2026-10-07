"""Audio-video offset correction (sang/sync.py, scripts/sync_offsets.py, the training loader).

CPU only; no SyncNet, video or GPU."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from sang.diagnose import shift_frames
from sang.sync import TICK, SyncOffsets, clip_key, merge, pick_clips, shift_ticks, shift_wav


def _row(video, clip, n=200, root="/c"):
    return {"clip": f"/talkvid/{video}/{clip}.mp4", "path": f"{root}/{video}/{clip}.pt", "n": n, "start": 0}


def test_feature_and_waveform_shifts_agree():
    feat = torch.zeros(40, 3)
    feat[10] = 1.0
    wav = torch.zeros(40 * TICK)
    wav[10 * TICK + 5] = 1.0
    for s, want in ((-4, 6), (3, 13)):              # SyncNet -2 frames (audio late) -> pull forward 4 ticks
        assert int(shift_ticks(feat, s)[:, 0].argmax()) == want
        assert int(shift_wav(wav, s).argmax()) == want * TICK + 5
    a = torch.arange(10.0)[:, None]
    assert torch.equal(shift_ticks(a, -2), shift_frames(a, -1, 2))           # one convention across modules
    assert shift_ticks(a, 2)[:, 0].tolist()[:3] == [0, 0, 0]                  # edge-padded
    assert shift_wav(torch.ones(100), 1, tick=40)[:40].sum() == 0              # zero-padded
    assert shift_wav(torch.ones(100), 9, tick=40).sum() == 0 and torch.equal(shift_ticks(a, 0), a)


def test_merge_rules():
    rows = ([_row("A", f"a{i}") for i in range(4)] + [_row("B", f"b{i}") for i in range(3)]
            + [_row("C", f"c{i}") for i in range(3)] + [_row("D", f"d{i}") for i in range(3)] + [_row("E", "e0")])
    m = lambda v, c, off, conf: {"key": f"{v}/{c}", "video": v, "offset": off, "dist": 7.0, "conf": conf}
    measured = [m("A", "a0", -2, 7.0), m("A", "a3", -2, 6.0),                 # consistent, audio 2 frames late
                m("B", "b0", 0, 8.0), m("B", "b2", -1, 7.0),                  # mean -0.5 -> 1 tick
                m("C", "c0", -15, 0.4), m("C", "c1", -1, 1.0),                # no sync / low confidence
                m("D", "d0", 0, 7.0), m("D", "d1", -4, 7.0)]                  # clips disagree
    res = merge(measured, rows)
    c = res["clips"]
    assert all(c[f"A/a{i}"]["keep"] and c[f"A/a{i}"]["shift_ticks"] == -4 for i in range(4))
    assert all(c[f"B/b{i}"]["shift_ticks"] == -1 and c[f"B/b{i}"]["offset"] == -0.5 for i in range(3))
    assert c["C/c0"]["why"] == "no sync" and c["C/c1"]["why"] == "low confidence" and c["C/c2"]["why"] == "no sync in video"
    assert c["D/d0"]["keep"] and c["D/d0"]["shift_ticks"] == 0 and c["D/d1"]["shift_ticks"] == -8
    assert not c["D/d2"]["keep"] and c["D/d2"]["why"] == "video offsets disagree"
    assert c["E/e0"]["why"] == "not measured"
    meta = res["meta"]
    assert meta["kept"] == 9 and meta["clips"] == 14 and meta["videos_inconsistent"] == 1 and meta["videos_usable"] == 3
    assert abs(meta["frac_kept_abs_ge2"] - 5 / 9) < 1e-9                      # A x4 and d1
    nan = merge([m("A", "a0", float("nan"), float("nan"))], rows[:1])["clips"]["A/a0"]
    assert not nan["keep"] and nan["why"] == "no sync"


def test_pick_clips_spreads_and_filters():
    rows = [_row("V", f"c{i}", n=300) for i in range(5)] + [_row("W", "w0", n=80), _row("W", "w1", n=150)]
    got = [clip_key(r) for r in pick_clips(rows, 2, min_frames=101)]
    assert got == ["V/c0", "V/c4", "W/w1"]
    assert [clip_key(r) for r in pick_clips(rows, 3)][:3] == ["V/c0", "V/c2", "V/c4"]


def test_offsets_file_filter_and_shift():
    rows = [_row("A", "a0"), _row("A", "a1"), _row("Z", "z0")]
    so = SyncOffsets({"clips": {"A/a0": {"keep": True, "shift_ticks": -3}, "A/a1": {"keep": False, "why": "no sync"}}})
    kept, why = so.filter(rows)
    assert [clip_key(r) for r in kept] == ["A/a0"] and why == {"no sync": 1, "not in offsets file": 1}
    assert len(so.filter(rows, keep_unknown=True)[0]) == 2
    assert so.shift(rows[0]) == -3 and so.shift(rows[1]) == 0 and so.shift(rows[2]) == 0


def test_training_loader_moves_the_audio(tmp_path):
    spec = importlib.util.spec_from_file_location("tm", REPO / "scripts" / "train_motion.py")
    tm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tm)
    from sang.motion_model import Norm
    T = 80
    m = torch.zeros(T, 70)
    m[:, 0] = 1.2
    p = tmp_path / "A" / "a0.pt"
    p.parent.mkdir()
    audio = torch.arange(2 * T, dtype=torch.float32)[:, None].repeat(1, 4)
    torch.save({"m": m.half(), "kp": torch.zeros(T, 63).half(), "audio": audio.half(), "n": T}, p)
    row = {"clip": "/x/A/a0.mp4", "path": str(p), "n": T}
    norm = Norm(torch.zeros(42), torch.ones(42), torch.zeros(63), torch.ones(63))
    plain = tm.MotionWindows([row], 64, 10, norm, 1, fixed=True, preload=False)
    moved = tm.MotionWindows([row], 64, 10, norm, 1, fixed=True, preload=False, shifts={"A/a0": -4})
    a0, a1 = plain[0]["audio"].float(), moved[0]["audio"].float()
    assert torch.equal(a1[:-4], a0[4:])                                       # audio pulled forward 2 frames


def test_merge_cli(tmp_path):
    cache = tmp_path / "cache"
    (cache / "sync").mkdir(parents=True)
    rows = [_row("A", "a0", root=str(cache)), _row("A", "a1", root=str(cache)), _row("B", "b0", root=str(cache))]
    (cache / "index_000.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    (cache / "sync" / "offsets_000.jsonl").write_text("\n".join(json.dumps(x) for x in (
        {"key": "A/a0", "video": "A", "offset": -2, "dist": 7, "conf": 6.5},
        {"key": "B/b0", "video": "B", "offset": 15, "dist": 12, "conf": 0.3})))
    out = subprocess.run([sys.executable, str(REPO / "scripts" / "sync_offsets.py"), "--merge", "--cache", str(cache)],
                         capture_output=True, encoding="utf-8", check=True).stdout
    assert "2 kept / 3" in out
    c = json.loads((cache / "sync_offsets.json").read_text())["clips"]
    assert c["A/a1"]["shift_ticks"] == -4 and c["A/a1"]["source"] == "video" and not c["B/b0"]["keep"]
