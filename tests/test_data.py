"""Data pipeline: audio/video alignment, the derived tick count, and the face crop."""
import glob
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from sang.config import load_config
from sang.data import (SR, audio_samples, clip_cpu_windows, clip_tokens, cpu_task,
                       mel_spectrogram, window_starts)
from sang.video import load_vidtok

CLIPS = "/beegfs/work_fast/shared/li_shared/archi_data/talkvid/clips/*/*.mp4"


def test_audio_span_matches_video_span():
    """N frames span N-1 intervals: 17 @ 25 fps is 0.64 s, not 0.68 s (Part VI, D8)."""
    assert audio_samples(17, 25.0, 16000) == 10240
    assert audio_samples(17, 25.0, 16000) / 16000 == (17 - 1) / 25.0


def test_config_derives_the_real_tick_count():
    """ta must match what WavLM actually emits, not a hardcoded default (Part VI, D12)."""
    cfg = load_config(REPO / "configs/train.yaml")
    assert cfg["ta"] == 31, cfg["ta"]
    assert cfg["spatial"] == cfg["res"] // 8


def test_clip_tokens_alignment():
    """Real clip -> aligned video tokens + WavLM features + ref (skips if no data)."""
    files = sorted(glob.glob(CLIPS))
    if not files:
        print("no TalkVid data; skipping")
        return
    from sang.codec import load_wavlm

    frames, res, k = 17, 128, 32768
    vidtok = load_vidtok(codebook=k, device="cpu")
    out = clip_tokens(files[0], vidtok, load_wavlm(device="cpu"), frames=frames, res=res)

    assert out["video"].shape == (1, (frames - 1) // 4 + 1, res // 8, res // 8)
    assert 0 <= int(out["video"].min()) and int(out["video"].max()) < k
    assert out["ref"].shape == (1, 3, res, res)
    assert out["audio"].shape[0] == 1 and out["audio"].shape[-1] == 1024
    assert out["audio"].shape[1] == 31  # 0.64 s of WavLM at 50 Hz


def test_window_starts_span_the_clip():
    st = window_starts(1000, 17, 8)
    assert len(st) == 8 and st[0] == 0 and st[-1] == 1000 - 17
    assert all(b > a for a, b in zip(st, st[1:]))
    assert window_starts(20, 17, 1) == [0]


def test_cpu_task_reports_bad_clip_instead_of_raising():
    path, result, err = cpu_task(("windows", "/nonexistent/clip.mp4", {}))
    assert result is None and err


def test_mel_is_cut_from_the_same_window_as_the_audio():
    """Regression: mels used to be cut from packed-from-t=0 starts after the video windows were
    spread across the clip, so every window but the first paired video with the wrong second of
    audio. Both must come from the same spread start."""
    files = sorted(glob.glob(CLIPS))
    if not files:
        print("no TalkVid data; skipping")
        return
    import torch
    wins = clip_cpu_windows(files[0], frames=17, res=64, fps=25, max_windows=4, with_mel=True)
    assert len(wins) >= 2
    w1 = wins[1]
    assert w1["start"] > 0                                          # spread, not packed
    assert w1["pixels"].shape == (17, 64, 64, 3) and w1["wav"].shape == (1, audio_samples(17, 25, SR))
    expect = mel_spectrogram(torch.from_numpy(w1["wav"]))[0].half()
    assert torch.equal(w1["mel"], expect)


if __name__ == "__main__":
    test_audio_span_matches_video_span()
    test_config_derives_the_real_tick_count()
    test_clip_tokens_alignment()
    test_window_starts_span_the_clip()
    test_cpu_task_reports_bad_clip_instead_of_raising()
    test_mel_is_cut_from_the_same_window_as_the_audio()
    print("ok")
