"""Data pipeline: audio/video alignment, the derived tick count, and the face crop."""
import glob
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from _pooled_helpers import crash_on_three, wedge_on_three, worker_pid
from sang.config import load_config
from sang.data import (MEL_VERSION, SR, SYNC_MEL_OFFSET, audio_samples, clip_cpu_windows,
                       clip_mels, clip_tokens, cpu_task, mel_spectrogram, pooled, window_starts)
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


def test_mel_is_wav2lip_format():
    """stable_syncnet.pt was trained on the Wav2Lip mel (hop 200, n_fft 800, 55-7600 Hz, preemphasis
    0.97, dB normalised to [-4, 4]). The natural-log power mel it was fed before had values in
    [-10, 9] and scored ground-truth windows 0.06 worse than this one (probe, 2026-09-10)."""
    import numpy as np
    import torch
    torch.manual_seed(0)
    wav = torch.randn(1, audio_samples(17, 25, SR)) * 0.1
    mel = mel_spectrogram(wav)
    assert mel.shape == (1, 80, 52), mel.shape                      # 16 frames @ 25 fps = 52 mel steps
    assert float(mel.min()) >= -4.0 and float(mel.max()) <= 4.0

    import librosa
    from scipy import signal
    y = signal.lfilter([1, -0.97], [1], wav[0].numpy().astype(np.float64))
    D = librosa.stft(y=y, n_fft=800, hop_length=200, win_length=800)
    S = 20 * np.log10(np.maximum(1e-5, librosa.filters.mel(sr=SR, n_fft=800, n_mels=80, fmin=55, fmax=7600) @ np.abs(D))) - 20
    ref = torch.from_numpy(np.clip(8 * ((S + 100) / 100) - 4, -4, 4)).float()
    assert torch.allclose(mel[0], ref, atol=2e-2), float((mel[0] - ref).abs().max())


def test_sync_mel_starts_60ms_after_the_wavlm_window():
    """The WavLM window sits on container time (decord matches ffmpeg to 1 ms). stable_syncnet's
    notion of "in sync" is 60 ms later: on 28 ground-truth windows its loss falls from 0.66 at
    offset 0 to 0.45 at +60 ms, a smooth V either side. Cut the SyncNet mel there, and only there,
    so the loss rewards true sync instead of pulling the lips 60 ms early."""
    files = sorted(glob.glob(CLIPS))
    if not files:
        print("no TalkVid data; skipping")
        return
    import numpy as np
    import torch
    from decord import AudioReader
    from sang.video import open_video
    assert MEL_VERSION == 3 and SYNC_MEL_OFFSET == 960                  # 60 ms @ 16 kHz
    wins = clip_cpu_windows(files[0], frames=17, res=64, fps=25, max_windows=4, with_mel=True)
    assert len(wins) >= 2
    w1 = wins[1]
    assert w1["start"] > 0                                              # spread, not packed
    native = open_video(files[0]).get_avg_fps()
    wav = AudioReader(files[0][:-4] + ".m4a", sample_rate=SR, mono=True)[:].asnumpy()
    n = audio_samples(17, 25, SR)
    a0 = int(w1["start"] / native * SR)
    assert np.array_equal(w1["wav"], wav[:, a0:a0 + n])                 # WavLM window unchanged
    expect = mel_spectrogram(torch.from_numpy(wav[:, a0 + SYNC_MEL_OFFSET:a0 + SYNC_MEL_OFFSET + n]))[0].half()
    assert w1["mel"].shape == (80, 52) and torch.equal(w1["mel"], expect)
    # The repair path (MEL_VERSION bump) must produce the identical mel from the same start.
    assert torch.equal(clip_mels(files[0], frames=17, fps=25, max_windows=4)[1], expect)


if __name__ == "__main__":
    test_audio_span_matches_video_span()
    test_config_derives_the_real_tick_count()
    test_clip_tokens_alignment()
    test_window_starts_span_the_clip()
    test_cpu_task_reports_bad_clip_instead_of_raising()
    test_mel_is_wav2lip_format()
    test_sync_mel_starts_60ms_after_the_wavlm_window()
    print("ok")


def test_pooled_reports_a_dead_worker_instead_of_hanging():
    """A worker that dies without a result must not wedge the cache build. Under
    multiprocessing.Pool that truncates the result frame, _handle_results returns silently and
    the parent blocks forever -- job 162848 held an H100 for 7h45m that way, at 0% GPU."""
    got = {i: (v, e) for i, v, e in pooled(list(range(6)), crash_on_three, workers=2,
                                           tasks_per_worker=100, timeout=60)}
    assert sorted(got) == list(range(6)), "every task must be accounted for"
    assert got[3][0] is None and "BrokenProcessPool" in got[3][1], got[3]
    assert [got[i][0] for i in (0, 1, 2, 4, 5)] == [0, 1, 2, 4, 5], "survivors must still run"


def test_pooled_times_out_a_wedged_task():
    """A task that never returns is bounded by `timeout`, not by the wall clock of the job."""
    got = {i: (v, e) for i, v, e in pooled(list(range(5)), wedge_on_three, workers=2,
                                           tasks_per_worker=100, timeout=5)}
    assert sorted(got) == list(range(5))
    assert got[3][0] is None and "TimeoutError" in got[3][1], got[3]
    assert got[4][0] == 4, "the pool is rebuilt and the remaining tasks finish"


def test_pooled_retires_workers_on_the_generation_boundary():
    """Worker lifetime stays bounded so the ~26 MB/clip decord+mediapipe leak cannot accumulate
    (job 162800 reached 118 GB RSS without this)."""
    pids = [v for _, v, e in pooled(list(range(6)), worker_pid, workers=1, tasks_per_worker=2,
                                    timeout=60)]
    assert len(pids) == 6 and all(pids)
    assert pids[0] == pids[1] and pids[2] == pids[3] and pids[4] == pids[5], pids
    assert len({pids[0], pids[2], pids[4]}) == 3, f"a fresh worker per generation: {pids}"
