"""Voice cloning plumbing (sang/voice.py): VAD, chunking, kNN matching, the voice bank, metrics.

CPU only; kNN-VC and the speaker encoder are replaced by small fakes, so no weights are needed."""
import sys
import wave
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sang.voice as V
from sang.voice import (FRAME_S, SR, VoiceBank, auto_k, cer, chunks, fit_length, knn_features, match_level,
                        rms_db, save_wav, speech_segments, tier)


def _speech(seconds: float, amp: float = 0.3, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(int(seconds * SR), generator=g) * amp


def _silence(seconds: float) -> torch.Tensor:
    return torch.randn(int(seconds * SR)) * 1e-4


def test_fit_length_and_level():
    w = torch.randn(1000)
    assert fit_length(w, 1200).numel() == 1200 and fit_length(w, 800).numel() == 800
    assert torch.equal(fit_length(w, 1200)[:1000], w)
    quiet, loud = torch.randn(SR) * 0.01, torch.randn(SR) * 0.2
    assert abs(rms_db(match_level(quiet, loud)) - rms_db(loud)) < 0.1


def test_speech_segments_bridge_and_drop():
    w = torch.cat([_silence(1), _speech(2), _silence(0.1), _speech(1), _silence(1), _speech(0.2), _silence(1)])
    segs = speech_segments(w)
    assert len(segs) == 1, segs                          # 0.1 s gap bridged, 0.2 s blip dropped
    a, b = segs[0]
    assert abs(a / SR - 1.0) < 0.05 and abs(b / SR - 4.1) < 0.05


def test_chunks_merge_short_tail():
    w = _speech(7.0)
    cs = chunks(w, [(0, w.numel())], chunk_s=3.0)
    assert [round(c.numel() / SR, 1) for c in cs] == [3.0, 4.0]          # 1 s tail < half a chunk -> merged
    w = _speech(7.5)
    cs = chunks(w, [(0, w.numel())], chunk_s=3.0)
    assert [round(c.numel() / SR, 1) for c in cs] == [3.0, 3.0, 1.5]     # 1.5 s = half: kept
    assert sum(c.numel() for c in chunks(w, [(0, SR), (2 * SR, w.numel())], chunk_s=2.0)) == w.numel() - SR


def test_knn_matches_brute_force_and_is_block_invariant():
    g = torch.Generator().manual_seed(0)
    bank, q = torch.randn(300, 16, generator=g), torch.randn(50, 16, generator=g)
    sim = F.normalize(q, dim=-1) @ F.normalize(bank, dim=-1).T
    want = bank[sim.topk(4, dim=-1).indices].mean(1)
    assert torch.allclose(knn_features(q, bank, 4), want, atol=1e-5)
    assert torch.allclose(knn_features(q, bank, 4, block=7), want, atol=1e-5)
    assert knn_features(q, bank[:2], 4).shape == (50, 16)                # k clamps to the bank size
    exact = knn_features(bank[:5], bank, 1)                              # a bank frame's nearest is itself
    assert torch.allclose(exact, bank[:5], atol=1e-5)


def test_auto_k_and_tier():
    assert auto_k(60) == 4 and auto_k(600) == 8 and auto_k(3600) == 16
    assert "too little" in tier(5) and "short" in tier(20) and "good" in tier(120) and "plateau" in tier(900)


def test_save_wav_round_trip(tmp_path):
    w = torch.sin(torch.linspace(0, 200, SR)) * 0.5
    p = save_wav(w, tmp_path / "x.wav")
    with wave.open(str(p)) as f:
        assert f.getframerate() == SR and f.getnchannels() == 1 and f.getnframes() == SR
        back = np.frombuffer(f.readframes(SR), dtype=np.int16) / 32767
    assert np.abs(back - w.numpy()).max() < 1e-3


def test_cer():
    assert cer("hello world", "hello world") == 0.0
    assert abs(cer("helo world", "hello world") - 1 / 11) < 1e-9
    assert cer("  Hello   World ", "hello world") == 0.0


class _FakeKnn:
    """get_features: one 16-d frame per 20 ms, all equal to a per-'speaker' vector (the amplitude)."""
    device = torch.device("cpu")

    def get_features(self, x, vad_trigger_level=0):
        n = max(1, x.shape[-1] // 320)
        return torch.full((n, 1024), float(x.abs().mean()))


class _FakeSV:
    """Speaker = sign pattern of the chunk's amplitude: loud chunks are 'A', quiet ones 'B'."""

    def embed(self, wavs):
        out = []
        for w in wavs:
            out.append(torch.tensor([1.0, 0.0]) if w.abs().mean() > 0.05 else torch.tensor([0.0, 1.0]))
        return torch.stack(out)


def test_bank_enrol_filters_other_speaker_dedupes_and_grows(tmp_path, monkeypatch):
    audio = {"a1.wav": torch.cat([_speech(6, 0.3, 1), _silence(1)]),        # speaker A
             "mixed.wav": torch.cat([_speech(6, 0.3, 2), _silence(1), _speech(3, 0.02, 3)]),  # A then B
             "a2.wav": _speech(9, 0.3, 4)}
    for name in audio:
        (tmp_path / name).write_bytes(b"x" * (len(name) + 1))              # real files: bank keys use stat()
    monkeypatch.setattr(V, "load_wav", lambda p, sr=SR: audio[Path(p).name])
    bank = VoiceBank(tmp_path / "bank")
    r = bank.enroll([tmp_path / "a1.wav", tmp_path / "mixed.wav"], _FakeKnn(), _FakeSV(), min_cos=0.5, log=lambda m: None)
    assert abs(r["added_s"] - 12.0) < 0.2, r                             # B's 3 s dropped
    assert abs(bank.seconds - 12.0) < 0.2
    again = VoiceBank(tmp_path / "bank")                                  # reload from disk
    assert again.feats.shape == bank.feats.shape and again.centroid is not None
    r = again.enroll([tmp_path / "a1.wav"], _FakeKnn(), _FakeSV(), log=lambda m: None)
    assert r["added_s"] == 0.0                                            # already enrolled
    r = again.enroll([tmp_path / "a2.wav"], _FakeKnn(), _FakeSV(), min_cos=0.5, log=lambda m: None)
    assert abs(again.seconds - 21.0) < 0.3                                # the bank grows with more audio
    summary = (tmp_path / "bank" / "bank.json").read_text()
    assert '"minutes"' in summary and "kept_s" in summary


def test_convert_keeps_length_and_level(monkeypatch):
    class Knn(_FakeKnn):
        def vocode(self, c):
            return torch.randn(1, c.shape[1] * 320 - 80) * 0.05               # a few samples short, other level
    bank = VoiceBank("/nonexistent/bank")
    bank.feats = torch.randn(500, 1024).half()
    src = _speech(3.013, 0.2)
    y = V.convert(src, bank, Knn())
    assert y.numel() == src.numel()
    assert abs(rms_db(y[: -SR // 10]) - rms_db(src)) < 1.0
