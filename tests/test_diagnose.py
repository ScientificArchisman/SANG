"""Lip-sync diagnosis statistics (sang/diagnose.py) on synthetic signals with known answers.

CPU only; no checkpoint, cache, audio models or SyncNet."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sang.diagnose import (RidgeProbe, by_group, corr, k_curve_expected, lag_curve, lag_summary, lang_group,
                           offset_stats, offset_summary, pca_basis, seed_stats, seeds_summary, shift_frames,
                           shifted_mouth_losses, stack_taps, taps, ticks_to_frames, verdict)
from sang.motion import REGIONS, T_DIM
from sang.motion_model import REF_DIM, MotionFlowTransformer


def _smooth(rng, n, k=9):
    x = np.convolve(rng.normal(size=n + k), np.ones(k) / k, mode="valid")[:n]
    return (x - x.mean()) / x.std()


def test_seed_stats_measure_what_randomness_costs():
    rng = np.random.default_rng(0)
    per = []
    for _ in range(40):
        m = _smooth(rng, 400)                                  # what the audio determines
        y = m + 0.6 * _smooth(rng, 400)                        # real take
        S = m + 0.8 * np.stack([_smooth(rng, 400) for _ in range(16)])
        per.append({**seed_stats(S, y), "true_mean_corr": corr(m, y)})
    s = seeds_summary(per)
    assert abs(s["r_inf"] - s["true_mean_corr"]) < 0.05       # r_sy / sqrt(r_ss) recovers corr(mean, real)
    assert abs(s["k16_corr"] - s["k16_expected"]) < 0.03
    assert s["k1_corr"] < s["k4_corr"] < s["k16_corr"]
    assert abs(k_curve_expected(0.6, 0.6, 10 ** 9) - 0.6 / 0.6 ** 0.5) < 1e-6
    assert any("[randomness: MAJOR]" in l for l in verdict({"seeds": s}))


def test_bias_is_flagged_when_samples_agree_but_miss():
    rng = np.random.default_rng(1)
    per = []
    for _ in range(30):
        m, y = _smooth(rng, 300), _smooth(rng, 300)
        y = 0.5 * m + y                                        # the model's mean only partly matches reality
        S = m + 0.1 * np.stack([_smooth(rng, 300) for _ in range(8)])
        per.append(seed_stats(S, y))
    lines = verdict({"seeds": seeds_summary(per)})
    assert any(l.startswith("[bias]") for l in lines) and any("not a factor" in l for l in lines)


def test_lag_curve_sign_and_summary():
    rng = np.random.default_rng(2)
    curves = []
    for _ in range(10):
        real = _smooth(rng, 300, k=5)
        gen = np.concatenate([np.full(2, real[0]), real[:-2]])  # gen[t + 2] = real[t]: generated lips are late
        c = lag_curve(gen, real, 6)
        assert int(np.argmax(c)) - 6 == 2
        curves.append(c)
    s = lag_summary(curves, 6)
    assert s["peak_lag"] == 2 and s["frac_abs_lag_ge2"] == 1.0 and s["corr_best_lag"] > s["corr_lag0"]
    assert any("[timing: SHIFT]" in l and "late" in l for l in verdict({"lag": s}))


def test_shift_frames_convention():
    a = torch.arange(10.0)[:, None]
    assert shift_frames(a, 1, 2)[:, 0].tolist() == [0, 0, 0, 1, 2, 3, 4, 5, 6, 7]      # delayed one frame
    assert shift_frames(a, -1, 2)[:, 0].tolist() == [2, 3, 4, 5, 6, 7, 8, 9, 9, 9]
    assert torch.equal(shift_frames(a, 0, 2), a)


class _AudioIsMouth:
    """A stand-in model whose mouth velocity IS the per-frame audio: its loss is lowest when the
    audio lines up with the motion, so the offset test must recover a known shift."""
    tpf = 2

    def __call__(self, x_t, t, audio, ref, prefix=None, prefix_keep=None, drop_audio=None):
        B, L = x_t.shape[:2]
        P = prefix.shape[1]
        frames = audio.reshape(B, P + L, 2, -1).mean(2)[:, P:]
        v = torch.zeros_like(x_t)
        v[..., REGIONS["mouth"]] = frames[..., :len(REGIONS["mouth"])]
        return torch.where(drop_audio.view(B, 1, 1), torch.zeros_like(v), v)


def test_offset_test_recovers_a_known_audio_delay():
    g = torch.Generator().manual_seed(0)
    n, P, L, mouth = 160, 10, 64, REGIONS["mouth"]
    y = 5 * torch.randn(n, T_DIM, generator=g)
    # at t = 0 the target is eps - y, so the loss is smallest where the audio frames equal -y
    true = (-y[:, mouth]).repeat_interleave(2, 0)
    given = shift_frames(true, 2, 2)                            # this clip's audio arrives 2 frames late
    shifts = list(range(-4, 5))
    losses, null = shifted_mouth_losses(_AudioIsMouth(), y, given, torch.zeros(1, REF_DIM), shifts, mouth, P, L,
                                        ts=(0.0,), n_noise=1, generator=torch.Generator().manual_seed(1))
    st = offset_stats(losses, shifts)
    assert st["best_shift"] == -2 and null > min(losses)
    aligned, _ = shifted_mouth_losses(_AudioIsMouth(), y, true, torch.zeros(1, REF_DIM), shifts, mouth, P, L,
                                      ts=(0.0,), n_noise=1, generator=torch.Generator().manual_seed(1))
    al = offset_stats(aligned, shifts)
    assert al["best_shift"] == 0 and al["sharpness_1"] > 1 and al["sharpness_2"] > 1     # a timing-sharp model
    s = offset_summary([st, st, {**st, "best_shift": 0}], shifts)
    assert s["frac_abs_best_ge2"] == 2 / 3 and s["median_best_shift"] == -2
    assert shifted_mouth_losses(_AudioIsMouth(), y[:70], given[:140], torch.zeros(1, REF_DIM), shifts, mouth, P, L) is None


def test_offset_test_runs_on_the_real_model():
    torch.manual_seed(0)
    m = MotionFlowTransformer(dim=32, depth=1, heads=2, audio_dim=8).eval()
    y, a, ref = torch.randn(150, T_DIM), torch.randn(300, 8), torch.randn(1, REF_DIM)
    out1 = shifted_mouth_losses(m, y, a, ref, [-1, 0, 1], REGIONS["mouth"], 10, 64, n_noise=2,
                                generator=torch.Generator().manual_seed(3))
    out2 = shifted_mouth_losses(m, y, a, ref, [-1, 0, 1], REGIONS["mouth"], 10, 64, n_noise=2,
                                generator=torch.Generator().manual_seed(3))
    assert out1 == out2 and len(out1[0]) == 3 and all(np.isfinite(out1[0]))


def test_taps_stack_and_ticks():
    assert taps(2) == [-2, -1, 0, 1, 2]
    t12 = taps(12)
    assert t12 == sorted(set(t12)) and {-12, -9, -6, 0, 6, 9, 12} <= set(t12) and set(range(-3, 4)) <= set(t12)
    X = torch.arange(5.0)[:, None]
    assert stack_taps(X, [-1, 0, 1])[0].tolist() == [0, 0, 1] and stack_taps(X, [-1, 0, 1])[4].tolist() == [3, 4, 4]
    h = torch.arange(10.0).view(5, 2)                          # 5 ticks of 2 dims
    f = ticks_to_frames(h, 3)
    assert f.shape == (3, 2) and f[0].tolist() == [1, 2] and f[2].tolist() == [4, 4.5]   # last tick paired with a zero


def test_ridge_probe_learns_within_clip_dynamics_and_pca():
    g = torch.Generator().manual_seed(0)
    w = torch.randn(12, generator=g)
    probe = RidgeProbe(12)
    for _ in range(30):
        X = torch.randn(200, 12, generator=g) + 10 * torch.randn(12, generator=g)    # clip-level offsets
        probe.add(X, X @ w + 50 * torch.randn(1, generator=g) + 0.1 * torch.randn(200, generator=g))
    wt = probe.solve(1e-3)
    X = torch.randn(200, 12, generator=g) + 3
    assert corr(RidgeProbe.predict(X, wt), (X @ w).numpy()) > 0.99
    _, V = pca_basis(torch.randn(1000, 20, generator=g), 8)
    assert V.shape == (20, 8) and torch.allclose(V.T @ V, torch.eye(8), atol=1e-4)


def test_language_groups_and_full_verdict():
    assert [lang_group(c) for c in ("en", "EN", "zh", None, "unk")] == ["en", "en", "other", "unk", "unk"]
    recs = [{"lang_group": "en", "r_sy": 0.70}] * 20 + [{"lang_group": "other", "r_sy": 0.55}] * 20 + [{"r_sy": None}]
    g = by_group(recs, "r_sy")
    assert abs(g["en"]["mean"] - 0.70) < 1e-9 and g["other"]["n"] == 20 and g["unk"]["n"] == 0
    R = {"seeds": {"r_sy": 0.64, "r_ss": 0.60, "r_inf": 0.83},
         "lag": {"peak_lag": 0, "frac_abs_lag_ge2": 0.3, "corr_lag0": 0.66, "corr_best_lag": 0.74},
         "offset": {"frac_abs_best_ge2": 0.25, "median_best_shift": 0.0, "sharpness_1": 0.02, "sharpness_2": 0.06,
                    "curve_min_shift": 0, "audio_benefit": 0.5},
         "syncnet": {"clips": 80, "frac_abs_off_ge2": 0.2, "frac_conf_lt3": 0.1, "r_sy_aligned": 0.68, "r_sy_offset": 0.45},
         "ref": {"mean_shift_open_minus_closed": 0.5, "r_closed": 0.6, "r_default": 0.64, "r_open": 0.62},
         "lang": {"r_sy": g},
         "probe": {"rows": [{"encoder": "wavlm-large", "layer": 24, "n_layers": 24, "window": 2, "val_corr": 0.55},
                            {"encoder": "openai/whisper-large-v3", "layer": 24, "n_layers": 32, "window": 12, "val_corr": 0.66}],
                   "baseline": {"encoder": "wavlm-large", "layer": 24, "window": 2, "val_corr": 0.55}}}
    text = "\n".join(verdict(R))
    for tag in ("[randomness: MAJOR]", "[data offsets (model's view): COMMON]", "[data offsets (SyncNet on real clips): COMMON]",
                "[reference shortcut: YES]", "[language gap: YES]", "better features/context exist", "[timing precision: LOW]"):
        assert tag in text, tag
    flat = {**R["offset"], "sharpness_1": 0.0001, "sharpness_2": 0.0002}     # loss ignores the shift: no verdict on offsets
    text = "\n".join(verdict({"offset": flat}))
    assert "inconclusive" in text and "COMMON" not in text
