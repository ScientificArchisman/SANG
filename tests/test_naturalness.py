"""Naturalness metrics: openness readouts, event detectors, concordance, typical frames.

CPU only; no LivePortrait, weights, phoneme model or data."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sang.model.target import REGIONS, T_DIM
from sang.eval.naturalness import (HOP, beat_alignment, bilabial_events, blink_events, ccc, closure_minima,
                              closure_offset, energy_db, eye_ratio, fit_readout, is_bilabial, lip_ratio,
                              load_readouts, pause_near, pauses)


def test_landmark_ratios():
    lmk = np.zeros((203, 2))
    lmk[48], lmk[66] = [0, 0], [10, 0]
    lmk[90], lmk[102] = [5, -1], [5, 1]
    for a, b, c, d in ((0, 12, 6, 18), (24, 36, 30, 42)):
        lmk[a], lmk[b], lmk[c], lmk[d] = [0, 0], [4, 0], [2, -0.5], [2, 0.5]
    assert abs(lip_ratio(lmk) - 0.2) < 1e-5 and abs(eye_ratio(lmk) - 0.25) < 1e-5


def test_readout_recovers_a_linear_map_and_round_trips(tmp_path):
    g = torch.Generator().manual_seed(0)
    Y = torch.randn(2000, T_DIM, generator=g) * 3 + 1
    cols = REGIONS["mouth"]
    w = torch.zeros(T_DIM); w[cols] = torch.randn(len(cols), generator=g)
    r = Y @ w + 0.5 + 0.01 * torch.randn(2000, generator=g)
    ro = fit_readout(Y, r, cols, np.arange(2000) // 20)
    assert ro.r2 > 0.99 and torch.allclose(ro.w, w, rtol=0.03, atol=1e-3)
    (tmp_path / "openness.json").write_text(json.dumps({"lip": ro.to_dict(), "eye": ro.to_dict()}))
    back = load_readouts(tmp_path / "openness.json")["lip"]
    assert torch.allclose(back(Y[:5]), ro(Y[:5]), atol=1e-5) and abs(back.r2 - ro.r2) < 1e-9


def test_blink_detector_finds_dips_and_ignores_long_closures():
    eye = np.full(300, 0.3)
    for s in (40, 120, 200):
        eye[s:s + 5] = [0.2, 0.08, 0.05, 0.1, 0.22]
    eye[250:280] = 0.05
    ev = blink_events(eye)
    assert [b for _, b, _ in ev] == [42, 122, 202], ev


def test_pauses_and_energy():
    n = 50
    wav = torch.randn(n * HOP) * 0.3
    wav[20 * HOP:30 * HOP] *= 1e-3
    wav[40 * HOP:41 * HOP] *= 1e-3
    p = pauses(energy_db(wav, n))
    assert p[20:30].all() and not p[40] and p.sum() == 10
    near = pause_near(p, reach=2)
    assert near[18:32].all() and not near[17] and not near[32]


def test_bilabial_tokens_and_events():
    assert is_bilabial("p") and is_bilabial("mʲ") and is_bilabial("b") and not is_bilabial("f")
    assert not is_bilabial("<pad>") and not is_bilabial("")
    post = torch.zeros(100, 3); post[:, 0] = 1
    post[20:24, 0], post[20:24, 1] = 0, 1
    assert bilabial_events(post, torch.tensor([1]), 50) == [11]


def test_closure_offset_and_minima():
    lip = np.full(60, 0.3); lip[22] = 0.0
    assert closure_offset(lip, [20]) == [2]
    assert closure_minima(lip, [20, 40], offset=2).tolist() == [0.0, 0.3]


def test_beat_alignment_perfect_and_chance():
    hb = np.arange(20, 480, 40)
    bas, chance = beat_alignment(hb, hb.copy(), 500)
    assert bas == 1.0 and chance < 0.6
    assert np.isnan(beat_alignment(np.array([], int), hb, 500)[0])


def test_ccc_scores_timing_level_and_amplitude():
    x = np.sin(np.linspace(0, 12, 200))
    assert abs(ccc(x, x) - 1) < 1e-9
    assert ccc(x, x + x.std()) < 0.9 and ccc(x, 0.5 * x) < 0.9 and ccc(x, -x) < -0.99

