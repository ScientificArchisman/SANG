"""Rule-based naturalness: readouts, event detectors, constraint projection in the sampler.

CPU only; no LivePortrait, weights, phoneme model or data."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sang.motion import REGIONS, T_DIM
from sang.motion_model import REF_DIM, MotionFlowTransformer, Norm, generate, project
from sang.naturalness import (FPS, HOP, Readout, beat_alignment, bilabial_events, blink_bounds, blink_events,
                              closure_bounds, closure_offset, energy_db, eye_ratio, fit_readout, is_bilabial,
                              lip_ratio, pauses, schedule_blinks)


def _unzero(m):
    g = torch.Generator().manual_seed(0)
    for mod in m.modules():
        if isinstance(mod, torch.nn.Linear) and mod.weight.abs().sum() == 0:
            mod.weight.data = torch.randn(mod.weight.shape, generator=g) * 0.2
            mod.bias.data = torch.randn(mod.bias.shape, generator=g) * 0.2


def test_landmark_ratios():
    lmk = np.zeros((203, 2))
    lmk[48], lmk[66] = [0, 0], [10, 0]          # mouth 10 wide
    lmk[90], lmk[102] = [5, -1], [5, 1]         # lips 2 apart
    for a, b, c, d in ((0, 12, 6, 18), (24, 36, 30, 42)):
        lmk[a], lmk[b], lmk[c], lmk[d] = [0, 0], [4, 0], [2, -0.5], [2, 0.5]
    assert abs(lip_ratio(lmk) - 0.2) < 1e-5 and abs(eye_ratio(lmk) - 0.25) < 1e-5


def test_readout_recovers_a_linear_map_and_its_z_form():
    g = torch.Generator().manual_seed(0)
    Y = torch.randn(2000, T_DIM, generator=g) * 3 + 1
    cols = REGIONS["mouth"]
    w = torch.zeros(T_DIM); w[cols] = torch.randn(len(cols), generator=g)
    r = Y @ w + 0.5 + 0.01 * torch.randn(2000, generator=g)
    ro = fit_readout(Y, r, cols, np.arange(2000) // 20)
    assert ro.r2 > 0.99 and torch.allclose(ro.w, w, rtol=0.03, atol=1e-3)   # ridge shrinks ~1%
    norm = Norm(Y.mean(0), Y.std(0), torch.zeros(63), torch.ones(63))
    a, c = ro.in_z(norm)
    z = norm.target(Y[:5])
    assert torch.allclose(z @ a + c, ro(Y[:5]), atol=1e-4)


def test_blink_detector_finds_dips_and_ignores_long_closures():
    eye = np.full(300, 0.3)
    for s in (40, 120, 200):
        eye[s:s + 5] = [0.2, 0.08, 0.05, 0.1, 0.22]
    eye[250:280] = 0.05                          # 1.2 s closed: not a blink
    ev = blink_events(eye)
    assert [b for _, b, _ in ev] == [42, 122, 202], ev


def test_pauses_and_energy():
    n = 50
    wav = torch.randn(n * HOP) * 0.3
    wav[20 * HOP:30 * HOP] *= 1e-3               # 10 silent frames (-60 dB)
    wav[40 * HOP:41 * HOP] *= 1e-3               # 1 silent frame: too short
    p = pauses(energy_db(wav, n))
    assert p[20:30].all() and not p[40] and p.sum() == 10


def test_bilabial_tokens_and_events():
    assert is_bilabial("p") and is_bilabial("mʲ") and is_bilabial("b") and not is_bilabial("f")
    assert not is_bilabial("<pad>") and not is_bilabial("")
    post = torch.zeros(100, 3); post[:, 0] = 1       # 2 s of ticks, token 1 = bilabial
    post[20:24, 0], post[20:24, 1] = 0, 1            # ticks 20-23 -> centre 21.5 -> 0.4425 s -> frame 11
    assert bilabial_events(post, torch.tensor([1]), 50) == [11]


def test_closure_offset_and_bounds():
    lip = np.full(60, 0.3); lip[22] = 0.0
    assert closure_offset(lip, [20]) == [2]
    ub = closure_bounds([20, 58], 2, 60, 0.05, width=3)
    assert np.isinf(ub[:21]).all() and (ub[21:24] == 0.05).all() and np.isinf(ub[24:59]).all()
    assert ub[59] == 0.05                                  # 58+2 = 60 is off the end; its window edge is not


def test_blink_bounds_shape():
    ub = blink_bounds([(10, 5)], 30, 0.3, 0.05)
    assert np.isinf(ub[:10]).all() and np.isinf(ub[15:]).all()
    assert abs(ub[12] - 0.05) < 1e-9 and ub[10] > ub[11] > ub[12]


def test_scheduler_respects_existing_blinks_and_snaps_to_pauses():
    rng = np.random.default_rng(0)
    n = 25 * 30
    pause = np.zeros(n, bool); pause[::100] = True
    add = schedule_blinks([], pause, n, [3.0], [5], rng)
    starts = [s for s, _ in add]
    assert 7 <= len(add) <= 11
    assert all(b - a >= 8 for a, b in zip(starts, starts[1:]))
    # the model already blinks every 2 s -> nothing to add for a 3 s interval
    own = list(range(10, n, 50))
    assert schedule_blinks(own, pause, n, [3.0], [5], np.random.default_rng(1)) == []
    # snapping: due times land within 15 frames of a pause frame
    pause2 = np.zeros(n, bool); pause2[::30] = True
    for s, _ in schedule_blinks([], pause2, n, [3.0], [5], np.random.default_rng(2)):
        assert s % 30 == 0


def test_beat_alignment_perfect_and_chance():
    hb = np.arange(20, 480, 40)
    bas, chance = beat_alignment(hb, hb.copy(), 500)
    assert bas == 1.0 and chance < 0.6
    assert np.isnan(beat_alignment(np.array([], int), hb, 500)[0])


def test_project_is_exact_and_minimal():
    g = torch.Generator().manual_seed(0)
    z = torch.randn(2, 8, T_DIM, generator=g)
    a1 = torch.zeros(T_DIM); a1[REGIONS["mouth"]] = torch.randn(18, generator=g)
    a2 = torch.zeros(T_DIM); a2[REGIONS["eyes"]] = torch.randn(15, generator=g)
    ub1 = torch.full((2, 8), float("inf")); ub1[:, 3] = -5.0
    ub2 = torch.full((2, 8), float("inf")); ub2[0, 5] = -2.0
    p = project(z, [(a1, ub1), (a2, ub2)])
    assert torch.all(p @ a1 <= ub1 + 1e-5) and torch.all(p @ a2 <= ub2 + 1e-5)
    free = torch.ones(2, 8, dtype=torch.bool); free[:, 3] = False; free[0, 5] = False
    assert torch.equal(p[free], z[free])                       # untouched where unconstrained
    assert torch.allclose(project(p, [(a1, ub1), (a2, ub2)]), p)   # idempotent


def test_generate_with_bounds_satisfies_them_and_is_unchanged_without():
    torch.manual_seed(0)
    m = MotionFlowTransformer(dim=32, depth=2, heads=2, audio_dim=16, max_frames=64).eval()
    _unzero(m)
    n = 40
    audio, ref = torch.randn(1, 2 * n, 16), torch.randn(1, REF_DIM)
    a = torch.zeros(T_DIM); a[REGIONS["mouth"]] = 1.0
    ub = torch.full((1, n), float("inf")); ub[0, [3, 17, 30]] = -4.0     # windows of 16: all three windows
    kw = dict(window=16, n_prefix=4, steps=5, cfg_audio=2.0)
    y0 = generate(m, audio, ref, n, generator=torch.Generator().manual_seed(1), **kw)
    y1 = generate(m, audio, ref, n, generator=torch.Generator().manual_seed(1), bounds=[], **kw)
    assert torch.allclose(y0, y1)
    y2 = generate(m, audio, ref, n, generator=torch.Generator().manual_seed(1), bounds=[(a, ub)], **kw)
    assert torch.all((y2 @ a)[0, [3, 17, 30]] <= -4.0 + 1e-4)
    assert not torch.allclose(y2, y0)
