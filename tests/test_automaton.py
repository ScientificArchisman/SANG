"""The gaze-head automaton (sang.model.automaton): it removes noise and keeps jumps, its shield never moves the mouth,
the sampler's shield and particles run, the eye-head panel and the hand-set baselines behave. CPU only."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

from sang.eval.naturalness import eye_head_panel
from sang.model.automaton import COLS, HANDSET, Shield, fit, handset
from sang.model.flow import REF_DIM, MotionFlowTransformer, generate, parse_spec, sample, sampler_kwargs
from sang.model.latent import fit_pca
from sang.model.target import WIDE_REGIONS
from test_train_eval import make_cache, train

MOUTH = WIDE_REGIONS["mouth"]


def synth(N, T=74, seed=0, noise=True):
    """Slow head strokes, piecewise-constant world gaze with 1-frame saccades (eyes counter-rotate, G = 0.002)."""
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(T).float()
    head = 2.0 * torch.sin((0.02 + 0.08 * torch.rand(N, 1, 3, generator=g)) * t[None, :, None]
                           + 6.28 * torch.rand(N, 1, 3, generator=g))
    jumps = (torch.rand(N, T, 1, generator=g) < 1 / 12) * 0.01 * torch.randn(N, T, 2, generator=g)
    gaze = jumps.cumsum(1) - 0.002 * head[..., [1, 0]]
    w = torch.zeros(N, T, 69)
    if noise:
        head = head + 0.05 * torch.randn(head.shape, generator=g)
        gaze = gaze + 0.001 * torch.randn(gaze.shape, generator=g)
    lid = -0.02 * (torch.rand(N, T, 1, generator=g) < 0.02).float()
    if noise:
        lid = lid + 0.0005 * torch.randn(lid.shape, generator=g)
    w[..., :3], w[..., [36, 37]], w[..., [48, 49]], w[..., [43, 52]] = head, gaze, gaze, lid.expand(-1, -1, 2)
    return w


def test_noise_goes_and_saccades_stay():
    gha, st = fit(synth(300), rounds=4, log=lambda s: None)
    assert st["median_fix"] >= 3 and abs(st["G"][0][0] - 0.002) < 0.001
    clean = synth(40, seed=1, noise=False)
    g = torch.Generator().manual_seed(2)
    noisy = clean.clone()
    noisy[..., :3] += 0.05 * torch.randn(40, 74, 3, generator=g)
    n = 0.001 * torch.randn(40, 74, 2, generator=g)
    noisy[..., [36, 37]] += n
    noisy[..., [48, 49]] += n
    out = noisy[..., COLS] + gha.correct(noisy)
    err = lambda a: (a - clean[..., COLS]).pow(2).mean((0, 1)).sqrt()
    before, after = err(noisy[..., COLS]), err(out)
    assert (after[:5] < 0.7 * before[:5]).all(), (before, after)
    cg = clean[..., 36]
    big = (cg.diff(dim=1)).abs() > 0.005
    kept = out[..., 3].diff(dim=1).abs()[big].mean() / cg.diff(dim=1).abs()[big].mean()
    assert 0.9 < kept < 1.1


def _latent_and_shield(n=1, k=40):
    lat = fit_pca(torch.randn(500, 69, generator=torch.Generator().manual_seed(0)), K=69)
    gha, _ = fit(synth(60), rounds=2, log=lambda s: None)
    return lat, Shield(gha, lat, k, n=n, ratio=False)


def test_shield_moves_head_and_eyes_never_the_mouth():
    lat, sh = _latent_and_shield()
    x0 = torch.randn(2, 30, 40)
    y = sh(x0, torch.randn(2, 5, 40))
    a, b = lat.decode(x0), lat.decode(y)
    assert (b[..., COLS] - a[..., COLS]).abs().max() > 1e-3
    assert torch.allclose(b[..., MOUTH], a[..., MOUTH], atol=1e-4)


def _tiny(k=40):
    torch.manual_seed(0)
    m = MotionFlowTransformer(dim=32, depth=1, heads=2, audio_dim=16, max_frames=64, t_dim=k).eval()
    for mod in m.modules():
        if isinstance(mod, torch.nn.Linear) and mod.weight.abs().sum() == 0:
            mod.weight.data.normal_(0, 0.2)
            mod.bias.data.normal_(0, 0.2)
    return m


def test_last_step_shield_keeps_the_sampled_mouth():
    lat, sh = _latent_and_shield(n=1)
    m, L = _tiny(), 20
    audio, ref = torch.randn(1, 2 * L, 16), torch.randn(1, REF_DIM)
    run = lambda s: sample(m, audio, ref, L, steps=3, generator=torch.Generator().manual_seed(1), shield=s)
    a, b = lat.decode(run(None)), lat.decode(run(sh))
    assert torch.allclose(b[..., MOUTH], a[..., MOUTH], atol=1e-4) and not torch.allclose(a[..., COLS], b[..., COLS])


def test_particles_return_one_clip():
    lat, sh = _latent_and_shield(n=1)
    m, n = _tiny(), 37
    y = generate(m, torch.randn(1, 2 * n, 16), torch.randn(1, REF_DIM), n, window=16, n_prefix=4, steps=2,
                 generator=torch.Generator().manual_seed(0), shield=sh, particles=3)
    assert y.shape == (1, n, 40) and torch.isfinite(y).all()
    sh.ratio = -1  # the control selects nothing: every particle survives in place
    assert torch.equal(sh.resample(torch.randn(6, 20, 40)), torch.arange(6))


def test_sampler_spec_carries_the_automaton_keys():
    cfg = {"cfg_audio": 2.0, "sample_steps": 4}
    assert sampler_kwargs(parse_spec("g=2,shield=2,particles=4,ratio=0"), cfg) == {
        "cfg_audio": 2.0, "steps": 4, "n_shield": 2, "particles": 4, "ratio": 0}
    assert sampler_kwargs(parse_spec("g=3"), cfg) == {"cfg_audio": 3.0, "steps": 4}


def test_panel_white_noise_and_smooth_motion():
    g = np.random.default_rng(0)
    noise = g.normal(size=(2000, 69))
    p = eye_head_panel(noise, noise)
    assert abs(p["head_rho"] + 2 / 3) < 0.05 and abs(p["gaze_rho"] + 2 / 3) < 0.05
    smooth = np.cumsum(np.cumsum(np.cumsum(g.normal(size=(2000, 69)), 0), 0), 0)
    assert eye_head_panel(smooth, noise)["head_rho"] > 0.9


def test_handset_keeps_constants_and_cuts_noise():
    x = np.random.default_rng(0).normal(size=(200, 3)) + 5.0
    for name in HANDSET:
        y = handset(x, name)
        assert y.shape == x.shape and np.allclose(handset(np.full((50, 2), 5.0), name), 5.0)
        # LivePortrait's process / observation ratio of 33 smooths only lightly
        assert y[20:-20].std() < (0.98 if name == "kalman" else 0.8) * x[20:-20].std(), name


def test_fit_automaton_end_to_end(tmp_path):
    cache = tmp_path / "cache"
    make_cache(cache)
    train(cache, tmp_path / "gen", "train_eval=false")
    out = tmp_path / "gha"
    log = subprocess.run([sys.executable, str(REPO / "scripts" / "fit_automaton.py"), "--gen",
                          str(tmp_path / "gen" / "last.pt"), "--out", str(out), "--n-train", "16", "--n-val", "8",
                          "--sang-windows", "8", "--rounds", "2", "--batch", "4", "--workers", "0"],
                         capture_output=True, encoding="utf-8", cwd=REPO)
    assert log.returncode == 0, log.stderr[-2000:]
    assert log.stdout.count("GATE G0") == 4 and log.stdout.count("GATE G1") == 3
    d = torch.load(out / "gha.pt", weights_only=False)
    assert d["sang"] is not None and torch.load(out / "gha_1state.pt", weights_only=False)["sang"] is None
