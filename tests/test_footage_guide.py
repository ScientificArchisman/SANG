"""The split first step and footage guidance (sang.model.scoring.FootageGuide). CPU only."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sang.model.flow import REF_DIM, flow_times, parse_spec, sample, sampler_kwargs
from sang.model.latent import MotionLatent
from sang.model.scoring import FootageGuide, guided_times
from sang.model.target import WIDE_DIM, WIDE_REGIONS
from test_motion_model import A, L, P, _model, _unzero

W, C, K = 5, 4, 24


def _setup(lam=0.5):
    torch.manual_seed(0)
    m = _model(demo_tokens=4, t_dim=K).eval()
    _unzero(m)
    lat = MotionLatent(K, WIDE_DIM)
    tgt = dict(y=torch.randn(W, L, K), prefix=torch.randn(W, P, K), audio=torch.randn(W, 2 * (P + L), A),
               ref=torch.randn(1, REF_DIM).expand(W, -1), demo=torch.randn(1, 4, 32).expand(W, -1, -1))
    return m, lat, FootageGuide(m, lat.decode, times=guided_times(6, 2), lam=lam, **tgt)


def test_split_and_the_guided_times():
    assert flow_times(6, 1) == flow_times(6)
    assert flow_times(4, 2) == pytest.approx([1.0, 0.875, 0.75, 0.5, 0.25, 0.0])
    assert guided_times(6, 2) == pytest.approx([11 / 12, 5 / 6]) and guided_times(6, 1) == pytest.approx([5 / 6])
    assert guided_times(6, 2, 0.5) == pytest.approx([11 / 12, 5 / 6, 4 / 6, 3 / 6])
    spec = parse_spec("g=2,steps=6,split=2,copies=8,guide=0.5")
    assert sampler_kwargs(spec, {"cfg_audio": 2.0, "sample_steps": 4}) == {"cfg_audio": 2.0, "steps": 6, "split": 2}


def test_a_guide_step_lowers_the_distance_and_leaves_the_mouth_alone():
    _, lat, gd = _setup(lam=0.1)
    mu = torch.randn(1, L, K).expand(C, -1, -1)
    x0 = mu + 3 * torch.randn(C, L, K)
    assert gd(x0, mu, 0.5) is None
    t = guided_times(6, 2)[0]
    dist = lambda x: float(gd.distance(gd.embed(x, mu), gd.target[round(t, 6)]).detach())
    assert dist(x0) > 0
    d = gd(x0, mu, t)
    assert d.shape == x0.shape and dist(x0 + d) < dist(x0)
    assert d.norm() <= 0.1 * (x0 - mu).norm() + 1e-5
    mouth = (lat.decode(x0 + d) - lat.decode(x0))[..., WIDE_REGIONS["mouth"]]
    assert mouth.abs().max() < 1e-4 * d.abs().max()


def test_a_zero_strength_guide_is_the_plain_sampler():
    m, _, gd = _setup()
    _, _, off = _setup(lam=0.0)
    kw = dict(prefix=torch.randn(C, P, K), steps=6, split=2, demo=torch.randn(1, 4, 32).expand(C, -1, -1))
    audio, ref = torch.randn(C, 2 * (P + L), A), torch.randn(C, REF_DIM)
    run = lambda g: sample(m, audio, ref, L, generator=torch.Generator().manual_seed(1), guide=g, **kw)
    base = run(None)
    assert torch.equal(base, run(off))
    assert not torch.allclose(base, run(gd), atol=1e-5)
    assert len(gd.log["d"]) == 2 and len(gd.log["mu_spread"]) == 1


def test_talker_builds_footage_windows_that_hide_their_own_stretch():
    from sang.render.pipeline import Talker
    m, lat, _ = _setup()
    tk = Talker.__new__(Talker)
    tk.cfg, tk.model, tk.latent, tk.dev, tk.lam, tk.gt = {"prefix": P, "frames": L}, m, lat, "cpu", 0.5, None
    tk.sampler = sampler_kwargs(parse_spec("steps=6,split=2"), {"cfg_audio": 2.0, "sample_steps": 6})
    n = P + L + 3 * 16
    gd = tk.footage_guide(torch.randn(1, REF_DIM), None, torch.randn(n, K), torch.randn(2 * n, A),
                          torch.randn(1, 4, 32), None, seed=0)
    assert gd.log["windows"] == 4 and len(gd.target) == 2
    assert tk.footage_guide(torch.randn(1, REF_DIM), None, torch.randn(P + L, K), torch.randn(2 * (P + L), A),
                            torch.randn(1, 4, 32), None, seed=0) is None
