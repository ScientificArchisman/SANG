"""Sampler options (report section A2-A4): mouth averaging, mouth temperature, autoguidance, guidance
interval, Sway step schedule, and the spec strings the scripts sweep. CPU only, tiny model."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sang.motion import REGIONS, T_DIM
from sang.motion_model import (REF_DIM, MotionFlowTransformer, cfg_combine, flow_times, generate, parse_spec, sample,
                               sampler_kwargs)
from sang.naturalness import guided_generate

MOUTH = REGIONS["mouth"]
CFG = {"cfg_audio": 2.0, "sample_steps": 4, "frames": 16, "prefix": 4}


def _model(seed=0):
    torch.manual_seed(seed)
    m = MotionFlowTransformer(dim=32, depth=1, heads=2, audio_dim=8).eval()
    g = torch.Generator().manual_seed(seed)
    for mod in m.modules():                                  # adaLN-Zero starts as the identity
        if isinstance(mod, torch.nn.Linear) and mod.weight.abs().sum() == 0:
            mod.weight.data = torch.randn(mod.weight.shape, generator=g) * 0.2
            mod.bias.data = torch.randn(mod.bias.shape, generator=g) * 0.2
    return m


def _inputs(B=1, L=16, P=4):
    g = torch.Generator().manual_seed(1)
    return (torch.randn(B, 2 * (P + L), 8, generator=g), torch.randn(B, REF_DIM, generator=g),
            torch.randn(B, P, T_DIM, generator=g))


def _gen(seed=3):
    return torch.Generator().manual_seed(seed)


def test_flow_times():
    assert flow_times(4) == pytest.approx([1.0, 0.75, 0.5, 0.25, 0.0])
    t = flow_times(10, sway=-1.0)
    assert t[0] == 1.0 and abs(t[-1]) < 1e-12 and all(a > b for a, b in zip(t, t[1:]))
    assert 1.0 - t[1] < 0.1                                   # more, smaller steps near the noise end
    with pytest.raises(ValueError):
        flow_times(4, sway=-1.5)


def test_defaults_match_the_original_euler_cfg_loop():
    m, (a, r, p) = _model(), _inputs()
    got = sample(m, a, r, 16, prefix=p, steps=4, cfg_audio=2.0, generator=_gen())
    x = torch.randn(1, 16, T_DIM, generator=_gen())          # the pre-change loop, written out
    keep, null = torch.ones(1, dtype=torch.bool), torch.ones(1, dtype=torch.bool)
    for i in range(4):
        t = torch.full((1,), 1.0 - i / 4)
        v = cfg_combine(m(x, t, a, r, prefix=p, prefix_keep=keep), m(x, t, a, r, prefix=p, prefix_keep=keep, drop_audio=null), 2.0)
        x = x - 0.25 * v
    assert torch.allclose(got, x, atol=1e-5)


def test_interval_autoguidance_and_temperature_degenerate_cases():
    m, (a, r, p) = _model(), _inputs()
    base = sample(m, a, r, 16, prefix=p, steps=4, cfg_audio=2.0, generator=_gen())
    no_guidance = sample(m, a, r, 16, prefix=p, steps=4, cfg_audio=1.0, generator=_gen())
    assert torch.allclose(sample(m, a, r, 16, prefix=p, steps=4, cfg_audio=2.0, generator=_gen(), g_tmax=-1.0), no_guidance)
    # guiding with the model itself adds (ag - 1)(v - v) = 0
    assert torch.allclose(sample(m, a, r, 16, prefix=p, steps=4, cfg_audio=2.0, generator=_gen(), ag_model=m, ag=3.0), base)
    weak = _model(seed=7)
    ag = sample(m, a, r, 16, prefix=p, steps=4, cfg_audio=2.0, generator=_gen(), ag_model=weak, ag=2.0)
    assert not torch.allclose(ag, base)
    ones = torch.ones(T_DIM)
    assert torch.allclose(sample(m, a, r, 16, prefix=p, steps=4, generator=_gen(), noise_scale=ones), base)
    cold = ones.clone()
    cold[MOUTH] = 0.5
    assert not torch.allclose(sample(m, a, r, 16, prefix=p, steps=4, generator=_gen(), noise_scale=cold), base)
    assert sample(m, a, r, 16, prefix=p, steps=6, generator=_gen(), sway=-0.8).shape == base.shape


def test_mouth_average_keeps_one_sample_outside_the_mouth():
    m = _model()
    g = torch.Generator().manual_seed(1)
    audio, ref = torch.randn(1, 2 * 40, 8, generator=g), torch.randn(1, REF_DIM, generator=g)
    K = 3
    got = generate(m, audio, ref, 40, window=16, n_prefix=4, steps=3, generator=_gen(), mouth_avg=K)
    many = generate(m, audio.repeat(K, 1, 1), ref.repeat(K, 1), 40, window=16, n_prefix=4, steps=3, generator=_gen())
    other = [i for i in range(T_DIM) if i not in MOUTH]
    assert got.shape == (1, 40, T_DIM)
    assert torch.allclose(got[0][:, MOUTH], many[:, :, MOUTH].mean(0), atol=1e-6)
    assert torch.allclose(got[0][:, other], many[0][:, other])


def test_specs():
    assert parse_spec("g=2,mouth=1.25,avg=4") == {"g": 2.0, "mouth": 1.25, "avg": 4.0} and parse_spec("") == {}
    with pytest.raises(ValueError):
        parse_spec("g=2,mouht=1.25")                          # typos fail loudly
    kw = sampler_kwargs(parse_spec("mouth=1.1,avg=4,tau=0.7,steps=6,sway=-0.8,gmax=0.7"), CFG)
    assert kw["mouth_avg"] == 4 and kw["steps"] == 6 and kw["sway"] == -0.8 and kw["g_tmax"] == 0.7
    assert kw["cfg_audio"][MOUTH[0]] == 1.1 and kw["cfg_audio"][0] == 2.0                  # g defaults to the run's
    assert kw["noise_scale"][MOUTH[0]] == 0.7 and kw["noise_scale"][0] == 1.0
    assert sampler_kwargs({}, CFG) == {"cfg_audio": 2.0, "cfg_rescale": 0.0, "steps": 4, "sway": 0.0,
                                       "g_tmin": 0.0, "g_tmax": 1.0, "mouth_avg": 1}
    assert sampler_kwargs({}, CFG, steps=9)["steps"] == 9
    with pytest.raises(ValueError):
        sampler_kwargs(parse_spec("ag_mouth=1.5"), CFG)       # autoguidance without a guide model
    m = _model()
    ag = sampler_kwargs(parse_spec("ag_mouth=1.5"), CFG, ag_model=m)["ag"]
    assert ag[MOUTH[0]] == 1.5 and ag[0] == 1.0


def test_guided_generate_passes_the_sampler_through():
    m = _model()
    g = torch.Generator().manual_seed(1)
    audio, ref = torch.randn(1, 2 * 30, 8, generator=g), torch.randn(1, REF_DIM, generator=g)
    samp = sampler_kwargs(parse_spec("mouth=1.25,steps=3,avg=2"), CFG)
    y, _ = guided_generate(m, audio, ref, 30, CFG, None, torch.zeros(30 * 640), None, "none", seed=5, sampler=samp)
    want = generate(m, audio, ref, 30, window=16, n_prefix=4, generator=torch.Generator().manual_seed(5), **samp)
    assert torch.allclose(y, want)
