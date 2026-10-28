"""Sampler options: mouth averaging, mouth temperature, and the spec strings the scripts take. CPU only, tiny model."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sang.motion import REGIONS, T_DIM
from sang.motion_model import REF_DIM, MotionFlowTransformer, flow_times, generate, parse_spec, sample, sampler_kwargs

MOUTH = REGIONS["mouth"]
CFG = {"cfg_audio": 2.0, "sample_steps": 4, "frames": 16, "prefix": 4}


def _model(seed=0):
    torch.manual_seed(seed)
    m = MotionFlowTransformer(dim=32, depth=1, heads=2, audio_dim=8).eval()
    g = torch.Generator().manual_seed(seed)
    for mod in m.modules():
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


def test_temperature_scales_the_starting_noise():
    m, (a, r, p) = _model(), _inputs()
    base = sample(m, a, r, 16, prefix=p, steps=4, generator=_gen())
    ones = torch.ones(T_DIM)
    assert torch.allclose(sample(m, a, r, 16, prefix=p, steps=4, generator=_gen(), noise_scale=ones), base)
    cold = ones.clone()
    cold[MOUTH] = 0.5
    assert not torch.allclose(sample(m, a, r, 16, prefix=p, steps=4, generator=_gen(), noise_scale=cold), base)


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
        parse_spec("g=2,mouht=1.25")
    kw = sampler_kwargs(parse_spec("mouth=1.1,avg=4,tau=0.7,steps=6,ramp=1.5"), CFG)
    assert kw["mouth_avg"] == 4 and kw["steps"] == 6 and "ramp" not in kw
    assert kw["cfg_audio"][MOUTH[0]] == 1.1 and kw["cfg_audio"][0] == 2.0
    assert kw["noise_scale"][MOUTH[0]] == 0.7 and kw["noise_scale"][0] == 1.0
    assert sampler_kwargs({}, CFG) == {"cfg_audio": 2.0, "steps": 4, "mouth_avg": 1}
    latent = {**CFG, "latent": "x.pt", "latent_k": 8}
    assert sampler_kwargs(parse_spec("g=3,steps=6"), latent) == {"cfg_audio": 3.0, "steps": 6, "mouth_avg": 1}
    for bad in ("mouth=1.25", "tau=0.5"):
        with pytest.raises(ValueError):
            sampler_kwargs(parse_spec(bad), latent)

