"""Motion latents and model spaces (sang.latent): ordered codes, exact full-rank PCA, nested-dropout training,
packing, and the map between a model's output and 70-d motion. CPU only."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sang.latent import MotionLatent, Space, curve, fit_ae, fit_pca
from sang.motion import M_DIM, T_DIM, to_target, wide
from sang.motion_model import Norm


def deltas(n=3000, seed=0):
    """Deltas with a few strong directions and weak noise, so ordering matters."""
    g = torch.Generator().manual_seed(seed)
    basis = torch.randn(4, 69, generator=g)
    return torch.randn(n, 4, generator=g) * torch.tensor([5.0, 3.0, 2.0, 1.0]) @ basis + 0.05 * torch.randn(n, 69, generator=g)


def motion(n=20, seed=0):
    g = torch.Generator().manual_seed(seed)
    m = torch.randn(n, M_DIM, generator=g) * 0.05
    m[:, 0] = 1.1 + 0.05 * torch.rand(n, generator=g)
    m[:, 1:4] = torch.randn(n, 3, generator=g) * 5
    return m


def test_pca_is_ordered_and_exact_at_full_rank():
    d = deltas()
    lat = fit_pca(d)
    assert lat.K == 69
    r = [c["r2"] for c in curve(lat, d, [1, 2, 4, 8, 69])]
    assert r == sorted(r) and r[2] > 0.95 and abs(r[-1] - 1) < 1e-5
    assert torch.allclose(lat.decode(lat.encode(d[:10])), d[:10], atol=1e-3)
    u = lat.encode(d)
    assert torch.allclose(u.mean(0), torch.zeros(69), atol=1e-3) and torch.allclose(u.std(0), torch.ones(69), atol=1e-3)


def test_nested_dropout_orders_the_autoencoder():
    torch.manual_seed(0)
    d = deltas()
    lat = fit_ae(d, K=8, hidden=64, steps=600, batch=512, device="cpu", log_every=0)
    r = {c["k"]: c["r2"] for c in curve(lat, d, [1, 2, 4, 8])}
    assert r[1] < r[2] < r[4] and r[4] > 0.8, r


def test_pack_round_trip_and_bad_kind():
    lat = fit_pca(deltas(), K=12)
    back = MotionLatent.unpack(lat.pack())
    d = deltas(50, seed=1)
    assert torch.allclose(back.encode(d, 5), lat.encode(d, 5)) and not back.training
    with pytest.raises(ValueError):
        MotionLatent(8, "vae")


def test_space_maps_motion_both_ways():
    m = motion(200)
    norm = Norm.fit(to_target(m), torch.randn(200, 63))
    s42 = Space(norm)
    z = s42.encode(m, m[:1])
    assert z.shape == (200, T_DIM)
    back = s42.decode(z, m[:1])
    assert torch.allclose(to_target(back), to_target(m), atol=1e-4)
    w = wide(m)
    lat = fit_pca(w - w[torch.randint(200, (200,))], K=69)
    assert fit_pca(w[:10] - w[:1]).K == 10
    sl = Space(norm, lat)
    u = sl.encode(m, m[:1], 69)
    assert u.shape == (200, 69) and torch.allclose(sl.decode(u, m[:1])[:, [0, 1, 2, 3, 4, 5]], m[:, :6], atol=1e-3)
    assert torch.allclose(sl.decode(u, m[:1])[:, 7:], m[:, 7:], atol=1e-3)
    assert sl.encode(m[:1], m[:1], 8).shape == (1, 8)
    with pytest.raises(ValueError):
        sl.decode(u, m[:1], ramp=1.5)

