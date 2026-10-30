"""Motion latents (sang.model.latent): ordered codes, exact full-rank PCA, packing, and the map between codes and
70-d motion. CPU only."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sang.model.latent import MotionLatent, fit_pca
from sang.model.target import M_DIM, wide


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
    err = [float(((lat.decode(lat.encode(d, k)) - d) ** 2).mean()) for k in (1, 2, 4, 8, 69)]
    assert err == sorted(err, reverse=True) and err[-1] < 1e-8
    assert torch.allclose(lat.decode(lat.encode(d[:10])), d[:10], atol=1e-3)
    u = lat.encode(d)
    assert torch.allclose(u.mean(0), torch.zeros(69), atol=1e-3) and torch.allclose(u.std(0), torch.ones(69), atol=1e-3)


def test_pack_round_trip_and_old_packs():
    lat = fit_pca(deltas(), K=12)
    back = MotionLatent.unpack(lat.pack())
    d = deltas(50, seed=1)
    assert torch.allclose(back.encode(d, 5), lat.encode(d, 5)) and not back.training
    old = MotionLatent.unpack({**lat.pack(), "hidden": 512})  # pca.pt and checkpoints written before the cleanup
    assert torch.allclose(old.encode(d, 5), lat.encode(d, 5))


def test_codes_map_motion_both_ways():
    m = motion(200)
    w = wide(m)
    lat = fit_pca(w - w[torch.randint(200, (200,))], K=69)
    assert fit_pca(w[:10] - w[:1]).K == 10
    u = lat.from_motion(m, m[:1], 69)
    back = lat.to_motion(u, m[:1])
    assert u.shape == (200, 69) and torch.allclose(back[:, :6], m[:, :6], atol=1e-3)
    assert torch.allclose(back[:, 7:], m[:, 7:], atol=1e-3)
    assert lat.from_motion(m[:1], m[:1], 8).shape == (1, 8)
    assert torch.allclose(MotionLatent.of({"latent": lat.pack()}).encode(w[:3], 5), lat.encode(w[:3], 5))
