"""Continuous (flow) track: head, objective, x0 estimate, sequential decode."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sang.diffusion import DiffusionHead, add_noise, flow_loss, flow_sample, x0_from_v
from sang.streaming_transformer import StreamingTalkingHead

B, TV, S, Z, D = 2, 5, 4, 4, 32   # tiny: 4x4 latent grid, 4 channels


def _model(**kw):
    return StreamingTalkingHead(64, dim=D, tv=TV, spatial=S, num_heads=2, num_layers=2,
                                factorized_head=False, audio_dim=16, ref_slices=2,
                                continuous=True, z_ch=Z, diff_depth=2, diff_hidden=32,
                                bridge_init="prev", **kw)


def test_head_and_x0_identity():
    torch.manual_seed(0)
    head = DiffusionHead(Z, D, hidden=32, depth=2)
    z0, h = torch.randn(B, 7, Z), torch.randn(B, 7, D)
    t = torch.rand(B)
    z_t, v = add_noise(z0, t)
    assert head(z_t, t, h).shape == (B, 7, Z)
    # x0_from_v inverts add_noise exactly when v is the true velocity
    assert torch.allclose(x0_from_v(z_t, t, v), z0, atol=1e-5)
    loss, parts, z0_hat = flow_loss(head, h, z0)
    assert torch.isfinite(loss) and z0_hat.shape == z0.shape and "diff" in parts
    loss.backward()
    assert head.in_proj.weight.grad is not None


def test_sampler_starts_from_the_training_distribution():
    """t_start=1 must start from noise; the old sampler started from the raw anchor at t=1,
    a point the head never sees in training."""
    torch.manual_seed(0)
    head = DiffusionHead(Z, D, hidden=32, depth=2)
    h, anchor = torch.randn(B, 7, D), torch.randn(B, 7, Z)
    torch.manual_seed(1); a = flow_sample(head, h, (7, Z), steps=2)
    torch.manual_seed(1); b = flow_sample(head, h, (7, Z), steps=2, anchor=anchor, t_start=1.0)
    assert torch.allclose(a, b)                          # anchor ignored at t_start=1
    c = flow_sample(head, h, (7, Z), steps=2, anchor=anchor, t_start=0.5)
    assert c.shape == (B, 7, Z) and not torch.allclose(a, c)


def test_forward_and_sequential_generate():
    torch.manual_seed(0)
    m = _model().train()
    lat = torch.randn(B, TV, Z, S, S)
    audio = torch.randn(B, 9, 16)
    loss, parts, z0_hat = m.forward_continuous(lat, audio)
    assert torch.isfinite(loss) and z0_hat.shape == (B, TV - 2, Z, S, S)
    loss.backward()
    m.eval()
    gen = m.generate_continuous(audio, lat[:, 0], ctx=lat[:, 1], steps=2)
    assert gen.shape == (B, TV - 2, Z, S, S)
    gen_cfg = m.generate_continuous(audio, lat[:, 0], ctx=lat[:, 1], steps=2, audio_cfg=2.0)
    assert gen_cfg.shape == gen.shape
    # slices are decoded sequentially: with "prev" the second content slice's prior is the
    # first *prediction*, so content slices must differ from each other
    assert not torch.allclose(gen[:, 0], gen[:, 1])


if __name__ == "__main__":
    test_head_and_x0_identity()
    test_sampler_starts_from_the_training_distribution()
    test_forward_and_sequential_generate()
    print("ok")
