"""Motion track: the LivePortrait port, the 42-d target, and the bidirectional flow transformer.

Runs on CPU with no LivePortrait install, no weights and no data."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sang.diffusion import DiffusionHead
from sang.motion import REGIONS, T_DIM, from_target, rotation_matrix, to_target
from sang.motion_model import REF_DIM, MotionFlowTransformer, Norm, flow_loss, generate, sample

B, L, P, A = 2, 16, 4, 32         # batch, target frames, prefix frames, audio feature dim


def _model(**kw):
    kw = {"dim": 32, "depth": 2, "heads": 2, "audio_dim": A, "max_frames": 64, **kw}
    return MotionFlowTransformer(**kw)


def _batch(prefix=True):
    n = (P if prefix else 0) + L
    b = {"target": torch.randn(B, L, T_DIM), "audio": torch.randn(B, 2 * n, A),
         "ref": torch.randn(B, REF_DIM)}
    if prefix:
        b["prefix"] = torch.randn(B, P, T_DIM)
    return b


def _unzero(m: torch.nn.Module) -> None:
    """adaLN-Zero makes a fresh model the identity; give the zero-initialised layers real weights."""
    g = torch.Generator().manual_seed(0)
    for mod in m.modules():
        if isinstance(mod, torch.nn.Linear) and mod.weight.abs().sum() == 0:
            mod.weight.data = torch.randn(mod.weight.shape, generator=g) * 0.2
            mod.bias.data = torch.randn(mod.bias.shape, generator=g) * 0.2


# ------------------------------------------------------------------ LivePortrait port
def test_rotation_matches_upstream_formula():
    """Literal numpy transcription of upstream src/utils/camera.py get_rotation_matrix."""
    pitch, yaw, roll = 12.0, -25.0, 7.5
    x, y, z = (np.deg2rad(a) for a in (pitch, yaw, roll))
    rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
    want = (rz @ ry @ rx).T
    got = rotation_matrix(torch.tensor([pitch]), torch.tensor([yaw]), torch.tensor([roll]))[0]
    assert np.allclose(got.numpy(), want, atol=1e-6)


def test_target_partition_and_round_trip():
    assert sorted(i for r in REGIONS.values() for i in r) == list(range(T_DIM)) and T_DIM == 42
    m = torch.randn(7, 70)
    m[:, 0] = 1.2
    m[:, 1:4] = torch.rand(7, 3) * 40 - 20
    assert torch.allclose(from_target(to_target(m), m), m, atol=1e-4)


# ------------------------------------------------------------------ model
def test_shapes_and_zero_init():
    m = _model()
    b = _batch()
    v = m(b["target"], torch.rand(B), b["audio"], b["ref"], prefix=b["prefix"])
    assert v.shape == (B, L, T_DIM)
    assert v.abs().max() == 0, "adaLN-Zero + zero output layer: v must start at exactly 0"
    v = m(b["target"], torch.rand(B), b["audio"][:, : 2 * L], b["ref"])      # no prefix
    assert v.shape == (B, L, T_DIM)


def test_frames_are_coupled_unlike_the_per_token_head():
    """The reason this module exists. dv_i/dx_j must be non-zero for j != i; for the old per-token
    DiffusionHead it is exactly zero, so a one-pass window would get independent per-frame noise."""
    m = _model()
    _unzero(m)
    b = _batch(prefix=False)
    x = b["target"].clone().requires_grad_(True)
    v = m(x, torch.full((B,), 0.5), b["audio"], b["ref"])
    v[0, 3].sum().backward()
    others = x.grad[0].abs().sum(-1)
    assert others[[i for i in range(L) if i != 3]].sum() > 0, "frames do not interact"

    head = DiffusionHead(T_DIM, 32, hidden=32, depth=2)
    _unzero(head)
    z = torch.randn(1, L, T_DIM, requires_grad=True)
    head(z, torch.full((1,), 0.5), torch.randn(1, L, 32))[0, 3].sum().backward()
    assert z.grad[0, [i for i in range(L) if i != 3]].abs().sum() == 0, "control: per-token head"


def test_attention_window_limits_reach():
    m = _model(depth=1, attn_window=2)
    _unzero(m)
    b = _batch(prefix=False)
    x = b["target"].clone().requires_grad_(True)
    m(x, torch.full((B,), 0.5), b["audio"], b["ref"])[0, 8].sum().backward()
    reach = x.grad[0].abs().sum(-1)
    assert reach[6:11].sum() > 0 and reach[:6].sum() == 0 and reach[11:].sum() == 0


def test_audio_dropout_and_prefix_dropout_change_the_output():
    m = _model()
    _unzero(m)
    b = _batch()
    t = torch.full((B,), 0.5)
    base = m(b["target"], t, b["audio"], b["ref"], prefix=b["prefix"])
    no_audio = m(b["target"], t, b["audio"], b["ref"], prefix=b["prefix"],
                 drop_audio=torch.ones(B, dtype=torch.bool))
    no_prefix = m(b["target"], t, b["audio"], b["ref"], prefix=b["prefix"],
                  prefix_keep=torch.zeros(B, dtype=torch.bool))
    assert not torch.allclose(base, no_audio) and not torch.allclose(base, no_prefix)


def test_overfits_one_batch():
    """The smallest thing that fails if the objective or the wiring is broken."""
    torch.manual_seed(0)
    m = _model()
    b = _batch()
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    first = None
    for step in range(150):
        loss, parts = flow_loss(m, b, p_audio=0.0, p_ref=0.0, p_prefix=0.0)
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = first if first is not None else float(loss.detach())
    assert loss.item() < 0.5 * first, f"{first:.3f} -> {loss.item():.3f}"
    assert {"fm_rot", "fm_brow", "fm_eyes", "fm_mouth", "vel"} <= set(parts)


def test_sample_and_long_generation():
    m = _model()
    _unzero(m)
    ref = torch.randn(1, REF_DIM)
    y = sample(m, torch.randn(1, 2 * L, A), ref, L, steps=3)
    assert y.shape == (1, L, T_DIM) and torch.isfinite(y).all()
    n = 37                                               # not a multiple of the window
    y = generate(m, torch.randn(1, 2 * n, A), ref, n, window=L, n_prefix=P, steps=2)
    assert y.shape == (1, n, T_DIM) and torch.isfinite(y).all()


def test_norm_round_trip():
    y, kp = torch.randn(100, T_DIM) * 3 + 1, torch.randn(100, 63)
    n = Norm.fit(y, kp)
    assert torch.allclose(n.untarget(n.target(y)), y, atol=1e-4)
    assert n.ref(kp[:2], y[:2]).shape == (2, REF_DIM)


def test_default_size():
    """Report, don't guess: the number that goes on the slide."""
    m = MotionFlowTransformer()
    n = sum(p.numel() for p in m.parameters())
    assert 40e6 < n < 80e6, n
    print(f"default MotionFlowTransformer: {n / 1e6:.1f} M parameters")
