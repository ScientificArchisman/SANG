"""Motion track: the LivePortrait port, the 69-d motion vector, the bidirectional flow transformer and its
objective (logit-normal t, Guidance-Free Training) over latent codes of any width.

Runs on CPU with no LivePortrait install, no weights and no data."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sang.model.flow import REF_DIM, MotionFlowTransformer, Norm, build, flow_loss, generate, sample, sample_t
from sang.model.target import T_DIM, WIDE_DIM, WIDE_REGIONS, from_wide, rotation_matrix, to_target, wide

B, L, P, A, K = 2, 16, 4, 32, 12


def _model(**kw):
    kw = {"dim": 32, "depth": 2, "heads": 2, "audio_dim": A, "max_frames": 64, "t_dim": K, **kw}
    return MotionFlowTransformer(**kw)


def _batch(prefix=True):
    n = (P if prefix else 0) + L
    b = {"target": torch.randn(B, L, K), "audio": torch.randn(B, 2 * n, A),
         "ref": torch.randn(B, REF_DIM)}
    if prefix:
        b["prefix"] = torch.randn(B, P, K)
    return b


def _unzero(m: torch.nn.Module) -> None:
    """adaLN-Zero makes a fresh model the identity; give the zero-initialised layers real weights."""
    g = torch.Generator().manual_seed(0)
    for mod in m.modules():
        if isinstance(mod, torch.nn.Linear) and mod.weight.abs().sum() == 0:
            mod.weight.data = torch.randn(mod.weight.shape, generator=g) * 0.2
            mod.bias.data = torch.randn(mod.bias.shape, generator=g) * 0.2


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


def test_wide_partition_and_round_trip():
    assert sorted(i for r in WIDE_REGIONS.values() for i in r) == list(range(WIDE_DIM)) and WIDE_DIM == 69
    m = torch.randn(7, 70)
    m[:, 0] = 1.2
    m[:, 1:4] = torch.rand(7, 3) * 40 - 20
    base = m[:1].clone()
    base[:, 6] = 0.3
    back = from_wide(wide(m), base)
    assert torch.allclose(back[:, [0, 1, 2, 3, 4, 5]], m[:, [0, 1, 2, 3, 4, 5]], atol=1e-4)
    assert torch.allclose(back[:, 7:], m[:, 7:], atol=1e-4) and torch.all(back[:, 6] == 0.3)


def test_shapes_and_zero_init():
    m = _model()
    b = _batch()
    v = m(b["target"], torch.rand(B), b["audio"], b["ref"], prefix=b["prefix"])
    assert v.shape == (B, L, K)
    assert v.abs().max() == 0, "adaLN-Zero + zero output layer: v must start at exactly 0"
    v = m(b["target"], torch.rand(B), b["audio"][:, : 2 * L], b["ref"])
    assert v.shape == (B, L, K)


def test_frames_are_coupled_unlike_the_per_token_head():
    """dv_i/dx_j must be non-zero for j != i; a per-frame denoiser would give a one-pass window independent
    per-frame noise."""
    m = _model()
    _unzero(m)
    b = _batch(prefix=False)
    x = b["target"].clone().requires_grad_(True)
    v = m(x, torch.full((B,), 0.5), b["audio"], b["ref"])
    v[0, 3].sum().backward()
    others = x.grad[0].abs().sum(-1)
    assert others[[i for i in range(L) if i != 3]].sum() > 0, "frames do not interact"


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
    assert set(parts) == {"fm", "vel", "loss"}


def test_sample_and_long_generation():
    m = _model()
    _unzero(m)
    ref = torch.randn(1, REF_DIM)
    y = sample(m, torch.randn(1, 2 * L, A), ref, L, steps=3)
    assert y.shape == (1, L, K) and torch.isfinite(y).all()
    n = 37
    y = generate(m, torch.randn(1, 2 * n, A), ref, n, window=L, n_prefix=P, steps=2)
    assert y.shape == (1, n, K) and torch.isfinite(y).all()


def test_first_window_is_the_training_dropped_prefix_configuration():
    """Training never puts target frames at positions 0..P-1: a dropped prefix is P null tokens
    there. generate() must reproduce that, not run the window with no prefix slot at all."""
    torch.manual_seed(0)
    m = _model().eval()
    _unzero(m)
    ref, audio = torch.randn(1, REF_DIM), torch.randn(1, 2 * L, A)
    y = generate(m, audio, ref, L, window=L, n_prefix=P, steps=3, generator=torch.Generator().manual_seed(3))
    junk = torch.randn(1, P, K)
    want = sample(m, torch.cat([torch.zeros(1, 2 * P, A), audio], 1), ref, L, prefix=junk, steps=3,
                  generator=torch.Generator().manual_seed(3), prefix_keep=torch.zeros(1, dtype=torch.bool))
    assert torch.allclose(y, want, atol=1e-5)
    old = sample(m, audio, ref, L, prefix=None, steps=3, generator=torch.Generator().manual_seed(3))
    assert not torch.allclose(y, old, atol=1e-3)


def test_start_prefix_continues_from_the_given_frames():
    torch.manual_seed(0)
    m = _model().eval()
    _unzero(m)
    ref, start = torch.randn(1, REF_DIM), torch.randn(1, P, K)
    n = 2 * L + 3
    audio = torch.randn(1, 2 * (P + n), A)
    y = generate(m, audio, ref, n, window=L, n_prefix=P, steps=2, start=start,
                 generator=torch.Generator().manual_seed(5))
    assert y.shape == (1, n, K)
    g = torch.Generator().manual_seed(5)
    w0 = sample(m, audio[:, : 2 * (P + L)], ref, L, prefix=start, steps=2, generator=g)
    w1 = sample(m, audio[:, 2 * L: 2 * (P + 2 * L)], ref, L, prefix=w0[:, -P:], steps=2, generator=g)
    assert torch.allclose(y[:, :L], w0, atol=1e-5) and torch.allclose(y[:, L:2 * L], w1, atol=1e-5)


def test_logit_normal_shift():
    torch.manual_seed(0)
    t0, t3 = sample_t(20000, "cpu", "logit_normal", 0.0), sample_t(20000, "cpu", "logit_normal", 0.3)
    assert abs(float(t0.median()) - 0.5) < 0.02 and float(t3.median()) > 0.55
    assert 0 < float(t3.min()) and float(t3.max()) < 1


class Counting(torch.nn.Module):
    """Records every beta / drop_audio a wrapped model is called with."""

    def __init__(self, m):
        super().__init__()
        self.m, self.calls = m, []
        self.t_dim, self.tpf = m.t_dim, m.tpf

    def forward(self, *a, **kw):
        self.calls.append((kw.get("beta"), kw.get("drop_audio")))
        return self.m(*a, **kw)


def test_gft_takes_guidance_as_an_input_and_samples_in_one_pass():
    m = _model().eval()
    b = _batch(prefix=False)
    t = torch.full((B,), 0.5)
    base = m(b["target"], t, b["audio"], b["ref"])
    assert torch.allclose(base, m(b["target"], t, b["audio"], b["ref"], beta=torch.full((B,), 0.3)))
    _unzero(m)
    v1, v3 = (m(b["target"], t, b["audio"], b["ref"], beta=torch.full((B,), x)) for x in (1.0, 0.3))
    assert not torch.allclose(v1, v3)
    drop = torch.ones(B, dtype=torch.bool)
    assert torch.allclose(m(b["target"], t, b["audio"], b["ref"], drop_audio=drop, beta=torch.full((B,), 0.3)),
                          m(b["target"], t, b["audio"], b["ref"], drop_audio=drop))
    c = Counting(m)
    sample(c, b["audio"], b["ref"], L, steps=3, cfg_audio=2.0)
    assert len(c.calls) == 3 and all(torch.allclose(beta, torch.full((B,), 0.5)) and d is None for beta, d in c.calls)


def test_gft_loss_trains_the_sampling_model_and_not_the_unconditional_pass():
    torch.manual_seed(0)
    m = _model()
    _unzero(m)
    b = _batch()
    c = Counting(m)
    loss, parts = flow_loss(c, b)
    assert len(c.calls) == 2 and c.calls[1][1].all() and c.calls[1][0] is None
    loss.backward()
    assert m.beta[-1].weight.grad.abs().sum() > 0 and torch.isfinite(loss)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    first = None
    for _ in range(120):
        loss, _ = flow_loss(m, b, p_audio=0.0, p_ref=0.0, p_prefix=0.0)
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = first if first is not None else float(loss.detach())
    assert float(loss.detach()) < 0.5 * first


def test_build_generates_latent_k_codes():
    cfg = {"latent": "x.pt", "latent_k": 12, "dim": 32, "layers": 1, "heads": 2, "audio_encoder": "wavlm-base"}
    m = build(cfg)
    b = {"target": torch.randn(B, L, 12), "audio": torch.randn(B, 2 * L, 768), "ref": torch.randn(B, REF_DIM)}
    loss, parts = flow_loss(m, b)
    assert torch.isfinite(loss) and set(parts) == {"fm", "vel", "loss"}
    assert sample(m, b["audio"], b["ref"], L, steps=2).shape == (B, L, 12)


def test_norm_describes_the_photo():
    m, kp = torch.randn(100, 70) * 3 + 1, torch.randn(100, 63)
    n = Norm.fit(to_target(m), kp)
    assert torch.allclose(n.target(to_target(m)).mean(0), torch.zeros(T_DIM), atol=1e-4)
    assert n.ref(kp[:2], m[:2]).shape == (2, REF_DIM)


def test_default_size():
    """Report, don't guess: the number that goes on the slide."""
    m = MotionFlowTransformer()
    n = sum(p.numel() for p in m.parameters())
    assert 40e6 < n < 80e6, n
    print(f"default MotionFlowTransformer: {n / 1e6:.1f} M parameters")

