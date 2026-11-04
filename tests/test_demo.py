"""Extra footage of the person (the demo): the model, the training windows and a training run. CPU only."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sang.data.dataset import MotionWindows
from sang.data.paths import load_index
from sang.model.flow import REF_DIM, Norm, flow_loss, generate
from test_motion_model import A, B, K, L, P, _batch, _model, _unzero
from test_train_eval import make_cache, train

M, N = 4, 20


def _demo(n=N, valid=None):
    mask = torch.zeros(B, n, dtype=torch.bool)
    mask[:, : (n if valid is None else valid)] = True
    return {"demo_motion": torch.randn(B, n, K), "demo_audio": torch.randn(B, 2 * n, A), "demo_mask": mask,
            "demo_keep": torch.ones(B, dtype=torch.bool)}


def test_without_footage_it_is_the_model_without_a_demo():
    torch.manual_seed(0)
    m = _model(demo_tokens=M).eval()
    _unzero(m)
    for blk in m.blocks:  # what training starts from: the cross-attention's output is zero
        torch.nn.init.zeros_(blk.xattn.out_proj.weight)
        torch.nn.init.zeros_(blk.xattn.out_proj.bias)
    b, t = _batch(), torch.full((B,), 0.5)
    tok = m.encode_demo(_demo())
    base = m(b["target"], t, b["audio"], b["ref"], prefix=b["prefix"])
    assert torch.allclose(base, m(b["target"], t, b["audio"], b["ref"], prefix=b["prefix"], demo=tok))
    _unzero(m)
    with_demo = m(b["target"], t, b["audio"], b["ref"], prefix=b["prefix"], demo=tok)
    assert not torch.allclose(base, with_demo, atol=1e-4)
    off = m(b["target"], t, b["audio"], b["ref"], prefix=b["prefix"], demo=tok,
            demo_keep=torch.zeros(B, dtype=torch.bool))
    assert torch.allclose(base, off, atol=1e-6)


def test_any_length_of_footage_gives_the_same_tokens_shape_and_padding_is_ignored():
    m = _model(demo_tokens=M).eval()
    short, long = m.encode_demo(_demo(5)), m.encode_demo(_demo(40))
    assert short.shape == long.shape == (B, M, 32)
    d = _demo(N, valid=7)
    a = m.encode_demo(d)
    d["demo_motion"][:, 7:] = 1e3  # padded frames must not matter
    d["demo_audio"][:, 14:] = 1e3
    assert torch.allclose(a, m.encode_demo(d), atol=1e-5)
    assert torch.isfinite(m.encode_demo(_demo(N, valid=0))).all()


def test_demo_loss_trains_the_encoder_and_sampling_uses_it():
    torch.manual_seed(0)
    m = _model(demo_tokens=M)
    b = {**_batch(), **_demo()}
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    first = None
    for _ in range(150):
        loss, _ = flow_loss(m, b, p_audio=0.0, p_ref=0.0, p_prefix=0.0, p_demo=0.0)
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = first if first is not None else float(loss.detach())
    assert float(loss.detach()) < 0.5 * first
    assert m.demo.motion.weight.grad.abs().sum() > 0
    m.eval()
    tok = m.encode_demo(b)
    ref, audio, n = b["ref"], torch.randn(B, 2 * 37, A), 37
    g = lambda: torch.Generator().manual_seed(1)
    y = generate(m, audio, ref, n, window=L, n_prefix=P, steps=2, generator=g(), demo=tok)
    y0 = generate(m, audio, ref, n, window=L, n_prefix=P, steps=2, generator=g())
    assert y.shape == (B, n, K) and not torch.allclose(y, y0, atol=1e-4)


def test_footage_comes_from_another_clip_of_the_same_person(tmp_path):
    make_cache(tmp_path)
    rows = load_index(tmp_path)
    norm = Norm(torch.zeros(REF_DIM - 63), torch.ones(REF_DIM - 63), torch.zeros(63), torch.ones(63))
    ds = MotionWindows(rows, 16, 4, norm, per_clip=3, fixed=True, demo=30, demo_min=10)
    for i in range(len(ds)):
        x = ds[i]
        j = i % len(ds.rows)
        (p,) = ds.peers[j]  # make_cache: two clips per speaker
        assert Path(ds.rows[p]["path"]).parent == Path(ds.rows[j]["path"]).parent and p != j
        n = int(x["demo_mask"].sum())
        assert bool(x["demo_keep"]) and 10 <= n <= 30 and not x["demo_mask"][n:].any()
        y2, yj, d = ds.mem[p][0], ds.mem[j][0], x["demo_motion"][:n]
        # a stretch of the other clip, as deltas to a frame of the target's clip: the same photo frame as the target
        photos = [y2[s] - d[0] for s in range(len(y2) - n + 1) if torch.allclose(y2[s:s + n] - d, (y2[s] - d[0]).expand(n, -1), atol=1e-4)]
        assert photos, "the footage is not a stretch of the other clip"
        is_row = lambda v: bool(((yj - v).abs().amax(1) < 1e-4).any())
        assert any(is_row(ph) and is_row(x["prefix"][0] + ph) for ph in photos)
        assert x["demo_audio"].shape == (60, 1024) and x["demo_motion"].shape == (30, 69)


def test_training_a_demo_model_reports_what_the_footage_is_worth(tmp_path):
    cache = tmp_path / "cache"
    make_cache(cache)
    out = train(cache, tmp_path / "run", "train_eval=false", "demo_tokens=4", "demo_frames=30", "demo_min_frames=10")
    assert "demo: 4 tokens from 10..30 frames" in out
    evals = [l for l in out.splitlines() if l.startswith("EVAL ")]
    assert evals and all("val_loss_nodemo" in l for l in evals)
    ck = torch.load(tmp_path / "run" / "last.pt", weights_only=False)
    assert any(k.startswith("demo.") for k in ck["ema"]) and ck["cfg"]["demo_tokens"] == 4
