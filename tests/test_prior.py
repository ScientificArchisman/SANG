"""Audio prior (flow from mu(audio) + noise) and batched CFG (sang/motion_model.py). CPU only."""
import json
import subprocess
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from sang.motion import T_DIM
from sang.motion_model import REF_DIM, MotionFlowTransformer, flow_loss, generate, sample


def tiny(prior: bool, seed: int = 0) -> MotionFlowTransformer:
    torch.manual_seed(seed)
    m = MotionFlowTransformer(dim=32, depth=2, heads=2, audio_dim=16, prior=prior, prior_width=16).eval()
    with torch.no_grad():                                    # zero-initialised heads: make v non-trivial
        for p in m.parameters():
            if p.ndim >= 1 and p.abs().sum() == 0:
                p.normal_(0, 0.05)
    return m


def inputs(B=2, P=3, L=8, seed=1):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(B, 2 * (P + L), 16, generator=g), torch.randn(B, REF_DIM, generator=g),
            torch.randn(B, P, T_DIM, generator=g))


def test_batched_cfg_matches_two_calls():
    for prior in (False, True):
        m = tiny(prior)
        audio, ref, prefix = inputs()
        kw = dict(prefix=prefix, steps=4, cfg_audio=2.0)
        a = sample(m, audio, ref, 8, generator=torch.Generator().manual_seed(5), cfg_batch=True, **kw)
        b = sample(m, audio, ref, 8, generator=torch.Generator().manual_seed(5), cfg_batch=False, **kw)
        assert (a - b).abs().max() < 1e-5, prior


def test_prior_sampling_starts_from_mu():
    m = tiny(True)
    audio, ref, prefix = inputs()
    mu = m.prior(audio, ref, 3 + 8)[:, 3:]
    with torch.no_grad():                                    # a flow that never moves: output = start
        m.out.weight.zero_()
        m.out.bias.zero_()
    y = sample(m, audio, ref, 8, prefix=prefix, steps=3, cfg_audio=1.0, noise_scale=0.0)
    assert torch.allclose(y, mu, atol=1e-6)                  # zero noise: exactly mu
    y1 = sample(m, audio, ref, 8, prefix=prefix, steps=3, cfg_audio=1.0, generator=torch.Generator().manual_seed(0))
    assert not torch.allclose(y1, mu)                        # unit noise around mu


def test_prior_constraints_hold_on_the_full_motion():
    m = tiny(True)
    audio, ref, prefix = inputs()
    a = torch.zeros(T_DIM)
    a[5] = 1.0
    ub = torch.full((2, 8), -0.3)                            # column 5 must stay <= -0.3 on every frame
    y = sample(m, audio, ref, 8, prefix=prefix, steps=4, cfg_audio=2.0, bounds=[(a, ub)],
               generator=torch.Generator().manual_seed(0))
    assert (y[..., 5] <= -0.3 + 1e-5).all()


def test_flow_loss_reports_and_trains_the_prior():
    m = tiny(True).train()
    audio, ref, prefix = inputs(B=4)
    batch = {"target": torch.randn(4, 8, T_DIM), "prefix": prefix.repeat(2, 1, 1)[:4], "audio": audio.repeat(2, 1, 1)[:4],
             "ref": ref.repeat(2, 1)[:4]}
    loss, parts = flow_loss(m, batch, t_dist="logit_normal", t_mean=0.3)
    assert "prior" in parts and torch.isfinite(loss)
    loss.backward()
    assert m.prior.out.weight.grad is not None and m.prior.out.weight.grad.abs().sum() > 0
    old = tiny(False).train()
    _, parts = flow_loss(old, batch)
    assert "prior" not in parts


def test_generate_with_prior_long_clip_and_mouth_average():
    m = tiny(True)
    g = torch.Generator().manual_seed(0)
    audio, ref = torch.randn(1, 2 * 30, 16, generator=g), torch.randn(1, REF_DIM, generator=g)
    y = generate(m, audio, ref, 30, window=8, n_prefix=3, steps=2, cfg_audio=2.0, mouth_avg=2,
                 generator=torch.Generator().manual_seed(0))
    assert y.shape == (1, 30, T_DIM) and torch.isfinite(y).all()


def test_training_with_prior_end_to_end(tmp_path):
    cache = tmp_path / "cache"
    rows, g = [], torch.Generator().manual_seed(0)
    for s in range(6):
        for c in range(2):
            p = cache / f"spk{s}" / f"c{c}.pt"
            p.parent.mkdir(parents=True, exist_ok=True)
            T = 90
            mm = torch.randn(T, 70, generator=g) * 0.05
            mm[:, 0] = 1.2
            torch.save({"m": mm.half(), "kp": torch.randn(T, 63, generator=g).half(),
                        "audio": torch.randn(2 * T, 1024, generator=g).half(), "n": T}, p)
            rows.append({"clip": f"/x/spk{s}/c{c}.mp4", "path": str(p), "n": T})
    (cache / "index_000.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    out = tmp_path / "run"
    over = [f"cache_dir={cache}", f"out_dir={out}", "dim=32", "layers=1", "heads=2", "max_steps=3", "warmup_steps=1",
            "eval_every=3", "log_every=1", "batch_size=2", "workers=0", "eval_batches=1", "sample_steps=2",
            "val_frac=0.3", "windows_per_clip=2", "prior=true", "prior_width=16", "t_dist=logit_normal", "t_mean=0.3"]
    log = subprocess.run([sys.executable, str(REPO / "scripts" / "train_motion.py"), *over],
                         capture_output=True, encoding="utf-8", cwd=REPO)
    assert log.returncode == 0, log.stderr[-2000:]
    assert "audio prior" in log.stdout and "prior" in log.stdout and "val_prior" in log.stdout
    ck = torch.load(out / "last.pt", weights_only=False)
    assert any(k.startswith("prior.") for k in ck["ema"]) and ck["cfg"]["prior"] is True
