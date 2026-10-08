"""Phase B training-recipe options: logit-normal t, contrastive flow matching, lip CCC and band-limited
spectral losses, extra EMAs, and an end-to-end CPU training run with all of them on."""
import importlib.util
import json
import random
import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from sang.motion import REGIONS, T_DIM
from sang.motion_model import (REF_DIM, MotionFlowTransformer, ema_weights, flow_loss, lip_ccc_loss, sample_t,
                               spectral_loss)

MOUTH = REGIONS["mouth"]


def _tm():
    spec = importlib.util.spec_from_file_location("tm", REPO / "scripts" / "train_motion.py")
    tm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tm)
    return tm


def test_logit_normal_shift():
    torch.manual_seed(0)
    assert sample_t(20000, "cpu").mean().item() == pytest.approx(0.5, abs=0.01)
    assert sample_t(20000, "cpu", "logit_normal", mean=0.5).median().item() == pytest.approx(0.622, abs=0.01)


def test_lip_ccc_and_spectral_losses():
    g = torch.Generator().manual_seed(0)
    x0 = torch.randn(4, 64, T_DIM, generator=g)
    a = torch.zeros(T_DIM)
    a[MOUTH] = torch.randn(len(MOUTH), generator=g)
    w = torch.ones(4)
    assert lip_ccc_loss(x0, x0, a, w).item() == pytest.approx(0.0, abs=1e-5)
    assert lip_ccc_loss(2 * x0, x0, a, w) > 0.1                      # right timing, wrong amplitude
    assert lip_ccc_loss(x0.roll(3, 1), x0, a, w) > 0.5               # wrong timing
    assert lip_ccc_loss(2 * x0, x0, a, torch.zeros(4)) == 0          # masked out
    k_max = 25                                                        # 10 Hz at 25 fps, 64 frames
    assert spectral_loss(x0, x0, MOUTH, k_max, w).item() == pytest.approx(0.0, abs=1e-6)
    t = torch.arange(64.0)
    hi = torch.cos(2 * torch.pi * 30 * t / 64)[None, :, None]         # bin 30 = 11.7 Hz: above the band
    lo = torch.cos(2 * torch.pi * 5 * t / 64)[None, :, None]          # bin 5 = 2 Hz: inside it
    assert spectral_loss(x0 + hi, x0, MOUTH, k_max, w).item() < 1e-5
    assert spectral_loss(x0 + lo, x0, MOUTH, k_max, w).item() > 0.01


def _batch(B=4, L=16, P=4, neg=True):
    g = torch.Generator().manual_seed(1)
    b = {"target": torch.randn(B, L, T_DIM, generator=g), "audio": torch.randn(B, 2 * (P + L), 8, generator=g),
         "ref": torch.randn(B, REF_DIM, generator=g), "prefix": torch.randn(B, P, T_DIM, generator=g)}
    if neg:
        b["neg_target"] = torch.randn(B, L, T_DIM, generator=g)
    return b


def _model():
    torch.manual_seed(0)
    m = MotionFlowTransformer(dim=32, depth=1, heads=2, audio_dim=8)
    for mod in m.modules():
        if isinstance(mod, torch.nn.Linear) and mod.weight.abs().sum() == 0:
            mod.weight.data = torch.randn(mod.weight.shape) * 0.2
    return m


def test_flow_loss_terms_add_up():
    m, b = _model(), _batch()
    torch.manual_seed(5)
    base, out0 = flow_loss(m, b)
    torch.manual_seed(5)
    with_cfm, out1 = flow_loss(m, b, lam_cfm=0.05)
    assert "cfm" not in out0 and torch.allclose(with_cfm, base - 0.05 * out1["cfm"], atol=1e-5)
    torch.manual_seed(5)
    _, out_batch = flow_loss(m, {k: v for k, v in b.items() if k != "neg_target"}, lam_cfm=0.05)
    assert not torch.allclose(out_batch["cfm"], out1["cfm"])             # falls back to another clip of the batch
    a = torch.zeros(T_DIM)
    a[MOUTH] = 1.0
    torch.manual_seed(5)
    full, out2 = flow_loss(m, b, lam_ccc=0.2, lip_a=a, lam_spec=0.3, aux_tmax=1.0, spec_hz=10.0)
    assert torch.allclose(full, base + 0.2 * out2["ccc"] + 0.3 * out2["spec"], atol=1e-5)
    full.backward()                                                    # the new terms are differentiable
    with pytest.raises(ValueError):
        flow_loss(m, b, lam_ccc=0.2)


def test_negative_window_is_elsewhere_in_the_clip():
    tm = _tm()
    rng = random.Random(0)
    for _ in range(50):
        sl = tm.neg_start(200, 64, 70, rng)
        assert sl.stop - sl.start == 64 and abs(sl.start - 70) >= 16 and 0 <= sl.start <= 136
    assert tm.neg_start(64, 64, 0, rng) == slice(0, 64)                   # no room: the same window


def test_ema_weights():
    ck = {"ema": {"w": 1}, "ema_extra": {"0.999": {"w": 2}}}
    assert ema_weights(ck) == {"w": 1} and ema_weights(ck, "0.999") == {"w": 2} and ema_weights(ck, 0.999) == {"w": 2}
    with pytest.raises(ValueError):
        ema_weights(ck, "0.9995")


def test_training_runs_with_every_phase_b_option(tmp_path):
    from sang.motion import to_target
    cache = tmp_path / "cache"
    rows, g = [], torch.Generator().manual_seed(0)
    for s in range(6):
        p = cache / f"spk{s}" / "c0.pt"
        p.parent.mkdir(parents=True)
        T = 120
        m = torch.randn(T, 70, generator=g) * 0.05
        m[:, 0] = 1.2
        torch.save({"m": m.half(), "kp": torch.randn(T, 63, generator=g).half(),
                    "audio": torch.randn(2 * T, 1024, generator=g).half(), "n": T}, p)
        rows.append({"clip": f"/x/spk{s}/c0.mp4", "path": str(p), "n": T})
    (cache / "index_000.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    w = [0.0] * T_DIM
    for i in MOUTH:
        w[i] = 1.0
    (cache / "openness.json").write_text(json.dumps({"lip": {"w": w, "b": 0.0}, "eye": {"w": [0.0] * T_DIM, "b": 0.0}}))
    out = tmp_path / "run"
    over = [f"cache_dir={cache}", f"out_dir={out}", "dim=32", "layers=1", "heads=2", "max_steps=4", "warmup_steps=1",
            "eval_every=4", "log_every=2", "batch_size=4", "workers=0", "eval_batches=1", "sample_steps=2",
            "val_frac=0.4", "t_dist=logit_normal", "t_mean=0.3", "lam_cfm=0.05", "lam_ccc=0.1", "lam_spec=0.2",
            "aux_tmax=1.0", "ema_extra=[0.999,0.9995]"]
    log = subprocess.run([sys.executable, str(REPO / "scripts" / "train_motion.py"), *over], capture_output=True,
                         encoding="utf-8", cwd=REPO)
    assert log.returncode == 0, log.stderr[-2000:]
    assert "lam_cfm" in log.stdout and "cfm " in log.stdout and "ccc " in log.stdout and "spec " in log.stdout
    ck = torch.load(out / "last.pt", weights_only=False)
    assert sorted(ck["ema_extra"]) == ["0.999", "0.9995"]
    assert ck["ema_extra"]["0.999"].keys() == ck["ema"].keys()
