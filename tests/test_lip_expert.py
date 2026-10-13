"""C2.0 motion-space speech expert (sang/lip_expert.py, scripts/lip_expert.py). CPU only."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from sang import lip_expert as le


def test_frame_audio_averages_the_two_ticks_of_each_frame():
    ticks = torch.arange(10, dtype=torch.float32)[:, None].repeat(1, 3)    # 5 frames x 2 ticks
    fa = le.frame_audio(ticks, n=4)
    assert fa.shape == (4, 3) and fa[0, 0] == 0.5 and fa[3, 0] == 6.5
    assert le.frame_audio(ticks, n=9).shape == (5, 3)                     # never past the audio


def test_pca_keeps_the_dominant_direction():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2000, 1)) @ np.array([[3.0, 4.0, 0.0]]) / 5 * 10 + 0.1 * rng.standard_normal((2000, 3))
    pca = le.fit_pca(x, 1)
    assert pca["explained"] > 0.99 and abs(abs(pca["basis"][0, 0]) - 0.6) < 0.01
    assert le.project(x, pca).shape == (2000, 1)


def test_r2_and_rescaled_r2():
    rng = np.random.default_rng(1)
    t = le.centre(rng.standard_normal((100, 4)))
    assert abs(le.r2([t], [t]) - 1) < 1e-12 and abs(le.clip_r2(np.zeros_like(t), t)) < 1e-12
    half = 0.5 * t                                                        # right shape, half the size
    assert abs(le.clip_r2(half, t) - 0.75) < 1e-9 and abs(le.clip_r2(half, t, rescale=True) - 1) < 1e-9


def test_verdict():
    assert le.verdict(0.05, 0.0) == "WEAK EXPERT"
    assert le.verdict(0.30, 0.20) == "GO" and le.verdict(0.30, 0.28) == "NO-GO" and le.verdict(0.30, 0.35) == "NO-GO"


def test_expert_learns_speech_that_the_mouth_carries():
    torch.manual_seed(0)
    rng = np.random.default_rng(2)
    W = rng.standard_normal((18, 4)).astype(np.float32)
    x = torch.from_numpy(np.cumsum(rng.standard_normal((64, 80, 18)), 1).astype(np.float32) * 0.1)
    t = torch.from_numpy(np.einsum("btd,dk->btk", x.numpy(), W))         # speech = a linear read of the mouth
    ex = le.LipExpert(18, 4, width=64)
    opt = torch.optim.Adam(ex.parameters(), 3e-3)
    for _ in range(300):
        loss = torch.nn.functional.mse_loss(ex(x), t)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        p = ex(x).numpy()
    assert le.r2(list(p), list(t.numpy())) > 0.9


def test_probe_runs_end_to_end_on_a_tiny_cache(tmp_path):
    """Fake cache -> tiny SANG checkpoint (2 steps) -> scripts/lip_expert.py on CPU."""
    cache = tmp_path / "cache"
    rows, g = [], torch.Generator().manual_seed(0)
    for s in range(8):
        for c in range(2):
            p = cache / f"spk{s}" / f"c{c}.pt"
            p.parent.mkdir(parents=True, exist_ok=True)
            T = 90
            m = torch.randn(T, 70, generator=g) * 0.05
            m[:, 0] = 1.2
            torch.save({"m": m.half(), "kp": torch.randn(T, 63, generator=g).half(),
                        "audio": torch.randn(2 * T, 1024, generator=g).half(), "n": T}, p)
            rows.append({"clip": f"/x/spk{s}/c{c}.mp4", "path": str(p), "n": T})
    (cache / "index_000.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    (cache / "talkvid_meta.json").write_text(json.dumps(
        {f"spk{s}/c{c}": {"language": "English", "person": str(s)} for s in range(8) for c in range(2)}))
    run = tmp_path / "run"
    over = [f"cache_dir={cache}", f"out_dir={run}", "dim=32", "layers=1", "heads=2", "max_steps=2", "warmup_steps=1",
            "eval_every=2", "log_every=1", "batch_size=2", "workers=0", "eval_batches=1", "sample_steps=2",
            "val_frac=0.3", "windows_per_clip=2"]
    log = subprocess.run([sys.executable, str(REPO / "scripts" / "train_motion.py"), *over],
                         capture_output=True, encoding="utf-8", cwd=REPO)
    assert log.returncode == 0, log.stderr[-2000:]
    out = tmp_path / "expert"
    log = subprocess.run([sys.executable, str(REPO / "scripts" / "lip_expert.py"), "--ckpt", str(run / "last.pt"),
                          "--device", "cpu", "--min-frames", "60", "--max-frames", "90", "--max-train-frames", "90",
                          "--steps", "20", "--batch", "8", "--k", "8", "--out", str(out)],
                         capture_output=True, encoding="utf-8", cwd=REPO)
    assert log.returncode == 0, (log.stdout[-2000:], log.stderr[-2000:])
    assert "VERDICT:" in log.stdout and "BY GROUP" in log.stdout
    res = json.loads((out / "probe.json").read_text())
    assert res["verdict"] in ("GO", "NO-GO", "WEAK EXPERT") and res["groups"]["all"]["clips"] >= 4
    ck = torch.load(out / "expert.pt", weights_only=False)
    ex = le.LipExpert(len(ck["cols"]), ck["k"])
    ex.load_state_dict(ck["state"])
    assert ex(torch.zeros(1, 64, len(ck["cols"]))).shape == (1, 64, 8)
