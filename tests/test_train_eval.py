"""train_eval: the val score on fixed training clips (EVAL_TRAIN lines) must not change the run. CPU only."""
import json
import subprocess
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from sang.model.latent import fit_pca


def make_cache(root: Path, n_spk: int = 6, T: int = 90) -> list[dict]:
    """A tiny motion cache with its index, sync offsets (all kept, no shift) and a motion latent."""
    rows, g = [], torch.Generator().manual_seed(0)
    for s in range(n_spk):
        for c in range(2):
            p = root / f"spk{s}" / f"c{c}.pt"
            p.parent.mkdir(parents=True, exist_ok=True)
            m = torch.randn(T, 70, generator=g) * 0.05
            m[:, 0] = 1.2
            torch.save({"m": m.half(), "kp": torch.randn(T, 63, generator=g).half(),
                        "audio": torch.randn(2 * T, 1024, generator=g).half(), "n": T}, p)
            rows.append({"clip": f"/x/spk{s}/c{c}.mp4", "path": str(p), "n": T})
    (root / "index_000.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    clips = {f"spk{s}/c{c}": {"keep": True, "offset": 0.0, "shift_ticks": 0} for s in range(n_spk) for c in range(2)}
    (root / "sync_offsets.json").write_text(json.dumps({"clips": clips}))
    torch.save(fit_pca(torch.randn(500, 69, generator=g), K=69).pack(), root / "pca.pt")
    return rows


def train(cache: Path, out: Path, *extra: str) -> str:
    over = [f"cache_dir={cache}", f"out_dir={out}", f"sync_offsets={cache / 'sync_offsets.json'}", "dim=32", "layers=1", "heads=2", "max_steps=4", "warmup_steps=1",
            "eval_every=2", "log_every=1", "batch_size=2", "workers=0", "eval_batches=1", "sample_steps=2",
            "val_frac=0.3", "windows_per_clip=2", f"latent={cache / 'pca.pt'}", *extra]
    log = subprocess.run([sys.executable, str(REPO / "scripts" / "train_motion.py"), *over],
                         capture_output=True, encoding="utf-8", cwd=REPO)
    assert log.returncode == 0, log.stderr[-2000:]
    return log.stdout


def test_train_eval_leaves_the_run_bit_identical(tmp_path):
    cache = tmp_path / "cache"
    make_cache(cache)
    off = train(cache, tmp_path / "off", "train_eval=false")
    on = train(cache, tmp_path / "on", "train_eval=true")
    assert "EVAL_TRAIN" not in off
    assert [l.split()[2] for l in on.splitlines() if l.startswith("EVAL_TRAIN")] == ["2", "4"]
    keep = lambda s: [l for l in s.splitlines() if l.startswith(("step", "EVAL "))]
    strip = lambda l: l.split(" it/s")[0].rsplit(" ", 1)[0]
    assert [strip(l) for l in keep(off)] == [strip(l) for l in keep(on)]
    a = torch.load(tmp_path / "off" / "last.pt", weights_only=False)
    b = torch.load(tmp_path / "on" / "last.pt", weights_only=False)
    for part in ("model", "ema"):
        assert all(torch.equal(a[part][k], b[part][k]) for k in a[part])



def test_init_seed_changes_the_weights_not_the_split(tmp_path):
    cache = tmp_path / "cache"
    make_cache(cache)
    a = train(cache, tmp_path / "a", "train_eval=false")
    b = train(cache, tmp_path / "b", "train_eval=false", "init_seed=1")
    split = lambda s: next(l for l in s.splitlines() if "unseen val speakers" in l)
    assert split(a) == split(b) and "init seed 1 (split seed 0)" in b and "init seed" not in a
    wa = torch.load(tmp_path / "a" / "last.pt", weights_only=False)["model"]
    wb = torch.load(tmp_path / "b" / "last.pt", weights_only=False)["model"]
    assert any(not torch.equal(wa[k], wb[k]) for k in wa)

