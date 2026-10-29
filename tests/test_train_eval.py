"""train_eval: the val score on fixed training clips (EVAL_TRAIN lines) must not change the run. CPU only."""
import subprocess
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))

from test_ref_frame import make_cache


def train(cache: Path, out: Path, *extra: str) -> str:
    over = [f"cache_dir={cache}", f"out_dir={out}", "sync_offsets=", "dim=32", "layers=1", "heads=2", "max_steps=4", "warmup_steps=1",
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

