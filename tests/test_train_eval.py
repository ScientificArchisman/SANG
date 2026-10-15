"""train_eval: the val score on fixed training clips (EVAL_TRAIN lines) must not change the run, and
scripts/plot_curves.py must read it. CPU only."""
import importlib.util
import subprocess
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))

from test_ref_frame import make_cache


def train(cache: Path, out: Path, *extra: str) -> str:
    over = [f"cache_dir={cache}", f"out_dir={out}", "dim=32", "layers=1", "heads=2", "max_steps=4", "warmup_steps=1",
            "eval_every=2", "log_every=1", "batch_size=2", "workers=0", "eval_batches=1", "sample_steps=2",
            "val_frac=0.3", "windows_per_clip=2", *extra]
    log = subprocess.run([sys.executable, str(REPO / "scripts" / "train_motion.py"), *over],
                         capture_output=True, encoding="utf-8", cwd=REPO)
    assert log.returncode == 0, log.stderr[-2000:]
    return log.stdout


def test_train_eval_leaves_the_run_bit_identical_and_plots(tmp_path):
    cache = tmp_path / "cache"
    make_cache(cache)
    off = train(cache, tmp_path / "off", "train_eval=false")
    on = train(cache, tmp_path / "on", "train_eval=true")
    assert "EVAL_TRAIN" not in off
    assert [l.split()[2] for l in on.splitlines() if l.startswith("EVAL_TRAIN")] == ["2", "4"]
    keep = lambda s: [l for l in s.splitlines() if l.startswith(("step", "EVAL "))]
    strip = lambda l: l.split(" it/s")[0].rsplit(" ", 1)[0]          # drop the timing fields
    assert [strip(l) for l in keep(off)] == [strip(l) for l in keep(on)]   # same losses, same val
    a = torch.load(tmp_path / "off" / "last.pt", weights_only=False)
    b = torch.load(tmp_path / "on" / "last.pt", weights_only=False)
    for part in ("model", "ema"):                                    # an eval mid-run, then two more steps
        assert all(torch.equal(a[part][k], b[part][k]) for k in a[part])

    spec = importlib.util.spec_from_file_location("plot_curves", REPO / "scripts" / "plot_curves.py")
    pc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pc)
    r = pc.parse(on)
    assert [s for s, _ in r["val"]] == [s for s, _ in r["train"]] == [2, 4]
    assert len(r["batches"]) == 4 and all("loss" in d for _, d in r["train"])
    log = tmp_path / "on.log"
    log.write_text(on)
    res = subprocess.run([sys.executable, str(REPO / "scripts" / "plot_curves.py"), str(log), "--name", "t",
                          "--out", str(tmp_path / "curves")], capture_output=True, encoding="utf-8", cwd=REPO)
    assert res.returncode == 0, res.stderr[-2000:]
    assert (tmp_path / "curves" / "t.png").stat().st_size > 0 and (tmp_path / "curves" / "t.pdf").exists()
