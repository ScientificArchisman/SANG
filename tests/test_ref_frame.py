"""ref_frame=median training: the reference is one of the clip's typical-opening frames (sang.eval.naturalness.
typical_frames, scripts/train_motion.MotionWindows). CPU only."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from sang.eval.naturalness import Readout, median_frame, typical_frames
from sang.model.flow import Norm
from sang.model.latent import fit_pca
from sang.model.target import REGIONS, T_DIM, to_target


def test_typical_frames_are_the_middle_band_by_rank():
    lip = np.arange(101, dtype=float)[::-1]
    idx = typical_frames(lip, band=0.2)
    assert sorted(lip[idx]) == list(range(40, 61))
    assert len(typical_frames(lip, band=0.0)) == 1 and lip[typical_frames(lip, 0.0)][0] == 50
    assert lip[median_frame(lip)] == 50
    assert len(typical_frames(np.ones(7), 0.2)) >= 1


def make_cache(root: Path, n_spk: int = 6, T: int = 90) -> list[dict]:
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
    w = torch.zeros(T_DIM)
    w[REGIONS["mouth"][1]] = 1.0
    we = torch.zeros(T_DIM)
    we[REGIONS["eyes"][1]] = 1.0
    (root / "openness.json").write_text(json.dumps({"lip": {"w": w.tolist(), "b": 0.0}, "eye": {"w": we.tolist(), "b": 0.0}}))
    torch.save(fit_pca(torch.randn(500, 69, generator=g), K=69).pack(), root / "pca.pt")
    return rows


def test_dataset_draws_the_reference_from_typical_frames(tmp_path):
    import sang.data.dataset as tm
    rows = make_cache(tmp_path)
    ms = [torch.load(r["path"], weights_only=True)["m"].float() for r in rows]
    ys = [to_target(m) for m in ms]
    kps = [torch.load(r["path"], weights_only=True)["kp"].float() for r in rows]
    norm = Norm.fit(torch.cat(ys), torch.cat(kps))
    lip = Readout(json.loads((tmp_path / "openness.json").read_text())["lip"]["w"], 0.0)
    ds = tm.MotionWindows(rows, 32, 4, norm, per_clip=20, preload=True, threads=2, ref_frame="median", lip=lip)
    for i in range(len(ds)):
        j = i % len(rows)
        item = ds[i]
        all_refs = norm.ref(kps[j], ms[j])
        k = int((all_refs - item["ref"]).abs().sum(1).argmin())
        assert k in set(typical_frames(lip(ys[j]).numpy(), 0.2).tolist())
    with pytest.raises(ValueError):
        tm.MotionWindows(rows, 32, 4, norm, per_clip=1, preload=False, ref_frame="median")


def test_training_with_median_reference_end_to_end(tmp_path):
    cache = tmp_path / "cache"
    make_cache(cache)
    out = tmp_path / "run"
    over = [f"cache_dir={cache}", f"out_dir={out}", "sync_offsets=", "dim=32", "layers=1", "heads=2", "max_steps=2", "warmup_steps=1",
            "eval_every=2", "log_every=1", "batch_size=2", "workers=0", "eval_batches=1", "sample_steps=2",
            "val_frac=0.3", "windows_per_clip=2", "ref_frame=median", f"latent={cache / 'pca.pt'}"]
    log = subprocess.run([sys.executable, str(REPO / "scripts" / "train_motion.py"), *over],
                         capture_output=True, encoding="utf-8", cwd=REPO)
    assert log.returncode == 0, log.stderr[-2000:]
    assert "typical one per clip" in log.stdout
    assert torch.load(out / "last.pt", weights_only=False)["cfg"]["ref_frame"] == "median"

