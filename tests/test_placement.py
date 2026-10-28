"""Head placement (`place: true`): a 45-d target = the 42 + the head's t_x, t_y and log scale relative to
the reference frame. Conversion, normalisation, model, sampler, dataset and an end-to-end CPU run."""
import subprocess
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

from sang.motion import EXPR_KP, M_DIM, PLACE, REGIONS, T_DIM, from_target, placement, regions, smooth_placement, to_target
from sang.motion_model import (REF_DIM, MotionFlowTransformer, Norm, flow_loss, generate, guidance_vector, parse_spec,
                               sampler_kwargs, t_dim_of)
from sang.naturalness import Readout
from test_ref_frame import make_cache


def motion(n: int = 12, seed: int = 0) -> torch.Tensor:
    """Random motion whose held parts (t_z, the 24 non-driven expression coordinates) are constant."""
    g = torch.Generator().manual_seed(seed)
    m = torch.randn(n, M_DIM, generator=g) * 0.05
    m[:, 0] = 1.0 + 0.2 * torch.rand(n, generator=g)
    m[:, 1:4] = 20 * torch.rand(n, 3, generator=g) - 10
    m[:, 6] = m[0, 6]
    held = [7 + 3 * k + a for k in range(21) if k not in EXPR_KP for a in range(3)]
    m[:, held] = m[0, held]
    return m


def test_round_trip_moves_the_head_relative_to_the_photo():
    m = motion()
    y = to_target(m, ref=m[:1])
    assert y.shape == (12, T_DIM + 3) and torch.allclose(y[0, PLACE], torch.zeros(3))
    assert torch.allclose(y[:, :T_DIM], to_target(m))
    assert torch.allclose(from_target(y, m[:1]), m, atol=1e-4)
    still = from_target(to_target(m), m[:1])
    assert torch.allclose(still[:, [0, 4, 5]], m[:1, [0, 4, 5]].expand(12, -1))
    assert torch.allclose(from_target(smooth_placement(y, amp=0.0), m[:1]), still, atol=1e-5)
    assert torch.allclose(placement(m)[:, 2], m[:, 0].log())


def test_smoothing_is_a_centred_mean_that_holds_the_ends():
    y = torch.zeros(5, T_DIM + 3)
    y[:, PLACE[0]] = torch.tensor([0.0, 0.0, 3.0, 0.0, 0.0])
    s = smooth_placement(y, k=3)
    assert torch.allclose(s[:, PLACE[0]], torch.tensor([0.0, 1.0, 1.0, 1.0, 0.0]))
    assert torch.equal(s[:, :T_DIM], y[:, :T_DIM])
    y42 = torch.randn(5, T_DIM)
    assert smooth_placement(y42, amp=0.0, k=3) is y42


def test_norm_readout_and_guidance_widen_to_45():
    y, kp = torch.randn(200, T_DIM + 3) * 2 + 1, torch.randn(200, 63)
    n = Norm.fit(y, kp)
    assert n.place and not Norm.fit(y[:, :T_DIM], kp).place
    assert torch.allclose(n.target(y[:, :T_DIM]), n.target(y)[:, :T_DIM])
    assert torch.allclose(n.untarget(n.target(y)), y, atol=1e-5)
    assert n.ref(kp[:2], y[:2]).shape == (2, REF_DIM)
    w = torch.zeros(T_DIM)
    w[REGIONS["mouth"]] = torch.randn(18)
    r = Readout(w, 0.3)
    assert torch.allclose(r(y), r(y[:, :T_DIM]))
    vec = guidance_vector(2.0, regions(T_DIM + 3), T_DIM + 3, mouth=1.25, place=1.0)
    assert vec.shape == (T_DIM + 3,) and torch.all(vec[PLACE] == 1.0) and torch.all(vec[REGIONS["mouth"]] == 1.25)
    assert set(regions(T_DIM + 3)) == {"rot", "brow", "eyes", "mouth", "place"} and "place" not in regions(T_DIM)


def test_model_loss_and_sampler_at_45():
    cfg = {"place": True, "cfg_audio": 2.0, "sample_steps": 2}
    td = t_dim_of(cfg)
    m = MotionFlowTransformer(dim=32, depth=1, heads=2, audio_dim=8, t_dim=td)
    B, P, L = 2, 3, 8
    b = {"target": torch.randn(B, L, td), "audio": torch.randn(B, 2 * (P + L), 8), "ref": torch.randn(B, REF_DIM),
         "prefix": torch.randn(B, P, td)}
    loss, parts = flow_loss(m, b)
    assert torch.isfinite(loss) and "fm_place" in parts
    kw = sampler_kwargs(parse_spec("g=2,mouth=1.25,tau=0.5,avg=2,steps=2,pamp=0.5,psmooth=3"), cfg)
    assert kw["cfg_audio"].shape == (td,) and kw["noise_scale"].shape == (td,)
    assert "pamp" not in kw and "psmooth" not in kw
    y = generate(m, torch.randn(1, 2 * (P + 20), 8), torch.randn(1, REF_DIM), 20, window=8, n_prefix=P,
                 start=torch.zeros(1, P, td), **kw)
    assert y.shape == (1, 20, td) and torch.isfinite(y).all()
    y = generate(m, torch.randn(1, 2 * 20, 8), torch.randn(1, REF_DIM), 20, window=8, n_prefix=P, steps=2)
    assert y.shape == (1, 20, td)


def test_dataset_appends_placement_relative_to_the_reference(tmp_path):
    import sang.data as tm
    rows = make_cache(tmp_path)
    base = tm.fit_norm(rows)
    norm = tm.fit_place_norm(rows, base)
    assert norm.place and torch.equal(norm.t_mean[:T_DIM], base.t_mean)
    ds = tm.MotionWindows(rows, 16, 4, norm, per_clip=3, preload=True, threads=2)
    ms = [torch.load(r["path"], weights_only=True)["m"].float() for r in rows]
    kps = [torch.load(r["path"], weights_only=True)["kp"].float() for r in rows]
    for i in range(len(ds)):
        item, j = ds[i], i % len(rows)
        assert item["target"].shape == (16, T_DIM + 3) and item["prefix"].shape == (4, T_DIM + 3)
        k = int((norm.ref(kps[j], to_target(ms[j])) - item["ref"]).abs().sum(1).argmin())
        raw = norm.untarget(torch.cat([item["prefix"], item["target"]]))
        s = int((norm.target(to_target(ms[j])) - torch.cat([item["prefix"], item["target"]])[:1, :T_DIM])
                .abs().sum(1).argmin())
        want = to_target(ms[j][s:s + 20], ref=ms[j][k:k + 1])
        assert torch.allclose(raw, want, atol=1e-3)


def test_training_with_placement_end_to_end(tmp_path):
    cache = tmp_path / "cache"
    make_cache(cache)
    over = [f"cache_dir={cache}", f"out_dir={tmp_path / 'run'}", "sync_offsets=", "dim=32", "layers=1", "heads=2", "max_steps=2",
            "warmup_steps=1", "eval_every=2", "log_every=1", "batch_size=2", "workers=0", "eval_batches=1",
            "sample_steps=2", "val_frac=0.3", "windows_per_clip=2", "place=true"]
    log = subprocess.run([sys.executable, str(REPO / "scripts" / "train_motion.py"), *over],
                         capture_output=True, encoding="utf-8", cwd=REPO)
    assert log.returncode == 0, log.stderr[-2000:]
    assert "head placement: 45-d target" in log.stdout and "fm_place" in log.stdout
    assert "std_r_place" in next(l for l in log.stdout.splitlines() if l.startswith("EVAL "))
    assert (cache / "norm45.pt").exists() and (cache / "norm42.pt").exists()
    ck = torch.load(tmp_path / "run" / "last.pt", weights_only=False)
    assert ck["norm"]["t_mean"].numel() == T_DIM + 3 and ck["model"]["out.weight"].shape[0] == T_DIM + 3

