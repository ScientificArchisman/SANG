"""Two-wave training: chains of windows as inference generates them, the active chain recap and the wave schedule.
CPU only."""
import copy
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "train"))
from sang.data.dataset import chained
from sang.model.flow import REF_DIM, generate
from sang.model.scoring import gen_fakes, roll_chain, self_prefix, set_loss, window_of
from test_motion_model import _model, _unzero
from test_train_eval import make_cache, train

B, L, P, K, A = 2, 8, 4, 3, 32


def _chain():
    g = torch.Generator().manual_seed(1)
    return {"prefix": torch.randn(B, P, 12, generator=g), "target": torch.randn(B, K * L, 12, generator=g),
            "audio": torch.randn(B, 2 * (P + K * L), A, generator=g), "ref": torch.randn(B, REF_DIM, generator=g)}


def test_roll_chain_is_exactly_how_generate_chains_windows():
    m = _model().eval()
    _unzero(m)
    b = _chain()
    want = generate(m, b["audio"], b["ref"], K * L, window=L, n_prefix=P, steps=2, start=b["prefix"],
                    generator=torch.Generator().manual_seed(7))
    got = roll_chain(m, b, K, L, steps=2, generator=torch.Generator().manual_seed(7))
    assert torch.allclose(torch.cat(got, 1), want, atol=1e-5)
    w = self_prefix(m, b, 2, L, steps=2, generator=torch.Generator().manual_seed(7))
    assert torch.allclose(w["prefix"], want[:, 2 * L - P:2 * L], atol=1e-5)
    assert torch.equal(w["target"], b["target"][:, 2 * L:3 * L]) and w["audio"].shape[1] == 2 * (P + L)
    one = {**b, "target": b["target"][:, :L], "audio": b["audio"][:, :2 * (P + L)]}
    assert all(torch.equal(window_of(one, 0, L)[k], one[k]) for k in one)
    assert torch.allclose(gen_fakes(m, window_of(b, 0, L), 2, 2.0, torch.Generator().manual_seed(7)), want[:, :L])


def test_the_chain_recap_trains_and_one_window_chains_are_the_old_loss():
    torch.manual_seed(0)
    policy = _model().eval()
    _unzero(policy)
    ref = copy.deepcopy(policy).requires_grad_(False)
    b = _chain()
    loss, parts = set_loss(policy, ref, b, m=2, steps=2, chain=L)
    loss.backward()
    assert torch.isfinite(loss) and sum(p.grad.abs().sum() for p in policy.parameters() if p.grad is not None) > 0
    one = {**b, "target": b["target"][:, :L], "audio": b["audio"][:, :2 * (P + L)]}
    torch.manual_seed(3)
    a, _ = set_loss(policy, ref, one, m=2, steps=2, chain=L)
    torch.manual_seed(3)
    c, _ = set_loss(policy, ref, one, m=2, steps=2)
    assert torch.allclose(a, c)


def test_chained_keeps_long_clips_and_remaps_footage_peers():
    rows = [{"n": n, "path": f"/v{v}/c{i}.pt"} for i, (n, v) in enumerate([(30, 0), (10, 0), (40, 0), (40, 1)])]
    ds = SimpleNamespace(rows=rows, mem=["a", "b", "c", "d"], peers=[[1, 2], [0, 2], [0, 1], []], L=8, P=4)
    c = chained(ds, 3)
    assert c.L == 24 and [r["n"] for r in c.rows] == [30, 40, 40] and c.mem == ["a", "c", "d"]
    assert c.peers == [[1], [0], []] and ds.L == 8 and len(ds.rows) == 4


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / "train" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_wave_schedule():
    tm = _load("train_motion")
    cfg = {"warmup_steps": 10, "max_steps": 110, "lr": 1.0, "min_lr": 0.1, "lam_vel": 1.0, "p_audio": 0.1,
           "p_ref": 0.1, "p_prefix": 0.5, "t_dist": "logit_normal", "t_mean": 0.8}
    old = [tm.lr_at(s, cfg) for s in range(110)]
    assert old[10] == 1.0 and abs(old[60] - 0.55) < 1e-9 and old[109] < 0.11
    w = {**cfg, "wave2": 60, "t_mean2": 0.3, "lam_vel2": None, "self_from": 50, "p_self": [0.1, 0.5], "self_depth": 2}
    assert tm.lr_at(59, w) == 1.0 and tm.lr_at(60, w) == 1.0 and abs(tm.lr_at(85, w) - 0.55) < 1e-9
    assert tm.objective(w, 59)["t_mean"] == 0.8 and tm.objective(w, 60)["t_mean"] == 0.3 == tm.objective(w)["t_mean"]
    assert tm.objective(w)["lam_vel"] == 1.0 and tm.objective(cfg)["t_mean"] == 0.8
    assert tm.self_share(49, w) == (0.0, 0) and tm.self_share(50, w) == (0.1, 1) and tm.self_share(109, w) == (0.5, 2)
    assert tm.self_share(100, cfg) == (0.0, 0)


def test_two_wave_base_and_chain_recap_run(tmp_path, monkeypatch, capsys):
    cache = tmp_path / "cache"
    make_cache(cache, T=160)
    out = train(cache, tmp_path / "gen", "train_eval=false", "demo_tokens=4", "demo_frames=30", "demo_min_frames=10",
                "wave2=2", "t_mean2=0.3", "lam_vel2=0.5", "self_from=1", "self_depth=1", "p_self=[0.5,0.5]")
    assert "chains of 2 windows" in out and " self " in out and "wave 2 at step 2" in out
    import train_set
    monkeypatch.setattr(sys, "argv", ["train_set.py", f"gen={tmp_path / 'gen' / 'last.pt'}",
                                      f"out_dir={tmp_path / 'set'}", "rounds=1", "alternations=1", "policy_steps=2",
                                      "persons=2", "m=2", "workers=0", "eval_batches=1", "log_every=1", "steps=2",
                                      "chain_k=2", "replay=1"])
    train_set.main()
    log = capsys.readouterr().out
    assert "chain_w1" in log and "gap_w1" in log and "chain_rot_r" in log and " replay " in log
    assert (tmp_path / "set" / "round1.pt").exists()
