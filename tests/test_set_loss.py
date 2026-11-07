"""Proper multi-sample fine-tuning (sang.model.scoring.energy / mbes / set_loss, PersonSets, scripts/train_set.py).
CPU only."""
import copy
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts"), str(Path(__file__).resolve().parent)]
from sang.data.dataset import MotionWindows, PersonSets
from sang.data.paths import load_index
from sang.model.flow import REF_DIM, Norm
from sang.model.scoring import block_scale, energy, haar, mbes, set_loss
from test_demo import M, _demo
from test_motion_model import _batch, _model, _unzero
from test_train_eval import make_cache, train


def test_the_energy_score_is_lowest_at_the_real_spread_and_pushes_a_narrow_sampler_out():
    g = torch.Generator().manual_seed(0)
    y, z = torch.randn(4000, 1, 20, generator=g), torch.randn(4000, 8, 20, generator=g)
    es = {s: float(energy(s * z, y).mean()) for s in (0.62, 1.0, 1.3)}
    assert es[1.0] < es[0.62] and es[1.0] < es[1.3]
    s = torch.tensor(0.62, requires_grad=True)
    energy(s * z, y).mean().backward()
    assert s.grad < 0  # descending the score widens the sampler


def test_the_energy_distance_is_zero_for_one_law_and_sees_a_shrunk_coordinate():
    g = torch.Generator().manual_seed(1)
    x, y = torch.randn(2000, 8, 30, generator=g), torch.randn(2000, 8, 30, generator=g)
    same = energy(x, y)
    assert abs(float(same.mean())) < 3 * float(same.std()) / 2000 ** 0.5
    se = float(same.std()) / 2000 ** 0.5
    part = x.clone()
    part[..., :6] *= 0.62  # 6 of 30 directions: seen, but weakly (about 3 standard errors over 2000 sets)
    assert 2 * se < float(energy(part, y).mean()) < float(energy(0.62 * x, y).mean())
    assert float(energy(0.62 * x, y).mean()) > 20 * se


def test_the_block_score_is_lowest_at_the_real_spread_and_sees_a_two_channel_shrink():
    g = torch.Generator().manual_seed(3)
    x = torch.randn(5, 16, 3, generator=g)
    assert torch.allclose(sum(h.pow(2).sum() for h in haar(x)), x.pow(2).sum(), rtol=1e-5)  # orthonormal
    n, m, T, C = 1500, 8, 16, 12
    y, z = torch.randn(n, T, C, generator=g), torch.randn(n, m, T, C, generator=g)
    start, dec = torch.zeros(n, 1, C), lambda u: u
    blocks = (dec, *block_scale(y, start, dec))
    score = {s: float(mbes(s * z, y, start, *blocks).mean()) for s in (0.62, 1.0, 1.3)}
    assert score[1.0] < score[0.62] and score[1.0] < score[1.3]
    narrow = z.clone()
    narrow[..., :2] *= 0.62  # 2 of 12 channels: the whole-window score dilutes it, the blocks do not
    zs = lambda dlt: float(dlt.mean() / (dlt.std() / n ** 0.5))
    whole = lambda u: energy(u.flatten(2), y.flatten(1)[:, None]) / 2
    z_es, z_mb = zs(whole(narrow) - whole(z)), zs(mbes(narrow, y, start, *blocks) - mbes(z, y, start, *blocks))
    assert z_mb > 3 and z_mb > 2 * abs(z_es)


def test_person_sets_hold_the_real_windows_out_of_the_footage(tmp_path):
    make_cache(tmp_path, n_spk=4)
    norm = Norm(torch.zeros(REF_DIM - 63), torch.ones(REF_DIM - 63), torch.zeros(63), torch.ones(63))
    ds = MotionWindows(load_index(tmp_path), 16, 4, norm, per_clip=3, fixed=True, demo=30, demo_min=10)
    ds.mem = [(y, r, torch.full_like(a, j)) for j, (y, r, a) in enumerate(ds.mem)]  # audio = the clip's index
    video = lambda j: Path(ds.rows[j]["path"]).parent
    sets = PersonSets(ds, 5, fixed=True)
    for i in range(len(sets)):
        x = sets[i]
        clips, foot = x["audio"][:, 0, 0].long(), x["demo_audio"][:, 0, 0].long()
        assert (foot == foot[0]).all() and not (clips == foot[0]).any()
        assert {video(int(c)) for c in clips} == {video(int(foot[0]))}
        n = int(x["demo_mask"][0].sum())
        assert (x["demo_mask"] == x["demo_mask"][0]).all() and x["demo_keep"].all()
        off = x["demo_motion"][:, :n] - x["demo_motion"][:1, :n]  # one stretch, each window's own photo frame
        assert torch.allclose(off, off[:, :1].expand_as(off), atol=1e-5)
    assert torch.equal(sets[3]["target"], sets[3]["target"])  # fixed: the val sets do not change


def test_both_scores_train_the_policy():
    torch.manual_seed(0)
    policy = _model(demo_tokens=M).eval()
    _unzero(policy)
    ref = copy.deepcopy(policy).requires_grad_(False)
    b = {**_batch(), **_demo()}
    loss, parts = set_loss(policy, ref, b, m=2, steps=2)
    loss.backward()
    grads = [p.grad for p in policy.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(gr).all() for gr in grads) and sum(gr.abs().sum() for gr in grads) > 0
    assert parts.keys() == {"path_kl", "ed"} and float(parts["path_kl"]) < 1e-8
    policy.zero_grad()
    blocks = (lambda u: u, *block_scale(b["target"], b["prefix"][:, -1:], lambda u: u))
    loss, parts = set_loss(policy, ref, b, m=2, steps=2, blocks=blocks)
    loss.backward()
    assert "mbes" in parts and sum(p.grad.abs().sum() for p in policy.parameters() if p.grad is not None) > 0


def test_train_set_runs_rounds_reports_the_gates_and_stops(tmp_path, monkeypatch, capsys):
    import train_set
    cache = tmp_path / "cache"
    make_cache(cache)
    train(cache, tmp_path / "gen", "train_eval=false", "demo_tokens=4", "demo_frames=30", "demo_min_frames=10")
    calls = iter([0.80, 0.85, 0.70, 0.80, 0.85])  # round 0, round 1 closer, round 2 away -> stop; then the mbes run
    def motion_eval(*a, **kw):
        r = next(calls)
        return {f"{s}_r_{q}": r for s in ("std", "vc") for q in ("mouth", "rot", "place")}
    monkeypatch.setattr(train_set, "motion_eval", motion_eval)
    monkeypatch.setattr(sys, "argv", ["train_set.py", f"gen={tmp_path / 'gen' / 'last.pt'}", f"out_dir={tmp_path / 'set'}",
                                      "rounds=3", "alternations=1", "policy_steps=2", "persons=2", "m=2", "score=es",
                                      "workers=0", "eval_batches=1", "log_every=1", "steps=2"])
    train_set.main()
    out = capsys.readouterr().out
    assert "STOP after round 2" in out and "es_s1.6" in out and " ed " in out
    assert "g1z_mbes_head0.62_slow" in out and "mbes_s1.6" in out
    assert sorted(p.name for p in (tmp_path / "set").iterdir()) == ["round1.pt", "round2.pt"]
    monkeypatch.setattr(sys, "argv", ["train_set.py", f"gen={tmp_path / 'gen' / 'last.pt'}", f"out_dir={tmp_path / 'mb'}",
                                      "rounds=1", "alternations=1", "policy_steps=2", "persons=2", "m=2",
                                      "workers=0", "eval_batches=1", "log_every=1", "steps=2"])
    train_set.main()
    out = capsys.readouterr().out
    assert "score mbes" in out and "mbes " in out and (tmp_path / "mb" / "round1.pt").exists()
