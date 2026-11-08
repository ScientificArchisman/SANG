"""The style memory and the analog calibration (sang.model.memory, sang.model.analog, scripts/train_style.py,
scripts/build_analog.py) and the Talker's analog modes. CPU only."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts"), str(Path(__file__).resolve().parent)]
from sang.data.paths import load_index
from sang.model.analog import MODES, correction, cross_fit, fit, loss, mix, moments, project, total
from sang.model.flow import REF_DIM, sample
from sang.model.memory import Memory, supcon
from test_demo import M
from test_motion_model import A, B, L, P, _model, _unzero
from test_train_eval import make_cache, train


def test_supcon_is_lowest_when_the_keys_group_by_speaker():
    g = torch.Generator().manual_seed(0)
    lab = torch.arange(8).repeat_interleave(4)
    centres = torch.nn.functional.normalize(torch.randn(8, 16, generator=g), dim=-1)
    z = torch.nn.functional.normalize(centres[lab] + 0.1 * torch.randn(32, 16, generator=g), dim=-1)
    assert supcon(z, lab) < supcon(z, lab[torch.randperm(32, generator=g)]) - 1


def _stats(seed, W=None, n=4000, k=5):
    """Statistics of v, r [1 step, 1 entry, n frames, k] with r = [v, 1] W + noise."""
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(1, 1, n, k, generator=g, dtype=torch.float64)
    W = torch.randn(k + 1, k, generator=g, dtype=torch.float64) if W is None else W
    r = torch.cat([v, torch.ones_like(v[..., :1])], -1) @ W + 0.1 * torch.randn(1, 1, n, k, generator=g,
                                                                                dtype=torch.float64)
    return total(moments(v, r)), W, v, r


def test_the_correction_is_the_kl_regularised_least_squares_fit():
    S, W, v, r = _stats(0)
    assert torch.allclose(fit(S)[0], W, atol=0.02)
    W_pop = torch.randn(1, 6, 5, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    lam = 2.0
    D = correction(S, W_pop, lam)
    # brute force: min_D ||e - phi D||^2 + lam ||phi D||^2 with e = y - phi W_pop is a least squares on stacked rows
    phi = torch.cat([v, torch.ones_like(v[..., :1])], -1)[0, 0]
    e = r[0, 0] - phi @ W_pop[0]
    X = torch.cat([phi, lam ** 0.5 * phi])
    Y = torch.cat([e, torch.zeros_like(e)])
    assert torch.allclose(D[0], torch.linalg.lstsq(X, Y).solution, atol=1e-6)
    assert loss(S, W_pop, D) < loss(S, W_pop, torch.zeros_like(D))  # the correction lowers its own data's loss
    assert torch.allclose(loss(S, W_pop, correction(S, W_pop, 0.0)), loss(S, fit(S), torch.zeros_like(D)))


def test_projected_statistics_are_those_of_the_projected_frames():
    S, _, v, r = _stats(2)
    U = torch.linalg.qr(torch.randn(5, 2, generator=torch.Generator().manual_seed(3), dtype=torch.float64))[0]
    direct = total(moments(v @ U, r @ U))
    for k, x in project(S, U).items():
        assert torch.allclose(x, direct[k], atol=1e-8), k


def test_mix_weighs_normalised_moments():
    a, b = _stats(4)[0], _stats(5, n=400)[0]
    m = mix(a, b, 0.25)
    assert torch.allclose(m["pp"], 0.25 * a["pp"] / a["n"] + 0.75 * b["pp"] / b["n"])
    assert torch.allclose(mix(a, None, 0.25)["py"], a["py"] / a["n"]) and torch.allclose(mix(a, b, 0.0)["yy"], b["yy"] / b["n"])


def test_sample_steps_with_the_correction():
    torch.manual_seed(0)
    m = _model().eval()
    _unzero(m)
    audio, ref, pre = torch.randn(B, 2 * (P + L), A), torch.randn(B, REF_DIM), torch.randn(B, P, m.t_dim)
    run = lambda c: sample(m, audio, ref, L, prefix=pre, steps=3, generator=torch.Generator().manual_seed(1), correct=c)
    seen = []
    assert torch.allclose(run(None), run(lambda i, v: seen.append(i) or v))
    assert seen == [0, 1, 2]
    assert not torch.allclose(run(None), run(lambda i, v: v + 0.1))


def test_cross_fit_never_lets_a_window_read_its_own_frames():
    torch.manual_seed(0)
    m = _model(demo_tokens=M).eval()
    T = 2 * (P + L + 3)
    codes, ticks, ref = torch.randn(T, m.t_dim), torch.randn(2 * T, A), torch.randn(REF_DIM)
    b, tok = cross_fit(m, codes, ticks, ref, P, L, stride=2)
    h = T // 2
    first = len(range(0, h - P - L + 1, 2))
    enc = lambda c, a: m.demo(c[None], a[None], torch.ones(1, len(c), dtype=torch.bool))
    assert torch.equal(b["target"][0], codes[P:P + L]) and torch.equal(b["target"][first], codes[h + P:h + P + L])
    assert torch.allclose(tok[0], enc(codes[h:], ticks[2 * h:])[0]) and torch.allclose(tok[first], enc(codes[:h], ticks[:2 * h])[0])
    assert cross_fit(m, codes[:P + L + 3], ticks, ref, P, L) is None  # halves shorter than a window


@pytest.fixture(scope="module")
def analog_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("analog")
    cache = root / "cache"
    make_cache(cache, n_spk=12, clips=3)
    train(cache, root / "gen", "train_eval=false", "demo_tokens=4", "demo_frames=30", "demo_min_frames=10")
    gen = root / "gen" / "last.pt"
    run = lambda script, *kv: subprocess.run([sys.executable, str(REPO / "scripts" / script), *kv],
                                             capture_output=True, encoding="utf-8", cwd=REPO)
    log = run("train_style.py", f"gen={gen}", f"out_dir={root / 'style'}", "dim=16", "out=8", "heads=2", "persons=2",
              "K=2", "max_steps=2", "warmup_steps=1", "workers=0", "log_every=1", "eval_every=2")
    assert log.returncode == 0, log.stderr[-2000:]
    mem = root / "style" / "memory.pt"
    log = run("build_analog.py", f"memory={mem}", f"out={root / 'analog.pt'}", "sampler=g=2,steps=2", "batch=2",
              "ranks=[2, 3]", "lams=[0.0, 1.0]", "alphas=[0.5]", "draws_own=2")
    assert log.returncode == 0, log.stderr[-2000:]
    return cache, gen, mem, root / "analog.pt", log.stdout


def test_the_memory_indexes_every_training_speaker(analog_run):
    cache, _, path, _, _ = analog_run
    mem = Memory(str(path))
    key, _ = mem.encoders("cpu")
    assert key.enc.audio is None and key.cfg["t_in"] == 2 * 42  # no voice: [residual, predicted mean] per frame
    assert set(mem.paths) <= set(mem.query) == {r["path"] for r in load_index(cache)}
    assert torch.allclose(mem.spk_z.norm(dim=-1), torch.ones(len(mem.spk_z)), atol=1e-3)
    assert len(mem.neighbours(mem.query_z[0], 3)) == 3


def test_build_analog_sums_every_speaker_and_gates_each_mode(analog_run):
    _, gen, mem_path, path, out = analog_run
    a, mem = torch.load(path, weights_only=False), Memory(str(mem_path))
    r = a["U"].shape[1]
    assert r in (2, 3) and a["spk_names"] == mem.spk_names and a["gen"] == str(gen)
    assert torch.allclose(a["U"].T @ a["U"], torch.eye(r), atol=1e-5)
    assert a["spk"]["pp"].shape == (len(mem.spk_names), 2, r + 1, r + 1) and (a["spk"]["n"] > 0).all()
    assert torch.allclose(a["W_pop"], fit(total({k: a["spk"][k].double() for k in a["spk"]})).float(), atol=1e-4)
    assert set(a["arms"]) == set(MODES)
    for m in MODES[1:]:
        assert f"GATE {m} " in out
    assert "win_own_matched_own_random" in out and "win_matched_random" in out


def test_the_talker_calibrates_by_mode(analog_run, monkeypatch):
    pytest.importorskip("cv2")
    from sang.model.target import M_DIM
    from sang.render.pipeline import Talker
    from test_pipeline import SRC, WAV, FakeWavLM, fake_codec
    _, gen, _, path, _ = analog_run
    codec, _ = fake_codec(monkeypatch)
    base = torch.zeros(1, M_DIM)
    base[:, 0] = 1.2
    codec.extract = lambda fr: {"m": base + 0.05 * torch.randn(len(fr), M_DIM, generator=torch.Generator().manual_seed(3)),
                                "span": (0, len(fr))}
    t = Talker(str(gen), "g=2,steps=2", device="cpu", codec=codec, wavlm=FakeWavLM(), analog=str(path))
    t.analog.a["arms"] = {m: {"alpha": 0.5, "lam": 0.0} for m in MODES}
    demo = (np.zeros((160, 100, 120, 3), np.uint8), torch.randn(160 * 640) * 0.1)  # two halves of one window each
    out = {}
    for mode in MODES:
        t.analog_mode = mode
        _, out[mode], timer = t.run(SRC, WAV, 75, demo=demo, render=False)
        assert torch.isfinite(out[mode]).all() and ("analog" in timer.t) == (mode != "none")
        assert ("d_norm" in t.analog_info) == (mode != "none") and ("nb_sim" in t.analog_info) == (mode not in ("none", "own"))
    assert not torch.allclose(out["none"], out["own"]) and not torch.allclose(out["own"], out["own_matched"])
    with pytest.raises(ValueError, match="steps"):
        Talker(str(gen), "g=2,steps=3", device="cpu", codec=codec, wavlm=FakeWavLM(), analog=str(path)).run(
            SRC, WAV, 75, demo=demo, render=False)
