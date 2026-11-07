"""The style memory (sang.model.memory, scripts/train_style.py) and its neighbours as footage: in training windows,
the set fine-tune and the Talker. CPU only."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts"), str(Path(__file__).resolve().parent)]
from sang.data.dataset import MotionWindows
from sang.data.paths import load_index
from sang.model.flow import REF_DIM, Norm, generate
from sang.model.memory import N_SPK, N_WIN, Memory, supcon
from test_demo import M, _demo
from test_motion_model import A, B, L, P, _model, _unzero
from test_train_eval import make_cache, train


def test_supcon_is_lowest_when_the_keys_group_by_speaker():
    g = torch.Generator().manual_seed(0)
    lab = torch.arange(8).repeat_interleave(4)
    centres = torch.nn.functional.normalize(torch.randn(8, 16, generator=g), dim=-1)
    z = torch.nn.functional.normalize(centres[lab] + 0.1 * torch.randn(32, 16, generator=g), dim=-1)
    assert supcon(z, lab) < supcon(z, lab[torch.randperm(32, generator=g)]) - 1


def test_the_neighbour_tag_starts_at_zero_and_old_checkpoints_load():
    m = _model(demo_tokens=M).eval()
    old = {k: v for k, v in m.state_dict().items() if k != "demo.nb"}
    m.demo.nb.data.fill_(1.0)
    m.load_state_dict(old)
    assert not m.demo.nb.any()
    d = _demo()
    nb = torch.zeros_like(d["demo_mask"])
    nb[:, 5:] = True
    assert torch.allclose(m.encode_demo(d), m.encode_demo({**d, "demo_nb": nb}))
    m.demo.nb.data.normal_()
    assert not torch.allclose(m.encode_demo(d), m.encode_demo({**d, "demo_nb": nb}))


def test_generate_reads_one_set_of_footage_tokens_per_window():
    torch.manual_seed(0)
    m = _model(demo_tokens=M).eval()
    _unzero(m)
    tok, other = m.encode_demo(_demo()), m.encode_demo(_demo())
    n, audio, ref = 2 * L + 3, torch.randn(B, 2 * (2 * L + 3), A), torch.randn(B, REF_DIM)
    run = lambda demo: generate(m, audio, ref, n, window=L, n_prefix=P, steps=2,
                                generator=torch.Generator().manual_seed(1), demo=demo)
    seen = []
    assert torch.allclose(run(tok), run(lambda pos: seen.append(pos) or tok))
    assert seen == [0, L, 2 * L]
    swapped = run(lambda pos: other if pos else tok)
    assert torch.allclose(swapped[:, :L], run(tok)[:, :L]) and not torch.allclose(swapped, run(tok))


@pytest.fixture(scope="module")
def style_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("style")
    cache = root / "cache"
    make_cache(cache)
    train(cache, root / "gen", "train_eval=false", "demo_tokens=4", "demo_frames=30", "demo_min_frames=10")
    gen = root / "gen" / "last.pt"
    log = subprocess.run([sys.executable, str(REPO / "scripts" / "train_style.py"), f"gen={gen}",
                          f"out_dir={root / 'style'}", "dim=16", "out=8", "heads=2", "persons=2", "K=2", "max_steps=2",
                          "warmup_steps=1", "workers=0", "log_every=1", "eval_every=2", "audio_dims=4"],
                         capture_output=True, encoding="utf-8", cwd=REPO)
    assert log.returncode == 0, log.stderr[-2000:]
    return cache, gen, root / "style" / "memory.pt", log.stdout


def test_the_style_key_trains_and_indexes_every_training_window(style_run):
    cache, _, path, out = style_run
    assert "EVAL step 2 " in out and "top1_half" in out and "memory -> " in out
    mem = Memory(str(path))
    key, _ = mem.encoders("cpu")
    assert key.enc.audio is None and key.cfg["t_in"] == 2 * 42  # no voice: [residual, predicted mean] per frame
    rows = {r["path"]: r for r in load_index(cache)}
    assert set(mem.paths) <= set(mem.query) == set(rows)  # training clips are retrievable, every clip is a query
    assert torch.allclose(mem.spk_z.norm(dim=-1), torch.ones(len(mem.spk_z)), atol=1e-3)
    assert all(int(mem.win_start[w]) + mem.L <= rows[mem.paths[int(mem.win_clip[w])]]["n"] for w in range(len(mem.win_key)))
    assert sum(map(len, mem.spk_windows)) == len(mem.win_key)
    z = mem.query_z[mem.query[mem.paths[0]]]
    own = int(mem.clip_spk[0])
    spk = mem.neighbours(z, N_SPK, exclude=own)
    assert own not in spk.tolist() and len(spk) == len(mem.spk_z) - 1
    a = torch.randn(2 * mem.L, 1024)
    ws = mem.windows(spk, mem.akey(a), 2)
    cand = torch.cat([mem.spk_windows[s] for s in spk.tolist()])
    assert set(ws.tolist()) <= set(cand.tolist())
    assert float((mem.win_key[ws] @ mem.akey(a)).min()) >= float((mem.win_key[cand] @ mem.akey(a)).sort().values[-2])
    yw, aw = mem.fetch(int(ws[0]))
    assert yw.shape == (mem.L, 69) and aw.shape == (2 * mem.L, 1024) and torch.allclose(yw.mean(0), torch.zeros(69), atol=1e-5)


def test_training_windows_carry_other_speakers_audio_matched_windows(style_run):
    cache, _, path, _ = style_run
    mem = Memory(str(path))
    norm = Norm(torch.zeros(REF_DIM - 63), torch.ones(REF_DIM - 63), torch.zeros(63), torch.ones(63))
    ds = MotionWindows(load_index(cache), 64, 10, norm, per_clip=2, fixed=True, demo=30, demo_min=10, memory=mem,
                       p_own=0.0, p_nb=0.0)
    mem.bind(ds)
    for i in range(len(ds.rows)):
        x, (p,) = ds[i], ds.peers[i]
        spk = mem.neighbours(mem.query_z[ds.query[p]], N_SPK, exclude=ds.own_spk[i])
        ws = mem.windows(spk, mem.akey(x["audio"][2 * 10:]), N_WIN)
        n = int(x["demo_mask"][:30].sum())
        centre = x["demo_motion"][:n].mean(0)
        for k, w in enumerate(ws.tolist()):
            yw, aw = mem.fetch(w)
            o = 30 + k * mem.L
            assert torch.allclose(x["demo_motion"][o:o + mem.L], yw + centre, atol=1e-5)
            assert torch.equal(x["demo_audio"][2 * o:2 * (o + mem.L)], aw)
        assert x["demo_nb"].sum() == len(ws) * mem.L and not x["demo_nb"][:30].any()
        assert torch.equal(x["demo_mask"], x["demo_nb"] | (torch.arange(len(x["demo_nb"])) < n))


def test_the_set_fine_tune_reads_the_memory_and_trains_the_tag(style_run, tmp_path, monkeypatch, capsys):
    import train_set
    _, gen, path, _ = style_run
    monkeypatch.setattr(sys, "argv", ["train_set.py", f"gen={gen}", f"out_dir={tmp_path}", f"memory={path}", "rounds=1",
                                      "alternations=1", "policy_steps=2", "persons=2", "m=2", "workers=0",
                                      "eval_batches=1", "log_every=1", "steps=2"])
    train_set.main()
    assert "memory " in capsys.readouterr().out
    ck = torch.load(tmp_path / "round1.pt", weights_only=False)
    assert ck["ema"]["demo.nb"].abs().sum() > 0


def test_the_talker_adds_the_neighbours_by_mode(style_run, monkeypatch):
    pytest.importorskip("cv2")
    from sang.model.target import M_DIM
    from sang.render.pipeline import Talker
    from test_pipeline import SRC, WAV, FakeWavLM, fake_codec
    _, gen, path, _ = style_run
    codec, _ = fake_codec(monkeypatch)
    base = torch.zeros(1, M_DIM)
    base[:, 0] = 1.2
    codec.extract = lambda fr: {"m": base + 0.05 * torch.randn(len(fr), M_DIM, generator=torch.Generator().manual_seed(3)),
                                "span": (0, len(fr))}
    t = Talker(str(gen), "g=2,steps=2", device="cpu", codec=codec, wavlm=FakeWavLM(), memory=str(path))
    for blk in t.model.blocks:  # 4 training steps leave the footage's cross-attention at about zero
        torch.nn.init.normal_(blk.xattn.out_proj.weight, std=0.2)
    demo = (np.zeros((80, 100, 120, 3), np.uint8), torch.randn(80 * 640) * 0.1)
    out = {}
    for mode in ("none", "style", "style_noaudio", "random"):
        t.mem_mode = mode
        _, out[mode], timer = t.run(SRC, WAV, 75, demo=demo, render=False)
        assert torch.isfinite(out[mode]).all() and ("memory" in timer.t) == (mode != "none")
        assert ("mem_sim" in t.mem_info) == (mode != "none")
    assert not torch.allclose(out["none"], out["style"])
