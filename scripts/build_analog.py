#!/usr/bin/env python3
"""Build the analog calibration (sang.model.analog), then gate it on unseen speakers.

    sbatch --mem=128gb bash_scripts/run.sh scripts/build_analog.py                    # runs/style/analog.pt

configs/build_analog.yaml on top of the memory's generator config. For every TalkVid training clip in the memory it
sums its speaker's sufficient statistics of the deployed sampler's residual. U is then the residual's top principal
directions, and W_pop the population fit. A window never reads its own frames as footage (training never shows it
them): a clip's windows read the tokens of the speaker's next clip (its first footage_s), relative to the clip's middle
frame, as the person's footage halves read each other's at test time (sang.model.analog.cross_fit).

GATE, on the unseen val speakers with three or more clips. The footage is the first two clips (own: each one's windows
with the other's tokens); the test is up to test_clips others, whose windows read both footage clips' tokens. Each
mode's correction is scored on the test windows, as its share of the oracle's gain (the correction fitted on the test
windows themselves):
    share = sum (loss_none - loss_mode) / sum (loss_none - loss_oracle)
The rank, lam and alpha are chosen on the even speakers and reported on the odd ones. win_<a>_<b> is the share of odd
speakers where mode a's loss is below b's. The neighbours carry the person only if own_matched beats own_random and
matched beats random.
"""
import argparse
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from sang.data.dataset import load_cfg
from sang.model.analog import KEYS, MODES, correction, fit, loss, mix, moments, project, residuals, total
from sang.model.flow import build, parse_spec, sampler_kwargs
from sang.model.memory import N_SPK, Memory, windows
from train_motion import setup_data

FPS = 25  # motion frames per second (WavLM's 50 Hz ticks / 2)


def clip_batch(d, ds, j, gen, dev, cfg, foot):
    """Clip j's windows (at most max_windows, spread over the clip) relative to its middle frame, and the footage
    tokens of the clips `foot` (their frames joined, at most footage_s) -> (batch, tokens [n, M, dim]) or None if the
    clip is too short."""
    y, refs, audio = ds.clip(j)
    k = len(y) // 2
    b = windows(d.latent.encode((y - y[k]).to(dev), d.k), audio.to(dev).float(), refs[k].to(dev), d.P, d.L,
                int(cfg["stride"]))
    if b is None:
        return None
    n = len(b["target"])
    if n > cfg["max_windows"]:
        idx = torch.linspace(0, n - 1, int(cfg["max_windows"])).round().long()
        b = {key: v[idx] for key, v in b.items()}
    fy = torch.cat([ds.clip(f)[0] for f in foot])
    fa = torch.cat([ds.clip(f)[2] for f in foot])
    n = min(len(fy), int(cfg["footage_s"] * FPS))
    fc = d.latent.encode((fy[:n] - y[k]).to(dev), d.k)
    tok = gen.demo(fc[None], fa[:2 * n].to(dev).float()[None], torch.ones(1, n, dtype=torch.bool, device=dev))
    return b, tok.expand(len(b["target"]), -1, -1)


def stats_of(gen, items, steps, g, draws, rng):
    """Summed full-space statistics of a list of clip_batch results."""
    b = {k: torch.cat([it[0][k] for it in items]) for k in items[0][0]}
    v, r = residuals(gen, b, torch.cat([it[1] for it in items]), steps, g, draws, rng)
    return total(moments(v, r))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs/build_analog.yaml"))
    ap.add_argument("overrides", nargs="*", help="key=value, parsed as YAML")
    args = ap.parse_args()
    rcfg = load_cfg(args.config, args.overrides)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mem = Memory(rcfg["memory"])
    ck = torch.load(mem.m["gen"], map_location="cpu", weights_only=False)
    cfg = {**ck["cfg"], **rcfg}
    torch.manual_seed(cfg["seed"])
    sk = sampler_kwargs(parse_spec(cfg["sampler"]), ck["cfg"])
    steps, g, draws = sk["steps"], sk["cfg_audio"], int(cfg["draws"])
    d = setup_data({**cfg, "demo_tokens": 0}, dev)
    gen = build(ck["cfg"]).to(dev).eval().requires_grad_(False)
    gen.load_state_dict(ck["ema"])
    if gen.demo is None:
        raise SystemExit(f"{mem.m['gen']} has no footage encoder: the calibration reads the footage")
    rng = torch.Generator(device=dev).manual_seed(int(cfg["seed"]))
    print(f"analog statistics of {mem.m['gen']} (step {ck['step']}), g={g:g}, steps={steps}, {draws} draws a window, "
          f"memory {cfg['memory']}: {len(mem.spk_names)} speakers, {len(mem.paths)} clips", flush=True)

    # every training speaker's full-space statistics
    row = {r["path"]: j for j, r in enumerate(d.train_ds.rows)}
    missing = [p for p in mem.paths if p not in row]
    if missing:
        raise SystemExit(f"{len(missing)} memory clips are not in the training split, e.g. {missing[0]}: rebuild the "
                         "memory from the same cache and split")
    S = len(mem.spk_names)
    clips = [[] for _ in range(S)]
    for c, s_ in enumerate(mem.clip_spk.tolist()):
        clips[s_].append(c)
    nxt = {c: cs[(i + 1) % len(cs)] for cs in clips if len(cs) > 1 for i, c in enumerate(cs)}
    spk, pending, ids, t0 = None, [], [], time.time()
    for c, p in enumerate(mem.paths):
        it = clip_batch(d, d.train_ds, row[p], gen, dev, cfg, [row[mem.paths[nxt[c]]]]) if c in nxt else None
        if it is not None:
            pending.append(it)
            ids.append(torch.full((len(it[0]["target"]),), int(mem.clip_spk[c]), device=dev))
        if pending and (sum(len(i) for i in ids) >= cfg["batch"] or c == len(mem.paths) - 1):
            b = {k: torch.cat([i[0][k] for i in pending]) for k in pending[0][0]}
            v, r = residuals(gen, b, torch.cat([i[1] for i in pending]), steps, g, draws, rng)
            m = moments(v, r)
            idx = torch.cat(ids).repeat_interleave(draws)
            if spk is None:
                spk = {k: torch.zeros((S, *m[k].shape[1:]), dtype=torch.float64, device=dev) for k in KEYS}
            for k in KEYS:
                spk[k].index_add_(0, idx, m[k])
            pending, ids = [], []
        if (c + 1) % 2000 == 0:
            print(f"  {c + 1}/{len(mem.paths)} clips, {time.time() - t0:.0f} s", flush=True)
    pop = total(spk)
    evals, evecs = torch.linalg.eigh(pop["yy"].sum(0) / pop["n"])
    evecs, evals = evecs.flip(-1), evals.flip(-1)
    print(f"statistics of {int((spk['n'] > 0).sum())}/{S} speakers (one-clip speakers have none) in "
          f"{time.time() - t0:.0f} s; residual variance in the top 2/4/6/12 directions: "
          + " / ".join(f"{float(evals[:r].sum() / evals.sum()):.2f}" for r in (2, 4, 6, 12)), flush=True)

    # the gate: footage = a val speaker's first two clips, test = its next clips
    by = {}
    for j, r in enumerate(d.val_ds.rows):
        by.setdefault(Path(r["path"]).parent.name, []).append(j)
    persons = []
    for name in sorted(by):
        js = sorted(by[name], key=lambda j: d.val_ds.rows[j]["path"])
        if len(js) < 3:
            continue
        a, c, tests = js[0], js[1], js[2:2 + int(cfg["test_clips"])]
        qs = [mem.query.get(d.val_ds.rows[j]["path"]) for j in (a, c)]
        own = [it for it in (clip_batch(d, d.val_ds, a, gen, dev, cfg, [c]),
                             clip_batch(d, d.val_ds, c, gen, dev, cfg, [a])) if it is not None]
        test = [it for it in (clip_batch(d, d.val_ds, j, gen, dev, cfg, [a, c]) for j in tests) if it is not None]
        if None in qs or not own or not test:
            continue
        z = F.normalize(mem.query_z[qs[0]] + mem.query_z[qs[1]], dim=0)
        persons.append({"own": stats_of(gen, own, steps, g, draws, rng),
                        "test": stats_of(gen, test, steps, g, draws, rng),
                        "matched": total(spk, mem.neighbours(z, N_SPK).to(dev)),
                        "random": total(spk, mem.neighbours(z, N_SPK, random.Random(f"{cfg['seed']}:{name}")).to(dev))})
    if len(persons) < 4:
        raise SystemExit(f"only {len(persons)} val speakers with footage and a test clip: no gate")

    def losses(st, W_pop, mode, alpha=0.0, lam=0.0):
        """Per person, the summed-over-steps test loss of a mode (st: the persons' projected statistics)."""
        out = []
        for pr in st:
            if mode == "none":
                D = torch.zeros_like(W_pop)
            elif mode == "oracle":
                D = correction(pr["test"], W_pop, 0.0)
            else:
                nb = pr["random"] if mode.endswith("random") else pr["matched"]
                D = correction(mix(pr["own"] if mode.startswith("own") else None,
                                   None if mode == "own" else nb, alpha), W_pop, lam)
            out.append(float(loss(pr["test"], W_pop, D).sum()))
        return torch.tensor(out)

    def share(l, l0, lo, part):
        return float((l0 - l)[part].sum() / (l0 - lo)[part].sum().clamp_min(1e-12))

    even = torch.arange(len(persons)) % 2 == 0
    best = {}
    for r in cfg["ranks"]:
        U = evecs[:, :r]
        W_pop = fit(project(pop, U))
        st = [{k: project(pr[k], U) for k in pr} for pr in persons]
        l0, lo = losses(st, W_pop, "none"), losses(st, W_pop, "oracle")
        for mode in MODES[1:]:
            alphas = cfg["alphas"] if mode.startswith("own_") else [1.0 if mode == "own" else 0.0]
            for alpha in alphas:
                for lam in cfg["lams"]:
                    l = losses(st, W_pop, mode, alpha, lam)
                    sh = share(l, l0, lo, even)
                    if sh > best.get((r, mode), (-1e9,))[0]:
                        best[(r, mode)] = (sh, alpha, lam, l, l0, lo)
        print(f"rank {r}: even share by mode " + "  ".join(f"{m} {best[(r, m)][0]:+.3f}" for m in MODES[1:]),
              flush=True)
    rank = max(cfg["ranks"], key=lambda r: max(best[(r, m)][0] for m in MODES[1:]))
    odd = ~even
    l0, lo = best[(rank, "own")][4:6]
    print(f"GATE rank {rank}, {int(odd.sum())} odd / {int(even.sum())} even val speakers (the oracle lowers the odd "
          f"test loss by {float((l0 - lo)[odd].mean()):.4g} of {float(l0[odd].mean()):.4g})", flush=True)
    arms = {"none": {"alpha": 1.0, "lam": 0.0}}
    for m in MODES[1:]:
        sh, alpha, lam, l, l0, lo = best[(rank, m)]
        arms[m] = {"alpha": alpha, "lam": lam}
        print(f"GATE {m:<11} alpha {alpha:.2f} lam {lam:g}: share even {sh:+.3f} odd {share(l, l0, lo, odd):+.3f}, "
              f"beats none in {float((l < l0)[odd].float().mean()):.2f} of odd speakers", flush=True)
    L = {m: best[(rank, m)][3] for m in MODES[1:]}
    for a, b in (("own_matched", "own_random"), ("matched", "random"), ("own_matched", "own")):
        print(f"GATE win_{a}_{b} {float((L[a] < L[b])[odd].float().mean()):.2f}", flush=True)

    U = evecs[:, :rank]
    out = Path(cfg["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"U": U.float().cpu(), "W_pop": fit(project(pop, U)).float().cpu(),
                "spk": {k: project(spk, U)[k].float().cpu() for k in KEYS}, "spk_names": mem.spk_names,
                "arms": arms, "g": g, "steps": steps, "stride": int(cfg["stride"]), "draws_own": int(cfg["draws_own"]),
                "memory": cfg["memory"], "gen": mem.m["gen"]}, out)
    print(f"analog -> {out}: rank {rank}, {S} speakers", flush=True)


if __name__ == "__main__":
    main()
