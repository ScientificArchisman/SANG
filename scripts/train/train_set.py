#!/usr/bin/env python3
"""Proper multi-sample fine-tuning of SANG's deployed sampler.

    python scripts/train/train_set.py gen=runs/demo/last.pt out_dir=runs/set [key=value ...]

configs/train_set.yaml on top of the generator run's own config. A batch is `persons` training windows (each reads
footage from another clip of that person, sang.data.dataset.PersonSets), each rolled out m times through the 6-step
sampler with gradients. The loss (sang.model.scoring.set_loss) scores the m samples against the real window, plus
kl x the path KL to the round's start:
  score=mbes     the multiscale-block energy score (sang.model.scoring.mbes), the default
  score=es       the plain energy score on the codes
Rounds: the KL reference is fixed per round. Training stops after the first round in which the TalkVid mouth or head
moves away from real, and the round before it is the one to use.
The active recap (chain_k > 1): windows are chains of chain_k windows and the score is on one of them, continuing the
policy's own rollout of the ones before (set_loss chain=L), as inference chains them. replay > 0 adds gamma x the
generator's own flow-matching loss on replay x persons fresh real windows per step (PPO-ptx's pretraining term).
EVAL lines (round 0 = the starting policy), on unseen val speakers, add to train_motion's rollout statistics:
  es_s<x>               energy score of 8 samples per input re-spread about their mean by x: lowest above x = 1 if
                        the 6-step sampler under-disperses and the score sees it
  mbes_s<x>             the same for the multiscale-block score
  chain_w<i>, gap_w<i>  (chain_k > 1) MBES of window i of a val chain continuing the policy's own windows, and its
                        excess over window i continuing the real frames: what the policy loses on its own prefix
  chain_rot_r           head rotation std over the whole generated chain / real (slow drift)
"""
import argparse
import copy
import math
import random
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from sang.data.dataset import PersonSets, chained, load_cfg
from sang.model.flow import build, flow_loss
from sang.model.scoring import block_scale, energy, gen_fakes, mbes, roll_chain, set_loss, window_of
from sang.model.target import WIDE_REGIONS
from train_motion import evaluate as motion_eval, forever, objective, setup_data


@torch.no_grad()
def chain_eval(policy, cdl, d, dev, n_batches: int, steps: int, g: float, blocks: tuple, m: int = 8) -> dict:
    """Per window i of a val chain: MBES of m samples continuing the policy's own windows (chain_w<i>) and its excess
    over m samples continuing the real frames (gap_w<i>), and the head rotation std of the whole chain / real."""
    acc = {}
    add = lambda k, v: acc.setdefault(k, []).append(float(v))
    rot = WIDE_REGIONS["rot"]
    for bi, b in enumerate(cdl):
        if bi >= n_batches:
            break
        batch = d.encode({k: v.to(dev) for k, v in b.items()})
        B, K = batch["target"].shape[0], batch["target"].shape[1] // d.L
        rep = {k: v.repeat_interleave(m, 0) for k, v in batch.items()}
        gen = torch.Generator(device=dev).manual_seed(bi)
        xs = roll_chain(policy, rep, K, d.L, steps, g, gen)
        score = lambda x, w: mbes(x.view(B, m, *x.shape[1:]), w["target"][::m], w["prefix"][::m, -1:], *blocks).mean()
        for i, x in enumerate(xs):
            w = window_of(rep, i, d.L)
            add(f"chain_w{i}", c := score(x, w))
            if i:
                add(f"gap_w{i}", c - score(gen_fakes(policy, w, steps, g, gen), w))
        gx = d.decode(torch.cat(xs, 1).float())[..., rot]
        add("chain_rot_r", gx.std(1).mean() / d.decode(batch["target"].float())[..., rot].std(1).mean())
    return {k: sum(v) / len(v) for k, v in acc.items()}


@torch.no_grad()
def set_eval(policy, sdl, d, dev, n_batches: int, steps: int, g: float, blocks: tuple, m: int = 8,
             spreads=(0.8, 1.0, 1.25, 1.6)) -> dict:
    acc = {}
    add = lambda k, v: acc.setdefault(k, []).append(float(v))
    for bi, b in enumerate(sdl):
        if bi >= n_batches:
            break
        batch = d.encode({k: v.to(dev).flatten(0, 1) for k, v in b.items()})
        y = batch["target"].float()
        rep = {k: v.repeat_interleave(m, 0) for k, v in batch.items()}
        x = gen_fakes(policy, rep, steps, g, torch.Generator(device=dev).manual_seed(bi)).float()
        x = x.view(y.shape[0], m, *y.shape[1:])
        xf = x.flatten(2)
        mu, mx = xf.mean(1, keepdim=True), x.mean(1, keepdim=True)
        start = batch["prefix"][:, -1:]
        for s in spreads:
            add(f"es_s{s:g}", energy(mu + s * (xf - mu), y.flatten(1)[:, None]).mean() / 2)
            add(f"mbes_s{s:g}", mbes(mx + s * (x - mx), y, start, *blocks).mean())
    return {k: sum(v) / len(v) for k, v in acc.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs/train_set.yaml"))
    ap.add_argument("overrides", nargs="*", help="key=value, parsed as YAML")
    args = ap.parse_args()
    rcfg = load_cfg(args.config, args.overrides)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(rcfg["gen"], map_location="cpu", weights_only=False)
    cfg = {**ck["cfg"], **rcfg}
    if dev == "cpu":
        cfg["bf16"] = False
    torch.manual_seed(cfg["seed"])
    random.seed(cfg["seed"])
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    m, score = int(cfg["m"]), cfg["score"]
    if m < 2 or score not in ("es", "mbes"):
        raise SystemExit("need m >= 2 samples per window and score es or mbes")
    d = setup_data(cfg, dev)
    K, replay = int(cfg["chain_k"]), int(cfg["replay"])
    load = lambda ds, n: forever(DataLoader(ds, batch_size=n, shuffle=True, num_workers=cfg["workers"],
                                            drop_last=True, pin_memory=dev == "cuda",
                                            persistent_workers=cfg["workers"] > 0))
    batches = load(PersonSets(chained(d.train_ds, K) if K > 1 else d.train_ds, 1), cfg["persons"])
    replays = load(d.train_ds, replay * cfg["persons"]) if replay else None
    sdl = DataLoader(PersonSets(d.val_ds, 8, fixed=True), batch_size=cfg["persons"], shuffle=False,
                     num_workers=min(2, cfg["workers"]))
    vdl = DataLoader(d.val_ds, batch_size=64, shuffle=False, num_workers=min(2, cfg["workers"]), drop_last=True)
    cdl = DataLoader(chained(d.val_ds, K), batch_size=cfg["persons"], shuffle=False,
                     num_workers=min(2, cfg["workers"])) if K > 1 else None

    policy = build(ck["cfg"]).to(dev)
    policy.load_state_dict(ck["ema"])
    policy.eval()
    g, steps = float(cfg["g"]), int(cfg["steps"])
    mcfg = {**ck["cfg"], "sample_steps": steps, "cfg_audio": g}
    amp = dict(device_type=dev, dtype=torch.bfloat16, enabled=cfg["bf16"])
    next_batch = lambda: d.encode({k: v.to(dev, non_blocking=True).flatten(0, 1) for k, v in next(batches).items()})
    real = [window_of(next_batch(), 0, d.L) for _ in range(max(1, 1024 // cfg["persons"]))]
    tgt, pre = (torch.cat([b[k] for b in real]) for k in ("target", "prefix"))
    perm = torch.randperm(len(tgt), device=tgt.device)
    blocks = (d.decode, *block_scale(tgt[perm], pre[perm][:, -1:], d.decode))
    del real, tgt, pre

    def eval_round(rnd: int) -> dict:
        with torch.autocast(**amp):
            rs = set_eval(policy, sdl, d, dev, cfg["eval_batches"], steps, g, blocks)
            if cdl is not None:
                rs.update(chain_eval(policy, cdl, d, dev, cfg["eval_batches"], steps, g, blocks))
        rm = motion_eval(policy, vdl, mcfg, dev, cfg["eval_batches"], d.encode, d.decode)
        policy.eval()
        dist = lambda parts: sum(abs(math.log(max(rm[f"{s}_r_{q}"], 1e-6))) for s in ("std", "vc") for q in parts)
        far = {"mouth": dist(("mouth",)), "head": dist(("rot", "place"))}
        print(f"EVAL round {rnd} " + " ".join(f"{k} {v:.4g}" for k, v in {**rs, **rm}.items())
              + f" dist_mouth {far['mouth']:.4f} dist_head {far['head']:.4f}", flush=True)
        return far

    Pn, A = int(cfg["policy_steps"]), int(cfg["alternations"])
    print(f"set fine-tuning from {cfg['gen']} (step {ck['step']}): {cfg['persons']} windows x m {m} samples, score "
          f"{score}, w_set {cfg['w_set']}, kl {cfg['kl']}, {cfg['rounds']} rounds x {A * Pn} steps, g={g:g}, "
          f"steps={steps}, chain_k {K}, replay {replay} x gamma {cfg['gamma']}", flush=True)
    rkw = objective(ck["cfg"])
    prev = eval_round(0)
    for rnd in range(1, int(cfg["rounds"]) + 1):
        ref = copy.deepcopy(policy).eval().requires_grad_(False)
        opt = torch.optim.AdamW(policy.parameters(), lr=cfg["lr"], weight_decay=0.0)
        for it in range(A):
            t0 = time.time()
            for step in range(it * Pn + 1, (it + 1) * Pn + 1):
                with torch.autocast(**amp):
                    loss, parts = set_loss(policy, ref, next_batch(), m, steps, g, cfg["kl"], cfg["w_set"],
                                           blocks=blocks if score == "mbes" else None, chain=d.L if K > 1 else 0)
                    if replays is not None:
                        fm, _ = flow_loss(policy, d.encode({k: v.to(dev, non_blocking=True)
                                                            for k, v in next(replays).items()}), **rkw)
                        loss, parts["replay"] = loss + cfg["gamma"] * fm, fm.detach()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                gnorm = torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg["grad_clip"])
                opt.step()
                if step % cfg["log_every"] == 0:
                    print(f"round {rnd} policy {step:>5} " + " ".join(f"{k} {float(v):.4f}" for k, v in parts.items())
                          + f" gnorm {float(gnorm):.3f} {cfg['log_every'] / (time.time() - t0):.2f} it/s", flush=True)
                    t0 = time.time()
        far = eval_round(rnd)
        sd = policy.state_dict()
        torch.save({"model": sd, "ema": sd, "cfg": ck["cfg"], "norm": ck["norm"], "latent": ck["latent"],
                    "step": ck["step"], "set": {"round": rnd, "gen": cfg["gen"], "cfg": cfg}}, out / f"round{rnd}.pt")
        away = [k for k in far if far[k] > prev[k]]
        if away:
            print(f"STOP after round {rnd}: the {' and '.join(away)} moved away from real "
                  f"({', '.join(f'{k} {prev[k]:.4f} -> {far[k]:.4f}' for k in away)}); use round {rnd - 1} "
                  f"(round 0 = {cfg['gen']})", flush=True)
            break
        prev = far
    print(f"done -> {out}", flush=True)


if __name__ == "__main__":
    main()
