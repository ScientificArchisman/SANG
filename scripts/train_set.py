#!/usr/bin/env python3
"""Proper multi-sample fine-tuning of SANG's deployed sampler (reports/SANG RL for video realism.md, idea 1).

    sbatch bash_scripts/run.sh scripts/train_set.py gen=runs/demo/last.pt out_dir=runs/set [key=value ...]

configs/train_set.yaml on top of the generator run's own config. A batch is `persons` training windows (each reads
footage from another clip of that person, sang.data.dataset.PersonSets), each rolled out m times through the 6-step
sampler with gradients. The loss (sang.model.scoring.set_loss) scores the m samples against the real window, plus
kl x the path KL to the round's start:
  score=mbes     the multiscale-block energy score (sang.model.scoring.mbes; reports/SANG style aware scoring
                 rules.md, rank 1); the default, runs/mbes
  score=es       the plain energy score on the codes (runs/set_es2)
Rounds: the KL reference is fixed per round; training stops after the first round in which the TalkVid mouth or head
moves away from real, and the round before it is the one to use.
EVAL lines (round 0 = the starting policy), on unseen val speakers, add to train_motion's rollout statistics:
  es_s<x>               energy score of 8 samples per input re-spread about their mean by x: lowest above x = 1 if
                        the 6-step sampler under-disperses and the score sees it
  mbes_s<x>             the same for the multiscale-block score
  g1z_{es,mbes}_<p>     gate G1 (rounds=0 runs only the EVAL): paired z over inputs of each score's change when the
                        samples are perturbed: the head x 0.62 (all, its 33-frame moving average "slow", the rest
                        "fast"), the head x 1.3, every sample re-spread x 1.25. Pass: |z| of mbes >= 3 on every head
                        x 0.62 and >= 3x that of es
"""
import argparse
import copy
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from sang.data.dataset import PersonSets, load_cfg
from sang.model.flow import build
from sang.model.scoring import block_scale, energy, gen_fakes, mbes, set_loss
from sang.model.target import WIDE_DIM
from train_motion import evaluate as motion_eval, setup_data

HEAD = [0, 1, 2, WIDE_DIM - 3, WIDE_DIM - 2, WIDE_DIM - 1]  # pitch, yaw, roll, t_x, t_y, log scale (target.wide)


def shrink(y: torch.Tensor, d, s: float, start: torch.Tensor | None = None, part: str = "all") -> torch.Tensor:
    """Codes y [..., T, K] with the head's motion scaled by s about the photo, or about start's pose: "all", "slow"
    (its 33-frame moving average) or "fast" (the rest). Diagnostic only, never a target."""
    with torch.autocast(y.device.type, enabled=False):
        w = d.decode(y.float())
        h = w[..., HEAD] - (d.decode(start.float())[..., HEAD] if start is not None else 0)
        slow = F.avg_pool1d(h.flatten(0, -3).transpose(1, 2), 33, 1, 16, count_include_pad=False).transpose(1, 2)
        slow = slow.reshape(h.shape)
        w[..., HEAD] += {"all": (s - 1) * h, "slow": (s - 1) * slow, "fast": (s - 1) * (h - slow)}[part]
        return d.latent.encode(w, d.k)


def forever(dl):
    while True:
        yield from dl


@torch.no_grad()
def set_eval(policy, sdl, d, dev, n_batches: int, steps: int, g: float, blocks: tuple, m: int = 8,
             spreads=(0.8, 1.0, 1.25, 1.6)) -> dict:
    acc, g1 = {}, {}
    add = lambda k, v: acc.setdefault(k, []).append(float(v))
    for bi, b in enumerate(sdl):
        if bi >= n_batches:
            break
        batch = d.encode({k: v.to(dev).flatten(0, 1) for k, v in b.items()})
        y = batch["target"].float()
        rep = {k: v.repeat_interleave(m, 0) for k, v in batch.items()}
        x = gen_fakes(policy, rep, steps, g, torch.Generator(device=dev).manual_seed(bi)).float()
        x = x.view(y.shape[0], m, *y.shape[1:])
        pre = batch["prefix"]
        xf = x.flatten(2)
        mu, mx = xf.mean(1, keepdim=True), x.mean(1, keepdim=True)
        start = pre[:, -1:]
        for s in spreads:
            add(f"es_s{s:g}", energy(mu + s * (xf - mu), y.flatten(1)[:, None]).mean() / 2)
            add(f"mbes_s{s:g}", mbes(mx + s * (x - mx), y, start, *blocks).mean())
        whole = lambda z: energy((z - start[:, None]).flatten(2), (y - start).flatten(1)[:, None]) / 2
        base = shrink(x, d, 1.0, start[:, None])  # the decode/encode round trip the perturbed samples go through
        e0, b0 = whole(base), mbes(base, y, start, *blocks)
        bm = base.mean(1, keepdim=True)
        moved = {**{f"head0.62_{p}": shrink(x, d, 0.62, start[:, None], p) for p in ("all", "slow", "fast")},
                 "head1.3": shrink(x, d, 1.3, start[:, None]), "spread1.25": bm + 1.25 * (base - bm)}
        for name, z in moved.items():
            g1.setdefault(f"es_{name}", []).append(whole(z) - e0)
            g1.setdefault(f"mbes_{name}", []).append(mbes(z, y, start, *blocks) - b0)
    out = {k: sum(v) / len(v) for k, v in acc.items()}
    for k, v in g1.items():
        v = torch.cat(v).float()
        out[f"g1z_{k}"] = float(v.mean() / (v.std() / len(v) ** 0.5))
    return out


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
    batches = forever(DataLoader(PersonSets(d.train_ds, 1), batch_size=cfg["persons"], shuffle=True,
                                 num_workers=cfg["workers"], drop_last=True, pin_memory=dev == "cuda",
                                 persistent_workers=cfg["workers"] > 0))
    sdl = DataLoader(PersonSets(d.val_ds, 8, fixed=True), batch_size=cfg["persons"], shuffle=False,
                     num_workers=min(2, cfg["workers"]))
    vdl = DataLoader(d.val_ds, batch_size=64, shuffle=False, num_workers=min(2, cfg["workers"]), drop_last=True)

    policy = build(ck["cfg"]).to(dev)
    policy.load_state_dict(ck["ema"])
    policy.eval()  # no dropout in rollouts; gradients still flow
    g, steps = float(cfg["g"]), int(cfg["steps"])
    mcfg = {**ck["cfg"], "sample_steps": steps, "cfg_audio": g}
    amp = dict(device_type=dev, dtype=torch.bfloat16, enabled=cfg["bf16"])
    next_batch = lambda: d.encode({k: v.to(dev, non_blocking=True).flatten(0, 1) for k, v in next(batches).items()})
    # mbes's c0, c_b: real spreads of ~1024 training windows, from the training loader (one 1024 batch with its
    # footage audio per worker ran out of host memory)
    real = [{k: b[k] for k in ("target", "prefix")} for b in (next_batch() for _ in range(max(1, 1024 // cfg["persons"])))]
    tgt, pre = (torch.cat([b[k] for b in real]) for k in ("target", "prefix"))
    perm = torch.randperm(len(tgt), device=tgt.device)
    blocks = (d.decode, *block_scale(tgt[perm], pre[perm][:, -1:], d.decode))
    del real, tgt, pre

    def eval_round(rnd: int) -> dict:
        with torch.autocast(**amp):
            rs = set_eval(policy, sdl, d, dev, cfg["eval_batches"], steps, g, blocks)
        rm = motion_eval(policy, vdl, mcfg, dev, cfg["eval_batches"], d.encode, d.decode)
        policy.eval()  # motion_eval leaves the model in train mode
        dist = lambda parts: sum(abs(math.log(max(rm[f"{s}_r_{q}"], 1e-6))) for s in ("std", "vc") for q in parts)
        far = {"mouth": dist(("mouth",)), "head": dist(("rot", "place"))}
        print(f"EVAL round {rnd} " + " ".join(f"{k} {v:.4g}" for k, v in {**rs, **rm}.items())
              + f" dist_mouth {far['mouth']:.4f} dist_head {far['head']:.4f}", flush=True)
        return far

    Pn, A = int(cfg["policy_steps"]), int(cfg["alternations"])
    print(f"set fine-tuning from {cfg['gen']} (step {ck['step']}): {cfg['persons']} windows x m {m} samples, score "
          f"{score}, w_set {cfg['w_set']}, kl {cfg['kl']}, {cfg['rounds']} rounds x {A * Pn} steps, g={g:g}, "
          f"steps={steps}", flush=True)
    prev = eval_round(0)
    for rnd in range(1, int(cfg["rounds"]) + 1):
        ref = copy.deepcopy(policy).eval().requires_grad_(False)
        opt = torch.optim.AdamW(policy.parameters(), lr=cfg["lr"], weight_decay=0.0)
        for it in range(A):
            t0 = time.time()
            for step in range(it * Pn + 1, (it + 1) * Pn + 1):
                with torch.autocast(**amp):
                    loss, parts = set_loss(policy, ref, next_batch(), m, steps, g, cfg["kl"], cfg["w_set"],
                                           blocks=blocks if score == "mbes" else None)
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
