#!/usr/bin/env python3
"""Proper multi-sample fine-tuning of SANG's deployed sampler (reports/SANG RL for video realism.md, ideas 1 and 2).

    sbatch bash_scripts/run.sh scripts/train_set.py gen=runs/demo/last.pt out_dir=runs/set [key=value ...]

configs/train_set.yaml on top of the generator run's own config. A batch is `persons` people x K windows
(sang.data.dataset.PersonSets: all K read one footage stretch from another clip of that person), each rolled out m
times through the 6-step sampler with gradients. The loss (sang.model.critic.set_loss) is the energy distance
between a person's K*m samples and its K real windows, plus kl x the path KL to the round's start:
  K = 1, m = 8   idea 1: 8 samples for one input against its real window (the energy score)
  K = 8, m = 1   idea 2: a person's generated windows against real windows of that person the generator never saw
  w_adv > 0      + the same distance in a set critic's learned feature space; the critic maximises it (R1 gp),
                 refit for critic_steps after every policy_steps, after critic_warmup steps against the start
  score=mbes     (K = 1) the multiscale-block energy score in place of the energy score (sang.model.critic.mbes;
                 reports/SANG style aware scoring rules.md, rank 1)
Rounds as in train_rl.py: the KL reference is fixed per round; training stops after the first round in which the
TalkVid mouth or head moves away from real, and the round before it is the one to use.
EVAL lines (round 0 = the starting policy), on unseen val speakers, add to train_motion's rollout statistics:
  ed_sang / ed_shrink   energy distance of 8-window person sets: SANG vs real, and real with the head x0.62 vs real
  gate                  ed_shrink / ed_sang: >= 0.25 if the distance sees head range (the critics: 0.002-0.007)
  ed_*_adv, gate_adv    the same in the set critic's space (w_adv > 0)
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
from sang.model.critic import block_scale, critic_from_generator, energy, gen_fakes, mbes, set_critic_loss, set_loss
from sang.model.flow import build
from sang.model.target import WIDE_DIM
from train_motion import evaluate as motion_eval, setup_data
from train_rl import forever

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


@torch.no_grad()
def set_eval(policy, critic, sdl, d, dev, n_batches: int, steps: int, g: float, blocks: tuple, K: int = 8, m: int = 8,
             spreads=(0.8, 1.0, 1.25, 1.6)) -> dict:
    acc, g1 = {}, {}
    add = lambda k, v: acc.setdefault(k, []).append(float(v))
    for bi, b in enumerate(sdl):
        if bi >= n_batches:
            break
        batch = d.encode({k: v.to(dev).flatten(0, 1) for k, v in b.items()})
        y = batch["target"].float()
        B0 = y.shape[0]
        P = B0 // K
        rep = {k: v.repeat_interleave(m, 0) for k, v in batch.items()}
        x = gen_fakes(policy, rep, steps, g, torch.Generator(device=dev).manual_seed(bi)).float()
        x = x.view(B0, m, *y.shape[1:])
        sets = lambda z, pre: (z - pre[:, -1:].float()).reshape(P, K, -1)  # from each window's start, as set_loss
        pre = batch["prefix"]
        # the head shrunk over the whole stretch (prefix too), and the same decode/encode round trip unshrunk
        (real, real_pre), (shr, shr_pre) = [(shrink(y, d, s), shrink(pre.float(), d, s)) for s in (1.0, 0.62)]
        add("ed_sang", energy(sets(x[:, 0], pre), sets(y, pre), True).mean())
        add("ed_shrink", energy(sets(shr, shr_pre), sets(real, real_pre), True).mean())
        if critic is not None:
            demo = critic.encode_demo(batch)
            f = lambda z, p: critic.features(z, {**batch, "prefix": p}, demo).view(P, K, -1)
            add("ed_sang_adv", energy(f(x[:, 0], pre), f(y, pre), True).mean())
            add("ed_shrink_adv", energy(f(shr, shr_pre), f(real, real_pre), True).mean())
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
    for sfx in ("", "_adv"):
        if f"ed_sang{sfx}" in out:
            out[f"gate{sfx}"] = out[f"ed_shrink{sfx}"] / out[f"ed_sang{sfx}"] if out[f"ed_sang{sfx}"] else float("nan")
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
    K, m, w_adv, score = int(cfg["K"]), int(cfg["m"]), float(cfg["w_adv"]), cfg["score"]
    if K * m < 2 or (w_adv and K < 2):
        raise SystemExit("need K*m >= 2 samples per person (and K >= 2 for the set critic)")
    if score not in ("es", "mbes") or (score == "mbes" and (K != 1 or w_adv)):
        raise SystemExit("score: es, or mbes with K = 1 and no set critic")
    d = setup_data(cfg, dev)
    real = next(iter(DataLoader(d.train_ds, batch_size=1024, shuffle=True, num_workers=cfg["workers"])))
    real = d.encode({k: real[k].to(dev) for k in ("target", "prefix")})
    blocks = (d.decode, *block_scale(real["target"], real["prefix"][:, -1:], d.decode))  # mbes's c0, c_b: real spreads
    batches = forever(DataLoader(PersonSets(d.train_ds, K), batch_size=cfg["persons"], shuffle=True,
                                 num_workers=cfg["workers"], drop_last=True, pin_memory=dev == "cuda",
                                 persistent_workers=cfg["workers"] > 0))
    sdl = DataLoader(PersonSets(d.val_ds, 8, fixed=True), batch_size=cfg["persons"], shuffle=False,
                     num_workers=min(2, cfg["workers"]))
    vdl = DataLoader(d.val_ds, batch_size=64, shuffle=False, num_workers=min(2, cfg["workers"]), drop_last=True)

    policy = build(ck["cfg"]).to(dev)
    policy.load_state_dict(ck["ema"])
    policy.eval()  # no dropout in rollouts; gradients still flow
    critic = critic_from_generator(ck).to(dev).eval().requires_grad_(False) if w_adv else None
    g, steps = float(cfg["g"]), int(cfg["steps"])
    mcfg = {**ck["cfg"], "sample_steps": steps, "cfg_audio": g}
    amp = dict(device_type=dev, dtype=torch.bfloat16, enabled=cfg["bf16"])
    next_batch = lambda: d.encode({k: v.to(dev, non_blocking=True).flatten(0, 1) for k, v in next(batches).items()})

    def eval_round(rnd: int) -> dict:
        with torch.autocast(**amp):
            rs = set_eval(policy, critic, sdl, d, dev, cfg["eval_batches"], steps, g, blocks)
        rm = motion_eval(policy, vdl, mcfg, dev, cfg["eval_batches"], d.encode, d.decode)
        policy.eval()  # motion_eval leaves the model in train mode
        dist = lambda parts: sum(abs(math.log(max(rm[f"{s}_r_{q}"], 1e-6))) for s in ("std", "vc") for q in parts)
        far = {"mouth": dist(("mouth",)), "head": dist(("rot", "place"))}
        print(f"EVAL round {rnd} " + " ".join(f"{k} {v:.4g}" for k, v in {**rs, **rm}.items())
              + f" dist_mouth {far['mouth']:.4f} dist_head {far['head']:.4f}", flush=True)
        return far

    def refit(n: int, first: int, opt, tag: str) -> None:
        critic.train().requires_grad_(True)
        for step in range(first + 1, first + n + 1):
            batch = next_batch()
            with torch.autocast(**amp):
                loss, parts = set_critic_loss(critic, batch, gen_fakes(policy, batch, steps, g).float(), K, cfg["gp"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), cfg["grad_clip"])
            opt.step()
            if step % cfg["log_every"] == 0:
                print(f"{tag} critic {step:>5} " + " ".join(f"{k} {float(v):.4f}" for k, v in parts.items()), flush=True)
        critic.eval().requires_grad_(False)

    copt = lambda: torch.optim.AdamW(critic.parameters(), lr=cfg["critic_lr"], weight_decay=cfg["weight_decay"])
    Pn, C, A = int(cfg["policy_steps"]), int(cfg["critic_steps"]), int(cfg["alternations"])
    print(f"set fine-tuning from {cfg['gen']} (step {ck['step']}): {cfg['persons']} persons x K {K} windows x m {m} "
          f"samples, score {score}, w_set {cfg['w_set']}, w_adv {w_adv}, kl {cfg['kl']}, {cfg['rounds']} rounds x {A} x ({Pn} policy"
          + (f" + {C} critic steps), critic warm-up {cfg['critic_warmup']}" if w_adv else " steps)")
          + f", g={g:g}, steps={steps}", flush=True)
    if w_adv:
        refit(int(cfg["critic_warmup"]), 0, copt(), "warmup")
    prev = eval_round(0)
    for rnd in range(1, int(cfg["rounds"]) + 1):
        ref = copy.deepcopy(policy).eval().requires_grad_(False)
        opt = torch.optim.AdamW(policy.parameters(), lr=cfg["lr"], weight_decay=0.0)
        c_opt = copt() if w_adv else None
        for it in range(A):
            t0 = time.time()
            for step in range(it * Pn + 1, (it + 1) * Pn + 1):
                with torch.autocast(**amp):
                    loss, parts = set_loss(policy, ref, next_batch(), K, m, steps, g, cfg["kl"], cfg["w_set"], w_adv,
                                           critic, blocks=blocks if score == "mbes" else None)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                gnorm = torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg["grad_clip"])
                opt.step()
                if step % cfg["log_every"] == 0:
                    print(f"round {rnd} policy {step:>5} " + " ".join(f"{k} {float(v):.4f}" for k, v in parts.items())
                          + f" gnorm {float(gnorm):.3f} {cfg['log_every'] / (time.time() - t0):.2f} it/s", flush=True)
                    t0 = time.time()
            if w_adv:
                refit(C, it * C, c_opt, f"round {rnd}")
        far = eval_round(rnd)
        sd = policy.state_dict()
        torch.save({"model": sd, "ema": sd, "cfg": ck["cfg"], "norm": ck["norm"], "latent": ck["latent"],
                    "step": ck["step"], "set": {"round": rnd, "gen": cfg["gen"], "cfg": cfg}}, out / f"round{rnd}.pt")
        if w_adv:
            torch.save({"critic": critic.state_dict(), "gen_cfg": ck["cfg"], "cfg": cfg, "step": rnd},
                       out / f"critic{rnd}.pt")
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
