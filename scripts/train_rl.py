#!/usr/bin/env python3
"""RL fine-tuning of SANG against the person critic: stage 4 of docs/person_critic_plan.md.

    sbatch bash_scripts/run.sh scripts/train_rl.py gen=runs/demo/last.pt critic=runs/critic/last.pt \\
        out_dir=runs/rl [key=value ...]

configs/train_rl.yaml on top of the generator run's own config. `rounds` rounds of
  (a) policy_steps of KL-regularised reward maximisation (sang.model.critic.rl_loss): the critic's reward on the
      clean sample, backpropagated through the Euler steps, minus kl times the path KL to the policy as it was when
      the round began; the critic is frozen;
  (b) critic_steps refitting the critic on the new policy's samples (critic_loss), so the reward stays
      log p_real / p_policy.
round<r>.pt loads like a generator (eval_hdtf.py --ckpt runs/rl/round<r>.pt), critic<r>.pt like a critic.
EVAL lines (round 0 = the starting policy), on unseen val speakers: train_critic's metrics for the critic against
the current policy, and train_motion's rollout statistics (std_r, vc_r, audio_gain, val_loss).
"""
import argparse
import copy
import random
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from sang.data.dataset import load_cfg
from sang.model.critic import critic_loss, gen_fakes, load_critic, rl_loss
from sang.model.flow import build
from train_critic import evaluate as critic_eval
from train_motion import evaluate as motion_eval, setup_data


def forever(dl):
    while True:
        yield from dl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs/train_rl.yaml"))
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
    d = setup_data(cfg, dev)
    batches = forever(DataLoader(d.train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["workers"],
                                 drop_last=True, pin_memory=dev == "cuda", persistent_workers=cfg["workers"] > 0))
    vdl = DataLoader(d.val_ds, batch_size=64, shuffle=False, num_workers=min(2, cfg["workers"]), drop_last=True)

    policy = build(ck["cfg"]).to(dev)
    policy.load_state_dict(ck["ema"])
    policy.eval()  # no dropout in rollouts; gradients still flow
    critic = load_critic(cfg["critic"], dev)
    g, steps = float(cfg["g"]), int(cfg["steps"])
    mcfg = {**ck["cfg"], "sample_steps": steps, "cfg_audio": g}
    kw = dict(p_clean=cfg["p_clean"], p_demo=cfg["p_demo"], shift=(int(cfg["shift_min"]), int(cfg["shift_max"])))
    amp = dict(device_type=dev, dtype=torch.bfloat16, enabled=cfg["bf16"])
    next_batch = lambda: d.encode({k: v.to(dev, non_blocking=True) for k, v in next(batches).items()})

    def eval_round(rnd: int) -> None:
        with torch.autocast(**amp):
            rc = critic_eval(critic, policy, vdl, d.encode, dev, cfg["eval_batches"], g, steps)
        rm = motion_eval(policy, vdl, mcfg, dev, cfg["eval_batches"], d.encode, d.decode)
        policy.eval()  # motion_eval leaves the model in train mode
        print(f"EVAL round {rnd} " + " ".join(f"{k} {v:.4g}" for k, v in {**rc, **rm}.items()), flush=True)

    print(f"RL from {cfg['gen']} (step {ck['step']}) against {cfg['critic']}: {cfg['rounds']} rounds x "
          f"({cfg['policy_steps']} policy + {cfg['critic_steps']} critic steps), kl {cfg['kl']}, "
          f"reward {cfg['w_real']} real + {cfg['w_sync']} sync, g={g:g}, steps={steps}", flush=True)
    eval_round(0)
    for rnd in range(1, int(cfg["rounds"]) + 1):
        ref = copy.deepcopy(policy).eval().requires_grad_(False)
        opt = torch.optim.AdamW(policy.parameters(), lr=cfg["lr"], weight_decay=0.0)
        critic.eval().requires_grad_(False)
        t0 = time.time()
        for step in range(1, int(cfg["policy_steps"]) + 1):
            with torch.autocast(**amp):
                loss, parts = rl_loss(policy, ref, critic, next_batch(), steps, g, cfg["kl"], cfg["w_real"],
                                      cfg["w_sync"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg["grad_clip"])
            opt.step()
            if step % cfg["log_every"] == 0:
                print(f"round {rnd} policy {step:>5} " + " ".join(f"{k} {float(v):.4f}" for k, v in parts.items())
                      + f" gnorm {float(gnorm):.3f} {cfg['log_every'] / (time.time() - t0):.2f} it/s", flush=True)
                t0 = time.time()
        critic.train().requires_grad_(True)
        copt = torch.optim.AdamW(critic.parameters(), lr=cfg["critic_lr"], weight_decay=cfg["weight_decay"])
        for step in range(1, int(cfg["critic_steps"]) + 1):
            batch = next_batch()
            with torch.autocast(**amp):
                loss, parts = critic_loss(critic, batch, gen_fakes(policy, batch, steps, g).float(), **kw)
            copt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), cfg["grad_clip"])
            copt.step()
            if step % cfg["log_every"] == 0:
                print(f"round {rnd} critic {step:>5} " + " ".join(f"{k} {float(v):.4f}" for k, v in parts.items()),
                      flush=True)
        critic.eval().requires_grad_(False)
        eval_round(rnd)
        sd = policy.state_dict()
        torch.save({"model": sd, "ema": sd, "cfg": ck["cfg"], "norm": ck["norm"], "latent": ck["latent"],
                    "step": ck["step"], "rl": {"round": rnd, "gen": cfg["gen"], "critic": cfg["critic"], "cfg": cfg}},
                   out / f"round{rnd}.pt")
        torch.save({"critic": critic.state_dict(), "gen_cfg": ck["cfg"], "cfg": cfg, "gen": str(out / f"round{rnd}.pt"),
                    "step": rnd}, out / f"critic{rnd}.pt")
    print(f"done -> {out}", flush=True)


if __name__ == "__main__":
    main()
