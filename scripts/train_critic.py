#!/usr/bin/env python3
"""Train the person critic (sang.model.critic) on TalkVid: stage 1 of docs/person_critic_plan.md.

    sbatch bash_scripts/run.sh scripts/train_critic.py gen=runs/demo/last.pt out_dir=runs/critic [key=value ...]

configs/train_critic.yaml on top of the generator run's own config (its data, latent, window and footage settings).
Real = a training window with its audio, photo, prefix and up to 10 s of another clip of the same person; fake = the
frozen generator's sample for the same inputs, made on the fly with the deployed sampler (gen_g, gen_steps); sync
fake = the real window with its audio shifted by shift_min..shift_max frames. The critic starts as a copy of the
generator with a zero two-logit head.

EVAL lines, on unseen val speakers:
  auc_real     real vs SANG (the training sampler) at t = 0;  auc_real_t  the same at a random noise level
  auc_sync     real vs the same windows with the audio 8 frames late, at t = 0
  rank_<s>     mean realism logit at t = 0 of real motion, real motion reversed in time, and SANG with each sampler
               of SAMPLERS (the generator sees the footage unless the name says otherwise); sync_<s> the same for the
               sync logit. Gate 1b compares this order with the HDTF FVD order.
"""
import argparse
import random
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from sang.data.dataset import load_cfg
from sang.model.critic import auc, critic_from_generator, critic_loss, gen_fakes, noised
from sang.model.flow import build
from train_motion import lr_at, setup_data

# name -> (guidance gamma, Euler steps, the generator sees the footage)
SAMPLERS = {"g2_s6": (2.0, 6, True), "g2_s6_nofootage": (2.0, 6, False), "g2_s12": (2.0, 12, True),
            "g1.5_s6": (1.5, 6, True), "g3_s6": (3.0, 6, True)}


@torch.no_grad()
def evaluate(critic, gen, loader, encode, dev, n_batches: int, g: float = 2.0, steps: int = 6) -> dict:
    """The EVAL metrics above for the critic against generator `gen` (also used per round by train_rl.py)."""
    was = critic.training
    critic.eval()
    train = f"g{g:g}_s{steps}"
    samplers = {train: (g, steps, True), **SAMPLERS}
    rank = {k: [] for k in ("real", "real_reversed", *samplers)}
    sync = {k: [] for k in rank}
    noisy_real, noisy_fake, late = [], [], []
    rng = torch.Generator(device=dev)
    for bi, batch in enumerate(loader):
        if bi >= n_batches:
            break
        batch = encode({k: v.to(dev) for k, v in batch.items()})
        x0 = batch["target"]
        B = x0.shape[0]
        kw = dict(prefix=batch["prefix"], demo=critic.encode_demo(batch), demo_keep=batch.get("demo_keep"))

        def score(x, t=None, audio=None):
            t = torch.zeros(B, device=dev) if t is None else t
            return critic(x, t, batch["audio"] if audio is None else audio, batch["ref"], **kw).float()

        xs = {"real": x0, "real_reversed": x0.flip(1)}
        for name, (gg, st, foot) in samplers.items():
            rng.manual_seed(bi)
            b = batch if foot or "demo_keep" not in batch else {**batch, "demo_keep": torch.zeros_like(batch["demo_keep"])}
            xs[name] = gen_fakes(gen, b, st, gg, rng).float()
        for name, x in xs.items():
            lg = score(x)
            rank[name].append(lg[:, 0])
            sync[name].append(lg[:, 1])
        late.append(score(x0, audio=batch["audio"].roll(16, 1))[:, 1])
        torch.manual_seed(bi)
        t = torch.rand(B, device=dev)
        noisy_real.append(score(noised(x0, t), t)[:, 0])
        noisy_fake.append(score(noised(xs[train], t), t)[:, 0])
    critic.train(was)
    cat = torch.cat
    out = {"auc_real": auc(cat(rank["real"]), cat(rank[train])), "auc_real_t": auc(cat(noisy_real), cat(noisy_fake)),
           "auc_sync": auc(cat(sync["real"]), cat(late))}
    out.update({f"rank_{k}": float(cat(v).mean()) for k, v in rank.items()})
    out.update({f"sync_{k}": float(cat(v).mean()) for k, v in sync.items()})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs/train_critic.yaml"))
    ap.add_argument("overrides", nargs="*", help="key=value, parsed as YAML")
    args = ap.parse_args()
    ccfg = load_cfg(args.config, args.overrides)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(ccfg["gen"], map_location="cpu", weights_only=False)
    cfg = {**ck["cfg"], **ccfg}  # the generator's data settings (and split seed), the critic's training settings
    if dev == "cpu":
        cfg["bf16"] = False
    torch.manual_seed(cfg["seed"])
    random.seed(cfg["seed"])
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    d = setup_data(cfg, dev)
    dl = DataLoader(d.train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["workers"],
                    drop_last=True, pin_memory=dev == "cuda", persistent_workers=cfg["workers"] > 0)
    vdl = DataLoader(d.val_ds, batch_size=64, shuffle=False, num_workers=min(2, cfg["workers"]), drop_last=True)

    gen = build(ck["cfg"]).to(dev).eval().requires_grad_(False)
    gen.load_state_dict(ck["ema"])
    critic = critic_from_generator(ck).to(dev).train()
    opt = torch.optim.AdamW(critic.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    g, steps = float(cfg["gen_g"]), int(cfg["gen_steps"])
    print(f"critic: {sum(p.numel() for p in critic.parameters()) / 1e6:.1f} M params from {cfg['gen']} (step "
          f"{ck['step']}); fakes g={g:g}, steps={steps}", flush=True)
    kw = dict(p_clean=cfg["p_clean"], p_demo=cfg["p_demo"], shift=(int(cfg["shift_min"]), int(cfg["shift_max"])),
              gp=float(cfg["gp"]))
    amp = dict(device_type=dev, dtype=torch.bfloat16, enabled=cfg["bf16"])

    step, t0 = 0, time.time()
    while step < cfg["max_steps"]:
        for batch in dl:
            batch = d.encode({k_: v.to(dev, non_blocking=True) for k_, v in batch.items()})
            for gr in opt.param_groups:
                gr["lr"] = lr_at(step, cfg)
            with torch.autocast(**amp):
                fake = gen_fakes(gen, batch, steps, g).float()
                loss, parts = critic_loss(critic, batch, fake, **kw)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(critic.parameters(), cfg["grad_clip"])
            opt.step()
            step += 1
            if step % cfg["log_every"] == 0:
                sps = cfg["log_every"] / (time.time() - t0)
                t0 = time.time()
                print(f"step {step:>6} lr {opt.param_groups[0]['lr']:.2e} "
                      + " ".join(f"{k_} {float(v):.4f}" for k_, v in parts.items())
                      + f" gnorm {float(gnorm):.3f} {sps:.2f} it/s", flush=True)
            if step % cfg["eval_every"] == 0 or step == cfg["max_steps"]:
                with torch.autocast(**amp):
                    r = evaluate(critic, gen, vdl, d.encode, dev, cfg["eval_batches"], g, steps)
                print(f"EVAL step {step} " + " ".join(f"{k_} {v:.4g}" for k_, v in r.items()), flush=True)
                torch.save({"critic": critic.state_dict(), "gen_cfg": ck["cfg"], "cfg": cfg, "gen": cfg["gen"],
                            "step": step, "eval": r}, out / "last.pt")
            if step >= cfg["max_steps"]:
                break
    print(f"done -> {out / 'last.pt'}", flush=True)


if __name__ == "__main__":
    main()
