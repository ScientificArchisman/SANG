#!/usr/bin/env python3
"""Train the style key, then build the style memory (sang.model.memory; research log §7.9).

    sbatch --time=04:00:00 bash_scripts/run.sh scripts/train_style.py                                 # runs/style
    sbatch --time=04:00:00 bash_scripts/run.sh scripts/train_style.py key=motion out_dir=runs/style_motion

configs/train_style.yaml on top of the generator run's config (gen), so the speaker split, sync offsets and latent are
the generator's. A batch is `persons` TalkVid speakers x K windows spread over each speaker's clips; the loss is a
supervised InfoNCE over speakers on the key of (audio, residual): key=residual (the real codes minus gen's mean from
the audio, photo and prefix) or key=motion (the codes).
EVAL on the unseen val speakers, 8 windows each:
  top1_win   a window's nearest other window is the same speaker's
  top1_half  the mean key of 4 windows finds the mean of the speaker's other 4 among all val speakers (chance =
             1 / speakers): how well a stretch of footage finds its person, the memory's first key
Then memory.pt: every training clip's style (the mean key of its windows), every speaker's, and the audio key of every
L-frame window at `stride` frames; the val clips' styles too, as the queries of a fine-tune's val windows.
"""
import argparse
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from sang.data.dataset import load_cfg
from sang.data.sync import clip_key
from sang.model.flow import build
from sang.model.memory import StyleKey, embed, key_input, supcon
from train_motion import lr_at, setup_data
from train_set import forever


class Speakers(Dataset):
    """K windows of one speaker (a video directory with two or more clips), over as many of its clips as possible;
    `speaker` labels them."""

    def __init__(self, ds, K: int, fixed: bool = False):
        by = {}
        for j, r in enumerate(ds.rows):
            by.setdefault(Path(r["path"]).parent.name, []).append(j)
        self.ds, self.K, self.fixed = ds, K, fixed
        self.groups = [v for v in by.values() if len(v) > 1]

    def __len__(self):
        return len(self.groups) * self.ds.per_clip

    def __getitem__(self, i):
        rng = random.Random(i) if self.fixed else random
        g = i % len(self.groups)
        clips = rng.sample(self.groups[g], len(self.groups[g]))
        items = [self.ds.window(clips[t % len(clips)], rng) for t in range(self.K)]
        return {**{k: torch.stack([w[k] for w in items]) for k in items[0]}, "speaker": torch.full((self.K,), g)}


def keys(key, gen, b, d, dev, kind, amp, generator=None):
    """A Speakers batch -> (unit keys [n, e], speaker labels [n])."""
    lab = b.pop("speaker").flatten().to(dev)
    batch = d.encode({k: v.to(dev, non_blocking=True).flatten(0, 1) for k, v in b.items()})
    with torch.autocast(**amp):
        x, a = key_input(gen, batch, kind, generator)
        return key(x, a), lab


@torch.no_grad()
def evaluate(key, gen, vdl, d, dev, kind, amp, tau) -> dict:
    key.eval()
    g = torch.Generator(device=dev).manual_seed(0)
    zs, labs = zip(*(keys(key, gen, b, d, dev, kind, amp, g) for b in vdl))
    key.train()
    z, lab = torch.cat(zs).float(), torch.cat(labs)
    sim = (z @ z.T).fill_diagonal_(float("-inf"))
    zz = z.view(-1, 8, z.shape[-1])
    a, b = F.normalize(zz[:, :4].mean(1), dim=-1), F.normalize(zz[:, 4:].mean(1), dim=-1)
    half = (a @ b.T).argmax(1) == torch.arange(len(zz), device=z.device)
    return {"val_supcon": float(supcon(z, lab, tau)), "top1_win": float((lab[sim.argmax(1)] == lab).float().mean()),
            "top1_half": float(half.float().mean()), "chance": 1.0 / len(zz)}


@torch.no_grad()
def build_memory(key, gen, d, cfg, dev, amp) -> dict:
    key.eval()
    kind, stride, P, L = cfg["key"], int(cfg["stride"]), d.P, d.L
    query_paths, query_z, clip_z, paths, shift, clip_spk, names = [], [], [], [], [], [], {}
    win_clip, win_start, means = [], [], []
    for ds, retrievable in ((d.train_ds, True), (d.val_ds, False)):
        for j, r in enumerate(ds.rows):
            y, refs, audio = ds.clip(j)
            k = len(y) // 2  # the clip's photo frame
            with torch.autocast(**amp):
                z = embed(key, gen, d.latent.encode((y - y[k]).to(dev), d.k), audio.to(dev), refs[k].to(dev), kind,
                          P, L, stride)
            if z is None:
                continue
            query_paths.append(r["path"])
            query_z.append(z.float().cpu())
            if not retrievable:
                continue
            starts = range(0, len(y) - L + 1, stride)
            win_clip += [len(paths)] * len(starts)
            win_start += list(starts)
            means.append(torch.stack([audio[2 * s:2 * (s + L)].float().mean(0) for s in starts]))
            paths.append(r["path"])
            shift.append(ds.shifts.get(clip_key(r), 0))
            clip_spk.append(names.setdefault(Path(r["path"]).parent.name, len(names)))
            clip_z.append(query_z[-1])
    spk = torch.tensor(clip_spk)
    cz = torch.stack(clip_z)
    spk_z = F.normalize(torch.zeros(len(names), cz.shape[1]).index_add_(0, spk, cz), dim=-1)
    mt = torch.cat(means)
    mu = mt.mean(0)
    sub = mt[torch.randperm(len(mt), generator=torch.Generator().manual_seed(0))[:50000]] - mu
    proj = torch.pca_lowrank(sub, q=int(cfg["audio_dims"]), center=False)[2]
    return {"L": L, "paths": paths, "shift": torch.tensor(shift), "clip_spk": spk, "spk_names": list(names),
            "spk_z": spk_z, "win_clip": torch.tensor(win_clip), "win_start": torch.tensor(win_start),
            "win_key": F.normalize((mt - mu) @ proj, dim=-1).half(), "a_mu": mu, "a_proj": proj,
            "query_paths": query_paths, "query_z": torch.stack(query_z).half(), "key": kind,
            "key_cfg": key.cfg, "key_state": key.state_dict(), "gen": cfg["gen"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs/train_style.yaml"))
    ap.add_argument("overrides", nargs="*", help="key=value, parsed as YAML")
    args = ap.parse_args()
    rcfg = load_cfg(args.config, args.overrides)
    if rcfg["key"] not in ("residual", "motion"):
        raise SystemExit("key=residual or key=motion")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(rcfg["gen"], map_location="cpu", weights_only=False)
    cfg = {**ck["cfg"], **rcfg}
    if dev == "cpu":
        cfg["bf16"] = False
    torch.manual_seed(cfg["seed"])
    random.seed(cfg["seed"])
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    d = setup_data({**cfg, "demo_tokens": 0, "memory": None}, dev)
    gen = build(ck["cfg"]).to(dev).eval().requires_grad_(False)
    gen.load_state_dict(ck["ema"])
    key = StyleKey(d.k, int(cfg["dim"]), int(cfg["out"]), int(cfg["heads"])).to(dev)
    opt = torch.optim.AdamW(key.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    amp = dict(device_type=dev, dtype=torch.bfloat16, enabled=cfg["bf16"])
    K, tau = int(cfg["K"]), float(cfg["tau"])
    batches = forever(DataLoader(Speakers(d.train_ds, K), batch_size=cfg["persons"], shuffle=True,
                                 num_workers=cfg["workers"], drop_last=True, pin_memory=dev == "cuda",
                                 persistent_workers=cfg["workers"] > 0))
    vset = Speakers(d.val_ds, 8, fixed=True)
    vdl = DataLoader(Subset(vset, range(len(vset.groups))), batch_size=cfg["persons"],
                     num_workers=min(2, cfg["workers"]))
    print(f"style key ({cfg['key']}): {sum(p.numel() for p in key.parameters()) / 1e6:.2f} M params, "
          f"{len(Speakers(d.train_ds, K).groups)} train / {len(vset.groups)} val speakers, {cfg['persons']} x {K} "
          f"windows a batch, residuals from {cfg['gen']} (step {ck['step']})", flush=True)
    print("EVAL step 0 " + " ".join(f"{k} {v:.4f}" for k, v in evaluate(key, gen, vdl, d, dev, cfg["key"], amp, tau).items()),
          flush=True)
    t0 = time.time()
    for step in range(1, int(cfg["max_steps"]) + 1):
        for gr in opt.param_groups:
            gr["lr"] = lr_at(step, cfg)
        z, lab = keys(key, gen, next(batches), d, dev, cfg["key"], amp)
        loss = supcon(z, lab, tau)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(key.parameters(), cfg["grad_clip"])
        opt.step()
        if step % cfg["log_every"] == 0:
            print(f"step {step:>6} supcon {float(loss):.4f} gnorm {float(gnorm):.3f} "
                  f"{cfg['log_every'] / (time.time() - t0):.1f} it/s", flush=True)
            t0 = time.time()
        if step % cfg["eval_every"] == 0 or step == cfg["max_steps"]:
            r = evaluate(key, gen, vdl, d, dev, cfg["key"], amp, tau)
            print(f"EVAL step {step} " + " ".join(f"{k} {v:.4f}" for k, v in r.items()), flush=True)
    t0 = time.time()
    mem = build_memory(key, gen, d, cfg, dev, amp)
    torch.save(mem, out / "memory.pt")
    print(f"memory -> {out / 'memory.pt'}: {len(mem['spk_names'])} speakers, {len(mem['paths'])} clips, "
          f"{len(mem['win_key'])} windows of {mem['L']} frames, {len(mem['query_paths'])} clip styles, "
          f"{time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
