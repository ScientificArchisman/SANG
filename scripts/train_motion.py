#!/usr/bin/env python3
"""Train the bidirectional MotionFlowTransformer on the motion cache (M1).

    sbatch bash_scripts/train_motion.sh                      # configs/train_motion.yaml
    sbatch bash_scripts/train_motion.sh --config configs/train_motion.yaml resume=runs/motion_bidir/last.pt

Validation is in motion space only -- no renderer in the loop -- and reports the two failure modes
this architecture is most exposed to, using Motar's (arXiv:2609.10317) rollout ratios, ideal 1.0:
  std_r  generated / GT temporal std     -> << 1 is the static, mean-seeking collapse
  vc_r   generated / GT frame-diff std   -> >> 1 is jitter, << 1 is frozen
plus audio_gain: mouth error with a DIFFERENT clip's audio over mouth error with the right audio.
~1.0 means the model ignores audio -- which is exactly how v3 failed (2.7% sensitivity).
HDTF FID/FVD/LSE are measured by rendering; see scripts/infer_motion.py.
"""
import argparse
import copy
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Dataset

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from sang.motion import REGIONS, to_target
from sang.motion_model import Norm, build, flow_loss, sample

TPF = 2


# ---------------------------------------------------------------------- data
def load_index(cache_dir: Path) -> list[dict]:
    rows = []
    for f in sorted(cache_dir.glob("index_*.jsonl")):
        rows += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    if not rows:
        raise SystemExit(f"no index_*.jsonl in {cache_dir}; run scripts/cache_motion.py first")
    return rows


def split_by_speaker(rows: list[dict], val_frac: float, seed: int):
    """Same rule as scripts/train.py: the speaker is the clip's parent directory, so val measures
    unseen identities, not unseen clips of seen people."""
    spk = sorted({Path(r["path"]).parent.name for r in rows})
    random.Random(seed).shuffle(spk)
    val = set(spk[: max(1, int(len(spk) * val_frac))])
    return ([r for r in rows if Path(r["path"]).parent.name not in val],
            [r for r in rows if Path(r["path"]).parent.name in val])


def fit_norm(rows: list[dict], n_clips: int = 500, seed: int = 0) -> Norm:
    pick = random.Random(seed).sample(rows, min(n_clips, len(rows)))
    ys, kps = [], []
    for r in pick:
        d = torch.load(r["path"], map_location="cpu", weights_only=True)
        ys.append(to_target(d["m"].float()))
        kps.append(d["kp"].float())
    return Norm.fit(torch.cat(ys), torch.cat(kps))


class MotionWindows(Dataset):
    """Random windows of P prefix + L target frames. The reference is a random frame of the same
    clip (Ditto's choice), which at inference becomes the source image.

    preload=True reads every clip once at start-up and keeps the normalised 42-d targets, reference
    features and fp16 audio in RAM (~2 MB per 20 s clip). Reading one .pt per SAMPLE from BeeGFS
    capped the first run at ~1.4 it/s: 256 network file opens per step."""

    def __init__(self, rows, L: int, P: int, norm: Norm, per_clip: int, fixed: bool = False,
                 preload: bool = True, threads: int = 16):
        self.rows = [r for r in rows if r["n"] >= L + P]
        self.L, self.P, self.norm, self.per_clip = L, P, norm, per_clip
        self.fixed = fixed                # val: the same windows every eval, so curves are comparable
        if not self.rows:
            raise ValueError(f"no clip has {L + P} frames")
        self.mem = None
        if preload:
            from concurrent.futures import ThreadPoolExecutor
            t0 = time.time()
            with ThreadPoolExecutor(threads) as ex:          # I/O-bound: threads hide network latency
                self.mem = list(ex.map(self._load, self.rows))
            gb = sum(a.numel() * a.element_size() for _, _, a in self.mem) / 1e9
            print(f"  preloaded {len(self.mem)} clips in {time.time() - t0:.0f} s, audio {gb:.1f} GB", flush=True)

    def _load(self, r):
        d = torch.load(r["path"], map_location="cpu", weights_only=True)
        y = self.norm.target(to_target(d["m"].float()))                  # [n, 42]  normalised
        ref = self.norm.ref(d["kp"].float(), to_target(d["m"].float()))  # [n, REF_DIM] one per frame
        return y, ref, d["audio"].half().contiguous()

    def __len__(self):
        return len(self.rows) * self.per_clip

    def __getitem__(self, i):
        j = i % len(self.rows)
        y_all, ref_all, audio = self.mem[j] if self.mem is not None else self._load(self.rows[j])
        n, span = y_all.shape[0], self.L + self.P
        rng = random.Random(i) if self.fixed else random
        s = rng.randint(0, n - span)
        k = rng.randint(0, n - 1)
        y = y_all[s:s + span]
        return {"prefix": y[: self.P], "target": y[self.P:], "ref": ref_all[k],
                "audio": audio[TPF * s: TPF * (s + span)]}          # fp16: half the bytes per batch; the model casts


# ---------------------------------------------------------------------- eval
@torch.no_grad()
def evaluate(model, loader, cfg, dev, max_batches: int) -> dict:
    model.eval()
    g = torch.Generator(device=dev).manual_seed(0)
    fm = n = 0.0
    stats = {k: [] for k in ("mse_mouth", "mse_rot", "std_r_mouth", "vc_r_mouth",
                             "std_r_rot", "vc_r_rot", "audio_gain")}
    mouth, rot = REGIONS["mouth"], REGIONS["rot"]

    def ratio(a, b):
        return float(a.std(1).mean() / b.std(1).mean().clamp_min(1e-6))

    for bi, batch in enumerate(loader):
        if bi >= max_batches:
            break
        batch = {k: v.to(dev) for k, v in batch.items()}
        torch.manual_seed(bi)                              # same t / noise every eval
        loss, _ = flow_loss(model, batch, lam_vel=cfg["lam_vel"], p_audio=0, p_ref=0, p_prefix=0,
                            t_dist=cfg["t_dist"])
        fm += float(loss)
        n += 1
        gt, L = batch["target"], batch["target"].shape[1]
        kw = dict(prefix=batch["prefix"], steps=cfg["sample_steps"], cfg_audio=cfg["cfg_audio"])
        g.manual_seed(bi)
        gen = sample(model, batch["audio"], batch["ref"], L, generator=g, **kw)
        g.manual_seed(bi)                                  # identical noise, someone else's audio
        other = sample(model, batch["audio"].roll(1, 0), batch["ref"], L, generator=g, **kw)
        e = (gen - gt).pow(2)
        stats["mse_mouth"].append(float(e[..., mouth].mean()))
        stats["mse_rot"].append(float(e[..., rot].mean()))
        for name, idx in (("mouth", mouth), ("rot", rot)):
            stats[f"std_r_{name}"].append(ratio(gen[..., idx], gt[..., idx]))
            stats[f"vc_r_{name}"].append(ratio(gen[:, 1:, idx] - gen[:, :-1, idx],
                                               gt[:, 1:, idx] - gt[:, :-1, idx]))
        stats["audio_gain"].append(float((other - gt)[..., mouth].pow(2).mean() / e[..., mouth].mean()))
    model.train()
    return {"val_loss": fm / max(1, n), **{k: sum(v) / max(1, len(v)) for k, v in stats.items()}}


# ---------------------------------------------------------------------- train
def lr_at(step, cfg):
    if step < cfg["warmup_steps"]:
        return cfg["lr"] * (step + 1) / cfg["warmup_steps"]
    p = min(1.0, (step - cfg["warmup_steps"]) / max(1, cfg["max_steps"] - cfg["warmup_steps"]))
    return cfg["min_lr"] + 0.5 * (cfg["lr"] - cfg["min_lr"]) * (1 + math.cos(math.pi * p))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs/train_motion.yaml"))
    ap.add_argument("overrides", nargs="*", help="key=value, parsed as YAML")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    for o in args.overrides:
        k, v = o.split("=", 1)
        val = yaml.safe_load(v)
        if isinstance(val, str):                  # YAML 1.1 reads "1e-4" as a string, not a float
            try:
                val = float(val)
            except ValueError:
                pass
        cfg[k] = val

    torch.manual_seed(cfg["seed"])
    random.seed(cfg["seed"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu":
        cfg["bf16"] = False                                # smoke tests only
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(cfg["cache_dir"])

    rows = load_index(cache)
    train_rows, val_rows = split_by_speaker(rows, cfg["val_frac"], cfg["seed"])
    norm_path = cache / "norm42.pt"
    if norm_path.exists():
        norm = Norm(**torch.load(norm_path, weights_only=True))
    else:
        norm = fit_norm(train_rows, seed=cfg["seed"])       # train speakers only: no val leakage
        torch.save({k: v for k, v in norm.state_dict().items()}, norm_path)
    L, P = cfg["frames"], cfg["prefix"]
    pre = cfg.get("preload", True)
    train_ds = MotionWindows(train_rows, L, P, norm, cfg["windows_per_clip"], preload=pre)
    val_ds = MotionWindows(val_rows, L, P, norm, 1, fixed=True, preload=pre)
    print(f"{len(train_ds.rows)} train / {len(val_ds.rows)} val clips "
          f"({len({Path(r['path']).parent.name for r in val_rows})} unseen val speakers)", flush=True)
    dl = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["workers"],
                    drop_last=True, pin_memory=dev == "cuda", persistent_workers=cfg["workers"] > 0)
    vdl = DataLoader(val_ds, batch_size=min(cfg["batch_size"], 64), shuffle=False,
                     num_workers=min(2, cfg["workers"]), drop_last=True)

    model = build(cfg).to(dev)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    decay = [p for n_, p in model.named_parameters() if p.ndim >= 2 and "pos" not in n_]
    no_decay = [p for n_, p in model.named_parameters() if p.ndim < 2 or "pos" in n_]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg["weight_decay"]},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=cfg["lr"], betas=tuple(cfg["betas"]))
    print(f"MotionFlowTransformer: {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M params", flush=True)

    step, best = 0, float("inf")
    if cfg.get("resume"):
        ck = torch.load(cfg["resume"], map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        step, best = ck["step"], ck.get("best", best)
        print(f"resumed at step {step}", flush=True)

    kw = {k: cfg[k] for k in ("lam_vel", "p_audio", "p_ref", "p_prefix", "t_dist")}
    t0 = time.time()
    t_data = t_gpu = 0.0                                   # seconds per logging window: waiting vs computing
    t_prev = time.perf_counter()
    while step < cfg["max_steps"]:
        for batch in dl:
            t_got = time.perf_counter()
            t_data += t_got - t_prev
            batch = {k: v.to(dev, non_blocking=True) for k, v in batch.items()}
            for gr in opt.param_groups:
                gr["lr"] = lr_at(step, cfg)
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=cfg["bf16"]):
                loss, parts = flow_loss(model, batch, **kw)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.lerp_(pm, 1.0 - cfg["ema"])
            step += 1
            if dev == "cuda":
                torch.cuda.synchronize()                   # so compute time is not billed to data
            t_gpu += time.perf_counter() - t_got

            if step % cfg["log_every"] == 0:
                sps = cfg["log_every"] / (time.time() - t0)
                t0 = time.time()
                msg = " ".join(f"{k} {float(v):.4f}" for k, v in parts.items())
                n_ = cfg["log_every"]
                print(f"step {step:>7} lr {opt.param_groups[0]['lr']:.2e} {msg} gnorm {float(gnorm):.3f} "
                      f"{sps:.1f} it/s  (data {t_data / n_:.3f} s + gpu {t_gpu / n_:.3f} s per step)", flush=True)
                t_data = t_gpu = 0.0

            if step % cfg["eval_every"] == 0 or step == cfg["max_steps"]:
                r = evaluate(ema, vdl, cfg, dev, cfg["eval_batches"])
                print("EVAL step {} ".format(step) + " ".join(f"{k} {v:.4f}" for k, v in r.items()), flush=True)
                ck = {"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                      "step": step, "best": best, "cfg": cfg, "norm": norm.state_dict()}
                torch.save(ck, out / "last.pt")
                if r["val_loss"] < best:
                    best = ck["best"] = r["val_loss"]
                    torch.save(ck, out / "best.pt")
                    (out / "best.json").write_text(json.dumps({"step": step, **r}, indent=1))
            t_prev = time.perf_counter()                   # eval and logging are not data wait
            if step >= cfg["max_steps"]:
                break
    print(f"done. best val_loss {best:.4f} -> {out / 'best.pt'}", flush=True)


if __name__ == "__main__":
    main()
