#!/usr/bin/env python3
"""Train the bidirectional MotionFlowTransformer on the motion cache.

    sbatch bash_scripts/run.sh scripts/train_motion.py out_dir=runs/<name> [key=value ...]   # configs/train_motion.yaml

The model generates the first latent_k codes of the motion latent at `latent` (scripts/fit_latent.py): everything
LivePortrait renders, relative to the photo. Guidance-Free Training: guidance is an input, one pass per sampling step.
  photo=wide            describe the photo by its 69 wide numbers instead of its 42-d target
  code_weight=spread    weight each code's loss by its variance: the MSE of the decoded motion in standardised units

Validation is in motion space only -- no renderer in the loop -- and reports the two failure modes
this architecture is most exposed to, using Motar's (arXiv:2609.10317) rollout ratios, ideal 1.0:
  std_r  generated / GT temporal std     -> << 1 is the static, mean-seeking collapse
  vc_r   generated / GT frame-diff std   -> >> 1 is jitter, << 1 is frozen
plus audio_gain: mouth error with a DIFFERENT clip's audio over mouth error with the right audio
(~1.0 = the model ignores audio). All are on the decoded 69-d motion.
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
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from sang.data.dataset import MotionWindows, fit_norm, load_cfg, split_by_speaker
from sang.data.paths import load_index
from sang.data.sync import SyncOffsets, clip_key
from sang.data.talkvid import clip_languages, filter_langs, parse_langs
from sang.model.flow import PHOTO_DIM, Norm, build, flow_loss, sample
from sang.model.latent import MotionLatent
from sang.model.target import WIDE_REGIONS


@torch.no_grad()
def evaluate(model, loader, cfg, dev, max_batches: int, encode, decode, w=None) -> dict:
    """Val flow loss, plus rollout statistics of sampled windows in the decoded motion's mouth, rot and place."""
    model.eval()
    g = torch.Generator(device=dev).manual_seed(0)
    fm = n = 0.0
    parts = {k: WIDE_REGIONS[k] for k in ("mouth", "rot", "place")}
    stats = {k: [] for p in parts for k in (f"mse_{p}", f"std_r_{p}", f"vc_r_{p}")}
    stats["audio_gain"] = []

    def ratio(a, b):
        return float(a.std(1).mean() / b.std(1).mean().clamp_min(1e-6))

    for bi, batch in enumerate(loader):
        if bi >= max_batches:
            break
        batch = encode({k: v.to(dev) for k, v in batch.items()})
        torch.manual_seed(bi)
        loss, _ = flow_loss(model, batch, lam_vel=cfg["lam_vel"], p_audio=0, p_ref=0, p_prefix=0, t_dist="uniform", w=w)
        fm += float(loss)
        n += 1
        L = batch["target"].shape[1]
        kw = dict(prefix=batch["prefix"], steps=cfg["sample_steps"], cfg_audio=cfg["cfg_audio"])
        g.manual_seed(bi)
        gen = decode(sample(model, batch["audio"], batch["ref"], L, generator=g, **kw))
        g.manual_seed(bi)
        other = decode(sample(model, batch["audio"].roll(1, 0), batch["ref"], L, generator=g, **kw))
        gt = decode(batch["target"])
        e = (gen - gt).pow(2)
        for name, idx in parts.items():
            stats[f"mse_{name}"].append(float(e[..., idx].mean()))
            stats[f"std_r_{name}"].append(ratio(gen[..., idx], gt[..., idx]))
            stats[f"vc_r_{name}"].append(ratio(gen[:, 1:, idx] - gen[:, :-1, idx], gt[:, 1:, idx] - gt[:, :-1, idx]))
        mouth = parts["mouth"]
        stats["audio_gain"].append(float((other - gt)[..., mouth].pow(2).mean() / e[..., mouth].mean()))
    model.train()
    return {"val_loss": fm / max(1, n), **{k: sum(v) / max(1, len(v)) for k, v in stats.items()}}


def lr_at(step, cfg):
    if step < cfg["warmup_steps"]:
        return cfg["lr"] * (step + 1) / cfg["warmup_steps"]
    p = min(1.0, (step - cfg["warmup_steps"]) / max(1, cfg["max_steps"] - cfg["warmup_steps"]))
    return cfg["min_lr"] + 0.5 * (cfg["lr"] - cfg["min_lr"]) * (1 + math.cos(math.pi * p))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs/train_motion.yaml"))
    ap.add_argument("--lang", nargs="+", default=None,
                    help="train AND validate on these languages only: TalkVid names or Whisper codes, e.g. "
                         "--lang English, --lang en zh; 'all' (default) = every language")
    ap.add_argument("--train-clips", type=int, default=None,
                    help="random subset of N training clips after the language filter (seeded)")
    ap.add_argument("overrides", nargs="*", help="key=value, parsed as YAML")
    args = ap.parse_args()
    if args.lang:
        args.overrides = [v for v in args.lang if "=" in v] + args.overrides
        args.lang = [v for v in args.lang if "=" not in v] or None
    cfg = load_cfg(args.config, args.overrides)
    if args.lang is not None:
        cfg["lang"] = args.lang
    if args.train_clips is not None:
        cfg["train_clips"] = args.train_clips
    langs = parse_langs(cfg.get("lang"))
    cfg["lang"] = langs or "all"

    init = cfg["seed"] if cfg.get("init_seed") is None else int(cfg["init_seed"])
    torch.manual_seed(init)
    random.seed(init)
    if init != cfg["seed"]:
        print(f"init seed {init} (split seed {cfg['seed']})", flush=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu":
        cfg["bf16"] = False
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(cfg["cache_dir"])

    rows = load_index(cache)
    train_rows, val_rows = split_by_speaker(rows, cfg["val_frac"], cfg["seed"])
    if langs is not None:
        lang_of = clip_languages(cache)
        n_t, n_v = len(train_rows), len(val_rows)
        train_rows, drop_t = filter_langs(train_rows, langs, lang_of)
        val_rows, drop_v = filter_langs(val_rows, langs, lang_of)
        have = {lang_of.get(clip_key(r)) for r in rows}
        missing = [l for l in langs if l not in have]
        if missing or not train_rows or not val_rows:
            raise SystemExit(f"--lang {langs}: no clips for {missing or langs} "
                             f"(languages in the cache: {sorted(l for l in have if l)})")
        print(f"languages {langs}: train {n_t} -> {len(train_rows)} clips, val {n_v} -> {len(val_rows)} clips; "
              f"dropped train {drop_t}, val {drop_v}", flush=True)
    width = PHOTO_DIM[cfg["photo"]]
    norm_path = cache / f"norm{width}.pt"
    if norm_path.exists():
        norm = Norm(**torch.load(norm_path, weights_only=True))
    else:
        norm = fit_norm(train_rows, width, seed=cfg["seed"])
        torch.save(dict(norm.state_dict()), norm_path)
    if not Path(cfg["latent"]).exists():
        raise SystemExit(f"no motion latent at {cfg['latent']}: run scripts/fit_latent.py first")
    latent = MotionLatent.unpack(torch.load(cfg["latent"], map_location="cpu", weights_only=False)).to(dev)
    k = int(cfg["latent_k"])
    if not 0 < k <= latent.K:
        raise SystemExit(f"latent_k must be in 1..{latent.K}")
    print(f"motion latent {cfg['latent']} (PCA, K {latent.K}): the model generates its first "
              f"{k} codes", flush=True)
    shifts = None
    if cfg.get("sync_offsets"):
        so = SyncOffsets.load(cfg["sync_offsets"])
        train_rows, why_t = so.filter(train_rows)
        val_rows, why_v = so.filter(val_rows)
        shifts = {clip_key(r): so.shift(r) for r in train_rows + val_rows}
        print(f"sync offsets {cfg['sync_offsets']}: audio moved in {sum(s != 0 for s in shifts.values())}/"
              f"{len(shifts)} clips; dropped train {why_t}, val {why_v}", flush=True)
    if cfg.get("train_clips"):
        n_t = len(train_rows)
        train_rows = sorted(random.Random(cfg["seed"]).sample(train_rows, min(int(cfg["train_clips"]), n_t)),
                            key=lambda r: r["path"])
        print(f"train_clips={cfg['train_clips']}: random subset {n_t} -> {len(train_rows)} train clips", flush=True)
    L, P = cfg["frames"], cfg["prefix"]
    data_kw = {"preload": cfg.get("preload", True), "shifts": shifts,
               "ref_frame": cfg.get("ref_frame", "random"), "ref_band": cfg.get("ref_band", 0.2)}
    if data_kw["ref_frame"] == "median":
        from sang.eval.naturalness import load_readouts
        data_kw["lip"] = load_readouts(cache / "openness.json")["lip"]
        print(f"reference frame: a typical one per clip (middle {data_kw['ref_band']:.0%} by lip opening)", flush=True)
    train_ds = MotionWindows(train_rows, L, P, norm, cfg["windows_per_clip"], **data_kw)
    val_ds = MotionWindows(val_rows, L, P, norm, cfg.get("val_windows_per_clip", 4), fixed=True, **data_kw)
    print(f"{len(train_ds.rows)} train / {len(val_ds.rows)} val clips "
          f"({len({Path(r['path']).parent.name for r in val_rows})} unseen val speakers)", flush=True)
    dl = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["workers"],
                    drop_last=True, pin_memory=dev == "cuda", persistent_workers=cfg["workers"] > 0)
    vdl = DataLoader(val_ds, batch_size=min(cfg["batch_size"], 64), shuffle=False,
                     num_workers=min(2, cfg["workers"]), drop_last=True)
    tdl = None
    if cfg.get("train_eval", False):
        sub = random.Random(cfg["seed"]).sample(train_ds.rows, min(len(val_ds.rows), len(train_ds.rows)))
        tev_ds = MotionWindows(sorted(sub, key=lambda r: r["path"]), L, P, norm, cfg.get("val_windows_per_clip", 4),
                               fixed=True, **data_kw)
        tdl = DataLoader(tev_ds, batch_size=min(cfg["batch_size"], 64), shuffle=False,
                         num_workers=min(2, cfg["workers"]), drop_last=True)
        print(f"train_eval: {len(tev_ds.rows)} fixed training clips scored like val at every eval", flush=True)

    encode = lambda b: {**b, "target": latent.encode(b["target"], k), "prefix": latent.encode(b["prefix"], k)}
    decode = latent.decode
    if cfg["code_weight"] not in ("equal", "spread"):
        raise SystemExit(f"code_weight must be equal or spread, got {cfg['code_weight']!r}")
    w = latent.spread(k) if cfg["code_weight"] == "spread" else None

    model = build(cfg).to(dev)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    decay = [p for n_, p in model.named_parameters() if p.ndim >= 2 and "pos" not in n_]
    no_decay = [p for n_, p in model.named_parameters() if p.ndim < 2 or "pos" in n_]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg["weight_decay"]},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=cfg["lr"], betas=tuple(cfg["betas"]))
    print(f"MotionFlowTransformer: {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M params, target "
          f"{model.t_dim}-d, GFT, photo {cfg['photo']} ({width}), code weights {cfg['code_weight']}", flush=True)

    step, best, bad = 0, float("inf"), 0
    if cfg.get("resume"):
        ck = torch.load(cfg["resume"], map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        step, best = ck["step"], ck.get("best", best)
        print(f"resumed at step {step}", flush=True)

    kw = {k_: cfg[k_] for k_ in ("lam_vel", "p_audio", "p_ref", "p_prefix", "t_dist")}
    kw.update(t_mean=cfg.get("t_mean", 0.0), t_scale=cfg.get("t_scale", 1.0), w=w)
    print(f"objective: t_dist {kw['t_dist']} (mean {kw['t_mean']}, scale {kw['t_scale']})", flush=True)
    t0 = time.time()
    t_data = t_gpu = 0.0
    t_prev = time.perf_counter()
    while step < cfg["max_steps"]:
        for batch in dl:
            t_got = time.perf_counter()
            t_data += t_got - t_prev
            batch = encode({k_: v.to(dev, non_blocking=True) for k_, v in batch.items()})
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
                torch.cuda.synchronize()
            t_gpu += time.perf_counter() - t_got

            if step % cfg["log_every"] == 0:
                sps = cfg["log_every"] / (time.time() - t0)
                t0 = time.time()
                msg = " ".join(f"{k_} {float(v):.4f}" for k_, v in parts.items())
                n_ = cfg["log_every"]
                print(f"step {step:>7} lr {opt.param_groups[0]['lr']:.2e} {msg} gnorm {float(gnorm):.3f} "
                      f"{sps:.1f} it/s  (data {t_data / n_:.3f} s + gpu {t_gpu / n_:.3f} s per step)", flush=True)
                t_data = t_gpu = 0.0

            if step % cfg["eval_every"] == 0 or step == cfg["max_steps"]:
                r = evaluate(ema, vdl, cfg, dev, cfg["eval_batches"], encode, decode, w)
                print("EVAL step {} ".format(step) + " ".join(f"{k_} {v:.4f}" for k_, v in r.items()), flush=True)
                if tdl is not None:
                    with torch.random.fork_rng(devices=[torch.cuda.current_device()] if dev == "cuda" else []):
                        rt = evaluate(ema, tdl, cfg, dev, cfg["eval_batches"], encode, decode, w)
                    print("EVAL_TRAIN step {} ".format(step) + " ".join(f"{k_.removeprefix('val_')} {v:.4f}"
                                                                      for k_, v in rt.items()), flush=True)
                ck = {"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                      "step": step, "best": best, "cfg": cfg, "norm": norm.state_dict(),
                      "latent": latent.pack()}
                torch.save(ck, out / "last.pt")
                if r["val_loss"] < best:
                    best = ck["best"] = r["val_loss"]
                    bad = 0
                    torch.save(ck, out / "best.pt")
                    (out / "best.json").write_text(json.dumps({"step": step, **r}, indent=1))
                else:
                    bad += 1
                if cfg.get("patience", 0) and bad >= cfg["patience"]:
                    print(f"early stop at step {step}: no val improvement for {bad} evals (best {best:.4f})", flush=True)
                    step = cfg["max_steps"]
            t_prev = time.perf_counter()
            if step >= cfg["max_steps"]:
                break
    print(f"done. best val_loss {best:.4f} -> {out / 'best.pt'}", flush=True)


if __name__ == "__main__":
    main()

