#!/usr/bin/env python3
"""C2.0 go/no-go (no SANG training): does SANG's mouth carry as much decodable speech as a real mouth?
Definitions in sang/lip_expert.py. ~30 min on one GPU (reading the training cache dominates).

    sbatch bash_scripts/job.sh scripts/lip_expert.py --ckpt runs/b1_lognorm03/last.pt \\
        --sync-offsets <cache>/sync_offsets.json

1. read the training clips (audio moved by their SyncNet offsets): mouth columns of the normalised
   target (the checkpoint's Norm) and frame-level WavLM features; PCA of the features
2. train the expert on random 64-frame windows of 95 % of the training videos; the other 5 % monitor it
3. on the val clips C0 / C1 score: R^2 on real motion, on SANG's motion (default sampler, one sample)
   and on real motion paired with another clip's audio (chance); per clip with 95 % intervals over videos
The expert, the PCA and the columns are saved to <out>/expert.pt for C2.1 (the training loss).
"""
import argparse
import importlib.util
import json
import random
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from sang.coartic import cluster_ci, video_of
from sang.lip_expert import LipExpert, centre, clip_r2, fit_pca, frame_audio, project, r2, verdict
from sang.motion import REGIONS, to_target
from sang.motion_model import Norm, build, ema_weights, parse_spec, sampler_kwargs
from sang.naturalness import guided_generate
from sang.paths import motion_cache
from sang.sync import clip_key, shift_ticks


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_clip(r: dict, so, norm, cols, max_frames: int):
    """-> (raw 42-d target [n, 42], normalised motion columns [m, d], frame WavLM [m, 1024], WavLM ticks)."""
    d = torch.load(r["path"], map_location="cpu", weights_only=True)
    n = min(max_frames, int(d["n"]))
    s = so.shift(r) if so is not None else 0
    ticks = shift_ticks(d["audio"][: 2 * n].float(), s)
    fa = frame_audio(ticks, n)
    y = to_target(d["m"][:n].float())
    mot = norm.target(y[: len(fa)])[:, cols]
    return y, d["kp"][:1].float(), mot.numpy(), fa.numpy(), ticks


@torch.no_grad()
def predict(expert, x: np.ndarray, dev: str) -> np.ndarray:
    return expert(torch.from_numpy(centre(x))[None].to(dev))[0].cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(REPO / "runs/b1_lognorm03/last.pt"))
    ap.add_argument("--sampler", default="g=2,mouth=1.25,tau=0.5,avg=4,steps=6")
    ap.add_argument("--sync-offsets", default=None, help="<cache>/sync_offsets.json (as for naturalness.py)")
    ap.add_argument("--dims", default="mouth", choices=["mouth", "all"], help="motion columns the expert reads")
    ap.add_argument("--k", type=int, default=64, help="WavLM principal components the expert predicts")
    ap.add_argument("--n", type=int, default=300, help="val clips")
    ap.add_argument("--train-clips", type=int, default=0, help="0 = every training clip")
    ap.add_argument("--max-frames", type=int, default=250, help="val clips (as C0 / C1)")
    ap.add_argument("--max-train-frames", type=int, default=500)
    ap.add_argument("--min-frames", type=int, default=125)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min-r2", type=float, default=0.1, help="WEAK EXPERT below this R^2 on real val motion")
    ap.add_argument("--min-headroom", type=float, default=0.15, help="GO if (R2_real - R2_gen) / R2_real reaches this")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=str(REPO / "runs/lip_expert"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    dev = args.device
    es = load_script("error_split")
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    cfg = ck["cfg"]
    model = build(cfg).to(dev).eval()
    model.load_state_dict(ema_weights(ck, None))
    norm = Norm(**ck["norm"]).to(dev)
    norm_cpu = Norm(**ck["norm"]).cpu()                     # ck was loaded onto the GPU
    samp = sampler_kwargs(parse_spec(args.sampler), cfg, None)
    cache = motion_cache(cfg["cache_dir"])
    cols = REGIONS["mouth"] if args.dims == "mouth" else list(range(42))
    train, val, so, groups_of = es.aligned_splits(cfg, cache, args.sync_offsets, None, args.min_frames)
    val = val[: args.n]
    if args.train_clips:
        train = rng.sample(train, min(args.train_clips, len(train)))
    print(f"{len(train)} training clips, {len(val)} val clips; expert reads {len(cols)} {args.dims} columns, "
          f"predicts {args.k} WavLM components; SANG {args.ckpt} (step {ck['step']}), sampler {args.sampler}", flush=True)

    # ---------------- 1. training data
    t0 = time.time()
    clips = []
    for i, r in enumerate(train, 1):
        try:
            _, _, mot, fa, _ = read_clip(r, so, norm_cpu, cols, args.max_train_frames)
            if len(fa) >= args.window:
                clips.append({"video": video_of(clip_key(r)), "mot": mot.astype(np.float32), "fa": fa.astype(np.float16)})
        except Exception as e:                                   # one bad clip never kills the run
            print(f"  skip {Path(r['clip']).name}: {type(e).__name__}: {e}", flush=True)
        if i % 2000 == 0:
            print(f"  read {i}/{len(train)} training clips, {time.time() - t0:.0f} s", flush=True)
    held = [c for c in clips if zlib.crc32(c["video"].encode()) % 20 == 0]
    fit = [c for c in clips if zlib.crc32(c["video"].encode()) % 20 != 0]
    frames = np.concatenate([c["fa"] for c in rng.sample(fit, min(800, len(fit)))]).astype(np.float32)
    pca = fit_pca(frames[np.random.default_rng(args.seed).permutation(len(frames))[:200_000]], args.k)
    for c in clips:
        c["tgt"] = project(c.pop("fa"), pca)
    scale = float(np.sqrt(np.mean(np.concatenate([centre(c["tgt"]).ravel() ** 2 for c in fit[:2000]]))))
    for c in clips:
        c["tgt"] /= scale
    print(f"{len(fit)} clips train the expert, {len(held)} monitor it; {args.k} components hold "
          f"{100 * pca['explained']:.1f} % of the WavLM variance; {time.time() - t0:.0f} s", flush=True)

    # ---------------- 2. train the expert
    expert = LipExpert(len(cols), args.k).to(dev)
    print(f"expert: {sum(p.numel() for p in expert.parameters()) / 1e6:.2f} M parameters", flush=True)
    opt = torch.optim.AdamW(expert.parameters(), lr=args.lr, weight_decay=1e-4)
    warm = max(1, args.steps // 20)                             # 5 % linear warm-up, then cosine to 0
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1.0, (i + 1) / warm) * 0.5 * (
        1 + np.cos(np.pi * min(1.0, max(0, i + 1 - warm) / max(1, args.steps - warm)))))
    L = args.window
    for step in range(1, args.steps + 1):
        xs, ts = [], []
        for c in rng.choices(fit, k=args.batch):
            s = rng.randrange(len(c["tgt"]) - L + 1)
            xs.append(centre(c["mot"][s:s + L]))
            ts.append(centre(c["tgt"][s:s + L]))
        x, t = torch.from_numpy(np.stack(xs)).to(dev), torch.from_numpy(np.stack(ts)).to(dev)
        loss = torch.nn.functional.mse_loss(expert(x), t)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 500 == 0 or step == args.steps:
            expert.eval()
            mon = (f"{r2([predict(expert, c['mot'], dev) for c in held], [centre(c['tgt']) for c in held]):.3f}"
                   if held else "n/a")
            expert.train()
            print(f"  step {step:>5}  loss {loss.item():.4f}  held-out training videos R^2 {mon}  "
                  f"{time.time() - t0:.0f} s", flush=True)
    expert.eval()

    # ---------------- 3. val: real vs SANG vs chance
    recs = []
    for i, r in enumerate(val, 1):
        try:
            y, kp0, mot, fa, ticks = read_clip(r, so, norm_cpu, cols, args.max_frames)
            m = len(fa)
            ref = norm.ref(kp0.to(dev), y[:1].to(dev))
            g, _ = guided_generate(model, ticks[None].to(dev), ref, len(y), cfg, norm, None, None, "none", args.seed, sampler=samp)
            gen = g[0, :m, cols].float().cpu().numpy()
            tgt = centre(project(fa, pca) / scale)
            recs.append({"clip": clip_key(r), "groups": sorted(groups_of(r)), "tgt": tgt,
                         "p_real": predict(expert, mot, dev), "p_gen": predict(expert, gen, dev),
                         "mot": mot})
        except Exception as e:
            print(f"  skip {Path(r['clip']).name}: {type(e).__name__}: {e}", flush=True)
        if i % 100 == 0:
            print(f"  val {i}/{len(val)} clips, {time.time() - t0:.0f} s", flush=True)
    for j, rec in enumerate(recs):                               # chance: real mouth, another clip's audio
        other = recs[(j + 1) % len(recs)]["tgt"]
        m = min(len(other), len(rec["mot"]))
        rec["p_chance"], rec["tgt_chance"] = predict(expert, rec["mot"][:m], dev), other[:m]
        for k in ("real", "gen"):
            rec[f"r2_{k}"] = clip_r2(rec[f"p_{k}"], rec["tgt"])
            rec[f"r2s_{k}"] = clip_r2(rec[f"p_{k}"], rec["tgt"], rescale=True)
        rec["r2_chance"] = clip_r2(rec["p_chance"], rec["tgt_chance"])

    def summary(rs: list[dict]) -> dict:
        if len(rs) < 4:
            return {"clips": len(rs)}
        vids = [video_of(x["clip"]) for x in rs]
        out = {"clips": len(rs), "videos": len(set(vids)),
               "r2_real": r2([x["p_real"] for x in rs], [x["tgt"] for x in rs]),
               "r2_gen": r2([x["p_gen"] for x in rs], [x["tgt"] for x in rs]),
               "r2_chance": r2([x["p_chance"] for x in rs], [x["tgt_chance"] for x in rs])}
        for k in ("r2_real", "r2_gen", "r2s_real", "r2s_gen", "r2_chance"):
            out[f"{k}_clip"] = cluster_ci([x[k] for x in rs], vids)
        out["diff_clip"] = cluster_ci([x["r2_real"] - x["r2_gen"] for x in rs], vids)
        out["headroom"] = (out["r2_real"] - out["r2_gen"]) / out["r2_real"] if out["r2_real"] > 0 else float("nan")
        return out

    groups = {g: summary([x for x in recs if g in x["groups"]]) for g in load_script("naturalness").GROUPS}
    a = groups["all"]
    v = verdict(a["r2_real"], a["r2_gen"], args.min_r2, args.min_headroom)
    print(f"\nVAL ({a['clips']} clips, {a['videos']} videos): R^2 of the WavLM components read back from the mouth")
    print(f"   real motion   {a['r2_real']:.3f}   per clip {a['r2_real_clip'][0]:.3f} ± {a['r2_real_clip'][1]:.3f}   "
          f"rescaled {a['r2s_real_clip'][0]:.3f}")
    print(f"   SANG motion   {a['r2_gen']:.3f}   per clip {a['r2_gen_clip'][0]:.3f} ± {a['r2_gen_clip'][1]:.3f}   "
          f"rescaled {a['r2s_gen_clip'][0]:.3f}")
    print(f"   chance        {a['r2_chance']:.3f}   (real mouth, another clip's audio)")
    print(f"   real - SANG per clip {a['diff_clip'][0]:+.3f} ± {a['diff_clip'][1]:.3f}; headroom {a['headroom']:+.1%}")
    pc = lambda key: [1 - float(sum(((x[key][:, j] - x["tgt"][:, j]) ** 2).sum() for x in recs))
                      / max(float(sum((x["tgt"][:, j] ** 2).sum() for x in recs)), 1e-12) for j in range(5)]
    print(f"   top 5 components, real: {' '.join(f'{x:.3f}' for x in pc('p_real'))}   "
          f"SANG: {' '.join(f'{x:.3f}' for x in pc('p_gen'))}")
    print(f"   VERDICT: {v}  (R^2 real >= {args.min_r2}; headroom >= {args.min_headroom:.0%})")
    print("\nBY GROUP")
    for g, x in groups.items():
        if x["clips"] >= 4:
            print(f"   {g:<20}{x['clips']:>5} clips   real {x['r2_real']:.3f}   SANG {x['r2_gen']:.3f}   "
                  f"chance {x['r2_chance']:+.3f}   headroom {x['headroom']:+.1%}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state": expert.state_dict(), "pca": pca, "scale": scale, "cols": cols, "k": args.k,
                "window": L, "norm": {k: v.cpu() for k, v in ck["norm"].items()},
                "ckpt": args.ckpt, "verdict": v}, out / "expert.pt")
    (out / "probe.json").write_text(json.dumps({"ckpt": args.ckpt, "step": ck["step"], "sampler": args.sampler,
                                                "dims": args.dims, "k": args.k, "verdict": v, "groups": groups,
                                                "clips": [{k: x[k] for k in ("clip", "groups", "r2_real", "r2_gen",
                                                                                "r2s_real", "r2s_gen", "r2_chance")}
                                                          for x in recs]}, indent=1, default=float))
    print(f"\nwrote {out / 'expert.pt'} and {out / 'probe.json'} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
