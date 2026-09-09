#!/usr/bin/env python3
"""Held-out generate eval: teacher-forced vs generate tok-acc, PSNR/SSIM vs GT and VidTok ceiling."""
import argparse
import glob
import json
import random
import sys
from pathlib import Path

import torch
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from sang.codec import load_mimi
from sang.data import clip_tokens
from sang.metrics import video_metrics
from sang.masking import per_slice_cosine_mask
from sang.model import build_talking_head
from sang.video import load_vidtok


def val_clips(cfg: dict) -> list[str]:
    clips = sorted(glob.glob(cfg["data_glob"]))
    random.Random(cfg["seed"]).shuffle(clips)
    if cfg.get("max_clips"):
        clips = clips[: cfg["max_clips"]]
    nval = max(1, int(len(clips) * cfg["val_frac"]))
    return clips[:nval]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs/train_stream_v3.yaml"))
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--out", default=str(REPO / "results/test/test.json"))
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    ckpt = Path(args.ckpt or cfg.get("resume", "runs/stream_v3/best.pt"))
    if not ckpt.is_absolute():
        ckpt = REPO / ckpt
    n = args.n if args.n is not None else 32

    device = "cuda" if torch.cuda.is_available() else "cpu"
    state = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = {**cfg, **state["cfg"]}

    vidtok = load_vidtok(codebook=cfg["codebook"], device=device)
    if cfg.get("audio_encoder"):
        from sang.codec import load_wavlm
        mimi = load_wavlm(cfg["audio_encoder"], device=device)
    else:
        mimi = load_mimi(device=device)
    fsq_codes = getattr(vidtok.regularization, "implicit_codebook", None)
    if fsq_codes is not None:
        fsq_codes = fsq_codes.to(device).float()
    model = build_talking_head(cfg, fsq_codes=fsq_codes).to(device).eval()
    model.load_state_dict(state["model"])

    clips = val_clips(cfg)[:n]
    rows = []
    for path in clips:
        try:
            d = clip_tokens(path, vidtok, mimi, frames=cfg["frames"], res=cfg["res"], start=0,
                            fps=cfg["fps"], audio_codebooks=cfg.get("audio_codebooks", 32),
                            face_cond=cfg.get("face_cond", False))
        except Exception as e:
            print(f"skip {Path(path).name}: {type(e).__name__} {e}", flush=True)
            continue
        idx, audio, gt = d["video"], d["audio"], d["pixels"]
        struct = d.get("struct")
        streaming = getattr(model, "arch_version", None) == "block_ar_v1"
        topk = {}
        if streaming:
            n_cond = getattr(model, "ref_slices", 1)
            if n_cond > 1:  # v3 window-0 eval: grid = [identity, ctx, content...], ctx = ref (static start)
                idx = torch.cat([idx[:, :1], idx[:, :1], idx], dim=1)
            B, tv, h, w = idx.shape
            mask = per_slice_cosine_mask(B, tv, h * w, idx.device, ref_slices=n_cond)
            hm, tgt = model(idx, audio, struct=struct, mask=mask)
            _, parts = model.compute_loss(hm, tgt)
            tf_acc = float(parts["acc_token"])
            if getattr(model, "factorized", False):
                topk = model.head.token_topk_acc(hm, tgt)
            gen = model.generate(audio, idx[:, 0], struct=struct,
                                 ctx=idx[:, 1] if n_cond > 1 else None,
                                 steps=cfg.get("decode_steps", 8),
                                 temperature=cfg.get("decode_temperature", 1.0),
                                 gumbel_temp=cfg.get("decode_gumbel", 4.5),
                                 audio_cfg=cfg.get("audio_cfg", 1.0))
        else:
            logits, target = model(idx, audio, struct=struct)
            tf_acc = (logits.argmax(-1) == target).float().mean().item()
            gen = model.generate(audio, idx[:, 0], tuple(idx.shape[1:]), struct=struct)
        ctv = (cfg["frames"] - 1) // 4 + 1
        off = gen.shape[1] - ctv  # leading conditioning slices (v2: 0, v3: 2)
        gen_c, gt_c = gen[:, off:], idx[:, off:]
        gen_acc = (gen_c.reshape(-1) == gt_c.reshape(-1)).float().mean().item()
        mot_acc = (gen_c[:, 1:].reshape(-1) == gt_c[:, 1:].reshape(-1)).float().mean().item()
        model.cpu()
        torch.cuda.empty_cache()
        rec = vidtok.decode(gt_c, decode_from_indices=True)
        pred = vidtok.decode(gen_c, decode_from_indices=True)
        model.to(device)
        psnr_g, ssim_g = video_metrics(gt, pred)
        psnr_c, ssim_c = video_metrics(gt, rec)
        rows.append({"clip": Path(path).name, "tf_acc": tf_acc, "gen_acc": gen_acc, "mot_acc": mot_acc,
                     "top1": topk.get(1, tf_acc), "top5": topk.get(5), "top10": topk.get(10),
                     "psnr": psnr_g, "ssim": ssim_g, "psnr_ceiling": psnr_c, "ssim_ceiling": ssim_c})
        tk = f" top1={topk.get(1, tf_acc):.3f} top5={topk.get(5, 0):.3f} top10={topk.get(10, 0):.3f}" if topk else ""
        print(f"{Path(path).name} tf={tf_acc:.3f} gen={gen_acc:.3f} mot={mot_acc:.3f}{tk} "
              f"PSNR {psnr_g:.2f}/{psnr_c:.2f} SSIM {ssim_g:.3f}/{ssim_c:.3f}", flush=True)

    keys = ["tf_acc", "gen_acc", "mot_acc", "top1", "top5", "top10", "psnr", "ssim", "psnr_ceiling", "ssim_ceiling"]
    mean = {k: sum(r.get(k, 0) or 0 for r in rows) / max(1, len(rows)) for k in keys}
    out = {"ckpt": str(ckpt), "step": state.get("step"), "n": len(rows), "mean": mean, "rows": rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"n={len(rows)} mean tf={mean['tf_acc']:.3f} gen={mean['gen_acc']:.3f} mot={mean['mot_acc']:.3f} "
          f"top1={mean.get('top1', 0):.3f} top5={mean.get('top5', 0):.3f} top10={mean.get('top10', 0):.3f} "
          f"PSNR {mean['psnr']:.2f} (ceiling {mean['psnr_ceiling']:.2f}) "
          f"SSIM {mean['ssim']:.3f} (ceiling {mean['ssim_ceiling']:.3f})")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
