#!/usr/bin/env python3
"""SANG Stage-0 diagnostics (masks, forward, causality, parity, overfit)."""
import argparse
import math
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from sang.config import load_config
from sang.model import build_talking_head
from sang.streaming_transformer import block_masks
from sang.video import load_vidtok


def _ok(msg):
    print(f"  [PASS] {msg}")


def _fail(msg):
    print(f"  [FAIL] {msg}")
    _fail.count += 1


_fail.count = 0


def _bool_mask_stats(name, allow):
    per_row = allow.sum(-1)
    dead = per_row == 0
    print(f"  {name}: shape={tuple(allow.shape)} "
          f"allowed/row min={per_row.min().item()} max={per_row.max().item()} "
          f"density={allow.float().mean().item():.4f}")
    if dead.any():
        _fail(f"{name}: {dead.sum().item()} query rows attend to NOTHING")
    else:
        _ok(f"{name}: no fully-masked query rows")
    return per_row


def model_tv(cfg: dict) -> int:
    """Slice count the model is actually built with: cfg["tv"] counts only content slices."""
    tv = cfg["tv"]
    if cfg.get("motion_ctx"):
        return tv + 2
    return tv + cfg.get("ref_slices", 1) - 1


def dummy_audio(cfg: dict, B: int, ta: int, device):
    """WavLM-shaped features [B, Ta, D]."""
    return torch.randn(B, ta, {"wavlm-base": 768, "wavlm-large": 1024}[cfg["audio_encoder"]],
                       device=device)


def cmd_masks(cfg, device):
    tv, r, ta = model_tv(cfg), cfg["spatial"] ** 2, cfg["ta"]
    self_allow, cross_allow = block_masks(
        tv, r, ta, device=device,
        cond_slices=2 if cfg.get("motion_ctx") else cfg.get("ref_slices", 1),
        audio_lookahead=cfg.get("audio_lookahead", 0))

    print("\n== self-attention ==")
    _bool_mask_stats("self_allow", self_allow)
    L = tv * r
    pos = torch.arange(L, device=device)
    slc = pos // r
    expected = slc.unsqueeze(1) >= slc.unsqueeze(0)
    if torch.equal(self_allow, expected):
        _ok("self_allow == block-causal spec")
    else:
        _fail(f"self_allow differs from spec in {(self_allow != expected).sum().item()} entries")

    if tv > 1 and not self_allow[:r, r:2 * r].any():
        _ok("slice 0 cannot see slice 1")
    else:
        _fail("slice 0 CAN see slice 1")

    print("\n== cross-attention ==")
    per_row = _bool_mask_stats("cross_allow", cross_allow)
    ticks_per_slice = per_row.view(tv, r)[:, 0]
    print(f"  tau(s) = {ticks_per_slice.tolist()}  (Ta={ta}, Tv={tv})")


def cmd_forward(cfg, device, fsq_codes):
    model = build_talking_head(cfg, fsq_codes=fsq_codes).to(device)
    B, tv, h, w = 2, model_tv(cfg), cfg["spatial"], cfg["spatial"]
    V, ta = cfg["codebook"], cfg["ta"]
    video = torch.randint(0, V, (B, tv, h, w), device=device)
    audio = dummy_audio(cfg, B, ta, device)
    struct = torch.randint(0, V, (B, tv, h, w), device=device) if cfg.get("face_cond") else None

    model.train()
    h_masked, target = model(video, audio, struct=struct, cond_drop=0.0)
    loss, parts = model.compute_loss(h_masked, target)
    logits_proxy = h_masked  # for grad check via loss

    print(f"\n  h_masked shape={tuple(h_masked.shape)} ce={parts['ce'].item():.4f}")
    if not torch.isfinite(loss):
        _fail("loss is non-finite")
    else:
        _ok("loss is finite")

    loss.backward()
    # mask_token is unused when bridge_init supplies the prior. Report, do not fail.
    inactive = set()
    if cfg.get("bridge_init"):
        inactive.add("backbone.mask_token")
    dead = [n for n, p in model.named_parameters()
            if p.requires_grad and (p.grad is None or p.grad.abs().max() == 0)]
    expected = [n for n in dead if n in inactive]
    unexpected = [n for n in dead if n not in inactive]
    if expected:
        print(f"  note: {len(expected)} structurally inactive for this config: {expected}")
    if unexpected:
        _fail(f"{len(unexpected)} params with unexpected zero grad: {unexpected[:8]}")
    else:
        _ok("all structurally active params received gradient")


def cmd_causality(cfg, device, fsq_codes):
    model = build_talking_head(cfg, fsq_codes=fsq_codes).to(device).eval()
    B, tv, h, w = 1, model_tv(cfg), cfg["spatial"], cfg["spatial"]
    V, ta = cfg["codebook"], cfg["ta"]
    video = torch.randint(0, V, (B, tv, h, w), device=device)
    audio = dummy_audio(cfg, B, ta, device)

    # Measured on continuous hidden states: logits_full() returns argmax token IDs, so any leak
    # too small to flip a token would read as zero.
    known = torch.ones(B, model.tv * model.r, dtype=torch.bool, device=device)
    continuous_audio = model.audio_proj is not None

    def hs(v, a):
        return model.hidden_states(v, a, known).reshape(B, model.tv, model.r, -1)

    with torch.no_grad():
        base = hs(video, audio)
        a2 = audio.clone()
        if continuous_audio:
            a2[:, -1, :] += 1.0                      # [B, Ta, D]: last TIME step
        else:
            a2[:, :, -1] = (a2[:, :, -1] + 7) % 2048  # [B, K, Ta]: last time step
        pert_a = hs(video, a2)
        v2 = video.clone()
        v2[:, -1] = (v2[:, -1] + 13) % V
        pert_v = hs(v2, audio)
        d1 = (base[:, 1] - pert_a[:, 1]).float().abs().max().item()
        d2 = (base[:, 1] - pert_v[:, 1]).float().abs().max().item()

    print(f"\n  max |dlogit(slice1)| future audio = {d1:.3e}, future slice = {d2:.3e}")
    if d1 > 1e-4:
        _fail("future audio leaks into slice 1")
    else:
        _ok("no future-audio leakage")
    if d2 > 1e-4:
        _fail("future slice leaks into slice 1")
    else:
        _ok("no future-slice leakage (within slice 1 prediction)")


def cmd_parity(cfg, device, fsq_codes):
    model = build_talking_head(cfg, fsq_codes=fsq_codes).to(device).eval()
    B, tv, h, w = 1, model_tv(cfg), cfg["spatial"], cfg["spatial"]
    V, ta = cfg["codebook"], cfg["ta"]
    video = torch.randint(0, V, (B, tv, h, w), device=device)
    audio = dummy_audio(cfg, B, ta, device)
    si = 1
    known = torch.zeros(B, tv * h * w, dtype=torch.bool, device=device)
    known[:, :si * h * w] = True

    with torch.no_grad():
        train_h = model.hidden_states(video, audio, known)
        gen_h = model.hidden_states_for_decode(video, audio, known)
    d = (train_h - gen_h).abs().max().item()
    print(f"\n  max |train_hidden - decode_hidden| = {d:.3e}")
    if d > 1e-4:
        _fail("train/infer representation mismatch")
    else:
        _ok("train and decode paths identical")


def cmd_overfit(cfg, device, fsq_codes, steps, lr, mask_ratio: float = 0.5):
    cfg = dict(cfg)
    cfg.update(dropout=0.0, label_smoothing=0.0, z_loss_weight=0.0, fsq_loss_weight=0.0, cond_dropout=0.0)
    model = build_talking_head(cfg, fsq_codes=fsq_codes).to(device)
    model.train()
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0

    B, tv, h, w = 1, model_tv(cfg), cfg["spatial"], cfg["spatial"]
    V, ta = cfg["codebook"], cfg["ta"]
    g = torch.Generator(device="cpu").manual_seed(0)
    video = torch.randint(0, V, (B, tv, h, w), generator=g).to(device)
    audio = dummy_audio(cfg, B, ta, device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)

    # mask_ratio must be > 0: at 0 every position is visible and CE reaches 0 by copying.
    print(f"\n  overfit 1 clip, lr={lr}, mask_ratio={mask_ratio}, supervise_all_motion=True")
    loss = None
    for step in range(1, steps + 1):
        h_masked, target = model(video, audio, struct=None, cond_drop=0.0,
                                 mask_ratio=mask_ratio, supervise_all_motion=True)
        loss, parts = model.compute_loss(h_masked, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9).item()
        opt.step()
        if step % max(1, steps // 20) == 0 or step == 1:
            print(f"  {step:6d} ce={loss.item():10.4f} gnorm={gnorm:10.4f} "
                  f"acc_dim={parts.get('acc_dim_mean', 0):.4f}")

    final = loss.item()
    print(f"\n  final CE = {final:.4f}  (ln V = {math.log(V):.4f})")
    if final > 1.0:
        _fail("CE did not collapse on single clip — structural bug")
    else:
        _ok(f"CE collapsed with {mask_ratio:.0%} of positions masked — model can fit "
            f"a single example through the masked path")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["masks", "forward", "causality", "parity", "overfit"])
    ap.add_argument("--config", default=str(REPO / "configs/train.yaml"))
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = load_config(args.config)
    torch.manual_seed(0)
    fsq_codes = None
    if cfg.get("factorized_head", True):
        vidtok = load_vidtok(codebook=cfg["codebook"], device="cpu")
        fsq_codes = getattr(vidtok.regularization, "implicit_codebook", None)
        if fsq_codes is not None:
            fsq_codes = fsq_codes.float()
            from sang.fsq_codec import FSQIndexCodec
            FSQIndexCodec.from_codebook(fsq_codes).self_check()
            if args.device != "cpu":
                fsq_codes = fsq_codes.to(args.device)
        del vidtok

    print(f"=== sanity: {args.cmd} ===")
    if args.cmd == "masks":
        cmd_masks(cfg, args.device)
    elif args.cmd == "forward":
        cmd_forward(cfg, args.device, fsq_codes)
    elif args.cmd == "causality":
        cmd_causality(cfg, args.device, fsq_codes)
    elif args.cmd == "parity":
        cmd_parity(cfg, args.device, fsq_codes)
    else:
        cmd_overfit(cfg, args.device, fsq_codes, args.steps, args.lr)

    print(f"\n=== {_fail.count} failures ===")
    sys.exit(1 if _fail.count else 0)


if __name__ == "__main__":
    main()
