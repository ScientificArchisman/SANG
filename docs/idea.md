# SANG Implementation Guide — Code for Every Recommended Change

**Companion to:** *Diagnosing the ln(V) Training Failure in SANG* (research report v1.0)
**Target repo state:** `arch_version = block_ar_v1`, Run C (`runs/stream_mask`, job 153688), val_acc ≈ 0.0001, train CE ≈ 10.47
**Document version:** 1.0 · August 2026

---

## How to use this document

Every section below gives you (a) the diagnosis in one line, (b) **complete drop-in code**, and (c) the exact command plus the number you should expect to see. Sections are ordered so that each one is testable before you move to the next. Do not skip Stage 0.

I do not have your actual source files — only `history.md`. So the code is written against the **signatures and structure described in your report** (`StreamingTalkingHead`, `StreamingBlockTransformer`, `block_masks()`, `_embed_grid()`, `token_loss()`, `scripts/train.py::evaluate`). Where I had to assume something, it is flagged with `# ASSUMPTION:`. Adjust names, not logic.

Three numeric cores in this document (FSQ digit decomposition, the per-slice cosine mask sampler, the MaskGIT unmasking schedule) were **executed and verified** before being written down; the verification output is quoted inline where relevant.

---

## Change ledger

| # | Change | Files | Why | Risk | Stage |
|---|--------|-------|-----|------|-------|
| 0 | Sanity + overfit harness | `scripts/sanity.py` (new) | Localize the ln(V) stall | none | 0 |
| 1 | Delete FSQ partial-credit MSE loss | `sang/model.py` | Prime suspect for uniform softmax | low | 1 |
| 2 | fp32 loss, z-loss off, LS off | `sang/model.py` | bf16 logsumexp over 32k; flatness | low | 1 |
| 3 | Factorized 5×8-way FSQ heads | `sang/fsq_codec.py`, `sang/fsq_head.py` (new) | 32768-way → 5 trivial 8-way problems | medium | 1 |
| 4 | Learned logit scale / untie head | `sang/fsq_head.py` | Decouple embed norm from logit scale | low | 1 |
| 5 | Attention-mask audit + fixes | `sang/streaming_transformer.py` | Inversion / fully-masked-row NaN | low | 1 |
| 6 | Per-slice cosine mask schedule | `sang/masking.py` (new) | Fixed 0.5 never trains the inference regime | medium | 2 |
| 7 | Single shared embed path | `sang/streaming_transformer.py` | Kills Bug-3 class by construction | medium | 2 |
| 8 | MaskGIT iterative decoding | `sang/streaming_transformer.py` | Conditional-independence trap | medium | 3 |
| 9 | Context corruption (scheduled sampling) | `sang/masking.py` | Exposure bias across slices | low | 3 |
| 10 | KV cache for committed slices | `sang/stream_layers.py` (new) | Streaming latency | high | 3 |
| 11 | Training-loop logging + gates | `scripts/train.py` | You currently cannot see learning | low | 2 |
| 12 | Real eval suite (FVD/LPIPS/Sync/CSIM) | `sang/eval_metrics.py`, `scripts/eval_full.py` (new) | Token accuracy is not a video metric | medium | 4 |
| 13 | Audio conditioning upgrade | `sang/audio_cond.py` (new) | 1 of 32 codebooks @ 12.5 Hz | medium | 4 |
| 14 | Config v2 | `configs/train_stream_v2.yaml` | — | low | all |

---

## Two structural observations that shape everything below

**Observation A — future slices are irrelevant to the current slice's logits.**
Your self-attention mask is `M[ℓ,ℓ'] = 1[s(ℓ) ≥ s(ℓ')]`. Slice `s` therefore **never attends to slices > s**. Whatever you put in future slices (mask tokens, ground truth, garbage) cannot change slice `s`'s output. This has a large consequence: you can train **all four motion slices simultaneously, each with its own independent mask ratio**, and every slice is in exactly the distribution it will face at inference. You do *not* need to pick a single target slice per sample and waste the other three. This is what makes Change 6 both correct and cheap.

**Observation B — your doc and your code disagree about `τ(s)`.**
§6.2 of `history.md` specifies `τ(s) = min(Ta, floor((s+1)·Ta/Tv))`, but §6.3's code is `ticks = ((slc + 1) * ta + tv - 1) // tv`, which is **ceil**. With `Ta=9, Tv=5`: floor gives `[1,3,5,7,9]`, ceil gives `[2,4,6,8,9]`. Both happen to be ≥1 so you dodged a fully-masked cross-attention row, but fix the discrepancy and add the clamp explicitly (Change 5) — if you ever set `Ta < Tv`, floor yields `τ(0)=0` and `softmax(all −inf) = NaN`.

---

# Stage 0 — Sanity and overfit harness

**Run this before changing anything else.** It answers, in under an hour, whether you have a *bug* or a *recipe* problem.

### `scripts/sanity.py` (new file)

```python
#!/usr/bin/env python3
"""
SANG Stage-0 diagnostics.

Subcommands:
  masks     - audit attention masks for inversion / fully-masked rows
  forward   - NaN/Inf, logit-scale and gradient-flow checks at init
  causality - future audio AND future slices must not affect earlier slices
  parity    - training forward and generate() must produce identical logits
  overfit   - single clip, all aux losses off: CE must go to ~0

Usage:
  python scripts/sanity.py masks --config configs/train_stream.yaml
  python scripts/sanity.py overfit --config configs/train_stream.yaml --steps 400
"""
import argparse
import math
import sys

import torch
import torch.nn.functional as F

# ASSUMPTION: these exist with these names in your repo.
from sang.model import build_talking_head, load_config
from sang.streaming_transformer import block_masks


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _ok(msg):
    print(f"  [PASS] {msg}")


def _fail(msg):
    print(f"  [FAIL] {msg}")
    _fail.count += 1


_fail.count = 0


def _bool_mask_stats(name, allow):
    """allow: bool tensor [Q, K] where True == attention IS permitted."""
    assert allow.dtype == torch.bool, f"{name}: expected bool allow-matrix"
    per_row = allow.sum(-1)
    dead = (per_row == 0)
    print(f"  {name}: shape={tuple(allow.shape)} "
          f"allowed/row min={per_row.min().item()} max={per_row.max().item()} "
          f"density={allow.float().mean().item():.4f}")
    if dead.any():
        _fail(f"{name}: {dead.sum().item()} query rows attend to NOTHING "
              f"-> softmax(-inf) = NaN. First dead row: {dead.nonzero()[0].item()}")
    else:
        _ok(f"{name}: no fully-masked query rows")
    return per_row


def cmd_masks(cfg, device):
    tv, r, ta = cfg["tv"], cfg["spatial"] ** 2, cfg["ta"]
    self_allow, cross_allow = block_masks(tv, r, ta, device=device)

    print("\n== self-attention ==")
    _bool_mask_stats("self_allow", self_allow)

    L = tv * r
    pos = torch.arange(L, device=device)
    slc = pos // r
    expected = slc.unsqueeze(1) >= slc.unsqueeze(0)
    if torch.equal(self_allow, expected):
        _ok("self_allow == 1[s(q) >= s(k)] (block-causal, bidirectional in-slice)")
    else:
        n_bad = (self_allow != expected).sum().item()
        _fail(f"self_allow differs from block-causal spec in {n_bad} entries")
        if torch.equal(self_allow, ~expected):
            _fail("  ...and it is EXACTLY the inverse. You have a sign inversion.")

    # in-slice must be fully bidirectional
    s0 = self_allow[:r, :r]
    if s0.all():
        _ok("slice 0 is fully bidirectional internally")
    else:
        _fail("slice 0 is NOT fully bidirectional -> you built raster-causal by mistake")

    # slice s must not see slice s+1
    if tv > 1 and not self_allow[:r, r:2 * r].any():
        _ok("slice 0 cannot see slice 1 (no future leakage)")
    else:
        _fail("slice 0 CAN see slice 1 -> future leakage; this reproduces Run A's tf_acc=1.0")

    print("\n== cross-attention (audio) ==")
    per_row = _bool_mask_stats("cross_allow", cross_allow)
    ticks_per_slice = per_row.view(tv, r)[:, 0]
    print(f"  tau(s) = {ticks_per_slice.tolist()}  (Ta={ta}, Tv={tv})")
    if (ticks_per_slice == 0).any():
        _fail("some slice sees ZERO audio ticks")
    if not torch.all(ticks_per_slice[1:] >= ticks_per_slice[:-1]):
        _fail("tau(s) is not non-decreasing in s")
    else:
        _ok("tau(s) is non-decreasing")
    if per_row.view(tv, r).std(dim=1).max() > 0:
        _fail("tau varies WITHIN a slice -> indexing bug in cross mask")
    else:
        _ok("tau is constant within each slice")

    print("\n== PyTorch attn_mask convention ==")
    print("  Reminder: for nn.MultiheadAttention / SDPA a BOOL attn_mask uses")
    print("  True == 'not allowed' (masked out). A FLOAT mask is ADDED to scores,")
    print("  so masked entries must be -inf and allowed entries 0.0.")
    print("  You must pass ~allow (bool) or allow.float().masked_fill(~allow, -inf).")
    print("  Mixing the two conventions silently produces uniform attention.")


def cmd_forward(cfg, device):
    model = build_talking_head(cfg).to(device)
    B, tv, h, w = 2, cfg["tv"], cfg["spatial"], cfg["spatial"]
    V, ta = cfg["codebook"], cfg["ta"]

    video = torch.randint(0, V, (B, tv, h, w), device=device)
    audio = torch.randint(0, 2048, (B, cfg["audio_codebooks"], ta), device=device)
    struct = torch.randint(0, V, (B, tv, h, w), device=device) if cfg["face_cond"] else None

    model.train()
    out = model(video, audio, struct=struct, cond_drop=0.0)
    logits, target = out[0], out[1]
    logits = logits.float()

    print(f"\n  logits shape={tuple(logits.shape)} dtype={logits.dtype}")
    if not torch.isfinite(logits).all():
        _fail(f"logits contain {(~torch.isfinite(logits)).sum().item()} non-finite values")
    else:
        _ok("logits are finite")

    std = logits.std().item()
    lse = logits.logsumexp(-1).mean().item()
    rng = (logits.max() - logits.min()).item()
    print(f"  logit std={std:.4f}  mean logsumexp={lse:.4f}  range={rng:.4f}")
    if std < 0.05:
        _fail("logit std < 0.05 at init -> near-uniform softmax. "
              "Tied head x RMSNorm scale mismatch. Add a learned logit scale (Change 4).")
    else:
        _ok("logit scale is reasonable at init")

    ce = F.cross_entropy(logits, target)
    print(f"  init CE = {ce.item():.4f}   (ln V = {math.log(cfg['codebook']):.4f})")

    ce.backward()
    dead, total = [], 0
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        total += 1
        if p.grad is None or p.grad.abs().max().item() == 0.0:
            dead.append(n)
    if dead:
        _fail(f"{len(dead)}/{total} params received ZERO gradient: {dead[:12]}")
    else:
        _ok(f"all {total} trainable params received gradient")


def cmd_causality(cfg, device):
    model = build_talking_head(cfg).to(device).eval()
    B, tv, h, w = 1, cfg["tv"], cfg["spatial"], cfg["spatial"]
    V, ta = cfg["codebook"], cfg["ta"]
    r = h * w

    video = torch.randint(0, V, (B, tv, h, w), device=device)
    audio = torch.randint(0, 2048, (B, cfg["audio_codebooks"], ta), device=device)

    with torch.no_grad():
        base = model.logits_full(video, audio)          # ASSUMPTION: [B, tv, r, V] helper

        # (1) perturb the LAST audio tick
        a2 = audio.clone()
        a2[:, :, -1] = (a2[:, :, -1] + 7) % 2048
        pert_a = model.logits_full(video, a2)
        d1 = (base[:, 1] - pert_a[:, 1]).abs().max().item()

        # (2) perturb the LAST video slice
        v2 = video.clone()
        v2[:, -1] = (v2[:, -1] + 13) % V
        pert_v = model.logits_full(v2, audio)
        d2 = (base[:, 1] - pert_v[:, 1]).abs().max().item()

    print(f"\n  max |dlogit(slice1)| wrt future AUDIO  = {d1:.3e}")
    print(f"  max |dlogit(slice1)| wrt future SLICE  = {d2:.3e}")
    for name, d in (("audio", d1), ("slice", d2)):
        if d > 1e-4:
            _fail(f"future {name} leaks into slice 1 -> causality violation")
        else:
            _ok(f"no future-{name} leakage")


def cmd_parity(cfg, device):
    """
    The test that would have caught Bug 3 immediately.
    Training forward with a given `known` mask must produce EXACTLY the logits
    that generate() sees at the corresponding decode step.
    """
    model = build_talking_head(cfg).to(device).eval()
    B, tv, h, w = 1, cfg["tv"], cfg["spatial"], cfg["spatial"]
    V, ta = cfg["codebook"], cfg["ta"]
    r = h * w

    video = torch.randint(0, V, (B, tv, h, w), device=device)
    audio = torch.randint(0, 2048, (B, cfg["audio_codebooks"], ta), device=device)

    si = 1
    known = torch.zeros(B, tv, r, dtype=torch.bool, device=device)
    known[:, :si] = True                       # slices < si visible, si and later masked

    with torch.no_grad():
        train_h = model.hidden_states(video, audio, known=known)      # ASSUMPTION helper
        gen_h = model.hidden_states_for_decode(video, audio, known=known)

    d = (train_h - gen_h).abs().max().item()
    print(f"\n  max |train_hidden - decode_hidden| = {d:.3e}")
    if d > 1e-4:
        _fail("train/infer representation MISMATCH. Both paths must call the same "
              "_embed_grid(); check mask-before-position ordering and struct handling.")
    else:
        _ok("train and decode paths are numerically identical")


def cmd_overfit(cfg, device, steps, lr):
    """
    Single clip, batch 1, ALL auxiliary machinery off.
    Expected: CE -> < 0.05 within a few hundred steps. If not, it is a BUG.
    """
    cfg = dict(cfg)
    cfg.update(dropout=0.0, label_smoothing=0.0, z_loss_weight=0.0,
               fsq_loss_weight=0.0, cond_dropout=0.0)

    model = build_talking_head(cfg).to(device)
    model.train()
    for m in model.modules():                     # belt and braces
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0

    B, tv, h, w = 1, cfg["tv"], cfg["spatial"], cfg["spatial"]
    V, ta = cfg["codebook"], cfg["ta"]
    g = torch.Generator(device="cpu").manual_seed(0)
    video = torch.randint(0, V, (B, tv, h, w), generator=g).to(device)
    audio = torch.randint(0, 2048, (B, cfg["audio_codebooks"], ta), generator=g).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)

    print(f"\n  overfitting 1 synthetic clip, lr={lr}, mask_ratio=0 (fully teacher-forced)")
    print(f"  {'step':>6} {'CE':>10} {'gnorm':>10} {'logit_std':>10}")
    for step in range(1, steps + 1):
        # mask_ratio=0 -> no masking at all; the model may trivially copy. That is FINE:
        # this stage only proves gradients reach the head and CE can reach 0.
        logits, target = model(video, audio, struct=None, cond_drop=0.0, mask_ratio=0.0)
        logits = logits.float()
        loss = F.cross_entropy(logits, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9).item()
        opt.step()
        if step % max(1, steps // 20) == 0 or step == 1:
            print(f"  {step:6d} {loss.item():10.4f} {gnorm:10.4f} {logits.std().item():10.4f}")

    final = loss.item()
    print(f"\n  final CE = {final:.4f}  (ln V = {math.log(V):.4f})")
    if final > 1.0:
        _fail("CE did not collapse on a SINGLE clip with no masking and no aux losses.\n"
              "        This is a structural bug, not a recipe problem. Check, in order:\n"
              "          1. attention mask inversion / fully-masked rows  (sanity.py masks)\n"
              "          2. tied-head logit scale                          (sanity.py forward)\n"
              "          3. loss indexing: logits[keep] vs target[keep] alignment\n"
              "          4. that `target` is not shifted by one slice")
    else:
        _ok("CE collapsed -> pipeline is structurally sound; the problem is the recipe")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["masks", "forward", "causality", "parity", "overfit"])
    ap.add_argument("--config", default="configs/train_stream.yaml")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg.setdefault("ta", 9)
    cfg.setdefault("spatial", 16)
    torch.manual_seed(0)

    print(f"=== sanity: {args.cmd} ===")
    if args.cmd == "masks":
        cmd_masks(cfg, args.device)
    elif args.cmd == "forward":
        cmd_forward(cfg, args.device)
    elif args.cmd == "causality":
        cmd_causality(cfg, args.device)
    elif args.cmd == "parity":
        cmd_parity(cfg, args.device)
    else:
        cmd_overfit(cfg, args.device, args.steps, args.lr)

    print(f"\n=== {_fail.count} failures ===")
    sys.exit(1 if _fail.count else 0)


if __name__ == "__main__":
    main()
```

### Run it

```bash
python scripts/sanity.py masks     --config configs/train_stream.yaml
python scripts/sanity.py forward   --config configs/train_stream.yaml
python scripts/sanity.py causality --config configs/train_stream.yaml
python scripts/sanity.py parity    --config configs/train_stream.yaml
python scripts/sanity.py overfit   --config configs/train_stream.yaml --steps 400 --lr 3e-4
```

### Decision gate

| Result | Meaning | Next |
|---|---|---|
| `overfit` CE → < 0.05 | Pipeline is sound | Go to Stage 1 (recipe) |
| `overfit` CE stays ≈ 10.4 | **Structural bug** | Fix whatever `masks`/`forward` flagged; do not proceed |
| `masks` reports inversion or dead rows | Found it | Change 5 |
| `forward` reports logit std < 0.05 | Tied-head scale | Change 4 |
| `parity` reports mismatch | Bug-3 residue | Change 7 |

---

# Stage 1 — Fix the loss and the head

## Change 1+2 — Rewrite `token_loss`

**Diagnosis.** `softmax(logits over 32768) @ C → MSE against C[target]` maps the 32768-simplex onto a 5-D expected-code vector. That map is massively many-to-one: a near-uniform distribution whose barycentre sits near the target has near-zero MSE. The loss therefore does not require a peaked distribution, and at weight 0.5 against a CE already at its maximum, it actively holds the softmax flat. **Delete it.**

### `sang/model.py` — replace `token_loss`

```python
import torch
import torch.nn.functional as F


def token_loss(logits, target, fsq_codes=None,
               label_smoothing=0.0, z_weight=0.0, fsq_weight=0.0):
    """
    DEPRECATED monolithic-vocabulary loss. Retained only so old checkpoints /
    the flat-AR baseline still run. `fsq_weight` is now hard-disabled.

    The FSQ partial-credit term is removed on purpose: the expected-code MSE is
    minimised by a broad softmax, which pins CE at ln(V). Use FactorizedFSQHead
    (sang/fsq_head.py) if you want partial credit -- there it is free and correct.
    """
    if fsq_weight:
        raise ValueError(
            "fsq_loss_weight is disabled. The expected-code MSE flattens the softmax "
            "and is the prime suspect for CE == ln(V). Set fsq_loss_weight: 0.0 and "
            "switch to FactorizedFSQHead for partial credit."
        )

    flat = logits.reshape(-1, logits.shape[-1]).float()   # fp32: bf16 logsumexp over
    tgt = target.reshape(-1)                              # 32k classes loses precision
    ce = F.cross_entropy(flat, tgt, label_smoothing=label_smoothing)

    total = ce
    parts = {"ce": ce.detach()}
    if z_weight:
        z = flat.logsumexp(-1).square().mean()
        total = total + z_weight * z
        parts["z"] = z.detach()
    parts["loss"] = total.detach()
    return total, parts
```

**Config immediately:** `fsq_loss_weight: 0.0`, `z_loss_weight: 0.0`, `label_smoothing: 0.0`. Re-introduce `z_loss_weight: 1.0e-4` and `label_smoothing: 0.05` only after you have a falling loss curve.

---

## Change 3 — Factorized FSQ heads

**Diagnosis.** `V = 32768 = 8^5`. A token is a 5-tuple of scalar levels. One 32768-way softmax is a needlessly brutal optimization target; five 8-way softmaxes have identical total entropy (`5·ln 8 = 10.3972 = ln 32768`, verified) but vastly better-conditioned gradients, give partial credit for free, and eliminate the 32k logsumexp entirely.

### `sang/fsq_codec.py` (new file)

The digit decomposition is derived **empirically from the FSQ codebook matrix you already load** (`fsq_codes`, shape `[V, 5]`). This requires zero knowledge of VidTok's internal basis ordering — verified to round-trip correctly for both least-significant-first and most-significant-first conventions, and for non-uniform level lists.

```python
"""
Mixed-radix / LUT decomposition of VidTok FSQ token indices.

FSQ (Mentzer et al., "Finite Scalar Quantization: VQ-VAE Made Simple", ICLR 2024)
represents each token as a tuple of independently quantized scalars. For
vidtok_fsq_causal_488_32768, V = 32768 = 8^5, i.e. 5 dimensions x 8 levels.

We derive the mapping index <-> digit-tuple directly from the implicit codebook
C in R^{V x n}, so we never have to guess whether the implementation is
least-significant-first or most-significant-first.
"""
import torch


class FSQIndexCodec:
    """Bidirectional map between FSQ token indices and per-dimension digits."""

    def __init__(self, levels, digits_lut, index_lut):
        self.levels = tuple(int(x) for x in levels)
        self.n_dims = len(self.levels)
        self.vocab = int(index_lut.numel())
        self.register_buffers(digits_lut, index_lut)

    def register_buffers(self, digits_lut, index_lut):
        self._digits_lut = digits_lut          # [V, n_dims] long
        self._index_lut = index_lut            # shape == levels, long

    # ---------------------------------------------------------------- build
    @classmethod
    def from_codebook(cls, C, atol=0.0):
        """
        C: [V, n_dims] float tensor -- the FSQ implicit codebook you already load
           as `fsq_codes` for the (now deleted) partial-credit loss.
        """
        C = C.detach().cpu()
        V, n = C.shape
        levels, digit_cols = [], []
        for d in range(n):
            vals = torch.unique(C[:, d])                       # sorted ascending
            levels.append(vals.numel())
            # exact match: vals came from unique() of this very column
            digit_cols.append(torch.searchsorted(vals, C[:, d].contiguous()))
        digits = torch.stack(digit_cols, dim=-1).long()        # [V, n]

        prod = 1
        for l in levels:
            prod *= l
        if prod != V:
            raise ValueError(f"prod(levels)={prod} != V={V}; codebook is not a clean "
                             f"FSQ lattice -- factorized heads do not apply.")

        index_lut = torch.full(tuple(levels), -1, dtype=torch.long)
        index_lut[tuple(digits.T)] = torch.arange(V, dtype=torch.long)
        if (index_lut < 0).any():
            raise ValueError("digit tuples are not a bijection onto [0, V); "
                             "duplicate code vectors in the codebook?")
        return cls(levels, digits, index_lut)

    def to(self, device):
        self._digits_lut = self._digits_lut.to(device)
        self._index_lut = self._index_lut.to(device)
        return self

    # ---------------------------------------------------------------- maps
    def to_digits(self, idx):
        """idx: [...] long -> [..., n_dims] long"""
        return self._digits_lut[idx.long()]

    def from_digits(self, digits):
        """digits: [..., n_dims] long -> [...] long"""
        return self._index_lut[tuple(digits.long().movedim(-1, 0))]

    # ---------------------------------------------------------------- checks
    def self_check(self, verbose=True):
        idx = torch.arange(self.vocab, device=self._digits_lut.device)
        rt = self.from_digits(self.to_digits(idx))
        ok = bool(torch.equal(rt, idx))
        if verbose:
            import math
            ent = sum(math.log(l) for l in self.levels)
            print(f"[FSQIndexCodec] V={self.vocab} levels={self.levels} "
                  f"roundtrip={ok} sum(ln levels)={ent:.4f} ln(V)={math.log(self.vocab):.4f}")
        if not ok:
            raise RuntimeError("FSQIndexCodec round-trip failed")
        return ok
```

> **Verified output** for a synthetic `[8,8,8,8,8]` FSQ codebook built both ways, plus a non-uniform `[8,8,8,6,5]` case:
> ```
> FSQ lsb_first=True:  V=32768 levels=(8,8,8,8,8) roundtrip=True matches_true_digits=True
> FSQ lsb_first=False: V=32768 levels=(8,8,8,8,8) roundtrip=True matches_true_digits=True
> FSQ non-uniform:     V=15360 derived levels=(8,8,8,6,5) roundtrip=True
> entropy check: sum(ln levels)=9.6395  ln(V)=9.6395
> ```

### `sang/fsq_head.py` (new file)

```python
"""Factorized prediction head over FSQ digit dimensions."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dt)


class FactorizedFSQHead(nn.Module):
    """
    Replaces the tied 32768-way Linear with `n_dims` small softmaxes.

    Parameter count: 5 * 512 * 8 = 20,480 (vs 16.7M for the tied 32k head).
    Random-guess CE (summed over dims) == ln(V), so the number stays directly
    comparable with your old curves: 10.3972 for V=32768.
    """

    def __init__(self, dim, codec, logit_scale_init=1.0, learn_scale=True):
        super().__init__()
        self.codec = codec
        self.levels = codec.levels
        self.n_dims = codec.n_dims
        self.norm = RMSNorm(dim)
        self.heads = nn.ModuleList([nn.Linear(dim, l) for l in self.levels])
        for h in self.heads:
            nn.init.normal_(h.weight, std=0.02)
            nn.init.zeros_(h.bias)
        # Change 4: decouple logit magnitude from embedding norm.
        self.logit_scale = nn.Parameter(
            torch.tensor(float(math.log(logit_scale_init))), requires_grad=learn_scale
        )

    # ------------------------------------------------------------------ core
    def forward(self, h):
        """h: [..., D] -> list of n_dims tensors [..., levels[d]] (fp32)."""
        z = self.norm(h)
        s = self.logit_scale.exp()
        return [head(z).float() * s for head in self.heads]

    # ------------------------------------------------------------------ loss
    def loss(self, h, target_idx, label_smoothing=0.0):
        """
        h:          [N, D] hidden states at supervised positions
        target_idx: [N]    FSQ token indices

        Returns (loss, parts). `loss` is the SUM over dims, so it is on the same
        scale as the old single-softmax CE: random == ln(V) == 10.3972.
        """
        tgt_digits = self.codec.to_digits(target_idx)        # [N, n_dims]
        logits = self.forward(h)

        total = h.new_zeros((), dtype=torch.float32)
        parts, correct = {}, None
        for d, lg in enumerate(logits):
            ce_d = F.cross_entropy(lg, tgt_digits[:, d], label_smoothing=label_smoothing)
            total = total + ce_d
            parts[f"ce_d{d}"] = ce_d.detach()
            pred_d = lg.argmax(-1)
            acc_d = (pred_d == tgt_digits[:, d]).float().mean()
            parts[f"acc_d{d}"] = acc_d.detach()
            correct = (pred_d == tgt_digits[:, d]) if correct is None \
                else correct & (pred_d == tgt_digits[:, d])

        parts["ce"] = total.detach()                    # comparable to old CE
        parts["acc_dim_mean"] = torch.stack(
            [parts[f"acc_d{d}"] for d in range(self.n_dims)]).mean()
        parts["acc_token"] = correct.float().mean().detach()
        parts["logit_scale"] = self.logit_scale.exp().detach()
        return total, parts

    # ------------------------------------------------------------ decoding
    @torch.no_grad()
    def argmax_tokens(self, h):
        logits = self.forward(h)
        digits = torch.stack([lg.argmax(-1) for lg in logits], dim=-1)
        return self.codec.from_digits(digits)

    @torch.no_grad()
    def sample_tokens(self, h, temperature=1.0, top_k=0):
        """
        Returns (token_idx [...], logprob [...]) where logprob is the joint
        log-probability of the sampled tuple = sum_d log p_d(digit_d).
        Used as the MaskGIT confidence score.
        """
        logits = self.forward(h)
        digits, logps = [], []
        for lg in logits:
            if temperature <= 0:
                d = lg.argmax(-1)
            else:
                sc = lg / temperature
                if top_k and top_k < sc.shape[-1]:
                    kth = sc.topk(top_k, dim=-1).values[..., -1:]
                    sc = sc.masked_fill(sc < kth, float("-inf"))
                probs = sc.softmax(-1)
                d = torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1) \
                         .reshape(probs.shape[:-1])
            digits.append(d)
            logps.append(lg.log_softmax(-1).gather(-1, d.unsqueeze(-1)).squeeze(-1))
        digits = torch.stack(digits, dim=-1)
        return self.codec.from_digits(digits), torch.stack(logps, dim=-1).sum(-1)


class CoupledFSQHead(FactorizedFSQHead):
    """
    Optional: relaxes the conditional-independence assumption ACROSS the 5 FSQ
    dims by conditioning head d on the digits of dims < d. Teacher-forced at
    train time, 5 sequential micro-steps at inference (negligible cost).

    Turn this on only if `acc_dim_mean` is healthy but `acc_token` lags badly --
    that gap is exactly the signature of cross-dim dependence.
    """

    def __init__(self, dim, codec, **kw):
        super().__init__(dim, codec, **kw)
        self.digit_emb = nn.ModuleList(
            [nn.Embedding(l, dim) for l in self.levels[:-1]]
        )

    def _step_logits(self, z, d, prev_digits):
        ctx = z
        for k in range(d):
            ctx = ctx + self.digit_emb[k](prev_digits[..., k])
        return self.heads[d](ctx).float() * self.logit_scale.exp()

    def loss(self, h, target_idx, label_smoothing=0.0):
        tgt = self.codec.to_digits(target_idx)
        z = self.norm(h)
        total = h.new_zeros((), dtype=torch.float32)
        parts, correct = {}, None
        for d in range(self.n_dims):
            lg = self._step_logits(z, d, tgt)            # teacher forcing
            ce_d = F.cross_entropy(lg, tgt[:, d], label_smoothing=label_smoothing)
            total = total + ce_d
            parts[f"ce_d{d}"] = ce_d.detach()
            ok = lg.argmax(-1) == tgt[:, d]
            parts[f"acc_d{d}"] = ok.float().mean().detach()
            correct = ok if correct is None else correct & ok
        parts["ce"] = total.detach()
        parts["acc_token"] = correct.float().mean().detach()
        return total, parts

    @torch.no_grad()
    def sample_tokens(self, h, temperature=1.0, top_k=0):
        z = self.norm(h)
        digits = torch.zeros(*z.shape[:-1], self.n_dims, dtype=torch.long, device=z.device)
        logp = torch.zeros(z.shape[:-1], device=z.device)
        for d in range(self.n_dims):
            lg = self._step_logits(z, d, digits)
            if temperature <= 0:
                pick = lg.argmax(-1)
            else:
                probs = (lg / temperature).softmax(-1)
                pick = torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1) \
                            .reshape(probs.shape[:-1])
            digits[..., d] = pick
            logp = logp + lg.log_softmax(-1).gather(-1, pick.unsqueeze(-1)).squeeze(-1)
        return self.codec.from_digits(digits), logp
```

### Wiring into `StreamingTalkingHead.__init__`

```python
from sang.fsq_codec import FSQIndexCodec
from sang.fsq_head import FactorizedFSQHead, CoupledFSQHead

# ... inside __init__, after video_emb is created ...
if cfg.get("factorized_head", True):
    # fsq_codes: [V, 5] implicit codebook, the same tensor you used to pass to token_loss
    codec = FSQIndexCodec.from_codebook(fsq_codes)
    codec.self_check()
    codec.to(device)
    HeadCls = CoupledFSQHead if cfg.get("coupled_fsq_head", False) else FactorizedFSQHead
    self.head = HeadCls(cfg["dim"], codec)
    self.norm = nn.Identity()          # the head owns its own RMSNorm
    self.factorized = True
else:
    self.norm = RMSNorm(cfg["dim"])
    self.head = nn.Linear(cfg["dim"], cfg["codebook"], bias=False)
    self.head.weight = self.video_emb.weight       # legacy tied head
    self.factorized = False
```

> **Note on weight tying.** With factorized heads, tying is meaningless (shapes differ) and is dropped. This is a feature: your `head.weight = video_emb.weight` tie forced one matrix to serve as both a 512-d lookup table and a 32768-way classifier, and any scale mismatch between RMSNorm'd activations (O(1)) and 0.02-std embeddings produces near-uniform logits at init. The `logit_scale` parameter in the head is the general remedy and is kept even in the factorized path.

---

## Change 5 — Attention masks

### `sang/streaming_transformer.py` — replace `block_masks()`

```python
def block_masks(tv, r, ta, device, min_ticks=1, return_float=False, dtype=torch.float32):
    """
    Returns (self_allow, cross_allow) as BOOL allow-matrices where
    True == attention IS permitted.

    IMPORTANT: PyTorch wants the opposite. Pass `~allow` for a bool attn_mask,
    or use return_float=True to get an additive mask (0 allowed / -inf blocked).

    self_allow  : [L, L]  block-causal, bidirectional within a slice
    cross_allow : [L, ta] causal audio, tau(s) ticks visible to slice s
    """
    L = tv * r
    pos = torch.arange(L, device=device)
    slc = pos // r

    self_allow = slc.unsqueeze(1) >= slc.unsqueeze(0)              # [L, L]

    # NOTE: history.md 6.2 specifies floor(), 6.3 implements ceil(). We use ceil
    # (the code path that actually ran) and clamp to >= min_ticks so that no
    # query row can ever be fully masked -> softmax(-inf) = NaN.
    ticks = ((slc + 1) * ta + tv - 1) // tv
    ticks = ticks.clamp(min=min_ticks, max=ta)
    aud = torch.arange(ta, device=device)
    cross_allow = aud.unsqueeze(0) < ticks.unsqueeze(1)            # [L, ta]

    # hard invariants -- cheap, and they would have caught a silent NaN path
    assert self_allow.any(dim=-1).all(), "self-attention has a fully-masked query row"
    assert cross_allow.any(dim=-1).all(), "cross-attention has a fully-masked query row"

    if return_float:
        def _f(a):
            m = torch.zeros(a.shape, device=device, dtype=dtype)
            return m.masked_fill(~a, float("-inf"))
        return _f(self_allow), _f(cross_allow)
    return self_allow, cross_allow
```

### At the call sites

```python
self_allow, cross_allow = block_masks(tv, r, ta, device=x.device)

# nn.MultiheadAttention with a BOOL mask: True == BLOCKED
h = self.self_attn(q, k, v, attn_mask=~self_allow, need_weights=False)[0]
h = self.cross_attn(q, ea, ea, attn_mask=~cross_allow[:, :ea.shape[1]], need_weights=False)[0]
```

**Never** mix conventions. If you migrate to `F.scaled_dot_product_attention` (Change 10), note it takes a bool mask where **True == KEEP** — the *opposite* of `nn.MultiheadAttention`. This inversion between the two APIs is a very common source of exactly your symptom.

---

# Stage 2 — Fix the training distribution

## Change 6 — Per-slice cosine mask schedule

**Diagnosis.** A fixed `Bernoulli(0.5)` on motion tokens means the model **always** sees half of the ground-truth tokens of the slice it is predicting. It learns spatial interpolation from visible in-slice neighbours. At inference the entire slice is masked — a regime it has literally never encountered. MaskGIT (Chang et al., CVPR 2022) samples the ratio per-sample from `γ(r) = cos(πr/2)`, `r ~ U(0,1)`; the paper found the cosine schedule "consistently outperforms the linear and other concave functions."

By **Observation A**, each slice can get an independent ratio and all four motion slices supervise simultaneously.

### `sang/masking.py` (new file)

```python
"""Mask sampling for block-causal masked-token training."""
import math

import torch


def cosine_gamma(u):
    """MaskGIT mask-scheduling function gamma(u) = cos(pi*u/2), u in [0,1]."""
    return torch.cos(math.pi * u / 2.0)


def per_slice_cosine_mask(B, tv, r, device, ref_slices=1, min_masked=1,
                          generator=None, dtype=torch.bool):
    """
    Independent MaskGIT cosine mask ratio for EVERY slice of EVERY sample.

    Valid because self-attention is block-causal: slice s never attends to
    slices > s, so each slice's mask is an independent training example and all
    tv-1 motion slices can be supervised in one forward pass.

    Returns: bool [B, tv*r], True == masked (hidden from the model, supervised).
    Slices [0, ref_slices) are never masked (reference identity).
    """
    u = torch.rand(B, tv, 1, device=device, generator=generator)
    gamma = cosine_gamma(u)                                   # (0, 1]
    n_mask = torch.ceil(gamma * r).clamp_(min_masked, r)      # [B, tv, 1]

    scores = torch.rand(B, tv, r, device=device, generator=generator)
    rank = scores.argsort(dim=-1).argsort(dim=-1)             # 0..r-1
    mask = rank < n_mask
    mask[:, :ref_slices] = False
    return mask.reshape(B, tv * r).to(dtype)


def inference_shaped_mask(B, tv, r, device, ref_slices=1, generator=None):
    """
    Ablation / alternative: pick a target slice s uniformly, reveal slices < s,
    mask slice s and everything after. Exactly one generate() step, but supervises
    only one slice per sample. Use per_slice_cosine_mask by default.
    """
    s = torch.randint(ref_slices, tv, (B, 1), device=device, generator=generator)
    slc = torch.arange(tv, device=device).unsqueeze(0)
    mask = (slc >= s).unsqueeze(-1).expand(B, tv, r).clone()
    mask[:, :ref_slices] = False
    return mask.reshape(B, tv * r)


def mixed_mask(B, tv, r, device, p_inference_shaped=0.0, **kw):
    if p_inference_shaped <= 0:
        return per_slice_cosine_mask(B, tv, r, device, **kw)
    use_inf = torch.rand(B, device=device) < p_inference_shaped
    a = per_slice_cosine_mask(B, tv, r, device, **kw)
    b = inference_shaped_mask(B, tv, r, device,
                              ref_slices=kw.get("ref_slices", 1))
    return torch.where(use_inf.unsqueeze(-1), b, a)


def corrupt_context(video_idx, mask, vocab, p_corrupt=0.0, ref_slices=1,
                    r=None, generator=None):
    """
    Change 9 -- scheduled-sampling-lite for exposure bias ACROSS slices.

    At inference, previous slices contain the model's own (imperfect) tokens; at
    training they are ground truth. Randomly replace a fraction of VISIBLE motion
    tokens with random tokens so the model learns to tolerate its own errors.
    This is the cheap discrete analogue of Self-Forcing rollout training.

    video_idx: [B, tv, h, w] long ; mask: bool [B, tv*r] ; returns corrupted copy.
    """
    if p_corrupt <= 0:
        return video_idx
    B, tv, h, w = video_idx.shape
    r = r or (h * w)
    flat = video_idx.reshape(B, tv * r).clone()

    visible = ~mask
    motion = torch.arange(tv * r, device=flat.device) >= ref_slices * r
    eligible = visible & motion.unsqueeze(0)

    hit = (torch.rand(flat.shape, device=flat.device, generator=generator) < p_corrupt) & eligible
    rnd = torch.randint(0, vocab, flat.shape, device=flat.device, generator=generator)
    flat = torch.where(hit, rnd, flat)
    return flat.reshape(B, tv, h, w)


def maskgit_decode_schedule(r, steps, schedule="cosine"):
    """
    Number of tokens STILL masked after each decode iteration t = 1..steps.
    Guarantees: strictly decreasing, >= 1 token revealed per step, ends at 0.
    """
    out, prev = [], r
    for t in range(1, steps + 1):
        u = t / steps
        if schedule == "cosine":
            frac = math.cos(math.pi / 2.0 * u)
        elif schedule == "linear":
            frac = 1.0 - u
        elif schedule == "square":
            frac = 1.0 - u ** 2
        else:
            raise ValueError(schedule)
        n = int(math.floor(r * frac))
        n = max(0, min(n, prev - 1))
        out.append(n)
        prev = n
    return out
```

> **Verified behaviour** (4096 samples, `tv=5, r=256`):
> ```
> mean ratio      = 0.637          (your fixed baseline was 0.500)
> P(ratio > 0.9)  = 0.283
> P(ratio == 1.0) = 0.056          <- full-slice masking, matches inference
> min/max         = 0.004 / 1.000
> slice-0 masked  = False
> any all-visible motion slice = False
> ```
> Under `Bernoulli(0.5)` the probability of a fully-masked 256-token slice is `2^-256`. That is the entire bug in one number: **you never once trained the task you evaluate.**
>
> And the decode schedule for `r=256`:
> ```
> steps= 8 remaining=[251,236,212,181,142,97,49,0]  revealed=[5,15,24,31,39,45,48,49] sum=256
> steps=12 remaining=[253,247,236,221,203,181,155,128,97,66,33,0] sum=256
> ```

---

## Change 7 — One shared embedding path

**Diagnosis.** Bug 3 existed because training and inference built token representations in two different places. The fix is not to keep them in sync by hand — it is to have **one function**.

### `sang/streaming_transformer.py`

```python
class StreamingTalkingHead(nn.Module):

    # ------------------------------------------------------------------
    # THE single place where tokens become vectors. Both training and
    # generate() call this. If it is right here, it is right everywhere.
    # ------------------------------------------------------------------
    def _embed_grid(self, grid, struct, known):
        """
        grid  : [B, tv, h, w] long   token ids (values at ~known positions ignored)
        struct: [B, tv, h, w] long or None
        known : [B, tv*r] bool       True == this cell's token is visible

        Returns [B, tv*r, D] BEFORE position embeddings. The backbone adds
        position afterwards, so masked cells become mask_token + pos(l) --
        the critical invariant from history.md 3.4.
        """
        B, tv, h, w = grid.shape
        r = h * w
        L = tv * r

        e = self.video_emb(grid.reshape(B, L))
        if struct is not None:
            s = self.struct_emb(struct.reshape(B, L))
            s = s.clone()
            s[:, :r] = 0.0                      # structure on motion slices only
            e = e + s

        mask_tok = self.backbone.mask_token.to(e.dtype)        # [1, 1, D]
        return torch.where(known.reshape(B, L).unsqueeze(-1), e, mask_tok)

    def _embed_audio(self, audio, n_ticks=None):
        a = audio[:, 0]                                        # codebook 0 (see Change 13)
        if n_ticks is not None:
            a = a[:, :n_ticks]
        t = torch.arange(a.shape[1], device=a.device)
        return self.audio_emb(a) + self.audio_pos(t).unsqueeze(0)

    def hidden_states(self, video_idx, audio, known, struct=None, n_ticks=None):
        """Backbone forward for an arbitrary `known` pattern. [B, tv, r, D]."""
        B, tv, h, w = video_idx.shape
        r = h * w
        e_v = self._embed_grid(video_idx, struct, known)
        e_a = self._embed_audio(audio, n_ticks)
        hs = self.backbone(e_v, e_a)               # adds pos, runs 8 layers
        return hs.view(B, tv, r, -1)

    # generate() uses the exact same function -> parity by construction
    hidden_states_for_decode = hidden_states

    # ------------------------------------------------------------------
    def forward(self, video_idx, audio, struct=None, cond_drop=0.0,
                mask=None, mask_ratio=None, p_corrupt=0.0):
        """
        Returns (h_masked [N, D], target [N]) -- hidden states at supervised
        positions, NOT logits. The head computes the (small) logits, so we never
        materialise an [N, 32768] tensor.
        """
        from sang.masking import per_slice_cosine_mask, corrupt_context

        B, tv, h, w = video_idx.shape
        r = h * w
        L = tv * r
        dev = video_idx.device

        # ---- structure conditioning dropout (per sample) ----
        if struct is not None and cond_drop > 0 and self.training:
            keep = (torch.rand(B, device=dev) >= cond_drop)
            struct = torch.where(keep.view(B, 1, 1, 1), struct,
                                 torch.zeros_like(struct))

        # ---- mask ----
        if mask is None:
            if mask_ratio is not None and mask_ratio == 0.0:
                mask = torch.zeros(B, L, dtype=torch.bool, device=dev)
                mask[:, r:] = True if False else False   # nothing masked: Stage-0 only
            elif mask_ratio is not None and mask_ratio > 0:
                motion = (torch.arange(L, device=dev) >= r).unsqueeze(0)
                mask = (torch.rand(B, L, device=dev) < mask_ratio) & motion  # legacy
            else:
                mask = per_slice_cosine_mask(B, tv, r, dev,
                                             ref_slices=self.ref_slices)

        # ---- Change 9: corrupt visible context ----
        if self.training and p_corrupt > 0:
            video_idx = corrupt_context(video_idx, mask, self.video_card,
                                        p_corrupt=p_corrupt,
                                        ref_slices=self.ref_slices, r=r)

        known = ~mask
        hs = self.hidden_states(video_idx, audio, known, struct=struct)   # [B,tv,r,D]

        # ---- gather supervised positions: masked AND motion ----
        sup = mask.view(B, tv, r).clone()
        sup[:, :self.ref_slices] = False
        h_masked = hs[sup]                                   # [N, D]
        target = video_idx.reshape(B, tv, r)[sup]            # [N]
        return h_masked, target
```

**Stage-0 caveat.** For the `overfit` subcommand with `mask_ratio=0.0` nothing is masked, so `sup` is empty. For that one diagnostic, supervise *all* motion positions instead:

```python
if mask_ratio == 0.0:                 # Stage-0 fully-teacher-forced diagnostic ONLY
    sup = torch.zeros(B, tv, r, dtype=torch.bool, device=dev)
    sup[:, self.ref_slices:] = True
```

### Training call site

```python
h_masked, target = model(video, audio, struct=struct,
                         cond_drop=cfg["cond_dropout"],
                         p_corrupt=cfg.get("context_corrupt", 0.0))
loss, parts = model.head.loss(h_masked, target,
                              label_smoothing=cfg["label_smoothing"])
if cfg["z_loss_weight"]:
    # z-loss is per-dim now; usually unnecessary with 8-way heads
    pass
```

---

# Stage 3 — Fix inference

## Change 8 — MaskGIT iterative decoding

**Diagnosis.** One-shot `argmax` over 256 positions assumes the tokens are conditionally independent given context. A face slice is strongly spatially correlated, so independent per-position argmax collapses to the marginal mode and produces incoherent output. MaskGIT's answer is iterative confidence-based decoding — generating a 256-token image in "as few as 8 iterations." Your `refine=1` is the degenerate `steps=1` case.

### `sang/streaming_transformer.py` — replace `generate()`

```python
@torch.no_grad()
def generate(self, audio, reference, shape, struct=None,
             steps=8, temperature=1.0, gumbel_temp=4.5,
             schedule="cosine", top_k=0, final_greedy=True, return_conf=False):
    """
    Block-autoregressive across slices, MaskGIT-iterative within each slice.

    audio     : [B, K, Ta]
    reference : [B, h, w]  ground-truth slice 0 (identity)
    shape     : (tv, h, w)
    steps     : MaskGIT iterations per slice. 1 == your old one-shot argmax.
                8-12 is the standard range.
    gumbel_temp: annealed Gumbel noise added to the confidence score, as in
                MaskGIT. Set 0.0 for deterministic confidence ordering.
    """
    from sang.masking import maskgit_decode_schedule

    tv, h, w = shape
    r = h * w
    B = reference.shape[0]
    dev = reference.device

    grid = torch.zeros(B, tv, h, w, dtype=torch.long, device=dev)
    grid[:, 0] = reference
    known = torch.zeros(B, tv, r, dtype=torch.bool, device=dev)
    known[:, :self.ref_slices] = True

    ta = audio.shape[-1]
    conf_log = []

    for si in range(self.ref_slices, tv):
        n_ticks = min(ta, max(1, ((si + 1) * ta + tv - 1) // tv))
        sched = maskgit_decode_schedule(r, steps, schedule)

        for t, n_still_masked in enumerate(sched, start=1):
            hs = self.hidden_states(grid, audio, known.reshape(B, tv * r),
                                    struct=struct, n_ticks=n_ticks)
            h_si = hs[:, si]                                       # [B, r, D]

            temp = 0.0 if (final_greedy and t == steps) else temperature
            tok, logp = self.head.sample_tokens(h_si, temperature=temp, top_k=top_k)

            # ---- confidence with annealed Gumbel noise (MaskGIT) ----
            conf = logp
            if gumbel_temp > 0:
                g = -torch.log(-torch.log(
                    torch.rand_like(conf).clamp_min(1e-10)).clamp_min(1e-10))
                conf = conf + gumbel_temp * (1.0 - t / steps) * g
            conf = conf.masked_fill(known[:, si], float("inf"))    # freeze decided cells

            # ---- reveal the (r - n_still_masked) most confident ----
            n_keep = r - n_still_masked
            order = conf.argsort(dim=-1, descending=True)
            new_known = torch.zeros_like(known[:, si])
            new_known.scatter_(1, order[:, :n_keep], True)

            newly = new_known & ~known[:, si]
            gflat = grid[:, si].reshape(B, r)
            grid[:, si] = torch.where(newly, tok, gflat).reshape(B, h, w)
            known[:, si] = new_known

            if return_conf:
                conf_log.append(logp[newly].mean().item() if newly.any() else float("nan"))

        assert known[:, si].all(), "slice not fully decoded -- check the schedule"

    return (grid, conf_log) if return_conf else grid
```

### Validation call site (`scripts/train.py::evaluate`)

```python
gen = model.generate(audio, video[:, 0], shape, struct=struct,
                     steps=cfg.get("decode_steps", 8),
                     temperature=cfg.get("decode_temperature", 1.0),
                     gumbel_temp=cfg.get("decode_gumbel", 4.5))
```

### Ablation you should run once

`steps ∈ {1, 4, 8, 12}` on a fixed 64-clip val subset. If quality is flat in `steps`, your model is not yet using in-slice context and the bottleneck is upstream. If it improves monotonically and saturates around 8–12, the fix landed.

---

## Change 10 — KV cache for committed slices

**Diagnosis.** You re-forward the entire 1280-token sequence for every slice, and now also for every MaskGIT step — `(tv-1) × steps` full forwards, i.e. 32 instead of 4. A cache is no longer optional.

The trick with MaskGIT-inside-block-AR: slice `si`'s K/V change every iteration, so only cache slices `< si`. After `si` is finalized, run **one commit pass** over its 256 tokens to append its K/V.

`nn.MultiheadAttention` cannot expose K/V, so this needs explicit projections. The helper below ports your existing checkpoint weights so you lose nothing.

### `sang/stream_layers.py` (new file)

```python
"""Explicit-projection attention layers with a per-slice KV cache."""
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from sang.fsq_head import RMSNorm


@dataclass
class KVCache:
    """Per-layer K/V for all COMMITTED (fully decoded) slices."""
    k: List[Optional[torch.Tensor]] = field(default_factory=list)   # [B, H, S, hd]
    v: List[Optional[torch.Tensor]] = field(default_factory=list)

    @classmethod
    def empty(cls, n_layers):
        return cls(k=[None] * n_layers, v=[None] * n_layers)

    def append(self, layer, k, v):
        if self.k[layer] is None:
            self.k[layer], self.v[layer] = k, v
        else:
            self.k[layer] = torch.cat([self.k[layer], k], dim=2)
            self.v[layer] = torch.cat([self.v[layer], v], dim=2)

    def get(self, layer):
        return self.k[layer], self.v[layer]

    def length(self):
        return 0 if self.k[0] is None else self.k[0].shape[2]


class CachedSelfAttention(nn.Module):
    def __init__(self, dim, heads, dropout=0.0):
        super().__init__()
        assert dim % heads == 0
        self.h, self.hd = heads, dim // heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.p = dropout

    def _split(self, x):
        B, S, _ = x.shape
        return x.view(B, S, self.h, self.hd).transpose(1, 2)      # [B,H,S,hd]

    def forward(self, x, attn_allow=None, cache=None, layer_idx=None, commit=False):
        """
        attn_allow: bool [Sq, Sk] True == KEEP.  NOTE: this is the SDPA
                    convention, the OPPOSITE of nn.MultiheadAttention.
        cache     : KVCache with committed slices, or None for full forward.
        commit    : if True, append this block's K/V to the cache.
        """
        B, S, D = x.shape
        q, k, v = self._split(self.q_proj(x)), self._split(self.k_proj(x)), self._split(self.v_proj(x))

        if cache is not None:
            ck, cv = cache.get(layer_idx)
            if commit:
                cache.append(layer_idx, k, v)
            if ck is not None:
                k = torch.cat([ck, k], dim=2)
                v = torch.cat([cv, v], dim=2)
            # Current block attends to ALL cached past + fully within itself,
            # which IS block-causal for a single-slice query block -> no mask.
            attn_allow = None

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_allow,
            dropout_p=self.p if self.training else 0.0)
        return self.out_proj(out.transpose(1, 2).reshape(B, S, D))


class CachedCrossAttention(nn.Module):
    def __init__(self, dim, heads, dropout=0.0):
        super().__init__()
        self.h, self.hd = heads, dim // heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.p = dropout

    def _split(self, x):
        B, S, _ = x.shape
        return x.view(B, S, self.h, self.hd).transpose(1, 2)

    def forward(self, x, mem, attn_allow=None):
        B, S, D = x.shape
        q = self._split(self.q_proj(x))
        k = self._split(self.k_proj(mem))
        v = self._split(self.v_proj(mem))
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_allow,
            dropout_p=self.p if self.training else 0.0)
        return self.out_proj(out.transpose(1, 2).reshape(B, S, D))


class SwiGLU(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        hidden = dim * mult
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class CachedStreamLayer(nn.Module):
    def __init__(self, dim, heads, dropout=0.1):
        super().__init__()
        self.n1, self.n2, self.n3 = RMSNorm(dim), RMSNorm(dim), RMSNorm(dim)
        self.self_attn = CachedSelfAttention(dim, heads, dropout)
        self.cross_attn = CachedCrossAttention(dim, heads, dropout)
        self.ffn = SwiGLU(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mem, self_allow=None, cross_allow=None,
                cache=None, layer_idx=None, commit=False):
        x = x + self.drop(self.self_attn(self.n1(x), self_allow,
                                         cache=cache, layer_idx=layer_idx, commit=commit))
        x = x + self.drop(self.cross_attn(self.n2(x), mem, cross_allow))
        x = x + self.drop(self.ffn(self.n3(x)))
        return x


# ---------------------------------------------------------------- migration
@torch.no_grad()
def port_mha_weights(old_mha: nn.MultiheadAttention, new_attn):
    """
    Copy nn.MultiheadAttention weights into the explicit q/k/v layers so existing
    checkpoints keep working. nn.MHA packs in_proj_weight as [3D, D] = [Wq;Wk;Wv].
    """
    D = new_attn.q_proj.in_features
    if old_mha._qkv_same_embed_dim:
        W, b = old_mha.in_proj_weight, old_mha.in_proj_bias
        new_attn.q_proj.weight.copy_(W[0:D]);      new_attn.k_proj.weight.copy_(W[D:2 * D])
        new_attn.v_proj.weight.copy_(W[2 * D:3 * D])
        if b is not None:
            new_attn.q_proj.bias.copy_(b[0:D]);    new_attn.k_proj.bias.copy_(b[D:2 * D])
            new_attn.v_proj.bias.copy_(b[2 * D:3 * D])
    else:
        new_attn.q_proj.weight.copy_(old_mha.q_proj_weight)
        new_attn.k_proj.weight.copy_(old_mha.k_proj_weight)
        new_attn.v_proj.weight.copy_(old_mha.v_proj_weight)
    new_attn.out_proj.weight.copy_(old_mha.out_proj.weight)
    if old_mha.out_proj.bias is not None:
        new_attn.out_proj.bias.copy_(old_mha.out_proj.bias)
```

### Cached `generate()` — the slice loop

```python
@torch.no_grad()
def generate_cached(self, audio, reference, shape, struct=None, steps=8, **kw):
    """
    Same output as generate(), but O(1) in committed history per step.
    VALIDATE FIRST: assert torch.equal(generate_cached(...), generate(...))
    with gumbel_temp=0.0, temperature=0.0 on a fixed seed.
    """
    from sang.masking import maskgit_decode_schedule
    from sang.stream_layers import KVCache

    tv, h, w = shape
    r, B, dev = h * w, reference.shape[0], reference.device
    grid = torch.zeros(B, tv, h, w, dtype=torch.long, device=dev)
    grid[:, 0] = reference
    known = torch.zeros(B, tv, r, dtype=torch.bool, device=dev)
    known[:, :self.ref_slices] = True

    cache = KVCache.empty(len(self.backbone.layers))
    ta = audio.shape[-1]

    # commit slice 0
    self._forward_slice(grid, struct, known, audio, 0, ta, cache, commit=True)

    for si in range(self.ref_slices, tv):
        n_ticks = min(ta, max(1, ((si + 1) * ta + tv - 1) // tv))
        for t, n_masked in enumerate(maskgit_decode_schedule(r, steps), start=1):
            h_si = self._forward_slice(grid, struct, known, audio, si, n_ticks,
                                       cache, commit=False)
            # ... identical confidence / reveal logic as generate() ...
        # commit the finalised slice so later slices read its true K/V
        self._forward_slice(grid, struct, known, audio, si, n_ticks, cache, commit=True)

    return grid
```

`_forward_slice` embeds only slice `si`'s 256 tokens (using `_embed_grid` restricted to that slice plus its positions) and runs the layers with `cache`. **Do not ship this until the equality assertion against the uncached path passes** — a silent cache bug is indistinguishable from a bad model.

---

# Stage 2/3 — Training loop

## Change 11 — `scripts/train.py`

```python
# ---------------------------------------------------------------- logging
import collections
import json
import time

meter = collections.defaultdict(float)
meter_n = 0

for step in range(start, total + 1):
    lr = lr_at(step, warmup, total, lr_max, lr_min)
    for gpar in opt.param_groups:
        gpar["lr"] = lr

    opt.zero_grad(set_to_none=True)
    for _ in range(grad_accum):
        video, audio, struct = next(train_iter)
        h_masked, target = model(video, audio, struct=struct,
                                 cond_drop=cfg["cond_dropout"],
                                 p_corrupt=cfg.get("context_corrupt", 0.0))
        loss, parts = model.head.loss(h_masked, target,
                                      label_smoothing=cfg["label_smoothing"])
        (loss / grad_accum).backward()
        for k, v in parts.items():
            meter[k] += float(v)
        meter_n += 1

    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
    opt.step()

    # ---- THE curve you were missing: train masked CE, logged separately ----
    if step % cfg.get("log_every", 50) == 0:
        m = {k: v / max(meter_n, 1) for k, v in meter.items()}
        chance = math.log(cfg["codebook"])
        print(f"step {step:7d} lr {lr:.2e} "
              f"ce {m['ce']:.4f} (chance {chance:.4f}, gap {chance - m['ce']:+.4f}) "
              f"acc_dim {m.get('acc_dim_mean', 0):.4f} (chance {1/8:.3f}) "
              f"acc_tok {m.get('acc_token', 0):.5f} (chance {1/cfg['codebook']:.2e}) "
              f"gnorm {gnorm:.3f} lscale {m.get('logit_scale', 1):.3f}",
              flush=True)
        meter.clear()
        meter_n = 0

    # ---- cheap teacher-forced diagnostic (every eval_every) ----
    if step % cfg["eval_every"] == 0:
        d = token_diagnostic(model, val_loader, cfg, n_batches=8)
        print(f"  [diag] val_ce {d['ce']:.4f} acc_dim {d['acc_dim_mean']:.4f} "
              f"acc_tok {d['acc_token']:.5f}", flush=True)

    # ---- expensive real eval (generate + decode + perceptual metrics) ----
    if step % cfg.get("full_eval_every", 5000) == 0:
        r = full_eval(model, fixed_val_subset, cfg)      # sang/eval_metrics.py
        print(f"  [full] fvd {r['fvd']:.1f} lpips {r['lpips']:.4f} "
              f"psnr {r['psnr']:.2f}/{r['psnr_ceiling']:.2f} "
              f"sync_c {r['sync_c']:.2f} sync_d {r['sync_d']:.2f} "
              f"csim {r['csim']:.3f} mot_acc {r['mot_acc']:.5f}", flush=True)
        # SELECT ON FVD + Sync-C, NEVER on token accuracy
        score = r["fvd"] - 10.0 * r["sync_c"]
        if score < best_score:
            best_score, best_step = score, step
            save_ckpt(model, opt, step, r, out_dir / "best.pt")
        json.dump({"step": step, **r}, open(out_dir / "last_eval.json", "w"), indent=2)

    # ---- early stopping: DISABLED while debugging ----
    # patience: 0 in the config means "never stop early"
```

### `token_diagnostic` helper

```python
@torch.no_grad()
def token_diagnostic(model, loader, cfg, n_batches=8):
    """Cheap teacher-forced masked CE on val. NOT a quality metric -- a learning probe."""
    model.eval()
    acc = collections.defaultdict(float)
    n = 0
    g = torch.Generator(device=cfg["device"]).manual_seed(1234)   # fixed masks
    for i, (video, audio, struct) in enumerate(loader):
        if i >= n_batches:
            break
        from sang.masking import per_slice_cosine_mask
        B, tv, h, w = video.shape
        mask = per_slice_cosine_mask(B, tv, h * w, video.device, generator=g)
        hm, tgt = model(video, audio, struct=struct, mask=mask, cond_drop=0.0)
        _, parts = model.head.loss(hm, tgt)
        for k, v in parts.items():
            acc[k] += float(v)
        n += 1
    model.train()
    return {k: v / max(n, 1) for k, v in acc.items()}
```

---

# Stage 4 — Real evaluation

## Change 12 — `sang/eval_metrics.py`

**Diagnosis.** Exact 32768-way token match is close to useless for generative video: perceptually identical frames routinely quantize to different token IDs. `mot_acc` can be ~0 for a good sample. Keep it as a *training probe*; select checkpoints on perceptual metrics.

```python
"""Perceptual / task metrics for talking-head generation.

Metric set (field standard):
  FVD            Frechet Video Distance                        lower better
  LPIPS          perceptual distance                           lower better
  PSNR / SSIM    reported as GAP TO THE VIDTOK CEILING         gap -> 0
  Sync-C/Sync-D  SyncNet lip-sync confidence/distance          C higher, D lower
                 (a.k.a. LSE-C / LSE-D, Prajwal et al., Wav2Lip, ACM MM 2020)
  CSIM           ArcFace cosine identity similarity            higher better

Install:
  pip install lpips facenet-pytorch scipy
  git clone https://github.com/joonson/syncnet_python  # for Sync-C/D
"""
import numpy as np
import scipy.linalg
import torch


# ------------------------------------------------------------------ Frechet
def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    diff = mu1 - mu2
    covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        off = np.eye(sigma1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma1 + off).dot(sigma2 + off))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))


def _stats(feats):
    f = feats.astype(np.float64)
    return f.mean(0), np.cov(f, rowvar=False)


class FVD:
    """
    Frechet Video Distance.

    IMPORTANT: published FVD numbers use the Kinetics-400 I3D logits-400 feature
    extractor. Pass that. The torchvision r3d_18 fallback below is fine for
    *relative* comparisons between your own runs but is NOT comparable to any
    number in a paper -- it is a different feature space.
    """

    def __init__(self, extractor=None, device="cuda"):
        self.device = device
        if extractor is not None:
            self.extractor = extractor
        else:
            import torchvision
            m = torchvision.models.video.r3d_18(weights="DEFAULT")
            m.fc = torch.nn.Identity()
            self.extractor = m.eval().to(device)
            self.comparable = False
            print("[FVD] WARNING: using r3d_18 features. Relative only; "
                  "not comparable to published FVD.")

    @torch.no_grad()
    def features(self, video):
        """video: [B, 3, T, H, W] in [-1, 1]."""
        x = (video.to(self.device) + 1) / 2
        mean = torch.tensor([0.43216, 0.394666, 0.37645], device=x.device).view(1, 3, 1, 1, 1)
        std = torch.tensor([0.22803, 0.22145, 0.216989], device=x.device).view(1, 3, 1, 1, 1)
        x = (x - mean) / std
        return self.extractor(x).float().cpu().numpy()

    def __call__(self, real_feats, fake_feats):
        m1, s1 = _stats(np.concatenate(real_feats, 0))
        m2, s2 = _stats(np.concatenate(fake_feats, 0))
        return frechet_distance(m1, s1, m2, s2)


# ------------------------------------------------------------------ LPIPS
class LPIPSMetric:
    def __init__(self, net="alex", device="cuda"):
        import lpips
        self.fn = lpips.LPIPS(net=net).to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, gen, gt):
        """gen/gt: [B, 3, T, H, W] in [-1, 1]. Averaged over frames."""
        B, C, T, H, W = gen.shape
        a = gen.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W).to(self.device)
        b = gt.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W).to(self.device)
        return float(self.fn(a, b).mean())


# ------------------------------------------------------------------ CSIM
class CSIM:
    """Identity preservation: cosine similarity of face embeddings, gen vs reference."""

    def __init__(self, device="cuda"):
        from facenet_pytorch import InceptionResnetV1
        self.net = InceptionResnetV1(pretrained="vggface2").eval().to(device)
        self.device = device

    @torch.no_grad()
    def __call__(self, gen_frames, ref_frame):
        """gen_frames: [N, 3, H, W] in [-1,1]; ref_frame: [1, 3, H, W]."""
        import torch.nn.functional as F
        g = F.interpolate(gen_frames.to(self.device), size=(160, 160),
                          mode="bilinear", align_corners=False)
        r = F.interpolate(ref_frame.to(self.device), size=(160, 160),
                          mode="bilinear", align_corners=False)
        eg = F.normalize(self.net(g), dim=-1)
        er = F.normalize(self.net(r), dim=-1)
        return float((eg @ er.T).mean())


# ------------------------------------------------------------------ SyncNet
def sync_scores(video_path, syncnet_dir, tmp_dir="/tmp/syncnet"):
    """
    Sync-C / Sync-D via joonson/syncnet_python (the metric used by Wav2Lip and
    essentially every talking-head paper since).

    NOTE: SyncNet expects ~224x224 face crops at 25 FPS with audio muxed in. Your
    128x128 output will score poorly partly for resolution reasons -- always
    report the SAME metric on the VidTok RECONSTRUCTION as a ceiling, exactly as
    you already do for PSNR/SSIM.
    """
    import os
    import re
    import subprocess

    ref = os.path.splitext(os.path.basename(video_path))[0]
    subprocess.run(
        ["python", "run_pipeline.py", "--videofile", video_path,
         "--reference", ref, "--data_dir", tmp_dir],
        cwd=syncnet_dir, check=True, capture_output=True)
    out = subprocess.run(
        ["python", "run_syncnet.py", "--videofile", video_path,
         "--reference", ref, "--data_dir", tmp_dir],
        cwd=syncnet_dir, check=True, capture_output=True, text=True).stdout

    conf = re.search(r"AV offset:\s*(-?\d+).*?Confidence:\s*([\d.]+)", out, re.S)
    dist = re.search(r"Min dist:\s*([\d.]+)", out)
    return {
        "sync_c": float(conf.group(2)) if conf else float("nan"),
        "sync_d": float(dist.group(1)) if dist else float("nan"),
        "av_offset": int(conf.group(1)) if conf else 0,
    }


# ------------------------------------------------------------------ ceiling
def ceiling_report(gen, gt, recon):
    """
    The only honest way to read PSNR/SSIM for a token model: as a fraction of the
    headroom the tokenizer actually leaves you.

    Your Run A: PSNR 9.67 with ceiling 24.06 -> achieved 0.0 of headroom (chance
    floor is ~9-10 dB for uncorrelated images). Track `psnr_frac`.
    """
    from sang.metrics import psnr, ssim
    p_gen, p_ceil = psnr(gen, gt), psnr(recon, gt)
    s_gen, s_ceil = ssim(gen, gt), ssim(recon, gt)
    floor_p = 9.5
    return {
        "psnr": p_gen, "psnr_ceiling": p_ceil,
        "psnr_frac": max(0.0, (p_gen - floor_p) / max(p_ceil - floor_p, 1e-6)),
        "ssim": s_gen, "ssim_ceiling": s_ceil,
        "ssim_frac": max(0.0, s_gen / max(s_ceil, 1e-6)),
    }
```

## `scripts/eval_full.py` (new file)

```python
#!/usr/bin/env python3
"""Full perceptual evaluation. Fixes the test.py OOM by staging VidTok on/off GPU."""
import argparse
import json

import torch

from sang.eval_metrics import FVD, CSIM, LPIPSMetric, ceiling_report, sync_scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--syncnet-dir", default=None)
    ap.add_argument("--out", default="results/test/full_eval.json")
    args = ap.parse_args()

    dev = "cuda"
    model = load_model(args.ckpt).to(dev).eval()

    # ---- PHASE 1: generate all token grids with ONLY the model resident ----
    grids, refs, gts, auds = [], [], [], []
    for video, audio, struct in val_iter(args.n):
        with torch.autocast("cuda", torch.bfloat16):
            g = model.generate(audio.to(dev), video[:, 0].to(dev),
                               shape=video.shape[1:], struct=struct,
                               steps=args.steps)
        grids.append(g.cpu()); gts.append(video.cpu()); auds.append(audio.cpu())

    # ---- Bug 4 fix: evict the model before the decoder arrives ----
    model.cpu()
    del model
    torch.cuda.empty_cache()

    # ---- PHASE 2: decode with ONLY VidTok resident ----
    vidtok = load_vidtok().to(dev).eval()
    fvd, lp, csim = FVD(device=dev), LPIPSMetric(device=dev), CSIM(device=dev)
    real_f, fake_f, rows = [], [], []

    for g, gt in zip(grids, gts):
        with torch.no_grad():
            px_gen = vidtok.decode(g.to(dev))
            px_gt = ground_truth_pixels(gt)
            px_rec = vidtok.decode(gt.to(dev))          # tokenizer ceiling
        real_f.append(fvd.features(px_gt))
        fake_f.append(fvd.features(px_gen))
        row = ceiling_report(px_gen, px_gt, px_rec)
        row["lpips"] = lp(px_gen, px_gt)
        row["lpips_ceiling"] = lp(px_rec, px_gt)
        row["csim"] = csim(px_gen[:, :, 0], px_gt[:, :, 0:1].squeeze(2))
        row["mot_acc"] = float((g[:, 1:] == gt[:, 1:]).float().mean())
        rows.append(row)

    res = {k: float(sum(r[k] for r in rows) / len(rows)) for k in rows[0]}
    res["fvd"] = fvd(real_f, fake_f)

    if args.syncnet_dir:
        res.update(sync_scores(mux_to_mp4(grids[0], auds[0]), args.syncnet_dir))

    print(json.dumps(res, indent=2))
    json.dump(res, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
```

**This also fixes Bug 4** (`test.py` OOM at 1 clip): generate everything with only the transformer resident, evict it, then decode with only VidTok resident.

---

## Change 13 — Audio conditioning

**Diagnosis.** You embed **1 of 32** Mimi codebooks at 12.5 Hz — about 1.8 audio ticks per 160 ms video slice. Mimi's first codebook is unusually strong (the Moshi paper describes distilling WavLM semantic information into it), so this is not indefensible, but it is thin for lip-sync, where the field's default conditioning is self-supervised speech features (Wav2Vec2 / HuBERT / WavLM) or Whisper encoder features at 50 Hz.

**Do not touch this until Stage 1–3 produce a falling loss.** Then A/B it.

### `sang/audio_cond.py` (new file)

```python
"""Audio conditioning front-ends. Swap via cfg['audio_cond']."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MimiCodebookStack(nn.Module):
    """
    Option A (cheap): use the first `n_codebooks` Mimi RVQ levels instead of 1.
    Level 0 carries distilled semantic/phonetic content; 1..7 add the acoustic
    detail that disambiguates similar phonemes. Requires no cache rebuild --
    you already store all 32 levels.
    """

    def __init__(self, dim, n_codebooks=8, card=2048, max_ticks=4096):
        super().__init__()
        self.n = n_codebooks
        self.emb = nn.ModuleList([nn.Embedding(card, dim) for _ in range(n_codebooks)])
        self.pos = nn.Embedding(max_ticks, dim)
        self.scale = nn.Parameter(torch.ones(n_codebooks))

    def forward(self, audio, n_ticks=None):
        a = audio[:, :self.n]
        if n_ticks is not None:
            a = a[..., :n_ticks]
        x = sum(self.scale[k] * self.emb[k](a[:, k]) for k in range(self.n))
        t = torch.arange(a.shape[-1], device=a.device)
        return x + self.pos(t).unsqueeze(0)


class SSLAudioAdapter(nn.Module):
    """
    Option B (better lip-sync, needs a cache rebuild): frozen SSL speech encoder.
    Wav2Vec2 / HuBERT / WavLM run at 50 Hz -- 4x your Mimi rate -- which is the
    resolution the lip-sync literature relies on.

    Cache change: store `ssl` float16 [T', C] alongside video/audio/struct.
    """

    def __init__(self, dim, feat_dim=768, target_ticks=20, max_ticks=4096):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(feat_dim), nn.Linear(feat_dim, dim))
        self.pos = nn.Embedding(max_ticks, dim)
        self.target_ticks = target_ticks

    def forward(self, feats, n_ticks=None):
        """feats: [B, T', C] float."""
        x = feats.transpose(1, 2)                                   # [B, C, T']
        x = F.interpolate(x, size=self.target_ticks, mode="linear", align_corners=False)
        x = self.proj(x.transpose(1, 2))                            # [B, Tt, D]
        if n_ticks is not None:
            x = x[:, :n_ticks]
        t = torch.arange(x.shape[1], device=x.device)
        return x + self.pos(t).unsqueeze(0)


@torch.no_grad()
def extract_ssl_features(wav_16k, model_name="facebook/hubert-base-ls960",
                         layer=9, device="cuda"):
    """
    Run once during cache build. Layer 9 of HuBERT-base is a common choice for
    phonetic content; sweep {6, 9, 12} if lip-sync underperforms.
    """
    from transformers import AutoModel
    m = AutoModel.from_pretrained(model_name, output_hidden_states=True).eval().to(device)
    out = m(wav_16k.to(device))
    return out.hidden_states[layer].half().cpu()          # [B, T', C]
```

### Aligning `τ(s)` cleanly

With `Ta = 9` and `Tv = 5`, `τ` is ragged. Choose `target_ticks = tv * ticks_per_slice` (e.g. `5 × 4 = 20`) and `τ(s) = (s+1) · ticks_per_slice` becomes exact:

```python
def block_masks_aligned(tv, r, ta, ticks_per_slice, device):
    assert ta == tv * ticks_per_slice, f"{ta} != {tv}*{ticks_per_slice}"
    L = tv * r
    slc = torch.arange(L, device=device) // r
    self_allow = slc.unsqueeze(1) >= slc.unsqueeze(0)
    ticks = (slc + 1) * ticks_per_slice
    aud = torch.arange(ta, device=device)
    cross_allow = aud.unsqueeze(0) < ticks.unsqueeze(1)
    return self_allow, cross_allow
```

---

## Change 14 — `configs/train_stream_v2.yaml`

```yaml
# ============================================================================
# SANG v2 -- post-diagnosis config.
# Diffs from train_stream.yaml are marked  # CHANGED / # NEW
# ============================================================================
data_glob: /beegfs/work_fast/shared/li_shared/archi_data/talkvid/clips/*/*.mp4
cache_dir: cache/e2_128_32768_win
out_dir: runs/stream_v2                 # CHANGED - do not overwrite stream_mask
max_clips: 8000
windows_per_clip: 8
val_frac: 0.05
seed: 0
workers: 4

frames: 17
res: 128
fps: 25
codebook: 32768
audio_codebooks: 32

model_type: streaming
frame0_loss_weight: 0

# ---- head -----------------------------------------------------------------
factorized_head: true                   # NEW  5 x 8-way instead of 1 x 32768-way
coupled_fsq_head: false                 # NEW  enable only if acc_token << acc_dim^5
learn_logit_scale: true                 # NEW  Change 4

# ---- masking --------------------------------------------------------------
mask_schedule: per_slice_cosine         # NEW  was: fixed Bernoulli(0.5)
mask_ratio: null                        # CHANGED  0.5 -> unused
p_inference_shaped: 0.0                 # NEW  ablation knob
context_corrupt: 0.0                    # NEW  raise to 0.05-0.15 in Stage 3

# ---- decoding -------------------------------------------------------------
decode_steps: 8                         # NEW  was refine=1 (one-shot argmax)
decode_temperature: 1.0                 # NEW
decode_gumbel: 4.5                      # NEW
decode_schedule: cosine                 # NEW

# ---- conditioning ---------------------------------------------------------
face_cond: true
cond_dropout: 0.2
audio_cond: mimi_cb0                    # NEW  mimi_cb0 | mimi_stack | ssl
audio_n_codebooks: 1                    # NEW  raise to 8 in the Stage-4 A/B

dim: 512
layers: 8
heads: 8
dropout: 0.1

# ---- loss -----------------------------------------------------------------
label_smoothing: 0.0                    # CHANGED  0.1 -> 0.0 (restore 0.05 later)
z_loss_weight: 0.0                      # CHANGED  1e-4 -> 0.0 (restore later)
fsq_loss_weight: 0.0                    # CHANGED  0.5 -> 0.0  *** the prime suspect ***

lr: 3.0e-4
min_lr: 3.0e-5
weight_decay: 0.1
betas: [0.9, 0.95]
warmup_steps: 300
max_steps: 120000
grad_clip: 1.0
batch_size: 16
grad_accum: 2

# ---- eval -----------------------------------------------------------------
log_every: 50                           # NEW  train CE curve
eval_every: 500                         # cheap teacher-forced diagnostic
full_eval_every: 5000                   # NEW  generate + decode + FVD/Sync/CSIM
patience: 0                             # CHANGED  12 -> 0 == early stopping OFF
select_on: fvd_sync                     # NEW  never select on token accuracy
device: cuda
```

---

# Runbook

```bash
# ---------------- Stage 0: is it a bug? (~1 hour) -------------------------
python scripts/sanity.py masks     --config configs/train_stream.yaml
python scripts/sanity.py forward   --config configs/train_stream.yaml
python scripts/sanity.py causality --config configs/train_stream.yaml
python scripts/sanity.py parity    --config configs/train_stream.yaml
python scripts/sanity.py overfit   --config configs/train_stream.yaml --steps 400
#   GATE: CE -> < 0.05. If not, fix the flagged bug before continuing.

# ---------------- Stage 1: loss + head (~2 hours) -------------------------
python -c "
from sang.fsq_codec import FSQIndexCodec
from sang.video import load_fsq_codes
FSQIndexCodec.from_codebook(load_fsq_codes()).self_check()"
#   Expect: roundtrip=True, levels=(8,8,8,8,8), sum(ln levels)=10.3972

python scripts/train.py --config configs/train_stream_v2.yaml \
  --set max_clips=1 windows_per_clip=1 patience=0 max_steps=3000 \
        out_dir=runs/overfit1 batch_size=1 grad_accum=1 dropout=0.0
#   GATE: ce (summed over dims) < 1.0 on a single clip within ~2k steps.

# ---------------- Stage 2: full data, new mask schedule -------------------
python scripts/train.py --config configs/train_stream_v2.yaml \
  --set max_clips=64 patience=0 max_steps=20000 out_dir=runs/stream_v2_small
#   GATE at 2k steps:  ce < 9.5   acc_dim_mean > 0.20   (chance: 10.397 / 0.125)
#   GATE at 20k steps: ce < 7.5   acc_dim_mean > 0.35

sbatch bash_scripts/train_stream.sh   # full run, configs/train_stream_v2.yaml

# ---------------- Stage 3: decoding ablation ------------------------------
for s in 1 4 8 12; do
  python scripts/eval_full.py --ckpt runs/stream_v2/best.pt --n 64 --steps $s \
    --out results/test/steps_$s.json
done
#   GATE: FVD must improve monotonically from steps=1 to steps=8.
#         Flat in steps  =>  the model is not using in-slice context yet.

# ---------------- Stage 4: real metrics -----------------------------------
python scripts/eval_full.py --ckpt runs/stream_v2/best.pt --n 256 --steps 8 \
  --syncnet-dir /path/to/syncnet_python --out results/test/full.json
```

## Expected numbers and decision gates

| Checkpoint | Metric | Chance / current | Target | If missed |
|---|---|---|---|---|
| Stage 0 overfit | CE | 10.40 | **< 0.05** | Structural bug — stop |
| Stage 1 overfit-1-clip | `ce` (sum over dims) | 10.40 | **< 1.0** @ 2k | Head/mask wiring |
| Stage 2 @ 2k steps | `ce` | 10.40 | **< 9.5** | Recipe: LR, mask, corruption |
| Stage 2 @ 2k steps | `acc_dim_mean` | 0.125 | **> 0.20** | Same |
| Stage 2 @ 20k | `acc_dim_mean` | 0.125 | **> 0.35** | Capacity / audio |
| Stage 3 | FVD(steps=8) vs FVD(steps=1) | — | **strictly better** | Not using in-slice context |
| Stage 4 | `psnr_frac` | 0.00 (Run A) | **> 0.4** | Tokenizer/resolution |
| Stage 4 | Sync-C | ~0 | **> 2**, ideally > 4 | Audio conditioning (Change 13) |
| Stage 4 | CSIM | — | **> 0.6** | Reference conditioning too weak |

The single most informative number in this whole document is `acc_dim_mean` against its 0.125 chance line. It is dense, it moves early, and unlike token accuracy it is not drowned in a 32768-way denominator. **If it lifts off 0.125, you are learning.** Your current setup could not have told you that.

---

# Caveats

- **I have not run any of this against your repo.** The three numeric cores (FSQ digit LUT, per-slice cosine sampler, MaskGIT schedule) were executed and verified standalone; everything else is written against the interfaces described in `history.md` and needs your names substituted. Treat `# ASSUMPTION:` markers as edit points.
- **The FSQ-loss diagnosis is a strong hypothesis, not a proven fact.** It is consistent with every symptom you report, but Stage 0 is what localizes the cause. It is entirely possible that more than one hypothesis is true at once (e.g. FSQ-loss flattening *and* an attention-mask issue) — which is why the ablations are additive rather than simultaneous.
- **`FSQIndexCodec.from_codebook` assumes VidTok's implicit codebook is a clean product lattice.** It raises if `prod(levels) != V`. If VidTok's FSQ uses a non-product or reordered layout, factorized heads do not apply and you should fall back to the single head plus the logit-scale fix. Run `self_check()` first — it costs milliseconds.
- **The KV-cache path (Change 10) is the highest-risk item here** and is deliberately last. Do not ship it without the exact-equality assertion against the uncached path at `temperature=0, gumbel_temp=0`.
- **FVD with `r3d_18` features is not comparable to published numbers.** For anything going in a paper, use the Kinetics-400 I3D extractor. Also note the download domains for I3D, SyncNet, ArcFace, and LPIPS weights are likely blocked by your cluster's egress allowlist — stage those files manually.
- **Target numbers in the gate table are engineering heuristics** calibrated to "is this learning at all," not published SOTA. Pull real TalkVid baseline figures from the source paper before treating any of them as a quality bar.
- **Section ordering matters more than any individual change.** If you apply Changes 6 and 8 without first resolving Stage 0, you will not be able to attribute the result.