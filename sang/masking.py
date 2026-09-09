"""Mask sampling for block-causal masked-token training."""
import math

import torch


def cosine_gamma(u: torch.Tensor) -> torch.Tensor:
    return torch.cos(math.pi * u / 2.0)


def per_slice_cosine_mask(B, tv, r, device, ref_slices=1, min_masked=1,
                          generator=None, dtype=torch.bool):
    """Independent MaskGIT cosine mask per slice. Returns bool [B, tv*r], True = masked."""
    u = torch.rand(B, tv, 1, device=device, generator=generator)
    gamma = cosine_gamma(u)
    n_mask = torch.ceil(gamma * r).clamp_(min_masked, r)

    scores = torch.rand(B, tv, r, device=device, generator=generator)
    rank = scores.argsort(dim=-1).argsort(dim=-1)
    mask = rank < n_mask
    mask[:, :ref_slices] = False
    return mask.reshape(B, tv * r).to(dtype)


def inference_shaped_mask(B, tv, r, device, ref_slices=1, generator=None):
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
    b = inference_shaped_mask(B, tv, r, device, ref_slices=kw.get("ref_slices", 1))
    return torch.where(use_inf.unsqueeze(-1), b, a)


def corrupt_context(video_idx, mask, vocab, p_corrupt=0.0, ref_slices=1, r=None, generator=None):
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


def maskgit_decode_schedule(r: int, steps: int, schedule: str = "cosine") -> list[int]:
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
