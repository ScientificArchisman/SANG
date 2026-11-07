"""Per-person renderer calibration from the footage (research log §3.38). LivePortrait's SPADE decoder is fitted to
reproduce the person's own footage frames, each rendered from the photo with that frame's own motion, so it learns
the teeth, inner mouth, wrinkles and skin that one closed-mouth photo cannot show.

  recon   zero-initialised LoRA on the decoder's convs, one render per frame, scored by image_score with m = 1: for a
          deterministic renderer the energy score is a multiscale distance to the real frame (MimicTalk-style
          reconstruction, arXiv 2410.06734)
  energy  the same LoRA plus zero-initialised noise inputs (a stochastic renderer), 4 renders per frame and the same
          score, whose spread term is now live: strictly proper, so its minimum is the real frames' law given the
          motion. Details the motion does not decide come out sharp instead of averaged, with no discriminator that
          could overfit 750 frames. At inference one noise map per clip, so details do not flicker between frames.
The warping network stays frozen; only the decoder's adapter trains. The adapter is removed when the context exits.
"""
import contextlib
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from sang.render.pipeline import crop_mask


class Noise:
    """The noisy convs' noise: fresh per sample while fitting; with shared = True one fixed map per conv (the same
    for every frame of a clip)."""

    def __init__(self, seed: int, device):
        self.seed, self.shared = seed, False
        self.gen = torch.Generator(device=device).manual_seed(seed)


class ConvLoRA(nn.Module):
    """base(x') + up(down(x')) with x' = x + s * noise (s only with a Noise); up = 0 and s = 0 at the start, so a
    fresh adapter changes nothing."""

    def __init__(self, base: nn.Conv2d, rank: int, noise: Noise | None = None, idx: int = 0):
        super().__init__()
        dev = next(base.parameters()).device
        self.base, self.noise, self.idx = base, noise, idx
        self.down = nn.Conv2d(base.in_channels, rank, base.kernel_size, base.stride, base.padding, base.dilation,
                              bias=False).to(dev)
        self.up = nn.Conv2d(rank, base.out_channels, 1, bias=False).to(dev)
        nn.init.zeros_(self.up.weight)
        self.s = nn.Parameter(torch.zeros(base.in_channels, device=dev)) if noise is not None else None

    def forward(self, x):
        if self.s is not None:
            B, _, H, W = x.shape
            n = self.noise
            if n.shared:
                g = torch.Generator(device=x.device).manual_seed(n.seed * 1000 + self.idx)
                z = torch.randn(1, 1, H, W, device=x.device, generator=g).expand(B, -1, -1, -1)
            else:
                z = torch.randn(B, 1, H, W, device=x.device, generator=n.gen)
            x = x + (self.s.view(1, -1, 1, 1) * z).to(x.dtype)
        return self.base(x) + self.up(self.down(x))


def haar2d(x: torch.Tensor, levels: int) -> list[torch.Tensor]:
    """Orthonormal 2D Haar levels of x [..., H, W]: per level the 3 detail bands [..., 3, h, w] (finest first), then
    the coarse mean [..., 1, h, w]."""
    out = []
    for _ in range(levels):
        a, b, c, d = x[..., 0::2, 0::2], x[..., 0::2, 1::2], x[..., 1::2, 0::2], x[..., 1::2, 1::2]
        out.append(torch.stack([a - b + c - d, a + b - c - d, a - b - c + d], -3) / 2)
        x = (a + b + c + d) / 2
    return out + [x.unsqueeze(-3)]


def image_scale(y: torch.Tensor, levels: int) -> tuple[torch.Tensor, torch.Tensor]:
    """image_score's constants from pairs of different real frames y [N, 3, S, S] (N >= 2): E||Y - Y'|| of the whole
    image and of every (level, band, RGB channel) block, in image_score's block order."""
    with torch.autocast(y.device.type, enabled=False):
        y = y.float()
        c0 = (y - y.roll(1, 0)).flatten(1).norm(dim=-1).mean()
        cb = torch.cat([(h - h.roll(1, 0)).flatten(-2).norm(dim=-1).mean(0).flatten() for h in haar2d(y, levels)])
        return c0, cb.clamp_min(1e-6)


def image_score(x: torch.Tensor, y: torch.Tensor, c0: torch.Tensor, cb: torch.Tensor, levels: int) -> torch.Tensor:
    """2D multiscale-block energy score per frame, the image analogue of sang.model.scoring.mbes: renders x
    [B, m, 3, S, S] and the real frame y [B, 3, S, S] (both masked).
      ES(whole image) + c0 x mean over blocks b of ES(b) / c_b
    with a block one RGB channel of one Haar band at one level. m = 1 (a deterministic renderer) leaves the distance
    term only. Strictly proper for m >= 2 samples of a stochastic renderer."""
    m = x.shape[1]
    i, j = torch.triu_indices(m, m, 1, device=x.device)

    def es(u, v):  # u [B, m, ..., n], v [B, ..., n] -> [B, ...]
        e = (u - v[:, None]).norm(dim=-1).mean(1)
        return e - 0.5 * (u[:, i] - u[:, j]).norm(dim=-1).mean(1) if m > 1 else e

    with torch.autocast(x.device.type, enabled=False):
        x, y = x.float(), y.float()
        blocks = torch.cat([es(a.flatten(-2), b.flatten(-2)).flatten(1)
                            for a, b in zip(haar2d(x, levels), haar2d(y, levels))], 1)
        return es(x.flatten(2), y.flatten(1)) + c0 * (blocks / cb).mean(1)


def to_crop(frames: np.ndarray, M_c2o, S: int, device, chunk: int = 32) -> torch.Tensor:
    """Full frames [T, H, W, 3] uint8 -> [T, 3, S, S] uint8 in the photo's crop. M_c2o maps a crop pixel to a frame
    pixel, as GpuPaster pastes it back, so a render from the photo lines up with the real frame."""
    from sang.render.liveportrait import as3x3
    T, H, W = frames.shape[:3]
    M = torch.from_numpy(as3x3(np.asarray(M_c2o, np.float64))).float().to(device)
    v, u = torch.meshgrid(torch.arange(S, device=device, dtype=torch.float32),
                          torch.arange(S, device=device, dtype=torch.float32), indexing="ij")
    p = M @ torch.stack([u.flatten(), v.flatten(), torch.ones(S * S, device=device)])
    grid = torch.stack([p[0] / (W - 1) * 2 - 1, p[1] / (H - 1) * 2 - 1], -1).view(1, S, S, 2)
    out = []
    for i in range(0, T, chunk):
        f = torch.from_numpy(np.ascontiguousarray(frames[i:i + chunk])).to(device).permute(0, 3, 1, 2).float()
        f = F.grid_sample(f, grid.expand(len(f), -1, -1, -1), mode="bilinear", padding_mode="zeros", align_corners=True)
        out.append(f.round().clamp(0, 255).to(torch.uint8))
    return torch.cat(out)


@contextlib.contextmanager
def calibrated(codec, state: dict, M_c2o, frames: np.ndarray, motion: torch.Tensor, mode: str = "recon",
               steps: int = 300, lr: float = 3e-4, rank: int = 8, renders: int = 8, hold: int = 50, levels: int = 6,
               seed: int = 0, log: dict | None = None):
    """Inside the context, codec's decoder is fitted to this person's footage; it is restored on exit.

    state: codec.source_state of the photo's crop; M_c2o its crop matrix (crop_image); frames [T, H, W, 3] uint8 the
    footage and motion [T, 70] its absolute LivePortrait motion (MotionCodec.extract). mode "recon" or "energy" (see
    the module docstring); renders = decoder passes per step (recon: that many frames x 1; energy: renders / 4 frames
    x 4 samples), so both arms cost the same per step. The last `hold` frames are held out: log gets rfit_es0 /
    rfit_es_<i> / rfit_es, the held-out score before, at each quarter and after the fit (comparable across arms: a
    deterministic renderer's energy score is its distance), rfit_s, and for energy rfit_noise, the mean noise gain
    (0 = it stayed deterministic)."""
    G, dev = codec.lp.spade_generator, codec.device
    T = min(len(frames), len(motion))
    if mode not in ("recon", "energy"):
        raise ValueError(f"render calibration mode '{mode}': recon or energy")
    if T - hold < renders:
        if log is not None:
            log["rfit_skipped"] = 1.0
        yield
        return
    t0 = time.perf_counter()
    m = 4 if mode == "energy" else 1
    noise = Noise(seed, dev) if mode == "energy" else None
    G.eval().requires_grad_(False)
    sites = []
    for blk in list(G.modules()):
        if hasattr(blk, "conv_0") and hasattr(blk, "conv_1"):  # LivePortrait's SPADEResnetBlock
            sites += [(blk, "conv_0"), (blk, "conv_1")]
    sites.append((G.conv_img, "0") if isinstance(G.conv_img, nn.Sequential) else (G, "conv_img"))
    saved = [(mod, name, getattr(mod, name)) for mod, name in sites]
    params = []
    try:
        for k, (mod, name, base) in enumerate(saved):
            lora = ConvLoRA(base, rank, noise if name in ("conv_0", "conv_1") else None, k)
            setattr(mod, name, lora)
            params += [p for p in (lora.down.weight, lora.up.weight, lora.s) if p is not None]
        amp = lambda: torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev != "cpu")
        with torch.no_grad():
            x_d = codec.driving_keypoints(state, motion[:T])
            S = G(feature=codec.warped(state, x_d[:1])).shape[-1]
            y_all = to_crop(frames[:T], M_c2o, S, dev)
            w = torch.from_numpy(crop_mask(S)).to(dev)
            g = torch.Generator(device=dev).manual_seed(seed)
            real = y_all[torch.randperm(T - hold, generator=g, device=dev)[:64]].float() / 255 * w
            c0, cb = image_scale(real, levels)
            del real
        held = torch.linspace(T - hold, T - 1, min(8, hold), device=dev).long()

        def render(idx):
            with torch.no_grad():
                f = codec.warped(state, x_d[idx])
            with amp():
                x = G(feature=f.repeat_interleave(m, 0))
            return x.float().view(len(idx), m, *x.shape[1:]) * w

        def score() -> float:  # the held-out frames, with fixed noise
            st = noise.gen.get_state() if noise else None
            if noise:
                noise.gen.manual_seed(seed + 1)
            with torch.no_grad():
                out = float(image_score(render(held), y_all[held].float() / 255 * w, c0, cb, levels).mean())
            if noise:
                noise.gen.set_state(st)
            return out

        before = score() if log is not None else None
        opt = torch.optim.Adam(params, lr=lr)
        every, nf = max(1, steps // 4), max(1, renders // m)
        for i in range(1, steps + 1):
            idx = torch.randint(0, T - hold, (nf,), generator=g, device=dev)
            loss = image_score(render(idx), y_all[idx].float() / 255 * w, c0, cb, levels).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if log is not None and i % every == 0 and i < steps:
                log[f"rfit_es_{i}"] = score()
        for p in params:
            p.requires_grad_(False)
        if log is not None:
            log.update(rfit_es0=before, rfit_es=score(), rfit_s=time.perf_counter() - t0)
            if noise:
                log["rfit_noise"] = float(torch.cat([p.abs() for p in params if p.dim() == 1]).mean())
        if noise:
            noise.shared = True
        yield
    finally:
        for mod, name, base in saved:
            setattr(mod, name, base)
