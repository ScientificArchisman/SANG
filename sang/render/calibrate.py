"""Shape-locked SPADE calibration of LivePortrait's decoder to one person (reports/SANG test time style transfer.md,
design 1). Every SPADE layer's modulation becomes

    gamma' = (1 + a) gamma + b,   beta' = (1 + c) beta + d

with one a, b, c, d per channel, all starting at 0 (8,000 channels over 18 layers, 32k values). The convs that predict
gamma and beta, the warping network, the appearance and motion extractors, stitching and conv_img stay frozen, so the
offsets only re-weight modulation the warped feature already supplies: they cannot add grain or move geometry.

Pairs: footage frame t's motion drives the TEST PHOTO; the real frame t, cropped with the box its motion was read in,
is the target. Data loss per pair:
  amplitude  L1 between per-channel log-amplitude spectra of Hann windows (1/8 of the map, half overlap) of the
             appearance extractor's three 2-D stages (64@256, 128@128, 256@64). A small misalignment moves phase, not
             amplitude, so it does not pay the decoder to blur (the L2 fit's failure).
  colour     L1 between 8x average-pooled RGB.
Pair weight exp(-r / median r), r = distance of the motion extractor's keypoints of the frozen render from the driving
keypoints (pairs the photo cannot reproduce count less). Regularisers: a Gaussian prior on the offsets, an 8x-pooled
L1 anchor on the photo rendered at its own pose, and a shape lock (the motion extractor's keypoints of each calibrated
render stay those of the frozen render). Two passes in temporal order, one clipped Adam step per window of 8 pairs.
accept = the data loss fell on held-out footage.
"""
import contextlib
import time

import numpy as np
import torch
import torch.nn.functional as F

EPS = 1e-3


def spades(G) -> list:
    """G's SPADE layers in module order (LivePortrait's SPADE: mlp_gamma / mlp_beta convs)."""
    return [m for m in G.modules() if hasattr(m, "mlp_gamma") and hasattr(m, "mlp_beta")]


def zeros(G) -> list[torch.Tensor]:
    """Offsets that change nothing: one [4, C] tensor (a, b, c, d) per SPADE layer."""
    dev = next(G.parameters()).device
    return [torch.zeros(4, s.mlp_gamma.out_channels, device=dev) for s in spades(G)]


@contextlib.contextmanager
def offsets(G, theta):
    """Inside, G's SPADE modulation carries the offsets theta (zeros() layout; None = unchanged)."""
    hooks, layers = [], spades(G)
    if theta is not None and [len(t[0]) for t in theta] != [s.mlp_gamma.out_channels for s in layers]:
        raise ValueError("offsets do not match this decoder's SPADE layers")
    try:
        dev = next(G.parameters()).device
        for s, t in zip(layers, theta or []):
            t = t.to(dev)
            for conv, (a, b) in ((s.mlp_gamma, t[:2]), (s.mlp_beta, t[2:])):
                hooks.append(conv.register_forward_hook(
                    lambda m, i, o, a=a, b=b: o * (1 + a.view(1, -1, 1, 1)) + b.view(1, -1, 1, 1)))
        yield
    finally:
        for h in hooks:
            h.remove()


def stages(Fx, img: torch.Tensor) -> list[torch.Tensor]:
    """The appearance extractor's 2-D stages of img [B, 3, S, S] in [0, 1], resized to its 256 input."""
    h = [Fx.first(F.interpolate(img, size=(256, 256), mode="bilinear", antialias=True, align_corners=False))]
    for blk in Fx.down_blocks:
        h.append(blk(h[-1]))
    return h


def log_amp(h: torch.Tensor, frac: int = 8) -> torch.Tensor:
    """Log-amplitude spectra of Hann windows 1/frac of the map wide, half overlap: [B, C, H, W] -> [B, C, n, n, k, k/2+1]."""
    k = max(2, h.shape[-1] // frac)
    p = h.float().unfold(2, k, k // 2).unfold(3, k, k // 2)
    w = torch.hann_window(k, periodic=False, device=h.device)
    return torch.log(torch.fft.rfft2(p * (w[:, None] * w)).abs() + EPS)


def data_loss(Fx, x, y, w, stage_w=(1.0, 0.5, 0.25), rgb: float = 10.0) -> torch.Tensor:
    """Per-pair data loss [B] of renders x and real crops y [B, 3, S, S] in [0, 1], both masked by w [S, S]."""
    x, y = x * w, y * w
    out = rgb * (F.avg_pool2d(x, 8) - F.avg_pool2d(y, 8)).abs().flatten(1).mean(1)
    with torch.no_grad():
        hy = stages(Fx, y)
    for s, a, b in zip(stage_w, stages(Fx, x), hy):
        if s:
            out = out + s * (log_amp(a) - log_amp(b)).abs().flatten(1).mean(1)
    return out


def blur(x: torch.Tensor, s: float = 2.0) -> torch.Tensor:
    """Separable Gaussian blur of [B, C, H, W] (std s px)."""
    r, c = int(3 * s), x.shape[1]
    g = torch.exp(-torch.arange(-r, r + 1, device=x.device, dtype=x.dtype) ** 2 / (2 * s * s))
    g = (g / g.sum()).view(1, 1, 1, -1).expand(c, 1, 1, -1)
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), g, groups=c)
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), g.transpose(2, 3), groups=c)


def real(d: dict, i) -> torch.Tensor:
    return d["Y"][i].float() / 255


@torch.no_grad()
def check(G, Fx, d: dict, w, seed: int = 0, bs: int = 8, **loss_kw) -> dict:
    """Pre-flight on pairs d: the data loss of frozen renders must RISE when white noise (std 0.25 / 0.5 / 1 x the
    render's RMS error) is added and when the render is blurred. If not, the loss rewards grain or blur, as the pixel
    energy score did (FVD 944). pre_ok = 1 when every difference is positive."""
    g = torch.Generator(device=w.device).manual_seed(seed)
    diff = {f"pre_noise_{f:g}": 0.0 for f in (0.25, 0.5, 1.0)} | {"pre_blur": 0.0}
    for i in range(0, len(d["X"]), bs):
        j = slice(i, i + bs)
        x, y = G(feature=d["X"][j]).float(), real(d, j)
        base = data_loss(Fx, x, y, w, **loss_kw)
        rms = ((x - y) * w).pow(2).flatten(1).mean(1).sqrt().view(-1, 1, 1, 1)
        for f in (0.25, 0.5, 1.0):
            z = torch.randn(x.shape, generator=g, device=x.device)
            diff[f"pre_noise_{f:g}"] += float((data_loss(Fx, (x + f * rms * z).clamp(0, 1), y, w, **loss_kw) - base).sum())
        diff["pre_blur"] += float((data_loss(Fx, blur(x), y, w, **loss_kw) - base).sum())
    out = {k: v / len(d["X"]) for k, v in diff.items()}
    return out | {"pre_ok": float(all(v > 0 for v in out.values()))}


def to_crop(frames: np.ndarray, M_c2o, S: int, device, chunk: int = 32) -> torch.Tensor:
    """Full frames [T, H, W, 3] uint8 -> [T, 3, S, S] uint8 in a crop. M_c2o maps a crop pixel to a frame pixel, as
    GpuPaster pastes it back."""
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


@torch.no_grad()
def pairs(codec, state: dict, frames: np.ndarray, motion: torch.Tensor, M_c2o, idx) -> dict:
    """Fitting pairs at frame indices idx: X the decoder inputs of the photo (state) driven by each frame's absolute
    motion [T, 70], Y the real frames [T, H, W, 3] cropped with the box that motion was read in (M_c2o), kp the
    driving keypoints."""
    kp = codec.driving_keypoints(state, motion[idx].float().to(codec.device))
    X = torch.cat([codec.warped(state, kp[i:i + 16]).float() for i in range(0, len(kp), 16)])
    S = codec.lp.spade_generator(feature=X[:1]).shape[-1]
    return {"X": X, "Y": to_crop(frames[idx], M_c2o, S, codec.device), "kp": kp}


def keypoints(lp, img: torch.Tensor) -> torch.Tensor:
    """LivePortrait's implicit keypoints [B, 21, 3] of renders img [B, 3, S, S] in [0, 1], with gradients
    (get_kp_info runs under no_grad)."""
    x = F.interpolate(img, size=(256, 256), mode="bilinear", antialias=True, align_corners=False)
    return lp.transform_keypoint(lp.motion_extractor(x))


def fit(G, Fx, kp_of, train: dict, held: dict | None, anchor: torch.Tensor, w: torch.Tensor, passes: int = 2,
        window: int = 8, lr: float = 3e-3, clip: float = 1.0, prior_s: float = 0.05, shape: float = 10.0,
        anchor_w: float = 10.0, **loss_kw) -> tuple[list[torch.Tensor], dict]:
    """Fit the offsets on train pairs (pairs()); returns (theta on the CPU, log). held = None skips the held-out check
    (the oracle, fitted on the test segment). anchor: the photo's decoder input at its own pose [1, 256, 64, 64];
    kp_of: renders -> keypoints (keypoints()). G is unchanged afterwards."""
    t0 = time.perf_counter()
    G.eval().requires_grad_(False)
    Fx.eval().requires_grad_(False)

    @torch.no_grad()
    def frozen(d):  # the frozen renders' keypoints, and pair weights from how far they are from the driving ones
        kp = torch.cat([kp_of(G(feature=d["X"][i:i + window]).float()) for i in range(0, len(d["X"]), window)])
        r = (kp - d["kp"]).norm(dim=-1).mean(-1)
        return kp, torch.exp(-r / r.median().clamp_min(1e-8))

    @torch.no_grad()
    def held_loss(theta):
        with offsets(G, theta):
            ls = torch.cat([data_loss(Fx, G(feature=held["X"][i:i + window]).float(), real(held, slice(i, i + window)),
                                      w, **loss_kw) for i in range(0, len(held["X"]), window)])
        return float((ls * wh).sum() / wh.sum())

    kp0, wt = frozen(train)
    with torch.no_grad():
        a0 = G(feature=anchor).float()
    theta = [t.requires_grad_() for t in zeros(G)]
    log = {}
    if held is not None:
        wh = frozen(held)[1]
        log["cal_held0"] = held_loss(None)
    opt = torch.optim.Adam(theta, lr=lr)
    P = len(train["X"])
    for _ in range(passes):
        for i in range(0, P, window):
            j = slice(i, i + window)
            with offsets(G, theta):
                x = G(feature=train["X"][j]).float()
                d = (wt[j] * data_loss(Fx, x, real(train, j), w, **loss_kw)).sum() / wt[j].sum()
                lock = (kp_of(x) - kp0[j]).pow(2).sum(-1).mean()
                anc = F.avg_pool2d((G(feature=anchor).float() - a0) * w, 8).abs().mean()
            prior = torch.cat([t.flatten() for t in theta]).pow(2).mean() / (2 * prior_s ** 2)
            loss = d + shape * lock + anchor_w * anc + prior
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(theta, clip)
            opt.step()
    theta = [t.detach() for t in theta]
    log.update(cal_data=d.item(), cal_lock=lock.item(), cal_anchor=anc.item(), cal_prior=prior.item(),
               cal_theta=float(torch.cat([t.flatten() for t in theta]).abs().mean()), cal_pairs=float(P))
    if held is not None:
        log["cal_held"] = held_loss(theta)
        log["cal_accept"] = float(log["cal_held"] < log["cal_held0"])
    log["cal_s"] = time.perf_counter() - t0
    return [t.cpu() for t in theta], log
