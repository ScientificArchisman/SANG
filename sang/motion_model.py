"""Bidirectional motion-flow transformer: audio + reference -> a 42-d motion trajectory.

Non-streaming by design. The whole window is generated jointly, so there is no exposure bias to
correct and no causal mask.

Why this is not the streaming backbone with the mask removed: sang.diffusion.DiffusionHead is a
per-token MLP, so given the backbone's hidden states every frame is denoised independently. That is
sound when frames are produced autoregressively (each chunk conditions on the previous chunk's
SAMPLE, which is where coherence comes from -- MAR and NOVA get it from multi-step masked
generation). Generating a window in one pass instead factorises p(m_1..m_T | audio) into
prod_i p(m_i | h_i): a mean trajectory plus independent per-frame noise, i.e. jitter. FLOAT
(ICCV'25), KDTalker (IJCV'25) and Ditto (MM'25) all solve it the same way, and so does this module:
the noisy trajectory itself is the transformer's input, so self-attention couples the noise across
frames. tests/test_motion_model.py checks that coupling directly.

Conditioning is frame-wise AdaLN (FLOAT sec. 4.2): each frame is modulated by ITS OWN condition
c_l = time(t_l) + audio_l + ref. FLOAT's ablation (Tab. 3, HDTF) against cross-attention under the
same mask: LSE-D 7.290 vs 7.757. audio_l is a symmetric window of WavLM ticks around frame l, so the
audio-visual alignment is built into the architecture instead of learned from positions.

Long clips follow FLOAT: each window is conditioned on the last P generated frames as a clean
prefix, and that prefix is dropped with p = 0.5 in training so the first window of a clip -- which
has none -- is in distribution.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

from sang.diffusion import SinusoidalTimeEmb, _modulate
from sang.motion import REGIONS, T_DIM
from sang.streaming_transformer import SwiGLU

REF_DIM = 63 + T_DIM          # canonical keypoints x_c + the reference frame's own target vector


class DiTBlock(nn.Module):
    """Pre-norm self-attention + SwiGLU, each modulated per FRAME by adaLN-Zero (shift, scale, gate)."""

    def __init__(self, dim: int, heads: int, dropout: float = 0.0):
        super().__init__()
        self.n1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.n2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = SwiGLU(dim)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)           # every block starts as the identity

    def forward(self, x: torch.Tensor, c: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        s1, k1, g1, s2, k2, g2 = self.ada(c).chunk(6, dim=-1)
        h = _modulate(self.n1(x), s1, k1)
        x = x + g1 * self.attn(h, h, h, attn_mask=mask, need_weights=False)[0]
        return x + g2 * self.ff(_modulate(self.n2(x), s2, k2))


class MotionFlowTransformer(nn.Module):
    """v_theta(x_t, t | audio, ref, prefix) over a whole window.

    x_t    [B, L, 42]          noisy target frames
    t      [B]                 flow time; 0 = clean, 1 = noise (sang.diffusion's convention)
    audio  [B, 2(P+L), 1024]   WavLM ticks at 50 Hz covering prefix + target frames
    ref    [B, REF_DIM]        normalised [x_c, reference target]
    prefix [B, P, 42] | None   clean previous frames (FLOAT's L'); None = no prefix at all
    -> v   [B, L, 42]
    """

    def __init__(self, dim: int = 512, depth: int = 8, heads: int = 8, audio_dim: int = 1024,
                 ticks_per_frame: int = 2, audio_kernel: int = 5, attn_window: int = 0,
                 max_frames: int = 512, dropout: float = 0.0):
        super().__init__()
        if audio_kernel % 2 == 0:
            raise ValueError("audio_kernel must be odd so the window is centred on its frame")
        self.tpf, self.attn_window = ticks_per_frame, attn_window
        self.in_proj = nn.Linear(T_DIM, dim)
        self.pos = nn.Embedding(max_frames, dim)
        self.prefix_tag = nn.Parameter(torch.zeros(dim))      # marks clean context frames
        self.null_prefix = nn.Parameter(torch.zeros(dim))     # stands in for a dropped prefix
        # A centred conv over frames IS the symmetric audio window: kernel 5 = +/-2 frames = +/-80 ms.
        self.audio = nn.Sequential(
            nn.Conv1d(audio_dim * ticks_per_frame, dim, audio_kernel, padding=audio_kernel // 2),
            nn.SiLU(), nn.Conv1d(dim, dim, 1))
        self.ref = nn.Sequential(nn.Linear(REF_DIM, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time = nn.Sequential(SinusoidalTimeEmb(dim), nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.null_audio = nn.Parameter(torch.zeros(dim))
        self.null_ref = nn.Parameter(torch.zeros(dim))
        self.blocks = nn.ModuleList([DiTBlock(dim, heads, dropout) for _ in range(depth)])
        self.out_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.out_ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.out = nn.Linear(dim, T_DIM)
        for lin in (self.out_ada[-1], self.out):
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)                 # v starts at exactly 0

    # ------------------------------------------------------------------ pieces
    def _audio_frames(self, audio: torch.Tensor, n_frames: int) -> torch.Tensor:
        """[B, >=tpf*F, D] ticks -> [B, F, dim], one centred window per frame."""
        B, _, D = audio.shape
        a = audio[:, : n_frames * self.tpf].float()
        if a.shape[1] < n_frames * self.tpf:                  # clip ended: pad with silence ticks
            a = F.pad(a, (0, 0, 0, n_frames * self.tpf - a.shape[1]))
        a = a.reshape(B, n_frames, self.tpf * D).transpose(1, 2)
        return self.audio(a).transpose(1, 2)

    def _mask(self, n: int, device) -> torch.Tensor | None:
        if self.attn_window <= 0:
            return None                                      # full bidirectional attention
        i = torch.arange(n, device=device)
        return (i[:, None] - i[None, :]).abs() > self.attn_window   # True = may NOT attend

    # ------------------------------------------------------------------ forward
    def forward(self, x_t, t, audio, ref, prefix=None, prefix_keep=None,
                drop_audio=None, drop_ref=None):
        B, L, _ = x_t.shape
        P = 0 if prefix is None else prefix.shape[1]
        n = P + L
        dev = x_t.device

        tok = self.in_proj(x_t.float())
        if P:
            ptok = self.in_proj(prefix.float()) + self.prefix_tag
            if prefix_keep is not None:                      # FLOAT: drop the whole prefix, p = 0.5
                ptok = torch.where(prefix_keep.view(B, 1, 1), ptok, self.null_prefix.expand_as(ptok))
            tok = torch.cat([ptok, tok], 1)
        tok = tok + self.pos(torch.arange(n, device=dev))[None]

        # frame-wise condition: prefix frames are clean, so their flow time is 0
        t_frames = torch.cat([torch.zeros(B, P, device=dev), t.float().view(B, 1).expand(B, L)], 1)
        c_time = self.time(t_frames.reshape(-1)).view(B, n, -1)
        c_audio = self._audio_frames(audio, n)
        if drop_audio is not None:
            c_audio = torch.where(drop_audio.view(B, 1, 1), self.null_audio.expand_as(c_audio), c_audio)
        if P and prefix_keep is not None:                    # a dropped prefix drops its audio too
            keep = torch.cat([prefix_keep.view(B, 1).expand(B, P), torch.ones(B, L, dtype=torch.bool, device=dev)], 1)
            c_audio = torch.where(keep.unsqueeze(-1), c_audio, self.null_audio.expand_as(c_audio))
        c_ref = self.ref(ref.float())
        if drop_ref is not None:
            c_ref = torch.where(drop_ref.view(B, 1), self.null_ref.expand_as(c_ref), c_ref)
        c = c_time + c_audio + c_ref.unsqueeze(1)

        mask = self._mask(n, dev)
        x = tok
        for blk in self.blocks:
            x = blk(x, c, mask)
        shift, scale = self.out_ada(c).chunk(2, dim=-1)
        return self.out(_modulate(self.out_norm(x), shift, scale))[:, P:]


# ---------------------------------------------------------------------- objective
def region_mse(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """Mean within each region, summed over regions (AVTR-1 eq. 5).

    A flat MSE over 42 columns gives head rotation 3/42 of the weight and the mouth 18/42; this
    gives rotation, brow, eyes and mouth one share each regardless of how many columns they have."""
    parts = {k: F.mse_loss(pred[..., idx].float(), target[..., idx].float()) for k, idx in REGIONS.items()}
    return sum(parts.values()), parts


def sample_t(B: int, device, dist: str = "uniform", mean: float = 0.0, scale: float = 1.0) -> torch.Tensor:
    """Flow times (t = 1 is noise). logit_normal: sigmoid(mean + scale * N(0, 1)) (SD3, arXiv 2403.03206);
    mean > 0 trains more at high noise, where the audio decides the coarse mouth timing."""
    if dist == "logit_normal":                               # AVTR-1, SD3: weight the middle of the path
        return torch.sigmoid(mean + scale * torch.randn(B, device=device))
    return torch.rand(B, device=device)                      # FLOAT


def lip_ccc_loss(x0_hat: torch.Tensor, x0: torch.Tensor, a: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Weighted mean over the batch of 1 - CCC(lip opening of x0_hat, of x0) over each window.

    a [42]: the lip-opening readout in NORMALISED target space (naturalness.Readout.in_z); its constant
    cancels in CCC. CCC scores timing and amplitude together, so unlike a pure sync loss it cannot be
    satisfied by exaggerating the mouth. w [B]: per-sample weight (0 = skip)."""
    o_hat, o = x0_hat.float() @ a.to(x0_hat).float(), x0.float() @ a.to(x0).float()        # [B, L]
    mh, m = o_hat.mean(1, keepdim=True), o.mean(1, keepdim=True)
    cov = ((o_hat - mh) * (o - m)).mean(1)
    ccc = 2 * cov / (o_hat.var(1, unbiased=False) + o.var(1, unbiased=False) + (mh - m).squeeze(1) ** 2 + 1e-6)
    return ((1 - ccc) * w).sum() / w.sum().clamp_min(1e-6)


def spectral_loss(x0_hat: torch.Tensor, x0: torch.Tensor, cols, k_max: int, w: torch.Tensor) -> torch.Tensor:
    """Weighted mean of L1 between the time-axis rFFTs (orthonormal) of x0_hat and x0 on `cols`, over
    bins 0..k_max only (FreDF, arXiv 2402.02399, band-limited here). Bins above k_max are left free:
    there extractor jitter dominates, and matching it would teach jitter."""
    f_hat = torch.fft.rfft(x0_hat[..., cols].float(), dim=1, norm="ortho")[:, : k_max + 1]
    f = torch.fft.rfft(x0[..., cols].float(), dim=1, norm="ortho")[:, : k_max + 1]
    per = (f_hat - f).abs().mean(dim=(1, 2))                                                # [B]
    return (per * w).sum() / w.sum().clamp_min(1e-6)


def flow_loss(model: MotionFlowTransformer, batch: dict, lam_vel: float = 1.0,
              p_audio: float = 0.1, p_ref: float = 0.1, p_prefix: float = 0.5,
              t_dist: str = "uniform", t_mean: float = 0.0, t_scale: float = 1.0,
              lam_cfm: float = 0.0, cfm_regions=("mouth",),
              lam_ccc: float = 0.0, lip_a: torch.Tensor | None = None,
              lam_spec: float = 0.0, spec_hz: float = 10.0, spec_regions=("mouth",),
              aux_tmax: float = 0.3, fps: float = 25.0) -> tuple[torch.Tensor, dict]:
    """batch: target [B,L,42], audio [B,2(P+L),1024], ref [B,REF_DIM], prefix [B,P,42] (optional),
    neg_target [B,L,42] (optional, another window of the same clip, for lam_cfm).

    Condition dropout rates are FLOAT's: audio 0.1, reference 0.1, previous-window context 0.5.

    Phase B options (all off by default; report 'SANG next round improvements' section 4):
      t_mean, t_scale  logit-normal time sampling (with t_dist='logit_normal').
      lam_cfm          contrastive flow matching (arXiv 2506.05350): minus lam * ||v - v_neg||^2 on
                       cfm_regions, v_neg = eps - x0_neg with the SAME noise and x0_neg from batch
                       'neg_target' (same clip, other audio) or else another clip of the batch. A convex
                       objective for lam < 1; it pushes the prediction to be specific to its own audio.
      lam_ccc          lip-opening CCC on the one-step clean estimate, for samples with t <= aux_tmax.
      lam_spec         band-limited (< spec_hz) spectral L1 on the clean estimate, same t mask."""
    x0 = batch["target"]
    B, dev = x0.shape[0], x0.device
    t = sample_t(B, dev, t_dist, t_mean, t_scale)
    eps = torch.randn_like(x0)
    tb = t.view(B, 1, 1)
    x_t = (1 - tb) * x0 + tb * eps
    prefix = batch.get("prefix")
    v = model(x_t, t, batch["audio"], batch["ref"], prefix=prefix,
              prefix_keep=(torch.rand(B, device=dev) >= p_prefix) if prefix is not None else None,
              drop_audio=torch.rand(B, device=dev) < p_audio,
              drop_ref=torch.rand(B, device=dev) < p_ref)
    loss, parts = region_mse(v, eps - x0)
    out = {"fm": loss.detach(), **{f"fm_{k}": p.detach() for k, p in parts.items()}}
    if lam_cfm > 0:
        x0_neg = batch.get("neg_target")
        x0_neg = x0.roll(1, 0) if x0_neg is None else x0_neg
        u_neg = eps - x0_neg
        cfm = sum(F.mse_loss(v[..., REGIONS[r]].float(), u_neg[..., REGIONS[r]].float()) for r in cfm_regions)
        loss = loss - lam_cfm * cfm
        out["cfm"] = cfm.detach()
    x0_hat = x_t - tb * v                                    # one-step clean estimate
    if lam_vel > 0:
        vel, _ = region_mse(x0_hat[:, 1:] - x0_hat[:, :-1], x0[:, 1:] - x0[:, :-1])
        loss = loss + lam_vel * vel
        out["vel"] = vel.detach()
    w = (t <= aux_tmax).float()                              # aux losses only where x0_hat is sharp
    if lam_ccc > 0:
        if lip_a is None:
            raise ValueError("lam_ccc needs lip_a, the lip-opening readout in normalised space")
        ccc = lip_ccc_loss(x0_hat, x0, lip_a, w)
        loss = loss + lam_ccc * ccc
        out["ccc"] = ccc.detach()
    if lam_spec > 0:
        cols = [c for r in spec_regions for c in REGIONS[r]]
        spec = spectral_loss(x0_hat, x0, cols, int(spec_hz * x0.shape[1] / fps), w)
        loss = loss + lam_spec * spec
        out["spec"] = spec.detach()
    out["loss"] = loss.detach()
    return loss, out


def ema_weights(ck: dict, which=None) -> dict:
    """A checkpoint's EMA state dict: the main one, or an extra decay kept with `ema_extra` (e.g. 0.999)."""
    if which in (None, "", "main"):
        return ck["ema"]
    extra = ck.get("ema_extra", {})
    key = f"{float(which):g}"
    if key not in extra:
        raise ValueError(f"checkpoint has extra EMAs {sorted(extra)} only, not {key}")
    return extra[key]


# ---------------------------------------------------------------------- sampling
def project(z: torch.Tensor, bounds) -> torch.Tensor:
    """Enforce a . z_f <= ub_f on every frame f, by the smallest move along a (normalised space).

    bounds: [(a [42], ub [B, L])], ub = +inf where a frame is free. The a's used here (lip and eye
    openness readouts, sang/naturalness.py) live on disjoint columns, so projecting one after the
    other satisfies all of them exactly."""
    for a, ub in bounds:
        a = a.to(z)
        over = ((z @ a) - ub.to(z)).clamp_min(0)                       # inf bound -> 0
        z = z - over.unsqueeze(-1) * a / (a @ a)
    return z


def guidance_vector(g: float, mouth: float | None = None, eyes: float | None = None,
                    brow: float | None = None, rot: float | None = None):
    """Audio-guidance scale per target region: a float when uniform, else a [42] tensor.

    The γ sweep (job 173799, 200 val clips) showed guidance sets mouth AMPLITUDE (lip-opening std vs
    real: 1.04 / 1.34 / 1.59 / 1.79 at γ 1 / 1.5 / 2 / 2.5) while correlation saturates at γ ≈ 1.5,
    so the mouth can take a lower scale than the rest (AVTR-1 also guides per region)."""
    per = {"mouth": mouth, "eyes": eyes, "brow": brow, "rot": rot}
    if all(v is None or v == g for v in per.values()):
        return float(g)
    vec = torch.full((T_DIM,), float(g))
    for name, v in per.items():
        if v is not None:
            vec[REGIONS[name]] = float(v)
    return vec


def cfg_combine(v_c: torch.Tensor, v_u: torch.Tensor, gamma, rescale: float = 0.0) -> torch.Tensor:
    """v_u + gamma * (v_c - v_u), gamma a float or a [42] per-coordinate vector.

    rescale (CFG-rescale, Lin et al. arXiv 2305.08891; 0 = off, they use 0.7): pulls each coordinate's
    temporal std of the guided velocity back to the conditional one's, blended by `rescale`, so
    guidance keeps its direction (timing) without inflating amplitude."""
    g = gamma.to(v_c).view(1, 1, -1) if isinstance(gamma, torch.Tensor) else gamma
    v = v_u + g * (v_c - v_u)
    if rescale > 0:
        ratio = v_c.std(dim=1, keepdim=True) / v.std(dim=1, keepdim=True).clamp_min(1e-6)
        v = rescale * (v * ratio) + (1.0 - rescale) * v
    return v


def flow_times(steps: int, sway: float = 0.0) -> list[float]:
    """steps + 1 flow times from 1 (noise) down to 0. sway = 0: uniform. sway < 0: F5-TTS's Sway
    Sampling (arXiv 2410.06885, u + s (cos(pi u / 2) - 1 + u) on the noise-to-data axis), which
    spends more steps near the noise end, where the condition is decided; s = -1 cut F5's WER
    2.84 -> 2.41 at 32 steps."""
    if not -1.0 <= sway <= 1.0:
        raise ValueError("sway must be in [-1, 1]")
    u = torch.linspace(0.0, 1.0, steps + 1, dtype=torch.float64)
    if sway:
        u = u + sway * (torch.cos(math.pi / 2 * u) - 1 + u)
    return (1.0 - u).tolist()


def _per_dim(g, like: torch.Tensor):
    return g.to(like).view(1, 1, -1) if isinstance(g, torch.Tensor) else g


@torch.no_grad()
def sample(model: MotionFlowTransformer, audio, ref, L: int, prefix=None, steps: int = 10,
           cfg_audio=2.0, generator: torch.Generator | None = None, bounds=None,
           prefix_keep: torch.Tensor | None = None, cfg_rescale: float = 0.0,
           ag_model: MotionFlowTransformer | None = None, ag=1.0, g_tmin: float = 0.0, g_tmax: float = 1.0,
           sway: float = 0.0, noise_scale=None) -> torch.Tensor:
    """Euler from t = 1 (noise) to 0. CFG on audio only, gamma = 2 (FLOAT Tab. 6).

    The unconditional branch drops audio and keeps the reference and prefix, so guidance pushes
    along the audio direction specifically rather than away from the speaker's identity.

    bounds (optional, see `project`): training-free constraints. Each step is written in its
    equivalent (x0, eps) form, x_next = (1 - t') x0 + t' eps, and x0 is projected onto the
    constraints first; with no bounds this is exactly the Euler step x - dt v. The last step lands
    on the projected x0, so the output satisfies the bounds exactly, while the earlier steps let
    the model re-fit the neighbouring frames around the constraint.

    prefix_keep [B] bool (optional): False = the prefix slot is present but DROPPED, exactly as
    flow_loss drops it with p_prefix -- null tokens, null audio, t = 0 -- which is how training
    represents 'no previous frames'.

    Sampler options (all off by default; report 'SANG next round improvements' section 2):
      ag_model, ag    autoguidance (Karras et al., arXiv 2406.02507): add (ag - 1)(v - v_guide), where
                      v_guide is a weaker, less-trained SANG given the SAME audio and reference.
                      ag a float or a [42] per-coordinate vector (see guidance_vector).
      g_tmin, g_tmax  guidance interval (arXiv 2404.07724): guide only while g_tmin <= t <= g_tmax.
      sway            step schedule, see flow_times.
      noise_scale     float or [42]: scales the starting noise per coordinate (a mouth temperature)."""
    B, dev = ref.shape[0], ref.device
    x = torch.randn(B, L, T_DIM, device=dev, generator=generator)
    if noise_scale is not None:
        x = x * _per_dim(noise_scale, x)
    keep = None
    if prefix is not None:
        keep = torch.ones(B, dtype=torch.bool, device=dev) if prefix_keep is None else prefix_keep.to(dev)
    null = torch.ones(B, dtype=torch.bool, device=dev)
    uniform_one = not isinstance(cfg_audio, torch.Tensor) and cfg_audio == 1.0
    ag_on = ag_model is not None and (isinstance(ag, torch.Tensor) or ag != 1.0)
    ts = flow_times(steps, sway)
    for i in range(steps):
        t_now, t_next = ts[i], ts[i + 1]
        t = torch.full((B,), t_now, device=dev)
        v_c = model(x, t, audio, ref, prefix=prefix, prefix_keep=keep)
        v = v_c
        if g_tmin <= t_now <= g_tmax:
            if not uniform_one or cfg_rescale > 0:
                v_u = model(x, t, audio, ref, prefix=prefix, prefix_keep=keep, drop_audio=null)
                v = cfg_combine(v_c, v_u, cfg_audio, cfg_rescale)
            if ag_on:
                v_g = ag_model(x, t, audio, ref, prefix=prefix, prefix_keep=keep)
                v = v + (_per_dim(ag, v) - 1.0) * (v_c - v_g)
        if bounds:
            x0 = project(x - t_now * v, bounds)
            eps = x + (1.0 - t_now) * v
            x = (1.0 - t_next) * x0 + t_next * eps
        else:
            x = x - (t_now - t_next) * v
    return x


@torch.no_grad()
def generate(model: MotionFlowTransformer, audio: torch.Tensor, ref: torch.Tensor, n_frames: int,
             window: int = 64, n_prefix: int = 10, steps: int = 10, cfg_audio=2.0,
             generator: torch.Generator | None = None, bounds=None,
             start: torch.Tensor | None = None, cfg_rescale: float = 0.0, mouth_avg: int = 1,
             **sampler) -> torch.Tensor:
    """Any length. Windows of `window` NEW frames, each conditioned on the previous `n_prefix`
    generated frames (FLOAT's L'). audio [B, 2*n_frames, 1024] -> [B, n_frames, 42].
    bounds: as in `sample`, with ub over the whole clip, [B, n_frames].
    sampler: ag_model, ag, g_tmin, g_tmax, sway, noise_scale -- passed to `sample`.

    mouth_avg = K > 1: draw K whole trajectories and average their 18 mouth coordinates; head, eyes
    and brows come from the first, so they keep their sample-to-sample variety. The model's mean
    mouth is closer to the real one than any single draw (diagnosis job 176834: lip corr 0.722 one
    sample, 0.746 mean of 2, 0.759 mean of 4).

    How the FIRST window starts (`start`):
      None     the prefix slot is present and dropped -- n_prefix null tokens at positions
               0..n_prefix-1, null audio, t = 0 -- which is exactly how training shows 'no previous
               frames' (flow_loss, p_prefix = 0.5). Training never saw target frames at positions
               0..n_prefix-1: the old first window put them there and jittered for ~8 frames in
               every demo (scripts/video_jitter.py: pixel acceleration 5.1x the real video's at
               frame 1, falling to 1.4x by frame 8, then ~0.5x; nothing at the window seam).
      [B, n_prefix, 42]
               a clean prefix the clip continues from, e.g. the source frame's own (normalised)
               target repeated: the video then starts at the photo's pose and expression. `audio`
               must then carry n_prefix frames BEFORE frame 0 (e.g. WavLM of prepended silence).
    """
    tpf, B, dev = model.tpf, ref.shape[0], ref.device
    if mouth_avg > 1:
        K = int(mouth_avg)
        rep = lambda z: None if z is None else z.repeat_interleave(K, 0)
        kb = None if not bounds else [(vec, ub.repeat_interleave(K, 0)) for vec, ub in bounds]
        y = generate(model, rep(audio), rep(ref), n_frames, window, n_prefix, steps, cfg_audio, generator, kb,
                     rep(start), cfg_rescale, 1, **sampler).view(B, K, n_frames, T_DIM)
        out = y[:, 0].clone()
        out[..., REGIONS["mouth"]] = y[..., REGIONS["mouth"]].mean(1)
        return out
    lead = 0 if start is None else start.shape[1]
    if start is not None and start.shape[1] != n_prefix:
        raise ValueError(f"start has {start.shape[1]} frames, the model chains {n_prefix}")
    out, pos = [], 0
    while pos < n_frames:
        keep = None
        if n_prefix == 0:
            prefix, a = None, audio[:, pos * tpf:(pos + window) * tpf]
        elif pos == 0 and start is None:
            prefix = torch.zeros(B, n_prefix, T_DIM, device=dev)
            keep = torch.zeros(B, dtype=torch.bool, device=dev)
            a = F.pad(audio[:, : window * tpf], (0, 0, n_prefix * tpf, 0))   # nulled anyway
        else:
            prefix = start if pos == 0 else torch.cat(out, 1)[:, -n_prefix:]
            a0 = lead + pos - n_prefix
            a = audio[:, a0 * tpf:(lead + pos + window) * tpf]            # _audio_frames pads a short tail
        wb = None
        if bounds:
            wb = []
            for vec, ub in bounds:
                u = ub[:, pos:pos + window]
                wb.append((vec, F.pad(u, (0, window - u.shape[1]), value=float("inf"))))
        y = sample(model, a, ref, window, prefix=prefix, steps=steps, cfg_audio=cfg_audio,
                   generator=generator, bounds=wb, prefix_keep=keep, cfg_rescale=cfg_rescale, **sampler)
        out.append(y[:, : n_frames - pos])                   # last window: generate full, keep what fits
        pos += window
    return torch.cat(out, 1)


class Norm(nn.Module):
    """Per-column standardisation for the 42-d target and the 63-d canonical keypoints."""

    def __init__(self, t_mean, t_std, kp_mean, kp_std):
        super().__init__()
        for k, v in (("t_mean", t_mean), ("t_std", t_std), ("kp_mean", kp_mean), ("kp_std", kp_std)):
            self.register_buffer(k, torch.as_tensor(v, dtype=torch.float32))

    def target(self, y):  return (y.float() - self.t_mean) / self.t_std
    def untarget(self, z): return z.float() * self.t_std + self.t_mean
    def kp(self, x):      return (x.float() - self.kp_mean) / self.kp_std

    def ref(self, kp: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Raw x_c [...,63] + raw reference target [...,42] -> normalised [..., REF_DIM]."""
        return torch.cat([self.kp(kp), self.target(y)], -1)

    @classmethod
    def fit(cls, targets: torch.Tensor, kps: torch.Tensor, floor: float = 1e-3) -> "Norm":
        return cls(targets.mean(0), targets.std(0).clamp_min(floor),
                   kps.mean(0), kps.std(0).clamp_min(floor))


def build(cfg: dict) -> MotionFlowTransformer:
    return MotionFlowTransformer(
        dim=cfg.get("dim", 512), depth=cfg.get("layers", 8), heads=cfg.get("heads", 8),
        audio_dim={"wavlm-base": 768, "wavlm-large": 1024}[cfg.get("audio_encoder", "wavlm-large")],
        audio_kernel=cfg.get("audio_kernel", 5), attn_window=cfg.get("attn_window", 0),
        max_frames=cfg.get("max_frames", 512), dropout=cfg.get("dropout", 0.0))


# ---------------------------------------------------------------------- sampler specs
SAMPLER_KEYS = {
    "g": "audio CFG scale on every region (default: the run's cfg_audio)",
    "mouth": "CFG scale on the 18 mouth coordinates", "eyes": "CFG on the eyes", "brow": "CFG on the brows",
    "rot": "CFG on head rotation", "rescale": "CFG-rescale (0 = off)",
    "steps": "Euler steps", "sway": "Sway Sampling coefficient in [-1, 0]",
    "gmin": "guide only while t >= gmin", "gmax": "guide only while t <= gmax (t = 1 is noise)",
    "ag": "autoguidance scale on every region (needs a guide checkpoint)", "ag_mouth": "autoguidance on the mouth only",
    "tau": "mouth temperature: starting-noise scale on the mouth", "temp": "starting-noise scale on every coordinate",
    "avg": "average the mouth over this many samples",
}


def parse_spec(s: str | None) -> dict:
    """'g=2,mouth=1.25,avg=4' -> {'g': 2.0, 'mouth': 1.25, 'avg': 4.0}. '' or None -> {}."""
    out = {}
    for kv in (s or "").split(","):
        if not kv.strip():
            continue
        k, _, v = kv.partition("=")
        k = k.strip()
        if k not in SAMPLER_KEYS:
            raise ValueError(f"unknown sampler key '{k}' in '{s}'; known: {', '.join(SAMPLER_KEYS)}")
        out[k] = float(v)
    return out


def sampler_kwargs(spec: dict, cfg: dict, ag_model: MotionFlowTransformer | None = None,
                   steps: int | None = None) -> dict:
    """A parsed spec -> keyword arguments for `generate` (and naturalness.guided_generate's `sampler`).
    Keys left out keep the run's defaults: cfg_audio, sample_steps, no autoguidance, uniform steps."""
    kw = {"cfg_audio": guidance_vector(spec.get("g", cfg["cfg_audio"]), mouth=spec.get("mouth"),
                                       eyes=spec.get("eyes"), brow=spec.get("brow"), rot=spec.get("rot")),
          "cfg_rescale": spec.get("rescale", 0.0),
          "steps": int(spec.get("steps", steps or cfg["sample_steps"])),
          "sway": spec.get("sway", 0.0), "g_tmin": spec.get("gmin", 0.0), "g_tmax": spec.get("gmax", 1.0),
          "mouth_avg": int(spec.get("avg", 1))}
    if "ag" in spec or "ag_mouth" in spec:
        if ag_model is None:
            raise ValueError("autoguidance (ag / ag_mouth) needs a guide model: pass --guide-ckpt")
        kw["ag_model"] = ag_model
        kw["ag"] = guidance_vector(spec.get("ag", 1.0), mouth=spec.get("ag_mouth"))
    if "tau" in spec or "temp" in spec:
        ns = torch.full((T_DIM,), float(spec.get("temp", 1.0)))
        if "tau" in spec:
            ns[REGIONS["mouth"]] = float(spec["tau"])
        kw["noise_scale"] = ns
    return kw


def load_ema(path, device: str = "cuda", norm: "Norm | None" = None) -> MotionFlowTransformer:
    """A checkpoint's EMA weights, ready to sample, e.g. an autoguidance guide. With `norm`, refuse a
    checkpoint whose target normalisation differs (its velocities would be in other units)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    if norm is not None and not torch.allclose(torch.as_tensor(ck["norm"]["t_std"]).to(norm.t_std), norm.t_std):
        raise ValueError(f"{path} was trained with a different target normalisation")
    m = build(ck["cfg"]).to(device).eval()
    m.load_state_dict(ck["ema"])
    return m
