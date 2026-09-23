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


def sample_t(B: int, device, dist: str = "uniform") -> torch.Tensor:
    if dist == "logit_normal":                               # AVTR-1, SD3: weight the middle of the path
        return torch.sigmoid(torch.randn(B, device=device))
    return torch.rand(B, device=device)                      # FLOAT


def flow_loss(model: MotionFlowTransformer, batch: dict, lam_vel: float = 1.0,
              p_audio: float = 0.1, p_ref: float = 0.1, p_prefix: float = 0.5,
              t_dist: str = "uniform") -> tuple[torch.Tensor, dict]:
    """batch: target [B,L,42], audio [B,2(P+L),1024], ref [B,REF_DIM], prefix [B,P,42] (optional).

    Condition dropout rates are FLOAT's: audio 0.1, reference 0.1, previous-window context 0.5."""
    x0 = batch["target"]
    B, dev = x0.shape[0], x0.device
    t = sample_t(B, dev, t_dist)
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
    if lam_vel > 0:
        x0_hat = x_t - tb * v                                # one-step clean estimate
        vel, _ = region_mse(x0_hat[:, 1:] - x0_hat[:, :-1], x0[:, 1:] - x0[:, :-1])
        loss = loss + lam_vel * vel
        out["vel"] = vel.detach()
    out["loss"] = loss.detach()
    return loss, out


# ---------------------------------------------------------------------- sampling
@torch.no_grad()
def sample(model: MotionFlowTransformer, audio, ref, L: int, prefix=None, steps: int = 10,
           cfg_audio: float = 2.0, generator: torch.Generator | None = None) -> torch.Tensor:
    """Euler from t = 1 (noise) to 0. CFG on audio only, gamma = 2 (FLOAT Tab. 6).

    The unconditional branch drops audio and keeps the reference and prefix, so guidance pushes
    along the audio direction specifically rather than away from the speaker's identity."""
    B, dev = ref.shape[0], ref.device
    x = torch.randn(B, L, T_DIM, device=dev, generator=generator)
    keep = torch.ones(B, dtype=torch.bool, device=dev) if prefix is not None else None
    null = torch.ones(B, dtype=torch.bool, device=dev)
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((B,), 1.0 - i * dt, device=dev)
        v = model(x, t, audio, ref, prefix=prefix, prefix_keep=keep)
        if cfg_audio != 1.0:
            v_u = model(x, t, audio, ref, prefix=prefix, prefix_keep=keep, drop_audio=null)
            v = v_u + cfg_audio * (v - v_u)
        x = x - dt * v
    return x


@torch.no_grad()
def generate(model: MotionFlowTransformer, audio: torch.Tensor, ref: torch.Tensor, n_frames: int,
             window: int = 64, n_prefix: int = 10, steps: int = 10, cfg_audio: float = 2.0,
             generator: torch.Generator | None = None) -> torch.Tensor:
    """Any length. Windows of `window` NEW frames, each conditioned on the previous `n_prefix`
    generated frames (FLOAT's L'). The first window has no prefix, which training covered by
    dropping the prefix half of the time. audio [B, 2*n_frames, 1024] -> [B, n_frames, 42]."""
    tpf = model.tpf
    out, pos = [], 0
    while pos < n_frames:
        if pos == 0 or n_prefix == 0:
            prefix, a0 = None, 0
        else:
            prev = torch.cat(out, 1)
            prefix, a0 = prev[:, -n_prefix:], pos - n_prefix
        a = audio[:, a0 * tpf:(pos + window) * tpf]          # _audio_frames pads a short tail
        y = sample(model, a, ref, window, prefix=prefix, steps=steps,
                   cfg_audio=cfg_audio, generator=generator)
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
