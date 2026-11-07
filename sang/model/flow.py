"""Bidirectional motion-flow transformer: audio + reference -> a motion trajectory.

Non-streaming by design. The whole window is generated jointly, so there is no exposure bias to
correct and no causal mask. The noisy trajectory itself is the transformer's input, so self-attention
couples the noise across frames (FLOAT, KDTalker and Ditto do the same); a per-frame denoiser would
factorise the window into a mean trajectory plus independent per-frame noise, i.e. jitter.

Conditioning is frame-wise AdaLN (FLOAT sec. 4.2): each frame is modulated by ITS OWN condition
c_l = time(t_l) + audio_l + ref. audio_l is a symmetric window of WavLM ticks around frame l, so the
audio-visual alignment is built into the architecture instead of learned from positions.

Long clips follow FLOAT: each window is conditioned on the last P generated frames as a clean
prefix, and that prefix is dropped with p = 0.5 in training so the first window of a clip -- which
has none -- is in distribution.

The target is the first `latent_k` codes of the PCA motion latent (sang.model.latent): everything LivePortrait
renders, relative to the photo. The loss is a plain MSE over the codes. The photo is described by its x_c and its
42-d target (sang.model.target.to_target; describing it by its 69 wide numbers, or weighting codes by their
variance, did no better on HDTF: docs/research_log.md 3.24).

Guidance is trained in (Guidance-Free Training, GFT, Chen et al., ICML 2025, arXiv 2501.15420): the network takes
a guidance temperature beta = 1/gamma as an extra condition and is trained so that
beta v(x | audio, beta) + (1 - beta) sg[v(x | no audio, 1)] is the conditional velocity. At sampling one pass
with beta = 1/gamma IS the classifier-free-guided velocity (HDTF: same metrics as CFG at the same gamma,
docs/research_log.md 3.22). The audio is still dropped with p_audio, so v(x | no audio) stays trained.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

from sang.model.target import T_DIM, to_target

REF_DIM = 63 + T_DIM


class SwiGLU(nn.Module):
    def __init__(self, dim: int, mult: float = 4.0):
        super().__init__()
        hid = int(dim * mult)
        self.w1 = nn.Linear(dim, hid, bias=False)
        self.w2 = nn.Linear(dim, hid, bias=False)
        self.w3 = nn.Linear(hid, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class Sinusoidal(nn.Module):
    """Scalar [N] -> [N, dim] sin/cos features."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t.reshape(-1, 1).float() * freqs.reshape(1, -1)
        return torch.cat([args.sin(), args.cos()], dim=-1)


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


class DiTBlock(nn.Module):
    """Pre-norm self-attention + SwiGLU, each modulated per FRAME by adaLN-Zero (shift, scale, gate).

    cross=True adds cross-attention to the demo tokens (DemoEncoder) between the two, with a zero-initialised output:
    a fresh demo model is exactly the model without one."""

    def __init__(self, dim: int, heads: int, dropout: float = 0.0, cross: bool = False):
        super().__init__()
        self.n1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.n2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = SwiGLU(dim)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)
        if cross:
            self.nx = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
            self.xattn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
            nn.init.zeros_(self.xattn.out_proj.weight)
            nn.init.zeros_(self.xattn.out_proj.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor, demo: torch.Tensor | None = None,
                demo_w: torch.Tensor | None = None) -> torch.Tensor:
        s1, k1, g1, s2, k2, g2 = self.ada(c).chunk(6, dim=-1)
        h = _modulate(self.n1(x), s1, k1)
        x = x + g1 * self.attn(h, h, h, need_weights=False)[0]
        if demo is not None:
            x = x + demo_w * self.xattn(self.nx(x), demo, demo, need_weights=False)[0]
        return x + g2 * self.ff(_modulate(self.n2(x), s2, k2))


class DemoEncoder(nn.Module):
    """Extra footage of the person -> n_tokens learned tokens (a Perceiver-style resampler).

    A frame token is the frame's motion codes (relative to the photo, like the target) plus its audio, so the
    tokens can carry how THIS person moves for the speech they make, not only how much. Two convolutions mix
    +-4 frames so velocities are visible; learned queries then cross-attend to all frames. No positional
    embedding, so any length of footage gives the same number of tokens. Nothing in it is hand-picked: what the
    tokens carry is learned from the flow loss alone."""

    def __init__(self, dim: int, t_dim: int, audio_dim: int, tpf: int, n_tokens: int, heads: int, depth: int = 2):
        super().__init__()
        self.tpf = tpf
        self.motion = nn.Linear(t_dim, dim)
        self.audio = nn.Linear(audio_dim * tpf, dim)
        self.mix = nn.Sequential(nn.Conv1d(dim, dim, 5, padding=2), nn.SiLU(), nn.Conv1d(dim, dim, 5, padding=2))
        self.queries = nn.Parameter(torch.randn(n_tokens, dim) * 0.02)
        self.kv_norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList(nn.ModuleList([nn.LayerNorm(dim), nn.MultiheadAttention(dim, heads, batch_first=True),
                                                   nn.LayerNorm(dim), SwiGLU(dim)]) for _ in range(depth))
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, motion: torch.Tensor, audio: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """motion [B, N, t_dim], audio [B, tpf*N, audio_dim], mask [B, N] (True = a real frame) -> [B, n_tokens, dim]."""
        B, N, _ = motion.shape
        a = audio[:, : N * self.tpf].float()
        if a.shape[1] < N * self.tpf:
            a = F.pad(a, (0, 0, 0, N * self.tpf - a.shape[1]))
        mask = mask.clone()
        mask[:, 0] |= ~mask.any(1)  # an empty demo would make attention NaN; its tokens are unused anyway
        h = (self.motion(motion.float()) + self.audio(a.reshape(B, N, -1))) * mask.unsqueeze(-1)
        h = self.kv_norm(h + self.mix(h.transpose(1, 2)).transpose(1, 2))
        q = self.queries.expand(B, -1, -1)
        for n1, att, n2, ff in self.layers:
            q = q + att(n1(q), h, h, key_padding_mask=~mask, need_weights=False)[0]
            q = q + ff(n2(q))
        return self.out_norm(q)


class MotionFlowTransformer(nn.Module):
    """v_theta(x_t, t | audio, ref, prefix, beta) over a whole window.

    x_t    [B, L, t_dim]       noisy latent codes
    t      [B]                 flow time; 0 = clean, 1 = noise
    audio  [B, 2(P+L), 1024]   WavLM ticks at 50 Hz covering prefix + target frames
    ref    [B, REF_DIM]        the photo: Norm.ref, normalised [x_c, its 42-d target]
    prefix [B, P, t_dim] | None  clean previous frames (FLOAT's L'); None = no prefix at all
    beta   [B] | None          guidance temperature 1/gamma (None = 1, no guidance)
    demo   [B, M, dim] | None  tokens of extra footage of the person (encode_demo); needs demo_tokens > 0
    -> v   [B, L, t_dim]
    """

    def __init__(self, dim: int = 512, depth: int = 8, heads: int = 8, audio_dim: int = 1024,
                 ticks_per_frame: int = 2, audio_kernel: int = 5, max_frames: int = 512, dropout: float = 0.0,
                 t_dim: int = 42, demo_tokens: int = 0):
        super().__init__()
        if audio_kernel % 2 == 0:
            raise ValueError("audio_kernel must be odd so the window is centred on its frame")
        self.tpf, self.t_dim = ticks_per_frame, t_dim
        self.in_proj = nn.Linear(t_dim, dim)
        self.pos = nn.Embedding(max_frames, dim)
        self.prefix_tag = nn.Parameter(torch.zeros(dim))
        self.null_prefix = nn.Parameter(torch.zeros(dim))
        self.audio = nn.Sequential(
            nn.Conv1d(audio_dim * ticks_per_frame, dim, audio_kernel, padding=audio_kernel // 2),
            nn.SiLU(), nn.Conv1d(dim, dim, 1))
        self.ref = nn.Sequential(nn.Linear(REF_DIM, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time = nn.Sequential(Sinusoidal(dim), nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.beta = nn.Sequential(Sinusoidal(dim), nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        nn.init.zeros_(self.beta[-1].weight)
        nn.init.zeros_(self.beta[-1].bias)
        self.null_audio = nn.Parameter(torch.zeros(dim))
        self.null_ref = nn.Parameter(torch.zeros(dim))
        self.demo = DemoEncoder(dim, t_dim, audio_dim, ticks_per_frame, demo_tokens, heads) if demo_tokens else None
        self.blocks = nn.ModuleList([DiTBlock(dim, heads, dropout, cross=bool(demo_tokens)) for _ in range(depth)])
        self.out_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.out_ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.out = nn.Linear(dim, t_dim)
        for lin in (self.out_ada[-1], self.out):
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)

    def _audio_frames(self, audio: torch.Tensor, n_frames: int) -> torch.Tensor:
        """[B, >=tpf*F, D] ticks -> [B, F, dim], one centred window per frame (a short clip is padded with silence)."""
        B, _, D = audio.shape
        a = audio[:, : n_frames * self.tpf].float()
        if a.shape[1] < n_frames * self.tpf:
            a = F.pad(a, (0, 0, 0, n_frames * self.tpf - a.shape[1]))
        a = a.reshape(B, n_frames, self.tpf * D).transpose(1, 2)
        return self.audio(a).transpose(1, 2)

    def encode_demo(self, batch: dict) -> torch.Tensor | None:
        """The demo tokens of a batch with demo_motion / demo_audio / demo_mask; None without a demo encoder or demo."""
        if self.demo is None or "demo_motion" not in batch:
            return None
        return self.demo(batch["demo_motion"], batch["demo_audio"], batch["demo_mask"])

    def forward(self, x_t, t, audio, ref, prefix=None, prefix_keep=None, drop_audio=None, drop_ref=None, beta=None,
                demo=None, demo_keep=None):
        """prefix_keep [B] bool: False drops the whole prefix (null tokens, null audio); drop_audio / drop_ref [B]
        bool: condition dropout. Where the audio is dropped the model always sees beta = 1. demo_keep [B] bool:
        False ignores that sample's demo, which is then exactly the model without footage."""
        B, L, _ = x_t.shape
        P = 0 if prefix is None else prefix.shape[1]
        n = P + L
        dev = x_t.device

        tok = self.in_proj(x_t.float())
        if P:
            ptok = self.in_proj(prefix.float()) + self.prefix_tag
            if prefix_keep is not None:
                ptok = torch.where(prefix_keep.view(B, 1, 1), ptok, self.null_prefix.expand_as(ptok))
            tok = torch.cat([ptok, tok], 1)
        tok = tok + self.pos(torch.arange(n, device=dev))[None]

        t_frames = torch.cat([torch.zeros(B, P, device=dev), t.float().view(B, 1).expand(B, L)], 1)
        c_time = self.time(t_frames.reshape(-1)).view(B, n, -1)
        c_audio = self._audio_frames(audio, n)
        if drop_audio is not None:
            c_audio = torch.where(drop_audio.view(B, 1, 1), self.null_audio.expand_as(c_audio), c_audio)
        if P and prefix_keep is not None:
            keep = torch.cat([prefix_keep.view(B, 1).expand(B, P), torch.ones(B, L, dtype=torch.bool, device=dev)], 1)
            c_audio = torch.where(keep.unsqueeze(-1), c_audio, self.null_audio.expand_as(c_audio))
        c_ref = self.ref(ref.float())
        if drop_ref is not None:
            c_ref = torch.where(drop_ref.view(B, 1), self.null_ref.expand_as(c_ref), c_ref)
        b = torch.ones(B, device=dev) if beta is None else beta.float().to(dev).expand(B)
        if drop_audio is not None:
            b = torch.where(drop_audio, torch.ones_like(b), b)
        c = c_time + c_audio + c_ref.unsqueeze(1) + self.beta(b * 1000.0).unsqueeze(1)

        demo_w = None
        if demo is not None:
            demo_w = (torch.ones(B, device=dev) if demo_keep is None else demo_keep.float()).view(B, 1, 1)
        x = tok
        for blk in self.blocks:
            x = blk(x, c, demo, demo_w)
        shift, scale = self.out_ada(c).chunk(2, dim=-1)
        return self.out(_modulate(self.out_norm(x), shift, scale)[:, P:])


def sample_t(B: int, device, dist: str = "uniform", mean: float = 0.0, scale: float = 1.0) -> torch.Tensor:
    """Flow times (t = 1 is noise). logit_normal: sigmoid(mean + scale * N(0, 1)) (SD3, arXiv 2403.03206);
    mean > 0 trains more at high noise, where the audio decides the coarse mouth timing."""
    if dist == "logit_normal":
        return torch.sigmoid(mean + scale * torch.randn(B, device=device))
    return torch.rand(B, device=device)


def flow_loss(model: MotionFlowTransformer, batch: dict, lam_vel: float = 1.0,
              p_audio: float = 0.1, p_ref: float = 0.1, p_prefix: float = 0.5, p_demo: float = 0.2,
              t_dist: str = "uniform", t_mean: float = 0.0, t_scale: float = 1.0) -> tuple[torch.Tensor, dict]:
    """batch: target [B,L,t_dim], audio [B,2(P+L),1024], ref [B,REF_DIM], prefix [B,P,t_dim] (optional), and for a
    demo model demo_motion [B,N,t_dim], demo_audio [B,2N,1024], demo_mask [B,N], demo_keep [B] (optional).

    Rectified flow (MSE), plus lam_vel times the same loss on frame-to-frame differences of the one-step clean
    estimate (FLOAT). Condition dropout rates are FLOAT's: audio 0.1, reference 0.1, previous-window context 0.5.
    The demo is dropped with p_demo, so the same model also runs with no footage at all.

    GFT: beta ~ U(0, 1) per sample, and the loss is on beta v(x | audio, beta) + (1 - beta) v_u, where
    v_u = v(x | no audio, 1) from the same network in eval mode without gradient (the paper's Algorithm 1)."""
    x0 = batch["target"]
    B, dev = x0.shape[0], x0.device
    t = sample_t(B, dev, t_dist, t_mean, t_scale)
    eps = torch.randn_like(x0)
    tb = t.view(B, 1, 1)
    x_t = (1 - tb) * x0 + tb * eps
    prefix = batch.get("prefix")
    drop = torch.rand(B, device=dev) < p_audio
    kw = dict(prefix=prefix, drop_ref=torch.rand(B, device=dev) < p_ref,
              prefix_keep=(torch.rand(B, device=dev) >= p_prefix) if prefix is not None else None)
    demo = model.encode_demo(batch) if hasattr(model, "encode_demo") else None
    if demo is not None:
        kw.update(demo=demo, demo_keep=batch["demo_keep"].bool() & (torch.rand(B, device=dev) >= p_demo))
    beta = torch.rand(B, device=dev)
    v = model(x_t, t, batch["audio"], batch["ref"], drop_audio=drop, beta=beta, **kw)
    training = model.training
    model.eval()
    with torch.no_grad():
        v_u = model(x_t, t, batch["audio"], batch["ref"], drop_audio=torch.ones_like(drop), **kw)
    model.train(training)
    v = beta.view(B, 1, 1) * v + (1 - beta.view(B, 1, 1)) * v_u
    mse = lambda a, b: F.mse_loss(a.float(), b.float())
    loss = mse(v, eps - x0)
    out = {"fm": loss.detach()}
    if lam_vel > 0:
        x0_hat = x_t - tb * v
        vel = mse(x0_hat[:, 1:] - x0_hat[:, :-1], x0[:, 1:] - x0[:, :-1])
        loss = loss + lam_vel * vel
        out["vel"] = vel.detach()
    out["loss"] = loss.detach()
    return loss, out


def flow_times(steps: int) -> list[float]:
    """steps + 1 uniform flow times from 1 (noise) down to 0."""
    return (1.0 - torch.linspace(0.0, 1.0, steps + 1, dtype=torch.float64)).tolist()


@torch.no_grad()
def sample(model: MotionFlowTransformer, audio, ref, L: int, prefix=None, steps: int = 10,
           cfg_audio: float = 2.0, generator: torch.Generator | None = None,
           prefix_keep: torch.Tensor | None = None, demo: torch.Tensor | None = None,
           demo_keep: torch.Tensor | None = None) -> torch.Tensor:
    """Euler from t = 1 (noise) to 0, one pass per step at beta = 1/cfg_audio: the guided velocity
    v_u + gamma (v_c - v_u), whose unconditional branch drops the audio and keeps the photo and the prefix.

    prefix_keep [B] bool (optional): False = the prefix slot is present but DROPPED, exactly as
    flow_loss drops it -- which is how training represents 'no previous frames'. demo: encode_demo tokens, or None;
    demo_keep [B] bool: False ignores that sample's demo."""
    B, dev = ref.shape[0], ref.device
    x = torch.randn(B, L, model.t_dim, device=dev, generator=generator)
    keep = None
    if prefix is not None:
        keep = torch.ones(B, dtype=torch.bool, device=dev) if prefix_keep is None else prefix_keep.to(dev)
    beta = torch.full((B,), 1.0 / float(cfg_audio), device=dev)
    ts = flow_times(steps)
    for i in range(steps):
        t = torch.full((B,), ts[i], device=dev)
        v = model(x, t, audio, ref, prefix=prefix, prefix_keep=keep, beta=beta, demo=demo, demo_keep=demo_keep)
        x = x - (ts[i] - ts[i + 1]) * v
    return x


@torch.no_grad()
def generate(model: MotionFlowTransformer, audio: torch.Tensor, ref: torch.Tensor, n_frames: int,
             window: int = 64, n_prefix: int = 10, steps: int = 10, cfg_audio: float = 2.0,
             generator: torch.Generator | None = None, start: torch.Tensor | None = None,
             demo: torch.Tensor | None = None) -> torch.Tensor:
    """Any length. Windows of `window` NEW frames, each conditioned on the previous `n_prefix`
    generated frames (FLOAT's L'). audio [B, 2*n_frames, 1024] -> [B, n_frames, t_dim].

    How the FIRST window starts (`start`):
      None     the prefix slot is present and dropped, exactly how training shows 'no previous frames'.
      [B, n_prefix, t_dim]
               a clean prefix the clip continues from, e.g. the source frame's own (normalised)
               target repeated: the video then starts at the photo's pose and expression. `audio`
               must then carry n_prefix frames BEFORE frame 0 (e.g. WavLM of prepended silence).
    """
    tpf, B, dev = model.tpf, ref.shape[0], ref.device
    lead = 0 if start is None else start.shape[1]
    if start is not None and start.shape[1] != n_prefix:
        raise ValueError(f"start has {start.shape[1]} frames, the model chains {n_prefix}")
    out, pos = [], 0
    while pos < n_frames:
        keep = None
        if pos == 0 and start is None:
            prefix = torch.zeros(B, n_prefix, model.t_dim, device=dev)
            keep = torch.zeros(B, dtype=torch.bool, device=dev)
            a = F.pad(audio[:, : window * tpf], (0, 0, n_prefix * tpf, 0))
        else:
            prefix = start if pos == 0 else torch.cat(out, 1)[:, -n_prefix:]
            a0 = lead + pos - n_prefix
            a = audio[:, a0 * tpf:(lead + pos + window) * tpf]
        y = sample(model, a, ref, window, prefix=prefix, steps=steps, cfg_audio=cfg_audio,
                   generator=generator, prefix_keep=keep, demo=demo)
        out.append(y[:, : n_frames - pos])
        pos += window
    return torch.cat(out, 1)


class Norm(nn.Module):
    """Standardisation of the photo input: its 42-d target and its 63-d canonical keypoints."""

    def __init__(self, t_mean, t_std, kp_mean, kp_std):
        super().__init__()
        for k, v in (("t_mean", t_mean), ("t_std", t_std), ("kp_mean", kp_mean), ("kp_std", kp_std)):
            self.register_buffer(k, torch.as_tensor(v, dtype=torch.float32))

    def target(self, y):
        return (y.float() - self.t_mean) / self.t_std

    def kp(self, x):
        return (x.float() - self.kp_mean) / self.kp_std

    def ref(self, kp: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """Raw x_c [N,63] + the photo's raw motion [N,70] -> normalised [N, REF_DIM]."""
        return torch.cat([self.kp(kp), self.target(to_target(m))], -1)

    @classmethod
    def fit(cls, targets: torch.Tensor, kps: torch.Tensor, floor: float = 1e-3) -> "Norm":
        return cls(targets.mean(0), targets.std(0).clamp_min(floor), kps.mean(0), kps.std(0).clamp_min(floor))


def build(cfg: dict) -> MotionFlowTransformer:
    return MotionFlowTransformer(
        dim=cfg.get("dim", 512), depth=cfg.get("layers", 8), heads=cfg.get("heads", 8),
        audio_kernel=cfg.get("audio_kernel", 5), max_frames=cfg.get("max_frames", 512), dropout=cfg.get("dropout", 0.0),
        t_dim=int(cfg["latent_k"]), demo_tokens=int(cfg.get("demo_tokens", 0)))


SAMPLER_KEYS = {
    "g": "audio guidance scale gamma (default: the run's cfg_audio), sampled in one pass at beta = 1/gamma",
    "steps": "Euler steps (default: the run's sample_steps)",
}


def parse_spec(s: str | None) -> dict:
    """'g=2,steps=6' -> {'g': 2.0, 'steps': 6.0}. '' or None -> {}."""
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


def sampler_kwargs(spec: dict, cfg: dict) -> dict:
    """A parsed spec -> keyword arguments for `generate`; keys left out keep the run's cfg_audio and sample_steps."""
    return {"cfg_audio": float(spec.get("g", cfg["cfg_audio"])), "steps": int(spec.get("steps", cfg["sample_steps"]))}
