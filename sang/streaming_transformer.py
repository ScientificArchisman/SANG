"""Block-causal streaming transformer with factorized FSQ head and MaskGIT decode."""
import torch
import torch.nn.functional as F
from torch import nn

from sang.fsq_codec import FSQIndexCodec
from sang.fsq_head import CoupledFSQHead, FactorizedFSQHead
from sang.masking import corrupt_context, maskgit_decode_schedule, mixed_mask, per_slice_cosine_mask


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(v + self.eps) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, dim: int, mult: float = 4.0):
        super().__init__()
        hid = int(dim * mult)
        self.w1 = nn.Linear(dim, hid, bias=False)
        self.w2 = nn.Linear(dim, hid, bias=False)
        self.w3 = nn.Linear(hid, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class StreamLayer(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float = 0.0):
        super().__init__()
        self.n1, self.n2, self.n3 = RMSNorm(dim), RMSNorm(dim), RMSNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ff = SwiGLU(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, e_a: torch.Tensor,
                self_allow: torch.Tensor, cross_allow: torch.Tensor) -> torch.Tensor:
        sa_mask = ~self_allow if self_allow is not None else None
        # need_weights=False lets MHA use the SDPA memory-efficient kernel instead of
        # materialising the full L x L attention matrix (1.6 GB/layer at L=7168, res 256).
        xn = self.n1(x)
        x = x + self.drop(self.self_attn(xn, xn, xn, attn_mask=sa_mask, need_weights=False)[0])
        ca_mask = ~cross_allow if cross_allow is not None else None
        x = x + self.drop(self.cross_attn(self.n2(x), e_a, e_a, attn_mask=ca_mask, need_weights=False)[0])
        x = x + self.drop(self.ff(self.n3(x)))
        return x


def slice_ticks(tv: int, ta: int, cond_slices: int = 1, device: torch.device | None = None,
                lookahead: int = 0) -> torch.Tensor:
    """Audio ticks visible after each grid slice. With cond_slices>1 (v3: identity+ctx), only the
    tv-cond_slices content slices consume the audio window; conditioning slices get the first ticks.
    lookahead: each slice may also see this many *future* audio ticks (EMO-style ±m window —
    the mouth pre-shapes before the phoneme, so strictly causal audio hurts lip-sync)."""
    slc = torch.arange(tv, device=device)
    if cond_slices > 1:
        ci = (slc - cond_slices).clamp(min=0)
        ctv = tv - cond_slices
        ticks = ((ci + 1) * ta + ctv - 1) // ctv
    else:
        ticks = ((slc + 1) * ta + tv - 1) // tv
    return (ticks + lookahead).clamp(min=1, max=ta)


def block_masks(tv: int, r: int, ta: int, device: torch.device, min_ticks: int = 1,
                cond_slices: int = 1, audio_lookahead: int = 0,
                return_float: bool = False, dtype=torch.float32):
    """Self [L,L] and cross [L,Ta] allow masks (True = may attend)."""
    L = tv * r
    pos = torch.arange(L, device=device)
    slc = pos // r
    self_allow = slc.unsqueeze(1) >= slc.unsqueeze(0)
    ticks = slice_ticks(tv, ta, cond_slices, device, lookahead=audio_lookahead)
    ticks = ticks.clamp(min=min_ticks, max=ta)
    aud = torch.arange(ta, device=device)
    cross_allow = aud.unsqueeze(0) < ticks.repeat_interleave(r).unsqueeze(1)
    assert self_allow.any(dim=-1).all(), "self-attention has a fully-masked query row"
    assert cross_allow.any(dim=-1).all(), "cross-attention has a fully-masked query row"
    if return_float:
        def _f(a):
            m = torch.zeros(a.shape, device=device, dtype=dtype)
            return m.masked_fill(~a, float("-inf"))
        return _f(self_allow), _f(cross_allow)
    return self_allow, cross_allow


class StreamingBlockTransformer(nn.Module):
    """Maps content embeddings [B,L,D] + audio [B,Ta,D] -> hidden [B,Tv,R,D]."""

    def __init__(self, dim: int = 512, tv: int = 5, r: int = 256, layers: int = 8,
                 heads: int = 8, dropout: float = 0.0, cond_slices: int = 1,
                 audio_lookahead: int = 0):
        super().__init__()
        self.tv, self.r, self.cond_slices = tv, r, cond_slices
        self.audio_lookahead = audio_lookahead
        # Spatial grid side, derived from r rather than hardcoded to 16. At r=256 (res 128) this is
        # bit-identical to the old `r // 16` / `16` pair; at r=1024 (v4, res 256) the old code
        # factorised a 32x32 grid as 64x16, so spatially adjacent latents stopped being positional
        # neighbours. See docs/v3_improvement_plan.md Part VI, D10.
        self.spatial = int(round(r ** 0.5))
        assert self.spatial * self.spatial == r, f"r={r} is not a square spatial grid"
        self.slice_emb = nn.Embedding(tv, dim)
        self.row_emb = nn.Embedding(self.spatial, dim)
        self.col_emb = nn.Embedding(self.spatial, dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList([StreamLayer(dim, heads, dropout) for _ in range(layers)])

    def _pos(self, device: torch.device) -> torch.Tensor:
        tv, r = self.tv, self.r
        slc = torch.arange(tv, device=device).repeat_interleave(r)
        intra = torch.arange(r, device=device).repeat(tv)
        row, col = intra // self.spatial, intra % self.spatial
        return self.slice_emb(slc) + self.row_emb(row) + self.col_emb(col)

    def forward(self, e_v: torch.Tensor, e_a: torch.Tensor) -> torch.Tensor:
        B, L, _ = e_v.shape
        tv, r = self.tv, self.r
        assert L == tv * r
        x = e_v + self._pos(e_v.device).unsqueeze(0)
        x = self.drop(x)
        self_allow, cross_allow = block_masks(tv, r, e_a.shape[1], e_v.device,
                                              cond_slices=self.cond_slices,
                                              audio_lookahead=self.audio_lookahead)
        for layer in self.layers:
            x = layer(x, e_a, self_allow, cross_allow)
        return x.view(B, tv, r, -1)


class StreamingTalkingHead(nn.Module):
    """Discrete cache -> embeddings -> block transformer -> factorized FSQ head."""

    arch_version = "block_ar_v1"

    def __init__(self, video_card: int = 32768, dim: int = 512,
                 tv: int = 5, spatial: int = 16, num_heads: int = 8, num_layers: int = 8,
                 dropout: float = 0.0,
                 factorized_head: bool = True, coupled_fsq_head: bool = False,
                 learn_logit_scale: bool = True, fsq_codes: torch.Tensor | None = None,
                 mask_schedule: str = "per_slice_cosine", p_inference_shaped: float = 0.0,
                 ref_slices: int = 1, audio_dim: int = 1024, audio_dropout: float = 0.0,
                 bridge_init: bool = False, audio_lookahead: int = 0,
                 continuous: bool = False, z_ch: int = 0, diff_depth: int = 6,
                 diff_hidden: int = 1024, use_struct_emb: bool = True):
        super().__init__()
        r = spatial * spatial
        self.video_card = video_card
        self.r = r
        self.tv = tv
        self.spatial = spatial
        self.ref_slices = ref_slices
        # continuous (v4): the grid holds VAE latents [B, tv, z_ch, h, w] and the head is a flow
        # MLP; otherwise it holds FSQ indices and the head is a factorized softmax.
        self.continuous = continuous
        self.z_ch = z_ch
        self.bridge_init = bridge_init  # False | "prev" | "ref"; see _bridge_prior
        self.audio_lookahead = audio_lookahead
        self.corrupt_from = 1  # slice 0 (identity) stays clean; ctx may carry generation error
        self.audio_dropout = audio_dropout
        self.mask_schedule = mask_schedule
        self.p_inference_shaped = p_inference_shaped
        self.video_emb = nn.Embedding(video_card, dim)
        # 16.8M params at video_card=32768; only built when structure conditioning is used.
        self.struct_emb = nn.Embedding(video_card, dim) if use_struct_emb else None
        self.audio_proj = nn.Linear(audio_dim, dim)
        self.audio_pos = nn.Embedding(1024, dim)  # 1024 ticks = 20 s at 50 Hz
        self.backbone = StreamingBlockTransformer(dim, tv, r, num_layers, num_heads, dropout,
                                                  cond_slices=ref_slices,
                                                  audio_lookahead=audio_lookahead)

        use_factorized = factorized_head and fsq_codes is not None
        self.factorized = use_factorized
        if continuous:
            from sang.diffusion import DiffusionHead
            self.latent_in = nn.Linear(z_ch, dim)
            self.norm = RMSNorm(dim)
            self.head = DiffusionHead(z_ch, dim, hidden=diff_hidden, depth=diff_depth)
        elif use_factorized:
            codec = FSQIndexCodec.from_codebook(fsq_codes)
            codec.to(fsq_codes.device)
            HeadCls = CoupledFSQHead if coupled_fsq_head else FactorizedFSQHead
            self.head = HeadCls(dim, codec, learn_scale=learn_logit_scale)
            self.norm = nn.Identity()
        else:
            self.norm = RMSNorm(dim)
            self.head = nn.Linear(dim, video_card, bias=False)
            self.head.weight = self.video_emb.weight

    def _embed_grid(self, grid: torch.Tensor, struct: torch.Tensor | None,
                    known: torch.Tensor) -> torch.Tensor:
        """Content embeddings before position add. known [B, tv*r] True = visible.
        Continuous mode: grid is [B, tv, z_ch, h, w] latents; discrete: [B, tv, h, w] indices."""
        r, L = self.r, self.tv * self.r
        if self.continuous:
            B = grid.shape[0]
            # [B, tv, z_ch, h, w] -> [B, tv*r, z_ch] -> project to dim
            e = self.latent_in(grid.reshape(B, self.tv, self.z_ch, r).permute(0, 1, 3, 2).reshape(B, L, self.z_ch).float())
            if struct is not None and self.struct_emb is not None:
                s = self.struct_emb(struct.reshape(B, -1).long())
                s = F.pad(s, (0, 0, L - s.shape[1], 0)) if s.shape[1] < L else s
                e = e + s
            if self.bridge_init:
                return torch.where(known.reshape(B, L).unsqueeze(-1), e,
                                   self._bridge_prior(e, r, self.tv))
            mask_tok = self.backbone.mask_token.to(e.dtype)
            return torch.where(known.reshape(B, L).unsqueeze(-1), e, mask_tok)
        B, tv, h, w = grid.shape
        e = self.video_emb(grid.reshape(B, L))
        if struct is not None and self.struct_emb is not None:
            s = self.struct_emb(struct.reshape(B, -1).long())
            if s.shape[1] < L:  # v3: struct covers content slices only; cond slices get no struct
                s = F.pad(s, (0, 0, L - s.shape[1], 0))
            else:  # v2: struct is grid-aligned; slice 0 is the reference (no struct)
                s = s.clone()
                s[:, :r] = 0.0
            e = e + s
        if self.bridge_init:
            return torch.where(known.reshape(B, L).unsqueeze(-1), e, self._bridge_prior(e, r, tv))
        mask_tok = self.backbone.mask_token.to(e.dtype)
        return torch.where(known.reshape(B, L).unsqueeze(-1), e, mask_tok)

    def _bridge_prior(self, e: torch.Tensor, r: int, tv: int) -> torch.Tensor:
        """Embedding that masked positions start from, instead of [MASK].

        "prev" (recommended): slice s starts from slice s-1, i.e. the previous frame. For the
        first content slice that is the motion-context slice, so the window starts from where the
        last one ended. generate() commits slices in order and block-causal attention only looks
        backwards, so slice s-1 is always already decoded when slice s is predicted.

        "ref" (the v3.1 setting, kept for reproducibility): every slice starts from slice 0.
        Measured on runs/stream_v3_161491: this makes ALL content slices byte-identical at
        generation time (spread 0.000000), so the model can only differentiate output frames
        through slice_emb and audio cross-attention -- and audio moves the hidden states by ~3%.
        That is the static, melting output. See docs/v3_improvement_plan.md Part VI, D1."""
        if str(self.bridge_init) == "prev":
            return torch.cat([e[:, :r], e[:, :-r]], dim=1)
        return e[:, :r].repeat(1, tv, 1)

    def _embed_audio(self, audio: torch.Tensor, n_ticks: int | None = None,
                     drop: torch.Tensor | None = None) -> torch.Tensor:
        """audio: WavLM features [B,Ta,D]. drop: bool [B] — null condition for audio CFG."""
        e = self.audio_proj(audio.float())
        if n_ticks is not None:
            e = e[:, :n_ticks]
        pos = self.audio_pos(torch.arange(e.shape[1], device=e.device)).unsqueeze(0)
        if drop is not None and drop.any():
            return torch.where(drop.view(-1, 1, 1), pos.expand_as(e), e + pos)
        return e + pos

    def sample_mask(self, B: int, device: torch.device) -> torch.Tensor:
        """Draw the training mask [B, tv*r] that forward() would draw.

        Exposed so auxiliary losses (pixel / SyncNet / TREPA) can share the CE pass's mask instead
        of running a fully-visible forward, which they could satisfy by copying the ground-truth
        token embedded at each position. See docs/v3_improvement_plan.md Part VI, D4."""
        if self.mask_schedule == "per_slice_cosine":
            return per_slice_cosine_mask(B, self.tv, self.r, device, ref_slices=self.ref_slices)
        return mixed_mask(B, self.tv, self.r, device, p_inference_shaped=self.p_inference_shaped,
                          ref_slices=self.ref_slices)

    def hidden_states(self, video_idx: torch.Tensor, audio: torch.Tensor, known: torch.Tensor,
                      struct: torch.Tensor | None = None, n_ticks: int | None = None,
                      audio_drop: torch.Tensor | None = None) -> torch.Tensor:
        e_v = self._embed_grid(video_idx, struct, known)
        e_a = self._embed_audio(audio, n_ticks, drop=audio_drop)
        return self.backbone(e_v, e_a)

    hidden_states_for_decode = hidden_states

    def compute_loss(self, h_masked: torch.Tensor, target: torch.Tensor, label_smoothing: float = 0.0):
        if self.factorized:
            return self.head.loss(h_masked, target, label_smoothing=label_smoothing)
        logits = self.head(self.norm(h_masked)).float()
        ce = F.cross_entropy(logits, target, label_smoothing=label_smoothing)
        acc = (logits.argmax(-1) == target).float().mean()
        return ce, {"ce": ce.detach(), "acc_dim_mean": acc.detach(), "acc_token": acc.detach(),
                    "loss": ce.detach()}

    def decode_pixels(self, h: torch.Tensor, vidtok, shape: tuple[int, int, int]) -> torch.Tensor:
        """Differentiable decode: hidden states -> expected FSQ codes -> VidTok decoder -> pixels.

        h: [N, dim] hidden states for the supervised (masked) tokens
        shape: (tv_content, h, w) of the content grid being decoded
        Returns [B, 3, T_content, H, W] in [-1, 1] with gradients flowing back to h."""
        if not self.factorized:
            raise RuntimeError("decode_pixels requires factorized FSQ head")
        B = h.shape[0] // (shape[0] * shape[1] * shape[2])
        codes = self.head.expected_codes(h)  # [N, n_dims] in FSQ normalized space
        # Reshape to VidTok latent grid: [B, n_dims, tv, h, w]
        z = codes.reshape(B, shape[0], shape[1], shape[2], -1).permute(0, 4, 1, 2, 3)
        z = vidtok.regularization.project_out(z)
        return vidtok.decoder(z)

    def forward(self, video_idx: torch.Tensor, audio: torch.Tensor,
                struct: torch.Tensor | None = None, cond_drop: float = 0.0,
                mask: torch.Tensor | None = None, mask_ratio: float | None = None,
                p_corrupt: float = 0.0, supervise_all_motion: bool = False):
        B, tv, h, w = video_idx.shape
        r, L, dev = self.r, self.tv * self.r, video_idx.device

        if struct is not None and cond_drop > 0 and self.training:
            keep = torch.rand(B, device=dev) >= cond_drop
            struct = torch.where(keep.view(B, 1, 1, 1), struct, torch.zeros_like(struct))

        if mask is None:
            if mask_ratio is not None and mask_ratio == 0.0:
                mask = torch.zeros(B, L, dtype=torch.bool, device=dev)
            elif mask_ratio is not None and mask_ratio > 0:
                motion = (torch.arange(L, device=dev) >= r).unsqueeze(0)
                mask = (torch.rand(B, L, device=dev) < mask_ratio) & motion
            elif self.mask_schedule == "per_slice_cosine":
                mask = per_slice_cosine_mask(B, tv, r, dev, ref_slices=self.ref_slices)
            else:
                mask = mixed_mask(B, tv, r, dev, p_inference_shaped=self.p_inference_shaped,
                                  ref_slices=self.ref_slices)

        if self.training and p_corrupt > 0:
            video_idx = corrupt_context(video_idx, mask, self.video_card, p_corrupt=p_corrupt,
                                        ref_slices=self.corrupt_from, r=r)

        audio_drop = None
        if self.training and self.audio_dropout > 0:
            audio_drop = torch.rand(B, device=dev) < self.audio_dropout

        known = ~mask
        hs = self.hidden_states(video_idx, audio, known, struct=struct, audio_drop=audio_drop)

        sup = mask.view(B, tv, r).clone()
        sup[:, :self.ref_slices] = False
        if supervise_all_motion:
            sup = torch.zeros(B, tv, r, dtype=torch.bool, device=dev)
            sup[:, self.ref_slices:] = True
        h_masked = hs[sup]
        target = video_idx.reshape(B, tv, r)[sup]
        return h_masked, target

    def _content_latents(self, video_lat: torch.Tensor) -> torch.Tensor:
        """[B, tv, z_ch, h, w] -> content positions as [B, Lc, z_ch] (slice-major, then spatial)."""
        B, _, z_ch, _, _ = video_lat.shape
        z = video_lat[:, self.ref_slices:]
        return z.reshape(B, -1, z_ch, self.r).permute(0, 1, 3, 2).reshape(B, -1, z_ch)

    def forward_continuous(self, video_lat: torch.Tensor, audio: torch.Tensor,
                           struct: torch.Tensor | None = None, cond_drop: float = 0.0):
        """Flow-matching training step on content latents.

        video_lat [B, tv, z_ch, h, w]: conditioning slices then content. The backbone sees the
        conditioning slices clean and, at each content position, the bridge prior (with "prev",
        the previous slice's latent -- teacher-forced, matching the sequential decoder in
        generate_continuous). Returns (loss, parts, z0_hat) where z0_hat [B, ctv, z_ch, h, w] is
        the one-step clean estimate, with graph, for pixel-space losses."""
        from sang.diffusion import flow_loss
        B, tv, z_ch, h, w = video_lat.shape
        r, dev = self.r, video_lat.device

        if struct is not None and cond_drop > 0 and self.training:
            keep = torch.rand(B, device=dev) >= cond_drop
            struct = torch.where(keep.view(B, 1, 1, 1), struct, torch.zeros_like(struct))
        audio_drop = None
        if self.training and self.audio_dropout > 0:
            audio_drop = torch.rand(B, device=dev) < self.audio_dropout

        known = torch.zeros(B, tv * r, dtype=torch.bool, device=dev)
        known[:, : self.ref_slices * r] = True
        hs = self.hidden_states(video_lat, audio, known, struct=struct, audio_drop=audio_drop)
        h_content = self.norm(hs[:, self.ref_slices:].reshape(B, -1, hs.shape[-1]))
        loss, parts, z0_hat = flow_loss(self.head, h_content, self._content_latents(video_lat))
        parts["loss"] = loss.detach()
        ctv = tv - self.ref_slices
        z0_hat = z0_hat.reshape(B, ctv, r, z_ch).permute(0, 1, 3, 2).reshape(B, ctv, z_ch, h, w)
        return loss, parts, z0_hat

    @torch.no_grad()
    def generate_continuous(self, audio: torch.Tensor, ref: torch.Tensor,
                            struct: torch.Tensor | None = None, ctx: torch.Tensor | None = None,
                            steps: int = 12, audio_cfg: float = 1.0, bridge_t: float = 1.0):
        """Sequential flow decode, one content slice at a time.

        ref / ctx [B, z_ch, h, w]. Each generated slice is written into the grid and marked known
        before the next is predicted, so with bridge_init "prev" the next slice's prior is the
        previous *prediction* -- the same relationship training has with the previous GT slice.
        (Decoding all slices jointly left the prior = ref for slices >= 3 at inference only.)
        bridge_t < 1 anchors each slice on the reference latent SDEdit-style; 1.0 = from noise.
        Returns content latents [B, ctv, z_ch, h, w]."""
        from sang.diffusion import flow_sample
        B, dev = ref.shape[0], ref.device
        tv, r, z_ch, sp = self.tv, self.r, self.z_ch, self.spatial
        grid = ref.reshape(B, 1, z_ch, sp, sp).repeat(1, tv, 1, 1, 1).clone()
        if self.ref_slices > 1:
            grid[:, 1] = (ref if ctx is None else ctx).reshape(B, z_ch, sp, sp)
        known = torch.zeros(B, tv * r, dtype=torch.bool, device=dev)
        known[:, : self.ref_slices * r] = True
        anchor = ref.reshape(B, z_ch, r).permute(0, 2, 1) if bridge_t < 1.0 else None
        null = torch.ones(B, dtype=torch.bool, device=dev) if audio_cfg != 1.0 else None

        for si in range(self.ref_slices, tv):
            hs = self.hidden_states(grid, audio, known, struct=struct)
            h_si = self.norm(hs[:, si])
            h_u = None
            if null is not None:
                h_u = self.norm(self.hidden_states(grid, audio, known, struct=struct, audio_drop=null)[:, si])
            z = flow_sample(self.head, h_si, (r, z_ch), steps=steps, anchor=anchor,
                            t_start=bridge_t, h_uncond=h_u, cfg=audio_cfg)
            grid[:, si] = z.permute(0, 2, 1).reshape(B, z_ch, sp, sp)
            known[:, si * r:(si + 1) * r] = True
        return grid[:, self.ref_slices:]

    @torch.no_grad()
    def logits_full(self, video_idx: torch.Tensor, audio: torch.Tensor,
                    struct: torch.Tensor | None = None) -> torch.Tensor:
        B = video_idx.shape[0]
        known = torch.ones(B, self.tv * self.r, dtype=torch.bool, device=video_idx.device)
        hs = self.hidden_states(video_idx, audio, known, struct=struct)
        flat = hs.reshape(B, -1, hs.shape[-1])
        if self.factorized:
            return self.head.argmax_tokens(flat).view(B, self.tv, self.r)
        return self.head(self.norm(flat)).argmax(-1).view(B, self.tv, self.r)

    @torch.no_grad()
    def generate(self, audio: torch.Tensor, ref: torch.Tensor,
                 struct: torch.Tensor | None = None, ctx: torch.Tensor | None = None,
                 steps: int = 8, temperature: float = 1.0, gumbel_temp: float = 4.5,
                 schedule: str = "cosine", top_k: int = 0, final_greedy: bool = True,
                 refine: int | None = None, audio_cfg: float = 1.0) -> torch.Tensor:
        """MaskGIT decode of the content slices. Grid = [ref, ctx?, content...]:
        ref [B,h,w] = identity anchor; ctx [B,h,w] = previous window's last slice (defaults to
        ref = static start). audio_cfg > 1 enables classifier-free guidance on the audio."""
        if refine is not None:
            steps = refine
        tv, r, B, dev = self.tv, self.r, ref.shape[0], ref.device
        h = w = self.spatial

        grid = torch.zeros(B, tv, h, w, dtype=torch.long, device=dev)
        grid[:, 0] = ref.reshape(B, h, w)
        if self.ref_slices > 1:
            grid[:, 1] = (ref if ctx is None else ctx).reshape(B, h, w)
        known = torch.zeros(B, tv, r, dtype=torch.bool, device=dev)
        known[:, :self.ref_slices] = True
        ta = audio.shape[1]
        ticks = slice_ticks(tv, ta, self.ref_slices, dev, lookahead=self.audio_lookahead)
        null_audio = torch.ones(B, dtype=torch.bool, device=dev) if audio_cfg != 1.0 else None

        for si in range(self.ref_slices, tv):
            n_ticks = int(ticks[si])
            sched = maskgit_decode_schedule(r, max(1, steps), schedule)

            for t, n_still_masked in enumerate(sched, start=1):
                hs = self.hidden_states(grid, audio, known.reshape(B, tv * r),
                                        struct=struct, n_ticks=n_ticks)
                h_si = hs[:, si]
                h_si_u = None
                if null_audio is not None:
                    h_si_u = self.hidden_states(grid, audio, known.reshape(B, tv * r),
                                                struct=struct, n_ticks=n_ticks,
                                                audio_drop=null_audio)[:, si]

                temp = 0.0 if (final_greedy and t == len(sched)) else temperature
                if self.factorized:
                    tok, logp = self.head.sample_tokens(h_si, temperature=temp, top_k=top_k,
                                                        h_uncond=h_si_u, guidance=audio_cfg)
                else:
                    lg = self.head(self.norm(h_si)).float()
                    if h_si_u is not None:
                        lg_u = self.head(self.norm(h_si_u)).float()
                        lg = lg_u + audio_cfg * (lg - lg_u)
                    if temp <= 0:
                        tok = lg.argmax(-1)
                    else:
                        tok = torch.multinomial((lg / temp).softmax(-1).reshape(-1, lg.shape[-1]), 1) \
                            .reshape(lg.shape[:-1])
                    logp = lg.log_softmax(-1).gather(-1, tok.unsqueeze(-1)).squeeze(-1)

                conf = logp
                if gumbel_temp > 0 and steps > 1:
                    g = -torch.log(-torch.log(torch.rand_like(conf).clamp_min(1e-10)).clamp_min(1e-10))
                    conf = conf + gumbel_temp * (1.0 - t / len(sched)) * g
                conf = conf.masked_fill(known[:, si], float("inf"))

                n_keep = r - n_still_masked
                order = conf.argsort(dim=-1, descending=True)
                new_known = torch.zeros_like(known[:, si])
                new_known.scatter_(1, order[:, :n_keep], True)

                newly = new_known & ~known[:, si]
                gflat = grid[:, si].reshape(B, r)
                grid[:, si] = torch.where(newly, tok, gflat).reshape(B, h, w)
                known[:, si] = new_known

        return grid


def _self_check():
    torch.manual_seed(0)
    B, h, w, V = 2, 16, 16, 128

    # single-ref layout
    tv = 5
    m = StreamingTalkingHead(V, dim=64, tv=tv, spatial=h, num_heads=4, num_layers=2,
                             factorized_head=False, audio_dim=32).train()
    video = torch.randint(0, V, (B, tv, h, w))
    audio = torch.randn(B, 9, 32)
    h_masked, tgt = m(video, audio)
    assert h_masked.ndim == 2 and tgt.ndim == 1 and h_masked.shape[0] == tgt.shape[0]
    loss, parts = m.compute_loss(h_masked, tgt)
    assert "ce" in parts
    m.eval()
    g1 = m.generate(audio, video[:, 0], steps=1, gumbel_temp=0.0, temperature=0.0)
    g8 = m.generate(audio, video[:, 0], steps=4, gumbel_temp=0.0, temperature=0.0)
    assert g1.shape == g8.shape == (B, tv, h, w)
    assert (g1[:, 0] == video[:, 0]).all(), "ref slice must be preserved"
    known = torch.zeros(B, tv * h * w, dtype=torch.bool)
    known[:, :h * w] = True
    d = (m.hidden_states(video, audio, known) - m.hidden_states_for_decode(video, audio, known)).abs().max()
    assert d.item() < 1e-5

    # motion_ctx layout: grid = [identity, ctx, content...], content-only struct
    tv3 = 7
    m3 = StreamingTalkingHead(V, dim=64, tv=tv3, spatial=h, num_heads=4, num_layers=2,
                              factorized_head=False, ref_slices=2, audio_dim=32,
                              audio_dropout=0.5).train()
    video3 = torch.randint(0, V, (B, tv3, h, w))
    feats = torch.randn(B, 12, 32)
    struct3 = torch.randint(0, V, (B, tv3 - 2, h, w))
    hm3, tg3 = m3(video3, feats, struct=struct3)
    assert hm3.shape[0] == tg3.shape[0] > 0
    m3.compute_loss(hm3, tg3)[0].backward()  # all paths (incl. audio_proj) receive grads
    m3.eval()
    g = m3.generate(feats, video3[:, 0], struct=struct3, ctx=video3[:, 1], steps=2,
                    gumbel_temp=0.0, temperature=0.0)
    assert g.shape == (B, tv3, h, w)
    assert (g[:, 0] == video3[:, 0]).all() and (g[:, 1] == video3[:, 1]).all(), "cond slices preserved"
    g_cfg = m3.generate(feats, video3[:, 0], steps=1, gumbel_temp=0.0, temperature=0.0,
                        audio_cfg=1.5)  # ctx defaults to ref; CFG dual-pass
    assert g_cfg.shape == (B, tv3, h, w)

    # causality: tokens in future slices must not change earlier hidden states
    known3 = torch.ones(B, tv3 * h * w, dtype=torch.bool)
    h_a = m3.hidden_states(video3, feats, known3)
    video3b = video3.clone()
    video3b[:, 4:] = torch.randint(0, V, (B, tv3 - 4, h, w))
    h_b = m3.hidden_states(video3b, feats, known3)
    assert (h_a[:, :4] - h_b[:, :4]).abs().max() < 1e-5, "future slices leaked into the past"

    # factorized coupled head with CFG guidance
    C = torch.tensor([[i, j] for i in range(8) for j in range(8)], dtype=torch.float)
    mh = StreamingTalkingHead(64, dim=64, tv=tv3, spatial=h, num_heads=4, num_layers=2,
                              ref_slices=2, audio_dim=32, fsq_codes=C, coupled_fsq_head=True).eval()
    assert mh.factorized
    v = torch.randint(0, 64, (B, tv3, h, w))
    hm, tg = mh(v, feats)
    loss, parts = mh.compute_loss(hm, tg)
    assert "acc_token" in parts
    g = mh.generate(feats, v[:, 0], ctx=v[:, 1], steps=2, audio_cfg=1.5)
    assert g.shape == (B, tv3, h, w) and g.min() >= 0 and g.max() < 64

    # bridge_init: masked content positions embed the identity ref token, not [MASK]
    mb = StreamingTalkingHead(V, dim=64, tv=tv3, spatial=h, num_heads=4, num_layers=2,
                              factorized_head=False, ref_slices=2, audio_dim=32,
                              bridge_init=True, audio_lookahead=2)
    known_c = torch.zeros(B, tv3 * h * w, dtype=torch.bool)
    known_c[:, :2 * h * w] = True  # cond slices known, all content masked
    e = mb._embed_grid(v, None, known_c)
    ref_e = mb.video_emb(v[:, 0].reshape(B, -1))
    assert torch.allclose(e[:, 2 * h * w:], ref_e.repeat(1, tv3 - 2, 1)), \
        "bridge_init: masked content must read the identity-ref embedding"
    mb.train()
    hm, tg = mb(v, feats, struct=struct3)
    mb.compute_loss(hm, tg)[0].backward()  # grads flow through the bridge prior
    mb.eval()
    g = mb.generate(feats, v[:, 0], struct=struct3, ctx=v[:, 1], steps=2,
                    gumbel_temp=0.0, temperature=0.0)
    assert g.shape == (B, tv3, h, w) and (g[:, :2] == v[:, :2]).all()

    # audio_lookahead: each slice sees `lookahead` extra future audio ticks (clamped at ta)
    t0 = slice_ticks(7, 34, 2)
    t4 = slice_ticks(7, 34, 2, lookahead=4)
    assert (t4 >= t0).all() and t4[-1] == t0[-1] == 34 and t4[2] == t0[2] + 4


if __name__ == "__main__":
    _self_check()
    print("ok")
