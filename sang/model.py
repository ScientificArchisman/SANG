import torch
import torch.nn.functional as F
from torch import nn
from moshi.modules.transformer import StreamingTransformer


class TalkingHead(nn.Module):
    """Flattened 1-D AR over VidTok tokens; frame 0 = reference (identity), audio via cross-attention (E1)."""

    def __init__(self, video_card: int = 32768, audio_card: int = 2048, dim: int = 512,
                 num_heads: int = 8, num_layers: int = 8, dropout: float = 0.0):
        super().__init__()
        self.bos = video_card
        self.video_emb = nn.Embedding(video_card + 1, dim)
        self.struct_emb = nn.Embedding(video_card, dim)
        self.audio_emb = nn.Embedding(audio_card, dim)
        self.audio_pos = nn.Embedding(4096, dim)
        self.drop = nn.Dropout(dropout)
        self.transformer = StreamingTransformer(
            d_model=dim, num_heads=num_heads, num_layers=num_layers,
            dim_feedforward=4 * dim, causal=True, cross_attention=True,
            positional_embedding="rope",
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, video_card, bias=False)

    def audio_cond(self, audio: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(audio.shape[-1], device=audio.device)
        return self.audio_emb(audio.long()).sum(1) + self.audio_pos(pos)

    def logits(self, seq: torch.Tensor, ca: torch.Tensor, add: torch.Tensor | None = None) -> torch.Tensor:
        e = self.video_emb(seq.long())
        if add is not None:
            e = e + add[:, : e.shape[1]]
        h = self.transformer(self.drop(e), cross_attention_src=ca)
        return self.head(self.drop(self.norm(h)))

    def struct_add(self, struct: torch.Tensor, r: int, cond_drop: float = 0.0) -> torch.Tensor:
        B = struct.shape[0]
        add = self.struct_emb(struct.reshape(B, -1).long())
        mask = torch.ones(B, add.shape[1], 1, device=add.device, dtype=add.dtype)
        mask[:, :r] = 0.0
        if self.training and cond_drop > 0:
            mask = mask * (torch.rand(B, 1, 1, device=add.device) >= cond_drop).to(add.dtype)
        return add * mask

    def forward(self, video_idx: torch.Tensor, audio: torch.Tensor,
                struct: torch.Tensor | None = None, cond_drop: float = 0.0):
        B, tv, h, w = video_idx.shape
        r = h * w
        target = video_idx.reshape(B, -1).long()
        inp = torch.cat([target.new_full((B, 1), self.bos), target[:, :-1]], dim=1)
        add = self.struct_add(struct, r, cond_drop) if struct is not None else None
        logits = self.logits(inp, self.audio_cond(audio), add)
        return logits[:, r:], target[:, r:]

    @torch.no_grad()
    def generate(self, audio: torch.Tensor, ref: torch.Tensor, shape,
                 struct: torch.Tensor | None = None) -> torch.Tensor:
        tv, h, w = shape
        r = h * w
        ca = self.audio_cond(audio)
        add = self.struct_add(struct, r) if struct is not None else None
        seq = torch.cat([audio.new_full((1, 1), self.bos).long(), ref.reshape(1, r).long()], dim=1)
        for _ in range(tv * h * w - r):
            nxt = self.logits(seq, ca, add)[:, -1].argmax(-1, keepdim=True)
            seq = torch.cat([seq, nxt], dim=1)
        return seq[:, 1:].reshape(1, tv, h, w)


def token_loss(logits: torch.Tensor, target: torch.Tensor, fsq_codes: torch.Tensor | None = None,
               label_smoothing: float = 0.0, z_weight: float = 0.0, fsq_weight: float = 0.0):
    """Flat-AR CE (+ optional z-loss). FSQ partial-credit MSE is disabled for streaming v2."""
    if fsq_weight:
        raise ValueError(
            "fsq_loss_weight is disabled: expected-code MSE flattens the softmax. "
            "Use FactorizedFSQHead (factorized_head: true) for FSQ partial credit."
        )
    flat = logits.reshape(-1, logits.shape[-1]).float()
    tgt = target.reshape(-1)
    ce = F.cross_entropy(flat, tgt, label_smoothing=label_smoothing)
    total, parts = ce, {"ce": ce.detach()}
    if z_weight:
        z = flat.logsumexp(-1).square().mean()
        total = total + z_weight * z
        parts["z"] = z.detach()
    parts["loss"] = total.detach()
    return total, parts


def build_talking_head(cfg: dict, fsq_codes: torch.Tensor | None = None) -> nn.Module:
    """Factory: flat AR (`TalkingHead`) or block streaming (`StreamingTalkingHead`)."""
    tv = (cfg["frames"] - 1) // 4 + 1
    spatial = cfg["res"] // 8
    common = dict(
        video_card=cfg["codebook"], dim=cfg["dim"], num_heads=cfg["heads"],
        num_layers=cfg["layers"], dropout=cfg.get("dropout", 0.0),
    )
    if cfg.get("model_type") == "streaming":
        from sang.streaming_transformer import StreamingTalkingHead
        # motion_ctx (v3): grid = [identity ref, prev-window tail, content...] — 2 extra cond slices
        # (v2 conflated the ref with content slice 0, so it added ref_slices-1).
        if cfg.get("motion_ctx"):
            ref_slices, tv = 2, tv + 2
        else:
            ref_slices = cfg.get("ref_slices", 1)
            tv = tv + ref_slices - 1
        audio_dim = {"wavlm-base": 768, "wavlm-large": 1024}.get(cfg.get("audio_encoder"), 0)
        continuous = cfg.get("continuous", False)
        z_ch = int(cfg.get("z_ch", 16)) if continuous else 0
        return StreamingTalkingHead(
            **common, tv=tv, spatial=spatial,
            ref_slices=ref_slices, audio_dim=audio_dim,
            audio_dropout=cfg.get("audio_dropout", 0.0),
            frame0_loss_weight=cfg.get("frame0_loss_weight", 0.0),
            factorized_head=cfg.get("factorized_head", True) and not continuous,
            coupled_fsq_head=cfg.get("coupled_fsq_head", False),
            learn_logit_scale=cfg.get("learn_logit_scale", True),
            fsq_codes=None if continuous else fsq_codes,
            mask_schedule=cfg.get("mask_schedule", "per_slice_cosine"),
            p_inference_shaped=cfg.get("p_inference_shaped", 0.0),
            bridge_init=cfg.get("bridge_init", False),
            audio_lookahead=cfg.get("audio_lookahead", 0),
            continuous=continuous, z_ch=z_ch,
            diff_depth=cfg.get("diff_depth", 3),
        )
    return TalkingHead(**common)
