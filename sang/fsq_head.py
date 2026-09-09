"""Factorized prediction head over FSQ digit dimensions."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dt)


class FactorizedFSQHead(nn.Module):
    """n_dims small softmaxes instead of one V-way head."""

    def __init__(self, dim: int, codec, logit_scale_init: float = 1.0, learn_scale: bool = True):
        super().__init__()
        self.codec = codec
        self.levels = codec.levels
        self.n_dims = codec.n_dims
        self.norm = RMSNorm(dim)
        self.heads = nn.ModuleList([nn.Linear(dim, l) for l in self.levels])
        for h in self.heads:
            nn.init.normal_(h.weight, std=0.02)
            nn.init.zeros_(h.bias)
        self.logit_scale = nn.Parameter(
            torch.tensor(float(math.log(logit_scale_init))), requires_grad=learn_scale
        )

    def forward(self, h: torch.Tensor) -> list[torch.Tensor]:
        z = self.norm(h)
        s = self.logit_scale.exp()
        return [head(z).float() * s for head in self.heads]

    def expected_codes(self, h: torch.Tensor) -> torch.Tensor:
        """Differentiable expected FSQ codes from logits: [N, n_dims] in quantized space.

        Uses softmax over each digit head, then maps to the actual quantized values.
        This is the key to backprop through the FSQ bottleneck into pixel-space losses."""
        logits = self.forward(h)
        codes = []
        for d, lg in enumerate(logits):
            probs = lg.softmax(-1)
            levels = self.levels[d]
            half = levels // 2
            vals = torch.arange(levels, device=lg.device, dtype=lg.dtype)
            vals = (vals - half) / half  # FSQ normalized space [-1, 1]
            codes.append((probs * vals).sum(-1))
        return torch.stack(codes, dim=-1)  # [N, n_dims]

    def loss(self, h: torch.Tensor, target_idx: torch.Tensor, label_smoothing: float = 0.0):
        tgt_digits = self.codec.to_digits(target_idx)
        logits = self.forward(h)
        total = h.new_zeros((), dtype=torch.float32)
        parts, correct = {}, None
        for d, lg in enumerate(logits):
            ce_d = F.cross_entropy(lg, tgt_digits[:, d], label_smoothing=label_smoothing)
            total = total + ce_d
            parts[f"ce_d{d}"] = ce_d.detach()
            pred_d = lg.argmax(-1)
            parts[f"acc_d{d}"] = (pred_d == tgt_digits[:, d]).float().mean().detach()
            correct = (pred_d == tgt_digits[:, d]) if correct is None \
                else correct & (pred_d == tgt_digits[:, d])
        parts["ce"] = total.detach()
        parts["acc_dim_mean"] = torch.stack([parts[f"acc_d{d}"] for d in range(self.n_dims)]).mean()
        parts["acc_token"] = correct.float().mean().detach()
        parts["logit_scale"] = self.logit_scale.exp().detach()
        parts["loss"] = total.detach()
        return total, parts

    @torch.no_grad()
    def token_topk_acc(self, h: torch.Tensor, target_idx: torch.Tensor, ks: tuple[int, ...] = (1, 5, 10)):
        """Joint token top-k under independent digit softmaxes (V=32768)."""
        logits = self.forward(h)
        lut = self.codec._digits_lut.to(h.device)
        scores = h.new_zeros(h.shape[0], self.codec.vocab)
        for d, lg in enumerate(logits):
            scores += lg.log_softmax(-1)[:, lut[:, d]]
        tgt = target_idx.long().unsqueeze(-1)
        topk = scores.topk(max(ks), dim=-1).indices
        hits = topk == tgt
        return {k: hits[:, :k].any(-1).float().mean().item() for k in ks}

    @torch.no_grad()
    def argmax_tokens(self, h: torch.Tensor) -> torch.Tensor:
        logits = self.forward(h)
        digits = torch.stack([lg.argmax(-1) for lg in logits], dim=-1)
        return self.codec.from_digits(digits)

    @torch.no_grad()
    def sample_tokens(self, h: torch.Tensor, temperature: float = 1.0, top_k: int = 0,
                      h_uncond: torch.Tensor | None = None, guidance: float = 1.0):
        logits = self.forward(h)
        if h_uncond is not None and guidance != 1.0:  # classifier-free: u + g*(c - u)
            logits = [u + guidance * (c - u) for c, u in zip(logits, self.forward(h_uncond))]
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
                d = torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1).reshape(probs.shape[:-1])
            digits.append(d)
            logps.append(lg.log_softmax(-1).gather(-1, d.unsqueeze(-1)).squeeze(-1))
        digits = torch.stack(digits, dim=-1)
        return self.codec.from_digits(digits), torch.stack(logps, dim=-1).sum(-1)


class CoupledFSQHead(FactorizedFSQHead):
    """Autoregressive across FSQ dims within each token."""

    def __init__(self, dim: int, codec, **kw):
        super().__init__(dim, codec, **kw)
        self.digit_emb = nn.ModuleList([nn.Embedding(l, dim) for l in self.levels[:-1]])

    def _step_logits(self, z, d, prev_digits):
        ctx = z
        for k in range(d):
            ctx = ctx + self.digit_emb[k](prev_digits[..., k])
        return self.heads[d](ctx).float() * self.logit_scale.exp()

    def expected_codes(self, h: torch.Tensor) -> torch.Tensor:
        """Differentiable expected FSQ codes through the coupled chain.

        Must not inherit FactorizedFSQHead's version: that reads heads[d](z) with no digit
        context, which these heads never see under the CE objective. Digit d is conditioned on
        the soft expectation over digits <d — the differentiable relaxation of _step_logits."""
        z = self.norm(h)
        ctx, codes = z, []
        for d, levels in enumerate(self.levels):
            lg = self.heads[d](ctx).float() * self.logit_scale.exp()
            probs = lg.softmax(-1)
            half = levels // 2
            vals = (torch.arange(levels, device=lg.device, dtype=lg.dtype) - half) / half
            codes.append((probs * vals).sum(-1))
            if d < len(self.digit_emb):  # carry the soft digit forward, as _step_logits does
                ctx = ctx + probs @ self.digit_emb[d].weight
        return torch.stack(codes, dim=-1)

    def loss(self, h, target_idx, label_smoothing=0.0):
        tgt = self.codec.to_digits(target_idx)
        z = self.norm(h)
        total = h.new_zeros((), dtype=torch.float32)
        parts, correct = {}, None
        for d in range(self.n_dims):
            lg = self._step_logits(z, d, tgt)
            ce_d = F.cross_entropy(lg, tgt[:, d], label_smoothing=label_smoothing)
            total = total + ce_d
            parts[f"ce_d{d}"] = ce_d.detach()
            ok = lg.argmax(-1) == tgt[:, d]
            parts[f"acc_d{d}"] = ok.float().mean().detach()
            correct = ok if correct is None else correct & ok
        parts["ce"] = total.detach()
        parts["acc_dim_mean"] = torch.stack([parts[f"acc_d{d}"] for d in range(self.n_dims)]).mean()
        parts["acc_token"] = correct.float().mean().detach()
        parts["loss"] = total.detach()
        return total, parts

    @torch.no_grad()
    def sample_tokens(self, h, temperature=1.0, top_k=0, h_uncond=None, guidance=1.0):
        z = self.norm(h)
        zu = self.norm(h_uncond) if (h_uncond is not None and guidance != 1.0) else None
        digits = torch.zeros(*z.shape[:-1], self.n_dims, dtype=torch.long, device=z.device)
        logp = torch.zeros(z.shape[:-1], device=z.device)
        for d in range(self.n_dims):
            lg = self._step_logits(z, d, digits)
            if zu is not None:  # classifier-free: u + g*(c - u)
                lu = self._step_logits(zu, d, digits)
                lg = lu + guidance * (lg - lu)
            if temperature <= 0:
                pick = lg.argmax(-1)
            else:
                probs = (lg / temperature).softmax(-1)
                pick = torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1).reshape(probs.shape[:-1])
            digits[..., d] = pick
            logp = logp + lg.log_softmax(-1).gather(-1, pick.unsqueeze(-1)).squeeze(-1)
        return self.codec.from_digits(digits), logp
