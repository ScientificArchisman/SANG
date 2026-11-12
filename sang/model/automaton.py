"""The gaze-head automaton (reports/SANG neurosymbolic jitter and gaze.md): a learned hybrid automaton over the head,
gaze and lid channels. SANG's sampler uses it as a shield in its last Euler steps, and its event grammar weights
particles across windows.

Structure (fixed, from physiology): two semi-Markov chains whose states are per-channel linear-Gaussian dynamics.
  head  hold (random walk) | stroke (local linear trend: angle plus velocity)         pitch, yaw, roll
  eyes  fixation | saccade | blink (random walks of very different sizes)             world gaze x, y; lid
World gaze is eye-in-head gaze plus a learned VOR term, w = e + G (yaw, pitch): in a fixation it is nearly constant,
so the eyes counter-rotate against the head. Every number is learned by hard EM on real motion (`fit`): process
variance per state and channel, observation variance per channel, duration pmfs, transitions and G.

Use: `segment` (semi-Markov Viterbi; each candidate segment is scored by its own Kalman likelihood, its filter starting
from the frame before, so the jump into a segment counts under that segment's state), then `smooth` (one switching
Kalman and RTS pass with the labels' dynamics). A saccade or blink has a large process variance, so its frames pass
almost unchanged: the automaton asks which state a frame is in, not how fast it moves. The mouth is never read or moved.
"""
import math

import numpy as np
import torch

from sang.model.target import WIDE_REGIONS, from_wide, wide

HEAD = [0, 1, 2]              # wide columns: pitch, yaw, roll (degrees)
EYEBALL = [36, 37, 48, 49]    # x, y of keypoints 11 and 15 (LivePortrait's eyeball keypoints)
LIDS = [43, 52]               # y of keypoints 13 and 16 (upper lids)
COLS = HEAD + EYEBALL + LIDS  # what the shield corrects
LOG2PI = math.log(2 * math.pi)


def _runs(row: np.ndarray) -> list[tuple[int, int, int]]:
    """Labels [T] -> [(start, end, state)] of the runs."""
    b = np.flatnonzero(np.diff(row)) + 1
    starts, ends = np.r_[0, b], np.r_[b, len(row)]
    return [(int(s), int(e), int(row[s])) for s, e in zip(starts, ends)]


class Chain:
    """One semi-Markov chain. trend[k]: state k is a local linear trend (q = its velocity's variance, v0 = the velocity
    variance at its onset); otherwise a random walk (q = the level's variance). r: observation variance per channel,
    shared by the states. No self transitions, so runs of a label are segments."""

    def __init__(self, names, trend, q, v0, r, logdur, logA, logpi):
        self.names, self.trend = list(names), [bool(t) for t in trend]
        self.q, self.v0, self.r, self.logdur, self.logA, self.logpi = q, v0, r, logdur, logA, logpi

    def state(self) -> dict:
        return {"names": self.names, "trend": self.trend, **{k: getattr(self, k).cpu() for k in
                                                              ("q", "v0", "r", "logdur", "logA", "logpi")}}

    @classmethod
    def load(cls, d: dict) -> "Chain":
        return cls(d["names"], d["trend"], *(d[k].float() for k in ("q", "v0", "r", "logdur", "logA", "logpi")))

    def to(self, dev) -> "Chain":
        for k in ("q", "v0", "r", "logdur", "logA", "logpi"):
            setattr(self, k, getattr(self, k).to(dev))
        return self

    @property
    def logsurv(self) -> torch.Tensor:
        """log P(duration >= d), d = 1..D."""
        return torch.logcumsumexp(self.logdur.flip(-1), -1).flip(-1)

    def seg_ll(self, y: torch.Tensor) -> torch.Tensor:
        """y [N, T, C] -> LL [N, M, T, T + 1]: LL[n, k, s, e] = log p(y[s:e] | one segment of state k). A fresh filter
        per start s, its level starting at y[s - 1] (y[0] with a broad prior at s = 0), summed over channels."""
        N, T, C = y.shape
        out = y.new_zeros(N, len(self.names), T, T + 1)
        on = torch.arange(T, device=y.device)
        r = self.r
        for k, trend in enumerate(self.trend):
            a, zero = float(trend), torch.zeros_like(self.q[k])
            ql, qv = (zero, self.q[k]) if trend else (self.q[k], zero)
            ml, mv, pll, plv, pvv = (y.new_zeros(N, T, C) for _ in range(5))
            for t in range(T):
                ml[:, t], mv[:, t], plv[:, t] = (y[:, t - 1] if t else y[:, 0]), 0.0, 0.0
                pll[:, t] = r if t else 1e4 * r
                pvv[:, t] = self.v0[k] if trend else 0.0
                ml, mv = ml + a * mv, a * mv
                pll, plv, pvv = pll + 2 * a * plv + a * a * pvv + ql, a * plv + a * a * pvv, a * a * pvv + qv
                S = pll + r
                e = y[:, t, None] - ml
                ll = -0.5 * (LOG2PI + S.log() + e * e / S).sum(-1)
                kl, kv = pll / S, plv / S
                ml, mv = ml + kl * e, mv + kv * e
                pll, plv, pvv = pll - kl * pll, plv - kl * plv, pvv - kv * plv
                out[:, k, :, t + 1] = out[:, k, :, t] + ll * (on <= t)
        return out

    def segment(self, y: torch.Tensor) -> torch.Tensor:
        """Semi-Markov Viterbi: the most likely labels [N, T]. The first and last segments may extend beyond the
        window, so they score by the duration's survival function instead of its pmf."""
        LL = self.seg_ll(y)
        N, M, T, _ = LL.shape
        D, dev = self.logdur.shape[1], y.device
        surv = self.logsurv
        V = y.new_full((N, T + 1, M), -math.inf)
        B = y.new_full((N, T + 1, M), -math.inf)  # best way to switch INTO state j right after frame e
        bs = torch.zeros(N, T + 1, M, dtype=torch.long, device=dev)
        bi = torch.zeros(N, T + 1, M, dtype=torch.long, device=dev)
        for e in range(1, T + 1):
            s = torch.arange(e, device=dev)
            dur = (surv if e == T else self.logdur)[:, (e - s - 1).clamp(max=D - 1)]
            dur[:, 0] = surv[:, min(e, D) - 1]
            prev = torch.cat([self.logpi.view(1, M, 1).expand(N, M, 1), B[:, 1:e].permute(0, 2, 1)], -1)
            V[:, e], bs[:, e] = (prev + LL[:, :, :e, e] + dur).max(-1)
            B[:, e], bi[:, e] = (V[:, e, :, None] + self.logA).max(1)
        lab = torch.zeros(N, T, dtype=torch.long, device=dev)
        rows, on = torch.arange(N, device=dev), torch.arange(T, device=dev)
        e, j = torch.full((N,), T, device=dev), V[:, T].argmax(-1)
        while bool((e > 0).any()):
            live = e > 0
            s = bs[rows, e, j]
            lab = torch.where(live[:, None] & (on >= s[:, None]) & (on < e[:, None]), j[:, None], lab)
            e, j = torch.where(live, s, e), torch.where(live, bi[rows, s, j], j)
        return lab

    def smooth(self, y: torch.Tensor, lab: torch.Tensor) -> torch.Tensor:
        """Posterior mean of the levels [N, T, C] under the labels' dynamics: one switching Kalman filter + RTS pass."""
        N, T, C = y.shape
        tr = torch.tensor(self.trend, device=y.device)[lab]
        a = tr.to(y.dtype)[..., None]
        q = self.q[lab]
        enter = torch.zeros_like(tr)
        enter[:, 1:] = lab[:, 1:] != lab[:, :-1]
        ql = torch.where(tr[..., None], 0.0, q)
        qv = torch.where(tr[..., None], q + enter[..., None] * self.v0[lab], 0.0)
        r = self.r
        ml, mv, pll, plv = y[:, 0].clone(), torch.zeros_like(y[:, 0]), 1e4 * r.expand(N, C), torch.zeros_like(y[:, 0])
        pvv = torch.where(tr[:, 0, None], self.v0[lab[:, 0]], 0.0)
        pred, filt = [], []
        for t in range(T):
            if t:
                at = a[:, t]
                ml, mv = ml + at * mv, at * mv
                pll, plv, pvv = (pll + 2 * at * plv + at * at * pvv + ql[:, t], at * plv + at * at * pvv,
                                 at * at * pvv + qv[:, t])
            pred.append((ml, mv, pll, plv, pvv))
            S = pll + r
            e = y[:, t] - ml
            kl, kv = pll / S, plv / S
            ml, mv = ml + kl * e, mv + kv * e
            pll, plv, pvv = pll - kl * pll, plv - kl * plv, pvv - kv * plv
            filt.append((ml, mv, pll, plv, pvv))
        sl, sv = filt[-1][0], filt[-1][1]
        out = [sl]
        for t in range(T - 2, -1, -1):
            fl, fv, fll, flv, fvv = filt[t]
            pl, pv, qll, qlv, qvv = pred[t + 1]
            at = a[:, t + 1]
            reg = qvv > 0  # a random walk's predicted velocity is exactly 0: invert the level alone
            det = torch.where(reg, qll * qvv - qlv * qlv, 1.0)
            ill = torch.where(reg, qvv / det, 1.0 / qll)
            ilv = torch.where(reg, -qlv / det, 0.0)
            ivv = torch.where(reg, qll / det, 0.0)
            gl, gv = fll + at * flv, flv + at * fvv  # rows of Pf F^T: [gl, at flv], [gv, at fvv]
            dl, dv = sl - pl, sv - pv
            sl = fl + (gl * ill + at * flv * ilv) * dl + (gl * ilv + at * flv * ivv) * dv
            sv = fv + (gv * ill + at * fvv * ilv) * dl + (gv * ilv + at * fvv * ivv) * dv
            out.append(sl)
        return torch.stack(out[::-1], 1)

    def grammar(self, y: torch.Tensor, lab: torch.Tensor) -> torch.Tensor:
        """Log-probability [N] of the discrete events only: the first state, the transitions, every duration (censored
        at the window's edges) and every saccade's jump size. No frame-level smoothness term, so it cannot pay for a
        flat head."""
        surv, D = self.logsurv, self.logdur.shape[1]
        out = []
        for n, row in enumerate(lab.cpu().numpy()):
            runs = _runs(row)
            lp = self.logpi[runs[0][2]]
            for i, (s, e, k) in enumerate(runs):
                d = min(e - s, D) - 1
                lp = lp + (surv[k, d] if s == 0 or e == len(row) else self.logdur[k, d])
                if i:
                    lp = lp + self.logA[runs[i - 1][2], k]
                if self.names[k] == "sacc" and s:
                    var = self.q[k] + 2 * self.r
                    j = y[n, s] - y[n, s - 1]
                    lp = lp - 0.5 * (LOG2PI + var.log() + j * j / var).sum()
            out.append(lp)
        return torch.stack(out)

    def lab_ll(self, y: torch.Tensor, lab: torch.Tensor, q=None, v0=None, r=None) -> torch.Tensor:
        """Per-channel log-likelihood [C] of chunks y [N, T, C] under their labels, every run filtered as `seg_ll`
        scores it (the level from the frame before, a trend's velocity from 0 with variance v0); frame 0 is left out.
        q, v0, r override the chain's own."""
        q, v0, r = (self.q if q is None else q), (self.v0 if v0 is None else v0), (self.r if r is None else r)
        N, T, C = y.shape
        tr = torch.tensor(self.trend, device=y.device)[lab][..., None]
        a_all, qs = tr.to(y.dtype), q[lab]
        ql, qv, v0s = torch.where(tr, 0.0, qs), torch.where(tr, qs, 0.0), torch.where(tr, v0[lab], 0.0)
        rr = r.expand(N, C)
        ml, mv, plv, pll, pvv = y[:, 0], torch.zeros_like(y[:, 0]), torch.zeros_like(y[:, 0]), 1e4 * rr, v0s[:, 0]
        out = y.new_zeros(C)
        for t in range(T):
            if t:
                new = lab[:, t, None] != lab[:, t - 1, None]
                ml, mv, pll = torch.where(new, y[:, t - 1], ml), torch.where(new, 0.0, mv), torch.where(new, rr, pll)
                plv, pvv = torch.where(new, 0.0, plv), torch.where(new, v0s[:, t], pvv)
            a = a_all[:, t]
            ml, mv = ml + a * mv, a * mv
            pll, plv, pvv = pll + 2 * a * plv + a * a * pvv + ql[:, t], a * plv + a * a * pvv, a * a * pvv + qv[:, t]
            S = pll + r
            e = y[:, t] - ml
            if t:
                out = out - 0.5 * (LOG2PI + S.log() + e * e / S).sum(0)
            kl, kv = pll / S, plv / S
            ml, mv = ml + kl * e, mv + kv * e
            pll, plv, pvv = pll - kl * pll, plv - kl * plv, pvv - kv * plv
        return out

    def mstep(self, y: torch.Tensor, lab: torch.Tensor, smooth_counts: float = 0.1) -> None:
        """Hard-EM update from labelled chunks. r, q and the trends' v0 by coordinate ascent on `lab_ll`, per channel
        (moment estimates of a trend's small velocity noise collapse, and the chain then cuts every stroke into
        pieces); durations, transitions and the first state by counting (censored runs give no duration)."""
        M, D = len(self.names), self.logdur.shape[1]
        mult = y.new_tensor([1.0, 0.25, 0.5, 0.71, 1.41, 2.0, 4.0])  # 1 first: a state with no frames keeps its q
        params = [("r", None)] + [("q", k) for k in range(M)] + [("v0", k) for k in range(M) if self.trend[k]]
        for _ in range(2):
            for name, k in params:
                full = getattr(self, name)
                cur = full if k is None else full[k]
                lls = []
                for m in mult:
                    cand = cur * m if k is None else torch.cat([full[:k], (cur * m)[None], full[k + 1:]])
                    lls.append(self.lab_ll(y, lab, **{name: cand}))
                best = cur * mult[torch.stack(lls).argmax(0)]
                setattr(self, name, best if k is None else torch.cat([full[:k], best[None], full[k + 1:]]))
        dur = np.full((M, D), smooth_counts)
        trans = np.ones((M, M)) - np.eye(M)
        first = np.ones(M)
        for row in lab.cpu().numpy():
            runs = _runs(row)
            first[runs[0][2]] += 1
            for i, (s, e, k) in enumerate(runs):
                if s > 0 and e < len(row):
                    dur[k, min(e - s, D) - 1] += 1
                if i:
                    trans[runs[i - 1][2], k] += 1
        with np.errstate(divide="ignore"):  # log 0 = -inf: no self transitions
            t = lambda x: torch.as_tensor(np.log(x), dtype=y.dtype, device=y.device)
            self.logA = t(trans / np.maximum(trans.sum(1, keepdims=True), 1e-12))
        self.logdur = t(dur / dur.sum(1, keepdims=True))
        self.logpi = t(first / first.sum())


class Automaton:
    """The head chain, the eye chain and the VOR gain G [2, 2] ((yaw, pitch) degrees -> eyeball units)."""

    def __init__(self, head: Chain, eye: Chain, G: torch.Tensor):
        self.head, self.eye, self.G = head, eye, G

    def state(self) -> dict:
        return {"head": self.head.state(), "eye": self.eye.state(), "G": self.G.cpu()}

    @classmethod
    def load(cls, d: dict) -> "Automaton":
        return cls(Chain.load(d["head"]), Chain.load(d["eye"]), d["G"].float())

    def to(self, dev) -> "Automaton":
        self.head.to(dev)
        self.eye.to(dev)
        self.G = self.G.to(dev)
        return self

    @staticmethod
    def signals(w: torch.Tensor):
        """Wide motion [..., T, 69] -> head [..., 3], eye-in-head gaze [..., 2] (mean of the two eyeballs), lid [..., 1]."""
        return (w[..., HEAD], 0.5 * (w[..., EYEBALL[:2]] + w[..., EYEBALL[2:]]),
                0.5 * (w[..., LIDS[0]] + w[..., LIDS[1]])[..., None])

    def eye_obs(self, gaze, head, lid):
        """World gaze and lid [..., 3]."""
        return torch.cat([gaze + head[..., [1, 0]] @ self.G.T, lid], -1)

    def correct(self, w: torch.Tensor, P: int = 0) -> torch.Tensor:
        """Wide motion [N, T, 69] (the first P frames a clamped prefix) -> the change [N, T - P, 9] on COLS that
        replaces the head, gaze and lids by their posterior mean. Both eyeballs move by the gaze's change, so their
        vergence is kept; the eyes are re-derived from the smoothed world gaze and the smoothed head, so they
        counter-rotate exactly as much as the head's correction asks."""
        head, gaze, lid = self.signals(w)
        hs = self.head.smooth(head, self.head.segment(head))
        ey = self.eye_obs(gaze, head, lid)
        es = self.eye.smooth(ey, self.eye.segment(ey))
        dg = es[..., :2] - hs[..., [1, 0]] @ self.G.T - gaze
        dl = es[..., 2:] - lid
        return torch.cat([hs - head, dg, dg, dl, dl], -1)[:, P:]

    def event_score(self, w: torch.Tensor, other: "Automaton | None" = None) -> torch.Tensor:
        """Event log-likelihood [N] of wide windows, segmented by this automaton; minus `other`'s on the same
        segmentation (the real-over-SANG ratio) when given."""
        head, gaze, lid = self.signals(w)
        lh = self.head.segment(head)
        ey = self.eye_obs(gaze, head, lid)
        le = self.eye.segment(ey)
        s = self.head.grammar(head, lh) + self.eye.grammar(ey, le)
        if other is not None:
            s = s - other.head.grammar(head, lh) - other.eye.grammar(other.eye_obs(gaze, head, lid), le)
        return s


def _geometric(mean: float, D: int) -> np.ndarray:
    p = 1.0 / mean
    pmf = p * (1 - p) ** np.arange(D)
    return np.log(pmf / pmf.sum())


def _batched(fn, *xs, batch: int = 512):
    return torch.cat([fn(*(x[i:i + batch] for x in xs)) for i in range(0, len(xs[0]), batch)])


def _init_chain(y: torch.Tensor, kind: str, states: bool, D: int) -> Chain:
    """Quantile initialisation (EM then learns every number)."""
    d = (y[:, 1:] - y[:, :-1]).flatten(0, 1)
    dd = (d[1:] - d[:-1])
    # the head is smooth between frames (second differences see its noise), gaze is piecewise constant (first ones do)
    cov = -(dd[1:] * dd[:-1]).mean(0) / 4 if kind == "head" else -(d[1:] * d[:-1]).mean(0)
    r = torch.maximum(cov, 1e-4 * (d * d).mean(0)).clamp_min(1e-12)  # a constant channel must not divide by 0
    pct = lambda x, p: torch.quantile(x[:: max(1, len(x) // 1_000_000)].float(), p, dim=0).to(y.dtype)
    lo = lambda v: torch.maximum(v, 1e-3 * r)
    d2, dd2 = d * d, dd * dd
    if not states:
        names, trend, q, v0, means = ["level"], [False], [lo(d2.mean(0) - 2 * r)], [r], [20.0]
    elif kind == "head":
        names, trend = ["hold", "stroke"], [False, True]
        q, v0, means = [lo(pct(d2, 0.25) - 2 * r), lo(pct(dd2, 0.5) - 6 * r)], [r, pct(d2, 0.75)], [10.0, 15.0]
    else:
        big = pct(d2, 0.99)
        sacc = torch.cat([big[:2], pct(d2, 0.75)[2:]])
        blink = torch.cat([100 * big[:2], pct(d2, 0.995)[2:]])
        names, trend = ["fix", "sacc", "blink"], [False, False, False]
        q, v0, means = [lo(pct(d2, 0.25)), sacc, blink], [r, r, r], [12.0, 2.0, 4.0]
    M = len(names)
    logA = torch.full((M, M), -math.inf if M == 1 else math.log(1.0 / max(M - 1, 1)), dtype=y.dtype)
    logA.fill_diagonal_(-math.inf)
    t = lambda x: torch.as_tensor(np.asarray(x), dtype=y.dtype, device=y.device)
    return Chain(names, trend, torch.stack(q), torch.stack(v0), r, t([_geometric(m, D) for m in means]),
                 logA.to(y.device), t(np.full(M, -math.log(M))))


def _vor(gaze, head, lab) -> tuple[torch.Tensor, float]:
    """G from within-fixation frames: d gaze ~ -G d(yaw, pitch); and the share of eye-in-head motion it explains."""
    dg = (gaze[:, 1:] - gaze[:, :-1])
    dh = (head[:, 1:] - head[:, :-1])[..., [1, 0]]
    sel = (lab[:, 1:] == 0) & (lab[:, 1:] == lab[:, :-1])
    X, Y = dh[sel].double(), dg[sel].double()
    if len(X) < 10:
        return torch.zeros(2, 2, dtype=gaze.dtype, device=gaze.device), 0.0
    G = -torch.linalg.lstsq(X, Y).solution.T
    share = 1 - float(((Y + X @ G.T) ** 2).sum() / (Y ** 2).sum().clamp_min(1e-30))
    return G.to(gaze.dtype), share


def fit(w: torch.Tensor, states: bool = True, vor: bool = True, rounds: int = 8, log=print) -> tuple[Automaton, dict]:
    """Hard EM on wide chunks [N, T, 69] (real motion through the generator's latent). states=False: one random-walk
    state per chain (a learned single-state Kalman); vor=False: G = 0. -> the automaton and its G0 statistics."""
    head, gaze, lid = Automaton.signals(w)
    D = w.shape[1]
    gha = Automaton(_init_chain(head, "head", states, D),
                    _init_chain(torch.cat([gaze, lid], -1), "eye", states, D),
                    torch.zeros(2, 2, dtype=w.dtype, device=w.device))
    share = 0.0
    for it in range(rounds):
        lh = _batched(gha.head.segment, head)
        ey = gha.eye_obs(gaze, head, lid)
        le = _batched(gha.eye.segment, ey)
        gha.head.mstep(head, lh)
        gha.eye.mstep(ey, le)
        if vor:
            gha.G, share = _vor(gaze, head, le)
        occ = lambda c, lab: " ".join(f"{n} {float((lab == k).float().mean()):.2f}" for k, n in enumerate(c.names))
        log(f"  EM round {it + 1}/{rounds}: head {occ(gha.head, lh)} | eyes {occ(gha.eye, le)} | VOR share {share:.2f}")
    fix = [e - s for row in le.cpu().numpy() for s, e, k in _runs(row) if k == 0 and s > 0 and e < D]
    sacc = sum(1 for row in le.cpu().numpy() for s, e, k in _runs(row) if gha.eye.names[k] == "sacc")
    stats = {"vor_share": share, "median_fix": float(np.median(fix)) if fix else float("nan"),
             "sacc_per_min": sacc * 60.0 * 25 / le.numel(), "r_head": gha.head.r.sqrt().tolist(),
             "r_eye": gha.eye.r.sqrt().tolist(), "G": gha.G.tolist()}
    return gha, stats


class Shield:
    """The automaton inside SANG's sampler (sang.model.flow.sample's `shield`): in the last n Euler steps the clean
    estimate's head, gaze and lid columns, decoded through the latent, are replaced by their posterior mean, with the
    minimum-norm code change that leaves the mouth's decoded columns as they were (weighted least squares, the mouth
    at 1e3). It also weights particles (sang.model.flow.generate's `particles`) by the real-over-SANG event ratio.
    The automaton itself runs on the CPU: a window is 74 frames."""

    def __init__(self, gha: Automaton, latent, k: int, n: int = 2, sang: Automaton | None = None, ratio: bool = True):
        self.gha, self.sang, self.latent, self.n, self.ratio = gha.to("cpu"), sang, latent, int(n), bool(ratio)
        W = (latent.sd[:, None] * latent.dec.weight[:, :k] * latent.c_sd[:k][None]).detach().double()
        rows = COLS + WIDE_REGIONS["mouth"]
        wts = torch.ones(len(rows), dtype=W.dtype, device=W.device)
        wts[len(COLS):] = 1e3
        self.lift = torch.linalg.pinv(wts[:, None] * W[rows])[:, :len(COLS)].float()  # [k, 9]

    @classmethod
    def load(cls, path: str, latent, k: int, n: int = 2, ratio: bool = True) -> "Shield":
        d = torch.load(path, map_location="cpu", weights_only=False)
        return cls(Automaton.load(d["real"]), latent, k, n, Automaton.load(d["sang"]) if d.get("sang") else None, ratio)

    def _wide(self, x, prefix):
        u = x if prefix is None else torch.cat([prefix, x], 1)
        return self.latent.decode(u).cpu(), 0 if prefix is None else prefix.shape[1]

    def __call__(self, x0: torch.Tensor, prefix: torch.Tensor | None = None) -> torch.Tensor:
        """Clean estimate [B, L, k] (+ its clean prefix [B, P, k]) -> corrected estimate [B, L, k]."""
        w, P = self._wide(x0, prefix)
        return x0 + self.gha.correct(w, P).to(x0) @ self.lift.T

    def resample(self, y: torch.Tensor, prefix: torch.Tensor | None = None,
                 generator: torch.Generator | None = None) -> torch.Tensor:
        """Particle indices [K]: systematic resampling by the tempered event score, the temperature the largest in
        (0, 1] that keeps the effective sample size >= K / 2."""
        if self.ratio and self.sang is None:
            raise ValueError("the particle ratio needs the automaton's SANG grammar (fit_automaton.py without --no-sang)")
        w, _ = self._wide(y, prefix)
        s = self.gha.event_score(w, self.sang if self.ratio else None).double()
        K = len(s)
        ess = lambda lam: float(1.0 / (torch.softmax(lam * s, 0) ** 2).sum())
        lam, lo, hi = 1.0, 0.0, 1.0
        if ess(1.0) < K / 2:
            for _ in range(30):
                lam = (lo + hi) / 2
                lo, hi = (lam, hi) if ess(lam) >= K / 2 else (lo, lam)
            lam = lo
        p = torch.softmax(lam * s, 0)
        dev = generator.device if generator is not None else "cpu"
        u = (torch.rand(1, generator=generator, device=dev).cpu().double() + torch.arange(K)) / K
        return torch.searchsorted(p.cumsum(0), u).clamp(max=K - 1).to(y.device)


HANDSET = ("one_euro", "kalman", "mean3", "savgol")


def handset(x: np.ndarray, name: str, fps: int = 25) -> np.ndarray:
    """Hand-set smoothers other talking-head systems use, as baselines on the shield's channels [n, c]: one_euro (1 Hz,
    beta 0: an EMA), kalman (LivePortrait's pykalman RTS, process 1e-5, observation 3e-7), mean3 (Ditto's k = 3 mean),
    savgol (SadTalker's Savitzky-Golay 13 / 2). The gains depend on ratios only, so no units are assumed."""
    x = np.asarray(x, np.float64)
    if name == "mean3":
        p = np.pad(x, ((1, 1), (0, 0)), mode="edge")
        return (p[:-2] + p[1:-1] + p[2:]) / 3
    if name == "savgol":
        from scipy.signal import savgol_filter
        return savgol_filter(x, 13, 2, axis=0) if len(x) >= 13 else x
    if name == "one_euro":
        a, y = 1.0 / (1.0 + fps / (2 * math.pi * 1.0)), x.copy()
        for t in range(1, len(x)):
            y[t] = a * x[t] + (1 - a) * y[t - 1]
        return y
    if name == "kalman":
        q, r = 1e-5, 3e-7
        mf, pf, pp = x.copy(), np.zeros(len(x)), np.zeros(len(x))
        pf[0] = r
        for t in range(1, len(x)):
            pp[t] = pf[t - 1] + q
            k = pp[t] / (pp[t] + r)
            mf[t] = mf[t - 1] + k * (x[t] - mf[t - 1])
            pf[t] = (1 - k) * pp[t]
        ms = mf.copy()
        for t in range(len(x) - 2, -1, -1):
            ms[t] = mf[t] + pf[t] / pp[t + 1] * (ms[t + 1] - mf[t])
        return ms
    raise ValueError(f"unknown hand-set smoother '{name}'; known: {', '.join(HANDSET)}")


def handset_motion(m: torch.Tensor, name: str) -> torch.Tensor:
    """A hand-set smoother on the shield's columns of a clip's motion [n, 70]; the mouth and the rest stay."""
    w = wide(m.float().cpu())
    w[:, COLS] = torch.as_tensor(handset(w[:, COLS].numpy(), name), dtype=w.dtype)
    return from_wide(w, m.float().cpu()).to(m)
