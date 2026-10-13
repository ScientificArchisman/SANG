"""C1 go/no-go: a Cohen-Massaro (1993) coarticulation prior for lip opening, and the test of whether it
knows something about the real lips that SANG does not (scripts/coartic_prior.py). Pure numpy.

Prior. Every phoneme i (CTC spike at frame t_i, token tok_i, sound class c_i from sang.error_split) pulls
the lips toward its target T(tok_i) with a dominance that decays with distance in time,
    D_i(t) = alpha[c_i] * exp(-theta[c_i] * |t - (t_i + lag)|)
and the predicted lip opening is the dominance-weighted average of the targets,
    p(t) = sum_i D_i(t) T(tok_i) / sum_i D_i(t).
Silence is a 'pause' pseudo-phoneme every `pause_step` frames inside each pause. Targets live in
per-clip z-scored lip opening; tokens seen fewer than `min_count` times share their class's target.

Fit on training clips: the lag with mean-at-frame targets; then alternately alpha / theta per class by
coordinate ascent on mean corr(p, real), and the targets by ridge least squares (p is linear in the
targets, so they have an exact solution given alpha / theta / lag; frame means would be blurred by the
neighbours twice).

Test on val clips. The question is whether the prior knows something SANG's MEAN prediction does not.
One SANG sample is the mean plus sampling noise, so any second estimate of the mouth (even another SANG
sample, or a prior that knows only SANG's mean) 'explains' part of the real lips by averaging that noise
out. So SANG's mean is estimated from K samples (seeds s .. s+K-1) and checked first:
    partial(B | A)        A, B = means of the first and second half of the samples: what another
                          half-batch still adds. Small -> the K-sample mean is pinned down.
    partial(prior | mean) corr of real and prior after removing the K-sample mean: the gate statistic
verdict(): INCONCLUSIVE if partial(B | A) >= max_baseline (rerun with more samples), else GO if
partial(prior | mean) >= gate, else NO-GO. Held-out blends z(base) + w z(other), w fitted on the other
fold's videos, give the same comparison in lip-corr units.
"""
import zlib

import numpy as np

from sang.diagnose import corr
from sang.error_split import sound_of

CLASSES = ("closure p/b/m", "labiodental f/v", "open vowel", "rounded vowel/w", "other vowel",
           "other consonant", "pause")
ALPHAS = (0.25, 0.5, 1.0, 2.0, 4.0)
THETAS = (0.15, 0.3, 0.5, 0.8, 1.2)                  # per frame: time constants 6.7 .. 0.8 frames


def zscore(x) -> np.ndarray:
    x = np.asarray(x, np.float64)
    return (x - x.mean()) / max(x.std(), 1e-8)


def events(spikes: list[tuple[int, str]], pause: np.ndarray, pause_step: int = 4) -> list[tuple[int, str, str]]:
    """Phoneme spikes plus pause pseudo-phonemes -> [(frame, token, class)]."""
    ev = [(f, t, sound_of(t)) for f, t in spikes]
    p = np.flatnonzero(np.asarray(pause, bool))
    ev += [(int(f), "<pause>", "pause") for f in p[::pause_step]]
    return sorted(ev)


def fit_targets(clips: list[dict], lag: int = 0, min_count: int = 20, shrink: float = 20.0) -> dict:
    """Mean z-scored real lip opening at each token's frame (+ lag) over training clips: the starting
    point for fit_targets_ls. clips: [{"lip": real lip [n], "events": events()}] -> {token or class:
    target}; a token's mean shrinks toward its class mean (weight count / (count + shrink)); tokens seen
    fewer than min_count times are left out and use their class's target."""
    tok, cls = {}, {}
    for c in clips:
        z = zscore(c["lip"])
        n = len(z)
        for f, t, k in c["events"]:
            j = f + lag
            if 0 <= j < n:
                tok.setdefault(t, []).append(z[j])
                cls.setdefault(k, []).append(z[j])
    out = {k: float(np.mean(v)) for k, v in cls.items()}
    for t, v in tok.items():
        k = "pause" if t == "<pause>" else sound_of(t)
        if len(v) >= min_count:
            w = len(v) / (len(v) + shrink)
            out[t] = float(w * np.mean(v) + (1 - w) * out[k])
    return out


def _weights(ev, n: int, alpha: dict, theta: dict, lag: int) -> np.ndarray:
    """[events, n] normalised dominance: column t holds each event's share of frame t."""
    f = np.array([e[0] for e in ev], np.float64) + lag
    a = np.array([alpha[e[2]] for e in ev])
    th = np.array([theta[e[2]] for e in ev])
    D = a[:, None] * np.exp(-th[:, None] * np.abs(np.arange(n)[None, :] - f[:, None]))
    return D / np.maximum(D.sum(0, keepdims=True), 1e-300)


def prior_trace(ev: list[tuple[int, str, str]], n: int, targets: dict, alpha: dict, theta: dict, lag: int = 0) -> np.ndarray:
    """Cohen-Massaro lip-opening prior [n] for one clip (module docstring); zeros without events."""
    if not ev:
        return np.zeros(n)
    T = np.array([targets.get(e[1], targets.get(e[2], 0.0)) for e in ev])
    return T @ _weights(ev, n, alpha, theta, lag)


def fit_targets_ls(clips: list[dict], start: dict, alpha: dict, theta: dict, lag: int, ridge: float = 10.0) -> dict:
    """Least-squares targets for the keys of `start` (tokens and classes), shrunk toward `start` with
    `ridge`: minimises sum over training frames of (z(real) - p)^2 + ridge * |T - start|^2."""
    keys = sorted(start)
    col = {k: i for i, k in enumerate(keys)}
    A, b = np.zeros((len(keys), len(keys))), np.zeros(len(keys))
    for c in clips:
        ev = c["events"]
        if not ev:
            continue
        z = zscore(c["lip"])
        W = _weights(ev, len(z), alpha, theta, lag)
        Xt = np.zeros((len(keys), len(z)))
        idx = [col.get(e[1], col.get(e[2])) for e in ev]
        ok = np.array([i is not None for i in idx])
        np.add.at(Xt, np.array([i for i in idx if i is not None], int), W[ok])
        A += Xt @ Xt.T
        b += Xt @ z
    T0 = np.array([start[k] for k in keys])
    T = np.linalg.solve(A + ridge * np.eye(len(keys)), b + ridge * T0)
    return {k: float(v) for k, v in zip(keys, T)}


def mean_corr(clips: list[dict], targets: dict, alpha: dict, theta: dict, lag: int) -> float:
    rs = [corr(prior_trace(c["events"], len(c["lip"]), targets, alpha, theta, lag), c["lip"]) for c in clips]
    return float(np.nanmean(rs))


def fit_prior(clips: list[dict], lags=range(-3, 4), passes: int = 2, log=print) -> dict:
    """Lag (frame-mean targets), then `passes` rounds of alpha / theta coordinate ascent and least-squares
    targets, each step kept only if it raises the mean training corr(prior, real lips)."""
    alpha = {k: 1.0 for k in CLASSES}
    theta = {k: 0.5 for k in CLASSES}
    best = None
    for lag in lags:
        tg = fit_targets(clips, lag)
        r = mean_corr(clips, tg, alpha, theta, lag)
        log(f"  lag {lag:+d}: train corr {r:.3f}")
        if best is None or r > best[0]:
            best = (r, lag, tg)
    r, lag, tg = best
    for p in range(passes):
        ls = fit_targets_ls(clips, tg, alpha, theta, lag)
        r_ls = mean_corr(clips, ls, alpha, theta, lag)
        if r_ls > r:
            tg, r = ls, r_ls
        for k in CLASSES:
            for grid, par in ((ALPHAS, alpha), (THETAS, theta)):
                keep, scores = par[k], []
                for v in grid:
                    par[k] = v
                    scores.append(mean_corr(clips, tg, alpha, theta, lag))
                i = int(np.nanargmax(scores))
                par[k] = grid[i] if scores[i] > r else keep
                r = max(r, scores[i])
        log(f"  pass {p + 1}: train corr {r:.3f} (least-squares targets {r_ls:.3f})")
    ls = fit_targets_ls(clips, tg, alpha, theta, lag)
    r_ls = mean_corr(clips, ls, alpha, theta, lag)
    if r_ls > r:
        tg, r = ls, r_ls
    return {"lag": lag, "targets": tg, "alpha": dict(alpha), "theta": dict(theta), "train_corr": r}


def residual(x, *on) -> np.ndarray:
    """x minus its least-squares fit a + sum_j b_j on_j."""
    x = np.asarray(x, np.float64)
    X = np.column_stack([np.ones_like(x)] + [np.asarray(o, np.float64) for o in on])
    return x - X @ np.linalg.lstsq(X, x, rcond=None)[0]


def partial_corr(real, other, *given) -> float:
    """corr(real, other | given): does `other` predict the part of the real lips that `given` misses?"""
    return corr(residual(real, *given), residual(other, *given))


def video_of(clip_key: str) -> str:
    return clip_key.split("/")[0]


def cluster_ci(x, clusters, z: float = 1.96) -> tuple[float, float]:
    """Mean of x and the half-width of its 95 % interval with cluster-robust standard error (clips of one
    video are not independent)."""
    x, clusters = np.asarray(x, np.float64), np.asarray(clusters)
    ok = np.isfinite(x)
    x, clusters = x[ok], clusters[ok]
    m = x.mean()
    s = sum(float(np.sum(x[clusters == c] - m)) ** 2 for c in np.unique(clusters))
    return float(m), float(z * np.sqrt(s) / len(x))


def blend_gain(clips: list[dict], base=("gen",), other: str = "prior", weights=np.linspace(0, 2, 41),
               folds: int = 2) -> dict:
    """Held-out blend sum z(base) + w z(other): w is chosen on the other folds (max mean corr with the real
    lips) and scored on this fold. Folds split by video when clips carry a "clip" key.
    clips: [{"lip", base..., other}] -> mean corr of the base alone and of the blend."""
    zs = [(zscore(c["lip"]), sum(zscore(c[k]) for k in base), zscore(c[other])) for c in clips]
    fold = np.array([zlib.crc32(video_of(c["clip"]).encode()) % folds if "clip" in c else i % folds
                     for i, c in enumerate(clips)])
    score = lambda idx, w: np.nanmean([corr(zs[i][1] + w * zs[i][2], zs[i][0]) for i in idx])
    idx = np.arange(len(zs))
    base_r, blend_r, ws = [], [], []
    for f in range(folds):
        test, train = idx[fold == f], idx[fold != f]
        if len(test) == 0 or len(train) == 0:
            continue
        w = weights[int(np.nanargmax([score(train, w) for w in weights]))]
        ws.append(round(float(w), 3))
        base_r += [corr(zs[i][1], zs[i][0]) for i in test]
        blend_r += [corr(zs[i][1] + w * zs[i][2], zs[i][0]) for i in test]
    if not base_r:
        return {}
    b, bl = np.array(base_r), np.array(blend_r)
    return {"base": float(np.nanmean(b)), "blend": float(np.nanmean(bl)), "gain": float(np.nanmean(bl - b)),
            "weights": ws, "better_clips_pct": float(100 * np.mean(bl > b))}


def verdict(partial_prior: float, baseline: float, gate: float = 0.1, max_baseline: float = 0.05) -> str:
    """GO / NO-GO on partial(prior | SANG mean), unless SANG's mean is not pinned down yet."""
    if not baseline < max_baseline:
        return "INCONCLUSIVE"
    return "GO" if partial_prior >= gate else "NO-GO"
