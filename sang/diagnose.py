"""Why does generated lip opening correlate with the real trajectory at only ~0.64?

Each hypothesis has a test that needs no retraining; scripts/diagnose_lips.py runs them on unseen
validation speakers. Everything here is pure numpy/torch so it can be unit-tested on CPU.

  seeds    randomness vs bias. With samples s_i = m + e_i (e_i independent of each other and of
           the real trajectory y):
               r_sy   = corr(s_i, y)              one sample against the real lips
               r_ss   = corr(s_i, s_j)            two samples against each other
               r_inf  = r_sy / sqrt(r_ss)         the model's MEAN prediction against the real lips
           corr(mean of k samples, y) = r_sy / sqrt(r_ss + (1 - r_ss) / k), which rises to r_inf.
           r_inf - r_sy is what randomness costs; 1 - r_inf is bias plus whatever the audio
           cannot determine. A calibrated sampler (its spread equals the real take-to-take
           spread) has r_sy ~ r_ss; r_sy > r_ss means too much mouth noise, r_sy < r_ss means
           the samples agree with each other more than with reality (bias).
  lag      timing. Cross-correlation of generated vs real lips over +/-L frames. A peak away from
           lag 0 is a learned shift; peaks scattered across clips are inconsistent timing.
  offset   data alignment. The model's own mouth flow loss with the audio shifted by d frames.
           The best d per clip estimates that clip's audio-video offset as the model sees it.
  ref      shortcut. Same seed, reference frame with the mouth open vs closed.
  lang     encoder language. The above, split English / other (WavLM-Large is English-only).
  probe    audio features. Ridge regression from audio features (layer x context width x
           encoder) to the real lip trajectory: how much lip timing the features carry at all.
"""
import math
from itertools import combinations

import numpy as np
import torch


# ---------------------------------------------------------------------- basic statistics
def corr(a, b) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if len(a) < 3 or a.std() < 1e-8 or b.std() < 1e-8:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def ccc(a, b) -> float:
    """Lin's concordance: correlation AND amplitude AND offset (1 = identical)."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    cov = float(np.mean((a - a.mean()) * (b - b.mean())))
    return 2 * cov / (a.var() + b.var() + (a.mean() - b.mean()) ** 2 + 1e-12)


def nanmean(xs) -> float:
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.mean(xs)) if xs else float("nan")


# ---------------------------------------------------------------------- seeds: randomness vs bias
def k_curve_expected(r_sy: float, r_ss: float, k: int) -> float:
    """corr(mean of k samples, real) implied by r_sy and r_ss (independent sample noise)."""
    if not (r_ss > 0):
        return float("nan")
    return r_sy / math.sqrt(r_ss + (1.0 - r_ss) / k)


def seed_stats(samples, real, ks=(1, 2, 4, 8, 16, 32)) -> dict:
    """samples [K, n] lip openness from K seeds, real [n] -> per-clip statistics."""
    S, y = np.asarray(samples, np.float64), np.asarray(real, np.float64)
    K = len(S)
    out = {"r_sy": nanmean([corr(s, y) for s in S]),
           "r_ss": nanmean([corr(S[i], S[j]) for i, j in combinations(range(K), 2)]) if K > 1 else float("nan"),
           "std_r": float(S.std(1).mean() / max(y.std(), 1e-8))}
    for k in ks:
        if k <= K:
            m = S[:k].mean(0)
            out[f"k{k}_corr"] = corr(m, y)
            out[f"k{k}_ccc"] = ccc(m, y)
            out[f"k{k}_std_r"] = float(m.std() / max(y.std(), 1e-8))
    return out


def seeds_summary(per_clip: list[dict]) -> dict:
    """Average the per-clip statistics, then derive r_inf and the expected k-curve."""
    keys = sorted({k for d in per_clip for k in d})
    s = {k: nanmean([d.get(k) for d in per_clip]) for k in keys}
    if s.get("r_ss", float("nan")) > 0:
        s["r_inf"] = min(1.0, s["r_sy"] / math.sqrt(s["r_ss"]))
        for k in (2, 4, 8, 16, 32):
            if f"k{k}_corr" in s:
                s[f"k{k}_expected"] = k_curve_expected(s["r_sy"], s["r_ss"], k)
    return s


# ---------------------------------------------------------------------- lag
def lag_curve(gen, real, max_lag: int = 6) -> np.ndarray:
    """c[l + max_lag] = corr(gen[t + l], real[t]). l > 0: the generated lips LAG the real ones
    by l frames (they move later); l < 0: they lead."""
    g, y = np.asarray(gen, np.float64), np.asarray(real, np.float64)
    n, out = len(g), []
    for l in range(-max_lag, max_lag + 1):
        out.append(corr(g[l:], y[:n - l]) if l >= 0 else corr(g[:n + l], y[-l:]))
    return np.array(out)


def lag_summary(curves: list[np.ndarray], max_lag: int) -> dict:
    """Per-clip curves -> mean curve, its peak, the per-clip peak histogram, lag-tolerant corr."""
    C = np.array([c for c in curves if np.isfinite(c).all()])
    if len(C) == 0:
        return {}
    lags = np.arange(-max_lag, max_lag + 1)
    mean = C.mean(0)
    best = lags[C.argmax(1)]
    hist = {int(l): int((best == l).sum()) for l in lags if (best == l).any()}
    return {"lags": lags.tolist(), "mean_curve": mean.round(4).tolist(),
            "peak_lag": int(lags[mean.argmax()]), "corr_lag0": float(mean[max_lag]),
            "corr_best_lag": float(C.max(1).mean()),
            "frac_abs_lag_ge2": float((np.abs(best) >= 2).mean()),
            "peak_hist": hist, "clips": int(len(C))}


# ---------------------------------------------------------------------- offset: audio shifted
def shift_frames(audio: torch.Tensor, delta: int, tpf: int = 2) -> torch.Tensor:
    """audio ticks [T, D] -> same length with out[tick] = audio[tick - delta * tpf], edge-padded.
    delta > 0 delays the audio by delta video frames."""
    T = audio.shape[0]
    idx = (torch.arange(T, device=audio.device) - delta * tpf).clamp(0, T - 1)
    return audio[idx]


@torch.no_grad()
def shifted_mouth_losses(model, y: torch.Tensor, audio: torch.Tensor, ref: torch.Tensor, shifts, mouth,
                         P: int, L: int, ts=(0.2, 0.35, 0.5, 0.65, 0.8), n_noise: int = 4,
                         generator: torch.Generator | None = None) -> tuple[list[float], float] | None:
    """The model's mouth flow loss on one REAL clip with its audio shifted by each d in `shifts`,
    plus the loss with audio dropped. Same (t, noise) draws for every shift (common random numbers),
    so differences between shifts are the audio's doing.

    y [n, 42] normalised targets, audio [2n, D] ticks, ref [1, REF_DIM]. Windows of L frames start at
    P, P + L, ... and use the dropped-prefix layout (null prefix tokens and audio), as in training.
    -> (losses per shift, loss without audio), or None if the clip is shorter than P + L."""
    n, dev, tpf = y.shape[0], y.device, model.tpf
    starts = list(range(P, n - L + 1, L))
    if not starts:
        return None
    W, T = len(starts), len(ts) * n_noise
    x0 = torch.stack([y[s:s + L] for s in starts])[:, None].expand(W, T, L, y.shape[1]).reshape(W * T, L, -1)
    eps = torch.randn(x0.shape, device=dev, generator=generator)
    t = torch.as_tensor(ts, dtype=torch.float32, device=dev).repeat(n_noise)[None].expand(W, T).reshape(-1)
    tb = t.view(-1, 1, 1)
    x_t = (1 - tb) * x0 + tb * eps
    target = (eps - x0)[..., mouth]
    refs = ref.expand(W * T, -1)
    prefix = torch.zeros(W * T, P, y.shape[1], device=dev)
    keep = torch.zeros(W * T, dtype=torch.bool, device=dev)

    def loss(a: torch.Tensor, drop: bool = False) -> float:
        aw = torch.stack([a[tpf * (s - P): tpf * (s + L)] for s in starts])[:, None]
        aw = aw.expand(W, T, *aw.shape[2:]).reshape(W * T, *aw.shape[2:])
        v = model(x_t, t, aw, refs, prefix=prefix, prefix_keep=keep,
                  drop_audio=torch.full((W * T,), drop, dtype=torch.bool, device=dev))
        return float((v[..., mouth] - target).pow(2).mean())

    return [loss(shift_frames(audio, d, tpf)) for d in shifts], loss(audio, drop=True)


def offset_stats(losses, shifts) -> dict:
    """Mouth flow loss per audio shift -> the best shift and how sharply the loss rises around 0."""
    L, d = np.asarray(losses, np.float64), list(shifts)
    i0 = d.index(0)
    rel = L / max(L[i0], 1e-12)
    out = {"best_shift": int(d[int(L.argmin())]), "rel_curve": rel.round(5).tolist()}
    if 1 in d and -1 in d:
        out["sharpness_1"] = float((rel[d.index(1)] + rel[d.index(-1)]) / 2 - 1)
    if 2 in d and -2 in d:
        out["sharpness_2"] = float((rel[d.index(2)] + rel[d.index(-2)]) / 2 - 1)
    return out


def offset_summary(per_clip: list[dict], shifts) -> dict:
    if not per_clip:
        return {}
    best = np.array([d["best_shift"] for d in per_clip])
    curve = np.array([d["rel_curve"] for d in per_clip]).mean(0)
    hist = {int(s): int((best == s).sum()) for s in shifts if (best == s).any()}
    return {"shifts": list(shifts), "mean_rel_curve": curve.round(5).tolist(),
            "curve_min_shift": int(list(shifts)[int(curve.argmin())]),
            "median_best_shift": float(np.median(best)),
            "frac_abs_best_ge2": float((np.abs(best) >= 2).mean()),
            "sharpness_1": nanmean([d.get("sharpness_1") for d in per_clip]),
            "sharpness_2": nanmean([d.get("sharpness_2") for d in per_clip]),
            "best_hist": hist, "clips": len(per_clip)}


# ---------------------------------------------------------------------- probe: ridge from audio features
def taps(w: int) -> list[int]:
    """Frame offsets for a +/-w context: dense within +/-3, then sparser out to +/-w."""
    dense = set(range(-min(w, 3), min(w, 3) + 1))
    far = {s * int(round(w * f)) for f in (0.5, 0.75, 1.0) for s in (-1, 1)} if w > 3 else set()
    return sorted(dense | far)


def stack_taps(X: torch.Tensor, offs: list[int]) -> torch.Tensor:
    """[n, d] -> [n, d * len(offs)]; row t holds X[t + o] for each offset (edge-padded)."""
    n = X.shape[0]
    base = torch.arange(n, device=X.device)
    return torch.cat([X[(base + o).clamp(0, n - 1)] for o in offs], 1)


def ticks_to_frames(h: torch.Tensor, n: int, tpf: int = 2) -> torch.Tensor:
    """[T, D] 50 Hz ticks -> [n, D] 25 fps frames (mean of each frame's ticks; zero-padded tail)."""
    need = n * tpf
    if h.shape[0] < need:
        h = torch.nn.functional.pad(h, (0, 0, 0, need - h.shape[0]))
    return h[:need].reshape(n, tpf, -1).mean(1)


class RidgeProbe:
    """Ridge regression accumulated clip by clip. Features and target are centred PER CLIP, so the
    probe learns within-clip dynamics -- what per-clip lip correlation measures -- and not
    identity offsets."""

    def __init__(self, dim: int, device="cpu"):
        self.xtx = torch.zeros(dim, dim, dtype=torch.float64, device=device)
        self.xty = torch.zeros(dim, dtype=torch.float64, device=device)

    def add(self, X: torch.Tensor, y: torch.Tensor) -> None:
        X = X.to(self.xtx) - X.to(self.xtx).mean(0)
        y = y.to(self.xty) - y.to(self.xty).mean()
        self.xtx += X.T @ X
        self.xty += X.T @ y

    def solve(self, lam_rel: float) -> torch.Tensor:
        d = self.xtx.shape[0]
        lam = lam_rel * float(torch.diagonal(self.xtx).mean())
        return torch.linalg.solve(self.xtx + lam * torch.eye(d, dtype=self.xtx.dtype, device=self.xtx.device), self.xty)

    @staticmethod
    def predict(X: torch.Tensor, w: torch.Tensor) -> np.ndarray:
        X = X.to(w) - X.to(w).mean(0)
        return (X @ w).cpu().numpy()


def pca_basis(frames: torch.Tensor, k: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """[N, D] -> (mean [D], components [D, k]) from a random subset of at most 50k rows."""
    g = torch.Generator().manual_seed(seed)
    N = frames.shape[0]
    pick = torch.randperm(N, generator=g)[: min(N, 50_000)]
    X = frames[pick.to(frames.device)].float()
    mu = X.mean(0)
    k = min(k, X.shape[1], X.shape[0])
    _, _, V = torch.pca_lowrank(X - mu, q=k, center=False)
    return mu, V[:, :k]


# ---------------------------------------------------------------------- grouping
def lang_group(code: str | None) -> str:
    if not code or code == "unk":
        return "unk"
    return "en" if code.lower().startswith("en") else "other"


def by_group(records: list[dict], key: str, group_key: str = "lang_group") -> dict:
    out = {}
    for g in sorted({r.get(group_key, "unk") for r in records}):
        vals = [r.get(key) for r in records if r.get(group_key, "unk") == g]
        out[g] = {"mean": nanmean(vals), "n": int(sum(v is not None and np.isfinite(v) for v in vals))}
    return out


# ---------------------------------------------------------------------- verdict
def verdict(R: dict) -> list[str]:
    """Aggregated results (whatever tests ran) -> ranked plain-English findings.

    Thresholds are deliberately round numbers; the printed values let a reader disagree."""
    lines = []

    s = R.get("seeds")
    if s and np.isfinite(s.get("r_inf", float("nan"))):
        gain = s["r_inf"] - s["r_sy"]
        tag = "MAJOR" if gain >= 0.08 else ("minor" if gain >= 0.03 else "not a factor")
        lines.append(f"[randomness: {tag}] one sample r_sy={s['r_sy']:.3f}, two samples r_ss={s['r_ss']:.3f}, "
                     f"model mean r_inf={s['r_inf']:.3f}: removing mouth randomness is worth +{gain:.3f}. "
                     + ("-> deterministic mouth anchor + residual flow, or average / lower mouth temperature."
                        if gain >= 0.08 else "-> the gap is mostly not sampling noise."))
        d = s["r_sy"] - s["r_ss"]
        if d > 0.05:
            lines.append(f"[dispersion] r_sy exceeds r_ss by {d:.3f}: the samples are noisier than real "
                         "take-to-take variation (over-dispersed mouth).")
        elif d < -0.05:
            lines.append(f"[bias] samples agree with each other (r_ss={s['r_ss']:.3f}) more than with the real "
                         f"lips (r_sy={s['r_sy']:.3f}): a consistent error, i.e. bias -> audio features, data, supervision.")
        else:
            lines.append(f"[dispersion] r_sy ~ r_ss (diff {d:+.3f}): the sampler's spread is about right.")
        lines.append(f"[ceiling for this model] even its mean prediction reaches only {s['r_inf']:.3f}; "
                     f"1 - r_inf = {1 - s['r_inf']:.3f} is bias plus what audio cannot determine (see probe).")

    lag = R.get("lag")
    if lag:
        tol = lag["corr_best_lag"] - lag["corr_lag0"]
        if lag["peak_lag"] != 0:
            lines.append(f"[timing: SHIFT] the mean cross-correlation peaks at lag {lag['peak_lag']:+d} frames "
                         f"(generated lips {'late' if lag['peak_lag'] > 0 else 'early'}): a learned constant offset.")
        lines.append(f"[timing] {lag['frac_abs_lag_ge2']:.0%} of clips peak at |lag| >= 2 frames; allowing the best lag "
                     f"per clip raises corr {lag['corr_lag0']:.3f} -> {lag['corr_best_lag']:.3f} (+{tol:.3f})."
                     + (" Per-clip timing errors are a real share of the gap." if tol >= 0.05 else ""))

    off = R.get("offset")
    if off:
        s1, s2, ab = off.get("sharpness_1", float("nan")), off.get("sharpness_2", float("nan")), off.get("audio_benefit", float("nan"))
        if np.isfinite(ab) and ab < 0.02:
            lines.append(f"[audio use: NEGLIGIBLE] dropping the audio changes the model's mouth loss only {ab:+.1%}.")
        elif np.isfinite(ab) and np.isfinite(s2):
            ratio = s2 / ab
            lines.append(f"[timing precision: {'LOW' if ratio < 0.2 else 'ok'}] dropping the audio raises the model's mouth "
                         f"loss {ab:+.1%}; shifting it 1 / 2 frames raises it {s1:+.1%} / {s2:+.1%}, i.e. an 80 ms error "
                         f"costs {ratio:.0%} of losing the audio."
                         + (" The model uses audio coarsely (blurred timing) -> sync-trained audio stream, lip-expert "
                            "loss, offset-corrected data." if ratio < 0.2 else ""))
        if not (s2 >= 0.005):
            lines.append(f"[data offsets (model's view): inconclusive] the loss changes only {s2:+.2%} for a 2-frame shift, so "
                         "per-clip best shifts are noise; rely on the SyncNet test.")
        else:
            tag = "COMMON" if off["frac_abs_best_ge2"] > 0.15 else "rare"
            lines.append(f"[data offsets (model's view): {tag}] the model fits {off['frac_abs_best_ge2']:.0%} of clips best "
                         f"with the audio shifted >= 2 frames (median shift {off['median_best_shift']:+.1f}).")
            if off["curve_min_shift"] != 0:
                lines.append(f"[data offsets] averaged over clips the loss is lowest at shift {off['curve_min_shift']:+d}: "
                             "the model's audio-motion alignment is itself shifted.")

    sn = R.get("syncnet")
    if sn and sn.get("clips"):
        tag = "COMMON" if sn["frac_abs_off_ge2"] > 0.15 else "rare"
        lines.append(f"[data offsets (SyncNet on real clips): {tag}] {sn['frac_abs_off_ge2']:.0%} of real clips are off by "
                     f">= 2 frames, {sn['frac_conf_lt3']:.0%} have confidence < 3. Lip corr on aligned clips "
                     f"{sn.get('r_sy_aligned', float('nan')):.3f} vs offset clips {sn.get('r_sy_offset', float('nan')):.3f}.")
        if sn["frac_abs_off_ge2"] > 0.15:
            lines.append("  -> correct per-clip offsets and filter low-confidence clips (report section 3a); "
                         "offset VAL clips also depress the measured corr itself.")

    ref = R.get("ref")
    if ref:
        tag = "YES" if abs(ref["mean_shift_open_minus_closed"]) >= 0.3 else "no"
        lines.append(f"[reference shortcut: {tag}] an open- vs closed-mouth reference moves the generated mean opening by "
                     f"{ref['mean_shift_open_minus_closed']:+.2f} real-std; corr {ref['r_closed']:.3f} (closed ref) / "
                     f"{ref['r_default']:.3f} (frame 0) / {ref['r_open']:.3f} (open ref).")

    lg = R.get("lang")
    if lg:
        en, ot = lg.get("r_sy", {}).get("en", {}), lg.get("r_sy", {}).get("other", {})
        if en.get("n", 0) >= 15 and ot.get("n", 0) >= 15:
            d = en["mean"] - ot["mean"]
            tag = "YES" if d >= 0.05 else "no"
            lines.append(f"[language gap: {tag}] lip corr English {en['mean']:.3f} (n={en['n']}) vs other "
                         f"{ot['mean']:.3f} (n={ot['n']}): {d:+.3f}."
                         + (" -> multilingual encoder + language embedding (report 5a)." if d >= 0.05 else ""))
        else:
            lines.append(f"[language] too few clips per group to judge (en n={en.get('n', 0)}, other n={ot.get('n', 0)}).")

    pr = R.get("probe")
    if pr and pr.get("rows"):
        rows = pr["rows"]
        base = pr.get("baseline")
        best = max(rows, key=lambda r: r["val_corr"] if np.isfinite(r["val_corr"]) else -1)
        lines.append(f"[audio features] best linear probe: {best['encoder']} layer {best['layer']} +/-{best['window']} frames "
                     f"-> lip corr {best['val_corr']:.3f}" + (f"; SANG's input (last WavLM layer, +/-2) {base['val_corr']:.3f}." if base else "."))
        if base and best["val_corr"] - base["val_corr"] >= 0.05:
            lines.append(f"  -> better features/context exist (+{best['val_corr'] - base['val_corr']:.3f} for a linear map): "
                         "change the audio front end (report 5a/5c).")
        if s and base and np.isfinite(s.get("r_inf", float("nan"))):
            if base["val_corr"] >= s["r_inf"] - 0.02:
                lines.append(f"  -> a LINEAR map on SANG's own input ({base['val_corr']:.3f}) matches or beats the model's mean "
                             f"({s['r_inf']:.3f}): the generator under-uses its audio (objective/supervision problem, "
                             "report sections 4 and 6), not only the features.")
    return lines
