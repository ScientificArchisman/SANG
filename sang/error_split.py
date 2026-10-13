"""C0: where does the remaining lip-sync error come from? (scripts/error_split.py)

Inputs per val clip: generated and real lip-opening traces g, y (one value per video frame, from the
openness readout) and the CTC phoneme posteriors of the clip's audio. Pure numpy, unit-tested on CPU.

  parts    exact split of the per-clip squared error in units of the real variance:
               mean((g - y)^2) / var(y) = bias^2 + (s - 1)^2 + 2 s (1 - r)
           bias = (mean g - mean y) / std y, s = std g / std y, r = corr(g, y). The last term is
           split again with r* = the correlation at the clip's best constant shift (|lag| <= L):
               timing = 2 s (r* - r)      removable by shifting the whole clip
               shape  = 2 s (1 - r*)      wrong movements, not a constant delay
           Averaged per clip, the parts are dominated by clips whose real lips barely move (small std y
           inflates bias^2 and (s - 1)^2), so the summary also pools them: sum_i var(y_i) part_i /
           sum_i var(y_i), the split of the total squared error in lip-opening units. In correlation
           units the gap 1 - r is timing (r* - r) plus shape (1 - r*).
  sound    every frame is labelled by the nearest phoneme spike (CTC argmax) within `reach` frames,
           or 'pause' when the audio is silent. Per class: share of frames, share of squared
           error (lip-opening units, so no clip dominates) and the signed error in units of the pooled
           real std (> 0 = lips too open, e.g. a missed p/b/m
           closure; < 0 = not open enough, e.g. a flat open vowel).
  speed    lip speed |dg| vs |dy|, compared quantile by quantile (each trace's own median and 90th
           percentile, no frame selection): median ratio >> 1 with 90th-pct ratio <= 1 means a busy
           mouth that still misses the fastest real movements (coarticulation smoothing).
           speed_r_fast, the generated speed on the real lips' fastest 20 % of frames, is biased low for
           any imperfect model (frames are selected on the real trace): compare models, not against 1.
  rate     phonemes per second of speech, from the same spikes.
"""
import numpy as np

from sang.diagnose import corr, lag_curve
from sang.naturalness import BILABIAL, FPS, _runs

SOUNDS = ("closure p/b/m", "labiodental f/v", "open vowel", "rounded vowel/w", "other vowel",
          "other consonant", "pause", "unlabelled")
PARTS = ("bias", "amplitude", "timing", "shape")
LABIODENTAL = set("fvɱ")
OPEN_VOWEL = set("aɑæɐʌɒɛɜ")
ROUNDED = set("uʊoɔɵwyʏøœ")                     # IPA y is the rounded vowel; the glide is j
OTHER_VOWEL = set("iɪeəɨɯɤɚɝ")


def sound_of(tok: str) -> str:
    c = tok[0]
    if c in BILABIAL:
        return "closure p/b/m"
    if c in LABIODENTAL:
        return "labiodental f/v"
    if c in OPEN_VOWEL:
        return "open vowel"
    if c in ROUNDED:
        return "rounded vowel/w"
    if c in OTHER_VOWEL:
        return "other vowel"
    return "other consonant"


def is_phone(tok: str) -> bool:
    return bool(tok) and not tok.startswith(("<", "[", "|"))


def phone_spikes(post: np.ndarray, tokens: dict, n: int) -> list[tuple[int, str]]:
    """CTC posteriors [ticks, V] -> [(video frame, token)], one per run of ticks with the same phone
    argmax. Tick k covers ~[20k, 20k + 25] ms, as in naturalness.bilabial_events."""
    ids = np.asarray(post).argmax(1)
    out = []
    for i in np.unique(ids):
        tok = tokens.get(int(i), "")
        if not is_phone(tok):
            continue
        for s, e in _runs(ids == i):
            f = int(((s + e - 1) / 2 * 0.02 + 0.0125) * FPS)
            if f < n:
                out.append((f, tok))
    return sorted(out)


def frame_sounds(spikes: list[tuple[int, str]], pause: np.ndarray, n: int, reach: int = 3) -> np.ndarray:
    """Per-frame sound class: 'pause' where the audio is silent, else the class of the nearest phoneme
    spike within `reach` frames (120 ms), else 'unlabelled'."""
    lab = np.array(["unlabelled"] * n, dtype=object)
    if spikes:
        f = np.array([s for s, _ in spikes])
        cls = np.array([sound_of(t) for _, t in spikes], dtype=object)
        for t in range(n):
            j = int(np.abs(f - t).argmin())
            if abs(int(f[j]) - t) <= reach:
                lab[t] = cls[j]
    lab[np.asarray(pause[:n], bool)] = "pause"
    return lab


def speech_rate(spikes: list, pause: np.ndarray, n: int) -> float:
    """Phonemes per second of non-silent audio."""
    speech = n - int(np.asarray(pause[:n], bool).sum())
    return len(spikes) / (speech / FPS) if speech > 0 else float("nan")


def split_clip(g, y, max_lag: int = 4, fast_q: float = 0.8) -> dict:
    """Per-clip error split (see the module docstring). Lag > 0: generated lips move later than real."""
    g, y = np.asarray(g, np.float64), np.asarray(y, np.float64)
    sy, sg = y.std(), g.std()
    if sy < 1e-8 or sg < 1e-8:
        return {}
    r = corr(g, y)
    curve = lag_curve(g, y, max_lag)
    k = int(np.nanargmax(curve))
    r_best = max(float(curve[k]), r)
    bias, s = (g.mean() - y.mean()) / sy, sg / sy
    parts = {"bias": bias ** 2, "amplitude": (s - 1) ** 2, "timing": 2 * s * (r_best - r), "shape": 2 * s * (1 - r_best)}
    dy, dg = np.abs(np.diff(y)), np.abs(np.diff(g))
    fast = dy >= np.quantile(dy, fast_q)
    return {"r": r, "r_best": r_best, "lag": k - max_lag, "bias": bias, "std_r": s, "slope": r * s, "sy": float(sy),
            "ccc": 2 * float(np.mean((g - g.mean()) * (y - y.mean()))) / (g.var() + y.var() + (g.mean() - y.mean()) ** 2),
            "mse_rel": float(np.mean((g - y) ** 2) / sy ** 2), "parts": parts,
            "speed_r": float(dg.mean() / max(dy.mean(), 1e-12)),
            "speed_r_fast": float(dg[fast].mean() / max(dy[fast].mean(), 1e-12)),
            "speed_q50_r": float(np.quantile(dg, 0.5) / max(np.quantile(dy, 0.5), 1e-12)),
            "speed_q90_r": float(np.quantile(dg, 0.9) / max(np.quantile(dy, 0.9), 1e-12))}


def frame_errors(g, y, sounds: np.ndarray) -> dict:
    """Per sound class: [frames, sum of squared error, sum of signed error], in lip-opening units."""
    e = np.asarray(g, np.float64) - np.asarray(y, np.float64)
    return {c: [int((sounds == c).sum()), float((e[sounds == c] ** 2).sum()), float(e[sounds == c].sum())]
            for c in SOUNDS if (sounds == c).any()}


def pool_sounds(per_clip: list[dict], scale: float = 1.0) -> dict:
    """Sum frame_errors over clips -> per class: % of frames, % of squared error, their ratio (> 1 = this
    sound is harder than average) and the mean signed error / scale (the pooled real std)."""
    tot = {c: [0, 0.0, 0.0] for c in SOUNDS}
    for fe in per_clip:
        for c, (k, se, e) in fe.items():
            tot[c][0] += k
            tot[c][1] += se
            tot[c][2] += e
    nf, ne = sum(v[0] for v in tot.values()), sum(v[1] for v in tot.values())
    out = {}
    for c, (k, se, e) in tot.items():
        if k:
            pf, pe = k / nf, se / ne
            out[c] = {"frames_pct": 100 * pf, "error_pct": 100 * pe, "ratio": pe / pf, "signed": e / k / scale}
    return out


def summarise(recs: list[dict]) -> dict:
    """Clip records (split_clip + rate) -> means, the four parts as % of the total error, lag histogram."""
    recs = [r for r in recs if r]
    if not recs:
        return {"clips": 0}
    m = lambda k: float(np.nanmean([r[k] for r in recs]))
    tot = {p: float(np.mean([r["parts"][p] for r in recs])) for p in PARTS}
    s = sum(tot.values())
    w = np.array([r["sy"] ** 2 for r in recs])
    pool = {p: float(np.sum(w * np.array([r["parts"][p] for r in recs]))) for p in PARTS}
    sp = sum(pool.values())
    sr, ab = np.array([r["std_r"] for r in recs]), np.abs([r["bias"] for r in recs])
    lags = np.array([r["lag"] for r in recs])
    return {"clips": len(recs), "r": m("r"), "r_best": m("r_best"), "ccc": m("ccc"), "std_r": m("std_r"),
            "slope": m("slope"), "bias": m("bias"), "mse_rel": m("mse_rel"),
            "speed_r": m("speed_r"), "speed_r_fast": m("speed_r_fast"),
            "speed_q50_r": float(np.median([r["speed_q50_r"] for r in recs])),
            "speed_q90_r": float(np.median([r["speed_q90_r"] for r in recs])),
            "parts": tot, "parts_pct": {p: 100 * v / s for p, v in tot.items()},
            "pooled_pct": {p: 100 * v / sp for p, v in pool.items()},
            "gap": {"total": 1 - m("r"), "timing": m("r_best") - m("r"), "shape": 1 - m("r_best")},
            "std_r_median": float(np.median(sr)), "std_r_low_pct": float(100 * (sr < 0.8).mean()),
            "std_r_high_pct": float(100 * (sr > 1.25).mean()), "abs_bias_median": float(np.median(ab)),
            "abs_bias_gt1_pct": float(100 * (ab > 1).mean()), "sy_mean": float(np.sqrt(w.mean())),
            "abs_lag_ge2_pct": float(100 * (np.abs(lags) >= 2).mean()),
            "lag_hist": {int(l): int((lags == l).sum()) for l in np.unique(lags)},
            "rate": m("rate") if "rate" in recs[0] else float("nan")}
