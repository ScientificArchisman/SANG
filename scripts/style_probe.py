#!/usr/bin/env python3
"""Can a clip's lip style be PREDICTED one-shot? (queue #22; follows the oracle of research log 3.19)

The speaker-style oracle (scripts/naturalness.py --lip-oracle) moves SANG's lip opening to the REAL
clip's median / 5-95 % range: val CCC 0.717 -> 0.768 / 0.778. Here the style comes instead from a
ridge regression fitted on TRAINING speakers, using only what exists at inference: the photo (frame
0's target, keypoints and openness), the whole clip's audio (mean WavLM, loudness, onset rate) and
SANG's own sample. It predicts the residual real - generated (lo, mid, hi) of the lip opening.

    sbatch bash_scripts/job.sh scripts/style_probe.py --ckpt runs/d1_97h/last.pt --sync-offsets $S_OLD \\
        --sampler $SP --lang all

Rows (val lip CCC after a level / range restyle; the same 300 val clips and samples as naturalness.py):
    model     SANG as is (d1: 0.717)
    const     the mean training residual: one global level correction, no per-clip information
    self      ridge on SANG's own sample (lo, mid, hi, std): a per-clip recalibration, no new input
    photo     self + the photo
    audio     self + the clip's audio
    all       self + photo + audio
    oracle    the real clip's own style (d1: 0.768 / 0.778)
Gates: all >= const + 0.015 -> one-shot style is predictable from photo and audio (build the
predictor); const or self >= model + 0.015 -> a free recalibration, no new model needed.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

FEATURES = {"const": (), "self": ("self",), "photo": ("self", "photo"), "audio": ("self", "audio"),
            "all": ("self", "photo", "audio")}
STYLE = ("lo", "mid", "hi")


# ---------------------------------------------------------------------- the probe (numpy only)
def ridge_fit(X: np.ndarray, Y: np.ndarray, lam: float):
    """Ridge on standardised X with an unpenalised intercept -> predict(X') function."""
    mu, sd = X.mean(0), X.std(0) + 1e-8
    Z = (X - mu) / sd
    beta = np.linalg.solve(Z.T @ Z + lam * len(Z) * np.eye(Z.shape[1]), Z.T @ (Y - Y.mean(0)))
    y0 = Y.mean(0)
    return lambda Xn: ((Xn - mu) / sd) @ beta + y0


def choose_lambda(X: np.ndarray, Y: np.ndarray, groups: np.ndarray, lams=tuple(10.0 ** np.arange(-3, 3.5, 0.5)),
                  folds: int = 5) -> float:
    """Ridge strength by speaker-grouped cross-validation (mean squared error of the residual)."""
    g = np.unique(groups)
    fold = {s: i % folds for i, s in enumerate(np.random.default_rng(0).permutation(g))}
    f = np.array([fold[s] for s in groups])
    err = []
    for lam in lams:
        e = 0.0
        for k in range(folds):
            tr, te = f != k, f == k
            e += float(((ridge_fit(X[tr], Y[tr], lam)(X[te]) - Y[te]) ** 2).sum())
        err.append(e)
    return float(lams[int(np.argmin(err))])


def design(clips: list[dict], parts) -> np.ndarray:
    return np.stack([np.concatenate([c[p] for p in parts]) if parts else np.zeros(0) for c in clips])


def probe(train: list[dict], val: list[dict], q: float = 5.0) -> dict:
    """train / val: per-clip dicts with feature vectors 'self', 'photo', 'audio', 'gen' and 'real' (lip
    styles), 'speaker'; val also 'gen_lip', 'real_lip'. -> {row: {'level', 'range' (mean val CCC),
    'r2_mid', 'lam'}}."""
    from sang.naturalness import ccc, lip_restyled
    res = lambda cs: np.array([[c["real"][k] - c["gen"][k] for k in STYLE] for c in cs])
    Ytr, Yva = res(train), res(val)
    groups = np.array([c["speaker"] for c in train])

    def score(pred_res, extra=None):
        out = {}
        for how in ("level", "range"):
            cc = []
            for c, d in zip(val, pred_res):
                st = {k: c["gen"][k] + d[i] for i, k in enumerate(STYLE)}
                st["hi"] = max(st["hi"], st["lo"] + 1e-6)
                cc.append(ccc(lip_restyled(c["gen_lip"], st, how, q), c["real_lip"]))
            out[how] = float(np.mean(cc))
        mid = STYLE.index("mid")
        ss = ((Yva[:, mid] - Yva[:, mid].mean()) ** 2).sum()
        out["r2_mid"] = float(1 - ((pred_res[:, mid] - Yva[:, mid]) ** 2).sum() / max(ss, 1e-12))
        return {**out, **(extra or {})}

    rows = {"model": {h: float(np.mean([ccc(c["gen_lip"], c["real_lip"]) for c in val])) for h in ("level", "range")}}
    for name, parts in FEATURES.items():
        if not parts:
            rows[name] = score(np.repeat(Ytr.mean(0, keepdims=True), len(val), 0))
            continue
        Xtr, Xva = design(train, parts), design(val, parts)
        lam = choose_lambda(Xtr, Ytr, groups)
        rows[name] = score(ridge_fit(Xtr, Ytr, lam)(Xva), {"lam": lam, "dims": int(Xtr.shape[1])})
    rows["oracle"] = score(Yva)
    return rows


# ---------------------------------------------------------------------- data (cluster)
def collect(rows: list[dict], model, norm, cfg, read, samp, so, max_frames: int, seed: int, dev: str) -> list[dict]:
    """One SANG sample per clip (mode none, the given seed: the same samples as naturalness.py) and the
    one-shot features."""
    from sang.motion import to_target
    from sang.naturalness import audio_onsets, energy_db, guided_generate, lip_style, load_wav, pauses
    from sang.sync import shift_ticks, shift_wav
    out = []
    for r in rows:
        try:
            d = torch.load(r["path"], map_location="cpu", weights_only=True)
            n = min(max_frames, int(d["n"]))
            s = so.shift(r) if so is not None else 0
            wav = shift_wav(load_wav(r["clip"], r.get("start", 0), n), s)
            audio = shift_ticks(d["audio"][: 2 * n].float(), s)
            y_real = to_target(d["m"][:n].float())
            kp0 = d["kp"][0].float()
            ref = norm.ref(kp0[None].to(dev), y_real[:1].to(dev))
            y, _ = guided_generate(model, audio[None].to(dev), ref, n, cfg, norm, wav, None, "none", seed, sampler=samp)
            gen_lip = read["lip"](norm.untarget(y[0]).cpu()).numpy()
            real_lip = read["lip"](y_real).numpy()
            db = energy_db(wav, n)
            talk = ~pauses(db)
            gen, real = lip_style(gen_lip), lip_style(real_lip)
            out.append({
                "speaker": Path(r["path"]).parent.name, "gen": gen, "real": real,
                "gen_lip": gen_lip, "real_lip": real_lip,
                "self": np.array([gen["lo"], gen["mid"], gen["hi"], gen_lip.std()]),
                "photo": np.concatenate([y_real[0].numpy(), kp0.numpy(),
                                         [float(read["lip"](y_real[:1])), float(read["eye"](y_real[:1]))]]),
                "audio": np.concatenate([audio.mean(0).numpy(), audio.std(0).numpy(),
                                         [db[talk].mean() if talk.any() else db.mean(), db.std(), 1 - talk.mean(),
                                          len(audio_onsets(db)) * 25.0 / n]]),
            })
        except Exception as e:
            print(f"  skip {Path(r['clip']).name}: {type(e).__name__}: {e}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sampler", default="g=2,mouth=1.25,tau=0.5,avg=4,steps=6")
    ap.add_argument("--n", type=int, default=300, help="val clips (the naturalness.py selection)")
    ap.add_argument("--n-train", type=int, default=2000, help="training clips to fit the probe on")
    ap.add_argument("--max-frames", type=int, default=250)
    ap.add_argument("--min-frames", type=int, default=125)
    ap.add_argument("--sync-offsets", default=None)
    ap.add_argument("--lang", nargs="+", default=None)
    ap.add_argument("--q", type=float, default=5.0, help="robust range percentiles q / 100-q")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="")
    ap.add_argument("--out", default=str(REPO / "results/style_probe"))
    args = ap.parse_args()

    from sang.motion_model import Norm, build, ema_weights, parse_spec, sampler_kwargs
    from sang.naturalness import load_readouts
    from sang.paths import motion_cache
    from sang.sync import SyncOffsets
    from sang.talkvid import clip_languages, filter_langs, parse_langs
    import importlib.util

    dev = "cuda"
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    cfg = ck["cfg"]
    model = build(cfg).to(dev).eval()
    model.load_state_dict(ema_weights(ck, None))
    norm = Norm(**ck["norm"]).to(dev)
    samp = sampler_kwargs(parse_spec(args.sampler), cfg)
    cache = motion_cache(cfg["cache_dir"])
    read = load_readouts(cache / "openness.json")

    spec = importlib.util.spec_from_file_location("tm", REPO / "scripts" / "train_motion.py")
    tm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tm)
    train, val = tm.split_by_speaker(tm.load_index(cache), cfg["val_frac"], cfg["seed"])
    lang_of = clip_languages(cache)
    langs = parse_langs(args.lang if args.lang is not None else cfg.get("lang"))
    if langs is not None:
        val, _ = filter_langs(val, langs, lang_of)
        train, _ = filter_langs(train, langs, lang_of)
    so = SyncOffsets.load(args.sync_offsets) if args.sync_offsets else None
    if so is not None:
        val, _ = so.filter(val)
        train, _ = so.filter(train)
    val = [r for r in sorted(val, key=lambda r: r["clip"]) if r["n"] >= args.min_frames][: args.n]
    train = [r for r in train if r["n"] >= args.min_frames]
    pick = np.random.default_rng(args.seed).permutation(len(train))[: args.n_train]
    train = [train[i] for i in sorted(pick)]
    print(f"{len(train)} training clips ({len({Path(r['path']).parent.name for r in train})} speakers), "
          f"{len(val)} val clips", flush=True)

    t0 = time.time()
    tr = collect(train, model, norm, cfg, read, samp, so, args.max_frames, args.seed, dev)
    va = collect(val, model, norm, cfg, read, samp, so, args.max_frames, args.seed, dev)
    print(f"samples and features: {len(tr)} train, {len(va)} val, {time.time() - t0:.0f} s", flush=True)
    rows = probe(tr, va, args.q)

    print(f"\n{'row':<8}{'CCC level':>11}{'CCC range':>11}{'R2 mid':>9}{'lambda':>9}{'dims':>7}")
    for k, r in rows.items():
        print(f"{k:<8}{r['level']:>11.3f}{r['range']:>11.3f}{r.get('r2_mid', float('nan')):>9.3f}"
              f"{r.get('lam', float('nan')):>9.3g}{r.get('dims', 0):>7d}")
    best = max(rows[k]["range"] for k in ("const", "self"))
    print(f"\ngate 1, one-shot style predictable: all {rows['all']['range']:.3f} vs const {rows['const']['range']:.3f} "
          f"(+0.015 needed) -> {'PASS' if rows['all']['range'] >= rows['const']['range'] + 0.015 else 'fail'}")
    print(f"gate 2, free recalibration: max(const, self) {best:.3f} vs model {rows['model']['range']:.3f} "
          f"(+0.015 needed) -> {'PASS' if best >= rows['model']['range'] + 0.015 else 'fail'}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    name = f"{Path(args.ckpt).parent.name}_step{ck['step']}{'_' + args.name if args.name else ''}.json"
    (out / name).write_text(json.dumps({"ckpt": args.ckpt, "sampler": args.sampler, "n_train": len(tr),
                                        "n_val": len(va), "rows": rows}, indent=1))
    print(f"\nwrote {out / name}", flush=True)


if __name__ == "__main__":
    main()
