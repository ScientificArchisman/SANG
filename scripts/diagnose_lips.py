#!/usr/bin/env python3
"""Diagnose SANG's lip-sync gap (generated vs real lip-opening correlation ~0.64) on unseen
validation speakers, with no retraining. See sang/diagnose.py for what each test means.

    sbatch --time=06:00:00 bash_scripts/job.sh scripts/diagnose_lips.py --ckpt runs/motion_12k_anneal/best.pt

Tests (--tests; default all, run in this order; the JSON is rewritten after each one):
  lang     Whisper language ID per clip; at the end every per-clip metric is split English / other
  seeds    K samples per clip: r_sy (sample vs real), r_ss (sample vs sample), r_inf (the model's
           mean vs real), the mean-of-k curve, and a wrong-audio floor
  lag      cross-correlation of generated (mean of K) vs real lips over +/- --max-lag frames
  offset   the model's mouth flow loss on REAL windows with the audio shifted -S..+S frames
  ref      same seed, reference frame with the mouth closed / frame 0 / mouth open
  probe    ridge probes audio features -> real lip trajectory, per encoder x layer x context width
  syncnet  SyncNet AV offset + confidence on the REAL val clips (slow: ~0.5-1 min per clip)

Minutes for everything except syncnet; syncnet adds ~1 h per 100 clips (--syncnet-n).

Needs <cache>/openness.json (scripts/calibrate_openness.py). lang and the Whisper probe need
openai/whisper-large-v3 in the HF cache (bash_scripts/install_voice.sh downloads it). Other probe
encoders must be downloaded on the LOGIN node first, e.g.
    huggingface-cli download facebook/wav2vec2-xls-r-300m
    huggingface-cli download utter-project/mHuBERT-147
A missing model is skipped with a message, never fatal. syncnet needs third_party/syncnet_python.
"""
import argparse
import importlib.util
import json
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from sang.bench import SYNCNET
from sang.codec import WAVLM, load_wavlm
from sang.diagnose import (RidgeProbe, by_group, corr, lag_curve, lag_summary, lang_group, nanmean, offset_stats,
                           offset_summary, pca_basis, seed_stats, seeds_summary, shifted_mouth_losses, stack_taps,
                           taps, ticks_to_frames, verdict)
from sang.motion import REGIONS, to_target
from sang.motion_model import Norm, build, ema_weights, generate, load_ema, parse_spec, sampler_kwargs
from sang.naturalness import SR, load_readouts, load_wav
from sang.paths import motion_cache
from sang.sync import SyncOffsets, real_clip_mp4, shift_ticks, shift_wav

TESTS = ("lang", "seeds", "lag", "offset", "ref", "probe", "syncnet")


def load_train_module():
    spec = importlib.util.spec_from_file_location("tm", REPO / "scripts" / "train_motion.py")
    tm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tm)
    return tm



# ---------------------------------------------------------------------- audio models
class WhisperLID:
    """Spoken-language ID from Whisper's first decoder step (the language-token distribution)."""

    def __init__(self, name: str, dev: str):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        from transformers.models.whisper.tokenization_whisper import LANGUAGES
        self.p = WhisperProcessor.from_pretrained(name, local_files_only=True)
        self.m = WhisperForConditionalGeneration.from_pretrained(
            name, local_files_only=True, torch_dtype=torch.float16).to(dev).eval()
        tok, self.dev = self.p.tokenizer, dev
        ids = {c: tok.convert_tokens_to_ids(f"<|{c}|>") for c in LANGUAGES}
        ids = {c: i for c, i in ids.items() if i is not None and i != tok.unk_token_id}
        self.codes, self.ids = list(ids), torch.tensor(list(ids.values()), device=dev)
        self.sot = tok.convert_tokens_to_ids("<|startoftranscript|>")

    @torch.no_grad()
    def __call__(self, wav: torch.Tensor) -> tuple[str, float]:
        x = self.p(wav.numpy(), sampling_rate=SR, return_tensors="pt").input_features.to(self.dev, torch.float16)
        logits = self.m(input_features=x, decoder_input_ids=torch.tensor([[self.sot]], device=self.dev)).logits[0, -1]
        p = logits.float()[self.ids].softmax(-1)
        j = int(p.argmax())
        return self.codes[j], float(p[j])


class AudioEncoder:
    """Hidden states of one audio encoder at chosen relative depths.

    'wavlm-large'/'wavlm-base' are called exactly as scripts/cache_motion.py does (raw waveform),
    so the last layer IS SANG's input. Whisper names use the encoder of the seq2seq model; anything
    else goes through AutoFeatureExtractor + AutoModel (wav2vec2 / HuBERT / XLS-R / mHuBERT)."""

    def __init__(self, name: str, dev: str, depths):
        self.name, self.dev = name, dev
        if "whisper" in name:
            from transformers import WhisperFeatureExtractor, WhisperModel
            self.fe = WhisperFeatureExtractor.from_pretrained(name, local_files_only=True)
            self.m = WhisperModel.from_pretrained(name, local_files_only=True,
                                                  torch_dtype=torch.float16).encoder.to(dev).eval()
            L, self.kind = self.m.config.encoder_layers, "whisper"
        elif name in WAVLM:
            self.m = load_wavlm(name, dev).m
            L, self.kind = self.m.config.num_hidden_layers, "raw"
        else:
            from transformers import AutoFeatureExtractor, AutoModel
            self.fe = AutoFeatureExtractor.from_pretrained(name, local_files_only=True)
            self.m = AutoModel.from_pretrained(name, local_files_only=True).to(dev).eval()
            L, self.kind = self.m.config.num_hidden_layers, "ssl"
        self.n_layers = L
        self.layers = sorted({min(L, max(1, round(d * L))) for d in depths})

    @torch.no_grad()
    def __call__(self, wav: torch.Tensor) -> dict[int, torch.Tensor]:
        if self.kind == "whisper":
            x = self.fe(wav.numpy(), sampling_rate=SR, return_tensors="pt").input_features.to(self.dev, torch.float16)
        elif self.kind == "raw":
            x = wav[None].to(self.dev)
        else:
            x = self.fe(wav.numpy(), sampling_rate=SR, return_tensors="pt").input_values.to(self.dev)
        hs = self.m(x, output_hidden_states=True).hidden_states
        return {l: hs[l][0].float() for l in self.layers}


# ---------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tests", nargs="+", default=list(TESTS), choices=TESTS)
    ap.add_argument("--n", type=int, default=300, help="val clips (same selection as naturalness.py)")
    ap.add_argument("--min-frames", type=int, default=125)
    ap.add_argument("--max-frames", type=int, default=250)
    ap.add_argument("--variant", default="g=2,mouth=1.25",
                    help="sampler spec as in naturalness.py --variants (e.g. 'g=2,mouth=1.25,avg=4'); '' = the run's defaults")
    ap.add_argument("--ema", default=None, help="extra EMA decay saved by training with ema_extra (e.g. 0.999); default = the main EMA")
    ap.add_argument("--guide-ckpt", default=None, help="autoguidance guide checkpoint (for ag / ag_mouth in --variant)")
    ap.add_argument("--steps", type=int, default=None, help="Euler steps (default: the run's sample_steps)")
    ap.add_argument("--k", type=int, default=8, help="samples per clip for the seeds test")
    ap.add_argument("--max-lag", type=int, default=6)
    ap.add_argument("--shifts", type=int, default=5, help="offset test: audio shifted -S..+S frames")
    ap.add_argument("--offset-noise", type=int, default=4, help="noise draws per t in the offset test")
    ap.add_argument("--whisper", default="openai/whisper-large-v3")
    ap.add_argument("--lang-file", default=None,
                    help="optional JSON {clip stem or clip path: language code}; overrides Whisper LID")
    ap.add_argument("--probe-encoders", nargs="+", default=["wavlm-large", "openai/whisper-large-v3"])
    ap.add_argument("--probe-depths", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--probe-windows", type=int, nargs="+", default=[2, 6, 12])
    ap.add_argument("--probe-train", type=int, default=600, help="train clips the probes are fitted on")
    ap.add_argument("--probe-dim", type=int, default=256, help="PCA dims per frame before stacking context")
    ap.add_argument("--syncnet-n", type=int, default=100, help="val clips checked with SyncNet")
    ap.add_argument("--sync-offsets", default=None,
                    help="<cache>/sync_offsets.json: drop no-sync clips and move each clip's audio by its SyncNet offset")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=str(REPO / "results/diagnose"))
    args = ap.parse_args()

    dev = args.device
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    cfg = ck["cfg"]
    model = build(cfg).to(dev).eval()
    model.load_state_dict(ema_weights(ck, args.ema))
    norm = Norm(**ck["norm"]).to(dev)
    cache = motion_cache(cfg["cache_dir"])
    read = load_readouts(cache / "openness.json")
    lip_read = read["lip"]
    samp = sampler_kwargs(parse_spec(args.variant), cfg, load_ema(args.guide_ckpt, dev, norm) if args.guide_ckpt else None,
                          steps=args.steps)
    steps = samp["steps"]
    P, L = cfg["prefix"], cfg["frames"]
    mouth = REGIONS["mouth"]

    tm = load_train_module()
    train_rows, val_rows = tm.split_by_speaker(tm.load_index(cache), cfg["val_frac"], cfg["seed"])
    so = SyncOffsets.load(args.sync_offsets) if args.sync_offsets else None
    if so is not None:
        train_rows, why_t = so.filter(train_rows)
        val_rows, why_v = so.filter(val_rows)
        print(f"sync offsets {args.sync_offsets}: dropped val {why_v}, train {why_t}", flush=True)
    val_rows = [r for r in sorted(val_rows, key=lambda r: r["clip"]) if r["n"] >= args.min_frames][: args.n]
    want_wav = bool({"lang", "probe"} & set(args.tests))

    def load_clip(r: dict) -> dict:
        d = torch.load(r["path"], map_location="cpu", weights_only=True)
        n = min(args.max_frames, int(d["n"]))
        y = to_target(d["m"][:n].float())
        s = so.shift(r) if so is not None else 0                     # WavLM ticks; 0 = uncorrected
        c = {"row": r, "n": n, "y": y, "kp": d["kp"][:n].float(), "audio": shift_ticks(d["audio"][: 2 * n].float(), s),
             "lip": lip_read(y).numpy(), "speaker": Path(r["path"]).parent.name, "shift": s}
        if want_wav:
            c["wav"] = shift_wav(load_wav(r["clip"], r.get("start", 0), n), s)
        return c

    t0 = time.time()
    clips = []
    for r in val_rows:
        try:
            clips.append(load_clip(r))
        except Exception as e:
            print(f"  skip {Path(r['clip']).name}: {type(e).__name__}: {e}", flush=True)
    per = [{"clip": Path(c["row"]["clip"]).stem, "speaker": c["speaker"], "n": c["n"], "sync_shift": c["shift"]}
           for c in clips]
    print(f"{len(clips)} val clips, {len({c['speaker'] for c in clips})} unseen speakers, loaded in "
          f"{time.time() - t0:.0f} s; guidance '{args.variant}', {steps} steps", flush=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (f"{Path(args.ckpt).parent.name}_step{ck['step']}{'_sync' if so is not None else ''}"
                          f"{f'_ema{args.ema}' if args.ema else ''}.json")
    R = {"ckpt": args.ckpt, "step": ck["step"], "variant": args.variant, "steps": steps, "clips": len(clips),
         "speakers": len({c["speaker"] for c in clips}), "tests": args.tests, "sync_offsets": args.sync_offsets}

    def save():
        out_path.write_text(json.dumps({"results": R, "per_clip": per}, indent=1, default=float))

    def ref_at(c: dict, k: int) -> torch.Tensor:
        return norm.ref(c["kp"][k:k + 1].to(dev), c["y"][k:k + 1].to(dev))

    @torch.no_grad()
    def sample_lips(c: dict, ref: torch.Tensor, k: int, seed: int, audio: torch.Tensor | None = None) -> np.ndarray:
        a = (c["audio"] if audio is None else audio)[None].to(dev).expand(k, -1, -1).contiguous()
        g = torch.Generator(device=dev).manual_seed(seed)
        y = generate(model, a, ref.expand(k, -1).contiguous(), c["n"], window=L, n_prefix=P, generator=g, **samp)
        return lip_read(norm.untarget(y)).cpu().numpy()               # [k, n]

    samples = {}

    # ------------------------------------------------------------------ lang
    if "lang" in args.tests:
        t0 = time.time()
        table = json.loads(Path(args.lang_file).read_text()) if args.lang_file else {}
        lid = None
        if len(table) < len(clips):
            try:
                lid = WhisperLID(args.whisper, dev)
            except Exception as e:
                print(f"[lang] Whisper '{args.whisper}' unavailable ({type(e).__name__}: {e}); "
                      "download it on the login node or pass --lang-file", flush=True)
        for c, rec in zip(clips, per):
            code = table.get(rec["clip"]) or table.get(c["row"]["clip"])
            if code is None and lid is not None:
                code, p = lid(c["wav"])
                rec["lang_p"] = round(p, 3)
                code = code if p >= 0.5 else "unk"
            rec["lang"] = code or "unk"
            rec["lang_group"] = lang_group(rec["lang"])
        del lid
        torch.cuda.empty_cache()
        counts = {}
        for rec in per:
            counts[rec["lang"]] = counts.get(rec["lang"], 0) + 1
        R["lang_counts"] = dict(sorted(counts.items(), key=lambda kv: -kv[1]))
        print(f"[lang] {R['lang_counts']}  ({time.time() - t0:.0f} s)", flush=True)
        save()

    # ------------------------------------------------------------------ seeds
    if "seeds" in args.tests:
        t0 = time.time()
        stats = []
        for i, (c, rec) in enumerate(zip(clips, per)):
            S = sample_lips(c, ref_at(c, 0), args.k, args.seed)
            samples[i] = S
            st = seed_stats(S, c["lip"])
            j = next((j for j in list(range(i + 1, len(clips))) + list(range(i))
                      if clips[j]["speaker"] != c["speaker"] and clips[j]["n"] >= c["n"]), None)
            if j is not None:
                w = sample_lips(c, ref_at(c, 0), 1, args.seed, audio=clips[j]["audio"][: 2 * c["n"]])[0]
                st["r_wrong_audio"] = corr(w, c["lip"])
            stats.append(st)
            rec.update({k: st[k] for k in ("r_sy", "r_ss", "std_r", f"k{args.k}_corr") if k in st})
            rec["r_wrong_audio"] = st.get("r_wrong_audio")
        R["seeds"] = seeds_summary(stats)
        s = R["seeds"]
        print(f"[seeds] K={args.k}: r_sy {s['r_sy']:.3f}  r_ss {s['r_ss']:.3f}  r_inf {s.get('r_inf', float('nan')):.3f}  "
              f"wrong-audio floor {s.get('r_wrong_audio', float('nan')):.3f}  ({time.time() - t0:.0f} s)", flush=True)
        for k in (1, 2, 4, 8, 16, 32):
            if f"k{k}_corr" in s:
                print(f"        mean of {k:>2}: corr {s[f'k{k}_corr']:.3f} (expected {s.get(f'k{k}_expected', s['r_sy']):.3f})  "
                      f"ccc {s[f'k{k}_ccc']:.3f}  std_r {s[f'k{k}_std_r']:.3f}", flush=True)
        save()

    # ------------------------------------------------------------------ lag
    if "lag" in args.tests:
        t0 = time.time()
        single, mean = [], []
        for i, (c, rec) in enumerate(zip(clips, per)):
            S = samples.get(i)
            if S is None:
                S = sample_lips(c, ref_at(c, 0), 1, args.seed)
            single.append(lag_curve(S[0], c["lip"], args.max_lag))
            mean.append(lag_curve(S.mean(0), c["lip"], args.max_lag))
            if np.isfinite(mean[-1]).all():
                rec["lag_best"] = int(np.argmax(mean[-1])) - args.max_lag
        g = lag_summary(mean, args.max_lag)
        R["lag_single"] = lag_summary(single, args.max_lag)
        if g:
            R["lag"] = {**g, "source": f"mean of {args.k if samples else 1} samples per clip"}
            print(f"[lag] mean curve peak {g['peak_lag']:+d}; corr lag0 {g['corr_lag0']:.3f} -> best lag "
                  f"{g['corr_best_lag']:.3f}; |lag|>=2 in {g['frac_abs_lag_ge2']:.0%} of clips; hist {g['peak_hist']}  "
                  f"({time.time() - t0:.0f} s)", flush=True)
            print("      " + "  ".join(f"{l:+d}:{v:.3f}" for l, v in zip(g["lags"], g["mean_curve"])), flush=True)
        else:
            print("[lag] no clip gave a finite curve", flush=True)
        save()

    # ------------------------------------------------------------------ offset
    if "offset" in args.tests:
        t0 = time.time()
        shifts = list(range(-args.shifts, args.shifts + 1))
        stats, benefit = [], []
        for i, (c, rec) in enumerate(zip(clips, per)):
            g = torch.Generator(device=dev).manual_seed(args.seed + i)
            res = shifted_mouth_losses(model, norm.target(c["y"].to(dev)), c["audio"].to(dev), ref_at(c, 0), shifts,
                                       mouth, P, L, n_noise=args.offset_noise, generator=g)
            if res is None:
                continue
            losses, null = res
            st = offset_stats(losses, shifts)
            st["audio_benefit"] = null / max(losses[shifts.index(0)], 1e-12) - 1
            stats.append(st)
            benefit.append(st["audio_benefit"])
            rec.update(offset_best=st["best_shift"], offset_sharp1=st.get("sharpness_1"))
        if stats:
            R["offset"] = {**offset_summary(stats, shifts), "audio_benefit": nanmean(benefit),
                           "convention": "shift d > 0 delays the audio by d frames"}
            o = R["offset"]
            print(f"[offset] loss-minimising audio shift: median {o['median_best_shift']:+.1f}, |shift|>=2 in "
                  f"{o['frac_abs_best_ge2']:.0%} of clips, hist {o['best_hist']}; mouth loss +{o['sharpness_1']:.1%} at "
                  f"+/-1, +{o['sharpness_2']:.1%} at +/-2; dropping audio +{o['audio_benefit']:.1%}  "
                  f"({time.time() - t0:.0f} s)", flush=True)
            print("         " + "  ".join(f"{d:+d}:{v:.4f}" for d, v in zip(o["shifts"], o["mean_rel_curve"])), flush=True)
        else:
            print(f"[offset] no clip has {P + L} frames", flush=True)
        save()

    # ------------------------------------------------------------------ ref
    if "ref" in args.tests:
        t0 = time.time()
        r_c, r_0, r_o, shift = [], [], [], []
        for c, rec in zip(clips, per):
            lip = c["lip"]
            k_open = int(np.argmin(np.abs(lip - np.percentile(lip, 95))))
            k_closed = int(np.argmin(np.abs(lip - np.percentile(lip, 5))))
            # one seed for all three, so the reference frame is the only difference
            gc, g0, go = (sample_lips(c, ref_at(c, k), 1, args.seed)[0] for k in (k_closed, 0, k_open))
            r_c.append(corr(gc, lip))
            r_0.append(corr(g0, lip))
            r_o.append(corr(go, lip))
            shift.append(float((go.mean() - gc.mean()) / max(lip.std(), 1e-8)))
            rec.update(ref_shift=shift[-1], ref_r_closed=r_c[-1], ref_r_open=r_o[-1])
        R["ref"] = {"r_closed": nanmean(r_c), "r_default": nanmean(r_0), "r_open": nanmean(r_o),
                    "mean_shift_open_minus_closed": nanmean(shift)}
        f = R["ref"]
        print(f"[ref] corr closed-ref {f['r_closed']:.3f} / frame-0 {f['r_default']:.3f} / open-ref {f['r_open']:.3f}; "
              f"mean opening open-minus-closed {f['mean_shift_open_minus_closed']:+.2f} real-std  ({time.time() - t0:.0f} s)",
              flush=True)
        save()

    # ------------------------------------------------------------------ probe
    if "probe" in args.tests:
        t0 = time.time()
        pool = [r for r in train_rows if r["n"] >= args.min_frames]
        pick = random.Random(args.seed).sample(pool, min(args.probe_train, len(pool)))
        train_clips = []
        for r in pick:
            try:
                d = torch.load(r["path"], map_location="cpu", weights_only=True)
                n = min(args.max_frames, int(d["n"]))
                y = to_target(d["m"][:n].float())
                wav = shift_wav(load_wav(r["clip"], r.get("start", 0), n), so.shift(r) if so is not None else 0)
                train_clips.append({"n": n, "lip": lip_read(y).numpy(), "wav": wav})
            except Exception as e:
                print(f"  [probe] skip {Path(r['clip']).name}: {type(e).__name__}", flush=True)
        n_fit = int(0.8 * len(train_clips))
        print(f"[probe] {len(train_clips)} train clips (fit {n_fit}, choose ridge strength on the rest), "
              f"{len(clips)} val clips; loaded in {time.time() - t0:.0f} s", flush=True)
        rows = []
        for name in args.probe_encoders:
            t1 = time.time()
            try:
                enc = AudioEncoder(name, dev, args.probe_depths)
            except Exception as e:
                print(f"[probe] skip encoder '{name}': {type(e).__name__}: {e}", flush=True)
                continue
            feats = {l: [] for l in enc.layers}
            for c in train_clips + clips:
                hs = enc(c["wav"])
                for l in enc.layers:
                    feats[l].append(ticks_to_frames(hs[l], c["n"]).half().cpu())
            n_layers = enc.n_layers
            del enc
            torch.cuda.empty_cache()
            print(f"[probe] {name}: layers {list(feats)} of {n_layers}, features in {time.time() - t1:.0f} s", flush=True)
            for l, F in feats.items():
                tr, va = F[: len(train_clips)], F[len(train_clips):]
                centred = torch.cat([x.float() - x.float().mean(0) for x in tr]).to(dev)
                _, V = pca_basis(centred, args.probe_dim, args.seed)
                del centred
                proj = lambda x: (x.to(dev).float() - x.to(dev).float().mean(0)) @ V
                Ptr, Pva = [proj(x) for x in tr], [proj(x) for x in va]
                for w in args.probe_windows:
                    offs = taps(w)
                    Xtr = [stack_taps(x, offs) for x in Ptr]
                    probe = RidgeProbe(Xtr[0].shape[1], dev)
                    for x, c in zip(Xtr[:n_fit], train_clips[:n_fit]):
                        probe.add(x, torch.as_tensor(c["lip"]))
                    best_lam, best_hold = None, -2.0
                    for lam in (1e-3, 1e-2, 1e-1, 1.0):
                        wts = probe.solve(lam)
                        hold = nanmean([corr(probe.predict(x, wts), c["lip"])
                                        for x, c in zip(Xtr[n_fit:], train_clips[n_fit:])])
                        if np.isfinite(hold) and hold > best_hold:
                            best_lam, best_hold = lam, hold
                    for x, c in zip(Xtr[n_fit:], train_clips[n_fit:]):
                        probe.add(x, torch.as_tensor(c["lip"]))
                    wts = probe.solve(best_lam or 1e-2)
                    key = f"probe|{name}|{l}|{w}"
                    vc = []
                    for x, c, rec in zip(Pva, clips, per):
                        rec[key] = corr(probe.predict(stack_taps(x, offs), wts), c["lip"])
                        vc.append(rec[key])
                    rows.append({"encoder": name, "layer": l, "n_layers": n_layers, "window": w, "taps": len(offs),
                                 "lam": best_lam, "hold_corr": best_hold, "val_corr": nanmean(vc), "key": key})
                    print(f"   {name:<28} layer {l:>2}/{n_layers}  +/-{w:<2} frames  val lip corr {rows[-1]['val_corr']:.3f}",
                          flush=True)
                    del Xtr, probe
                del Ptr, Pva
            del feats
        base = next((r for r in rows if r["encoder"] == "wavlm-large" and r["layer"] == r["n_layers"] and r["window"] == 2), None)
        R["probe"] = {"rows": rows, "baseline": base, "train_clips": len(train_clips)}
        print(f"[probe] done in {time.time() - t0:.0f} s", flush=True)
        save()

    # ------------------------------------------------------------------ syncnet
    if "syncnet" in args.tests and not (SYNCNET / "run_syncnet.py").exists():
        print(f"[syncnet] skipped: no syncnet_python at {SYNCNET} (bash_scripts/install_motion.sh)", flush=True)
    elif "syncnet" in args.tests:
        from sang.bench import lse
        t0 = time.time()
        tmp = out_dir / "syncnet_tmp"
        offs, confs = [], []
        todo = list(zip(clips, per))[: args.syncnet_n]
        print(f"[syncnet] {len(todo)} real clips, ~0.5-1 min each"
              + (" (measures the RAW clip: --sync-offsets does not apply here)" if so is not None else ""), flush=True)
        for j, (c, rec) in enumerate(todo, 1):
            mp4 = tmp / f"{rec['clip']}.mp4"
            try:
                real_clip_mp4(c["row"], c["n"], mp4)
                off, dist, conf = lse(mp4, tmp / f"{rec['clip']}_work")
            except Exception as e:
                print(f"  [syncnet] {rec['clip']}: {type(e).__name__}: {e}", flush=True)
                off, dist, conf = float("nan"), float("nan"), float("nan")
            rec.update(sn_offset=off, sn_dist=dist, sn_conf=conf)
            print(f"  [syncnet] {j}/{len(todo)} {rec['clip']:<40} offset {off:+.0f}  conf {conf:.2f}  "
                  f"model's shift {rec.get('offset_best', float('nan')):+.0f}  ({time.time() - t0:.0f} s)", flush=True)
            if np.isfinite(off):
                offs.append(off)
                confs.append(conf)
            shutil.rmtree(tmp / f"{rec['clip']}_work", ignore_errors=True)
            mp4.unlink(missing_ok=True)
        shutil.rmtree(tmp, ignore_errors=True)
        ok = [rec for rec in per if np.isfinite(rec.get("sn_offset", float("nan")))]
        o = np.array(offs)
        R["syncnet"] = {"clips": len(ok)}
        if len(ok):
            R["syncnet"].update(
                hist={int(k): int((o == k).sum()) for k in sorted(set(o.astype(int).tolist()))},
                frac_abs_off_ge1=float((np.abs(o) >= 1).mean()), frac_abs_off_ge2=float((np.abs(o) >= 2).mean()),
                frac_conf_lt3=float((np.array(confs) < 3).mean()), conf_median=float(np.median(confs)),
                r_sy_aligned=nanmean([rec.get("r_sy") for rec in ok if abs(rec["sn_offset"]) <= 1]),
                r_sy_offset=nanmean([rec.get("r_sy") for rec in ok if abs(rec["sn_offset"]) >= 2]),
                agreement_with_offset_test=corr([rec["sn_offset"] for rec in ok if "offset_best" in rec],
                                                [rec["offset_best"] for rec in ok if "offset_best" in rec]))
        sn = R["syncnet"]
        print(f"[syncnet] {sn['clips']} real clips: offsets {sn.get('hist')}, |off|>=2 {sn.get('frac_abs_off_ge2', float('nan')):.0%}, "
              f"conf<3 {sn.get('frac_conf_lt3', float('nan')):.0%}; r_sy aligned {sn.get('r_sy_aligned', float('nan')):.3f} "
              f"vs offset {sn.get('r_sy_offset', float('nan')):.3f}  ({time.time() - t0:.0f} s)", flush=True)
        save()

    # ------------------------------------------------------------------ language split + verdict
    if "lang" in args.tests:
        keys = ["r_sy", "r_ss", f"k{args.k}_corr", "std_r", "lag_best", "offset_best", "offset_sharp1", "ref_shift",
                "sn_offset"] + [r["key"] for r in R.get("probe", {}).get("rows", [])]
        R["lang"] = {k: by_group(per, k) for k in keys if any(k in rec for rec in per)}
    lines = verdict(R)
    R["verdict"] = lines
    save()
    print("\n================ DIAGNOSIS ================", flush=True)
    for line in lines:
        print(line, flush=True)
    if "lang" in R:
        for k in ("r_sy", f"k{args.k}_corr"):
            if k in R["lang"]:
                print(f"  {k} by language: " + ", ".join(f"{g} {v['mean']:.3f} (n={v['n']})" for g, v in R["lang"][k].items()))
        if R.get("probe", {}).get("rows"):
            b = R["probe"].get("baseline")
            top = max(R["probe"]["rows"], key=lambda r: r["val_corr"] if np.isfinite(r["val_corr"]) else -1)
            for row, label in ((b, "SANG input"), (top, "best probe")):
                if row and row["key"] in R["lang"]:
                    print(f"  {label} ({row['encoder']} L{row['layer']} +/-{row['window']}) by language: "
                          + ", ".join(f"{g} {v['mean']:.3f} (n={v['n']})" for g, v in R["lang"][row["key"]].items()))
    print(f"\nwrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
