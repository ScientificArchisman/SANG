#!/usr/bin/env python3
"""Does the cloned voice get closer to the person as we give it more of their audio?

    sbatch bash_scripts/job.sh scripts/eval_voice.py --targets 20
    sbatch bash_scripts/job.sh scripts/eval_voice.py --targets 20 --asr openai/whisper-large-v3

On unseen TalkVid validation speakers (the training split's own hash rule). For each target speaker T:
  - one of T's clips is held out as the reference of what T sounds like;
  - T's other clips go through the SAME enrolment as scripts/enroll_voice.py (VAD, chunks, speaker
    filter), and banks are built from the first 5 / 10 / 30 / 60 s / all of that speech;
  - clips of OTHER val speakers S are converted into each bank.

Reported per bank size (mean over pairs; speaker-verifier cosines with WavLM-base-plus-sv):
  sim_target   converted vs T's held-out clip        higher = more like T
  sim_source   converted vs the source speaker S     lower  = less of S left (leakage)
  env_corr     loudness-envelope correlation, converted vs source, 25 fps: ~1 = timing kept
  cer          (with --asr) Whisper transcript of converted vs of source: intelligibility
and two fixed rows: 'real T vs T' (ceiling: T's other clip vs the held-out clip) and 'source vs T'
(floor: the unconverted source). TalkVid speakers have ~1-2 min each, so this measures the
seconds-to-a-minute range; use your own long recordings for the minutes range.
"""
import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from sang.paths import MOTION_CACHE
from sang.voice import (FRAME_S, SR, SpeakerEncoder, auto_k, cer, chunks, fit_length, knn_features,
                        load_knnvc, load_wav, match_level, save_wav, speech_segments)


def envelope(w: torch.Tensor, hop: int = SR // 25) -> np.ndarray:
    n = w.numel() // hop
    return (10 * torch.log10(w[: n * hop].view(n, hop).pow(2).mean(1) + 1e-10)).numpy()


class Asr:
    """Whisper via transformers, language auto-detected (TalkVid is multilingual, no transcripts)."""

    def __init__(self, name: str, device: str):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        self.p = WhisperProcessor.from_pretrained(name, local_files_only=True)
        self.m = WhisperForConditionalGeneration.from_pretrained(
            name, local_files_only=True, torch_dtype=torch.float16).to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, wav: torch.Tensor) -> str:
        x = self.p(wav.numpy(), sampling_rate=SR, return_tensors="pt").input_features.to(self.device, torch.float16)
        return self.p.batch_decode(self.m.generate(x, task="transcribe"), skip_special_tokens=True)[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(MOTION_CACHE))
    ap.add_argument("--targets", type=int, default=20, help="target speakers")
    ap.add_argument("--sources", type=int, default=3, help="source clips converted per target")
    ap.add_argument("--budgets", type=float, nargs="+", default=[5, 10, 30, 60, 1e9], help="bank seconds; 1e9 = all")
    ap.add_argument("--src-seconds", type=float, default=8.0)
    ap.add_argument("--k", type=int, default=None, help="fixed k; default auto_k(bank seconds)")
    ap.add_argument("--min-cos", type=float, default=0.75)
    ap.add_argument("--asr", default=None, help="e.g. openai/whisper-large-v3 (download on the login node first)")
    ap.add_argument("--examples", type=int, default=4, help="pairs to save as wav")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(REPO / "results/voice_eval"))
    args = ap.parse_args()

    import importlib.util
    spec = importlib.util.spec_from_file_location("tm", REPO / "scripts" / "train_motion.py")
    tm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tm)
    _, val = tm.split_by_speaker(tm.load_index(Path(args.cache)), args.val_frac, args.seed)
    by_spk = defaultdict(list)
    for r in sorted(val, key=lambda r: r["clip"]):
        by_spk[Path(r["path"]).parent.name].append(r["clip"])
    spk = [s for s, cl in sorted(by_spk.items()) if len(cl) >= 3]
    rng = random.Random(args.seed)
    rng.shuffle(spk)
    print(f"{len(by_spk)} val speakers, {len(spk)} with >= 3 clips", flush=True)

    dev = "cuda"
    knnvc = load_knnvc(dev)
    sv = SpeakerEncoder(dev)
    asr = Asr(args.asr, dev) if args.asr else None
    out = Path(args.out)
    (out / "examples").mkdir(parents=True, exist_ok=True)

    rows, n_ex, t0 = defaultdict(list), 0, time.time()
    ceiling, floor = [], []
    targets = spk[: args.targets]
    for ti, T in enumerate(targets):
        clips, pool_s = by_spk[T], 0.0
        try:
            held = load_wav(Path(clips[0]).with_suffix(".m4a"))
            e_held = sv.embed([held])[0]
            # enrolment, exactly as scripts/enroll_voice.py: VAD -> chunks -> speaker filter -> layer-6 frames
            pieces = []
            for c in clips[1:]:
                w = load_wav(Path(c).with_suffix(".m4a"))
                pieces += chunks(w, speech_segments(w))
            emb = sv.embed(pieces)
            cen = F.normalize(emb.median(0).values, dim=0)
            kept = [p for p, c in zip(pieces, (emb @ cen).tolist()) if c >= args.min_cos]
            with torch.inference_mode():
                pool = torch.cat([knnvc.get_features(p[None], vad_trigger_level=0).half().cpu() for p in kept])
            ceiling.append(float(sv.embed([torch.cat(kept)])[0] @ e_held))
            pool_s = pool.shape[0] * FRAME_S
            sources = rng.sample([s for s in spk if s != T], min(args.sources, len(spk) - 1))
            for S in sources:
                src = load_wav(Path(by_spk[S][0]).with_suffix(".m4a"))
                seg = speech_segments(src)
                if not seg:
                    continue
                a = seg[0][0]
                src = src[a: a + int(args.src_seconds * SR)]
                e_src = sv.embed([src])[0]
                floor.append(float(e_src @ e_held))
                ref_txt = asr(src) if asr else None
                with torch.inference_mode():
                    q = knnvc.get_features(src[None], vad_trigger_level=0)
                for b in args.budgets:
                    if b < 1e9 and b > pool_s:
                        continue
                    bank = pool[: int(min(b, pool_s) / FRAME_S)].to(dev)
                    k = args.k or auto_k(bank.shape[0] * FRAME_S)
                    with torch.inference_mode():
                        y = knnvc.vocode(knn_features(q, bank, k)[None].to(dev)).float().cpu().squeeze()
                    y = match_level(fit_length(y, src.numel()), src)
                    e_y = sv.embed([y])[0]
                    key = "all" if b >= 1e9 else f"{int(b)} s"
                    r = {"sim_target": float(e_y @ e_held), "sim_source": float(e_y @ e_src),
                         "env_corr": float(np.corrcoef(envelope(y), envelope(src))[0, 1]),
                         "bank_s": bank.shape[0] * FRAME_S, "k": k}
                    if asr:
                        r["cer"] = cer(asr(y), ref_txt)
                    rows[key].append(r)
                    if n_ex < args.examples * len(args.budgets):
                        tag = f"{ti:02d}_{S[:12]}_to_{T[:12]}"
                        save_wav(src, out / "examples" / f"{tag}_source.wav")
                        save_wav(held, out / "examples" / f"{tag}_target_real.wav")
                        save_wav(y, out / "examples" / f"{tag}_converted_{key.replace(' ', '')}.wav")
                        n_ex += 1
        except Exception as e:
            print(f"  skip {T}: {type(e).__name__}: {e}", flush=True)
        print(f"  [{ti + 1}/{len(targets)}] {T}: enrolled {pool_s:.0f} s, {time.time() - t0:.0f} s elapsed", flush=True)

    order = [f"{int(b)} s" for b in args.budgets if b < 1e9] + ["all"]
    table = {k: {m: float(np.nanmean([r[m] for r in rows[k]])) for m in rows[k][0]} | {"pairs": len(rows[k])}
             for k in order if rows.get(k)}
    cols = ["sim_target", "sim_source", "env_corr"] + (["cer"] if asr else [])
    print("\n" + f"{'bank':<10}{'pairs':>7}{'mean s':>9}" + "".join(f"{c:>12}" for c in cols))
    for k, v in table.items():
        print(f"{k:<10}{v['pairs']:>7}{v['bank_s']:>9.1f}" + "".join(f"{v[c]:>12.3f}" for c in cols))
    print(f"{'real T vs T':<26}{np.mean(ceiling):>12.3f}   (ceiling: T's own speech vs its held-out clip)")
    print(f"{'source vs T':<26}{np.mean(floor):>12.3f}   (floor: unconverted source vs T)")
    dst = out / f"voice_eval_{time.strftime('%Y%m%d_%H%M')}.json"
    dst.write_text(json.dumps({"args": vars(args), "table": table, "ceiling": float(np.mean(ceiling)),
                               "floor": float(np.mean(floor)), "targets": targets}, indent=1))
    print(f"\nwrote {dst}; example wavs in {out / 'examples'}", flush=True)


if __name__ == "__main__":
    main()
