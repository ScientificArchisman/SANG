#!/usr/bin/env python3
"""Build or grow a person's voice bank for voice cloning (sang/voice.py, kNN-VC).

    sbatch bash_scripts/job.sh scripts/enroll_voice.py --name alice --audio /path/to/alice_recordings
    # later, with more recordings: the same line again; only new files are added
    python scripts/enroll_voice.py --name alice --audio more/ --ref alice_reference.wav

Each file goes through: energy VAD -> ~3 s chunks -> speaker-verification filter (chunks unlike the
person are dropped: interviewer, music, noise) -> kNN-VC WavLM layer-6 frames appended to
voices/<name>/bank.pt. voices/<name>/bank.json summarises minutes, the k that will be used and
the quality tier (kNN-VC improves up to ~5 min of clean speech; under ~30 s it is weak).

--ref: a short clean recording that is certainly the person; the filter then compares every chunk
to it. Without it, the bank's first enrolment takes the median chunk as the person, which assumes
they are the majority speaker in what you give it.
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from sang.voice import AUDIO_EXT, SpeakerEncoder, VoiceBank, load_knnvc, tier


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="bank name -> voices/<name>/")
    ap.add_argument("--audio", nargs="+", required=True, help="audio/video files or folders (searched recursively)")
    ap.add_argument("--ref", default=None, help="a clean recording that is certainly this person")
    ap.add_argument("--min-cos", type=float, default=0.75, help="speaker-filter threshold (cosine to the centroid)")
    ap.add_argument("--no-sv", action="store_true", help="skip the speaker filter (only if every file is clean, solo speech)")
    ap.add_argument("--banks", default=str(REPO / "voices"))
    args = ap.parse_args()

    files = []
    for a in map(Path, args.audio):
        files += sorted(f for f in (a.rglob("*") if a.is_dir() else [a]) if f.suffix.lower() in AUDIO_EXT)
    if not files:
        raise SystemExit(f"no audio files ({sorted(AUDIO_EXT)}) under {args.audio}")

    dev = "cuda"
    knnvc = load_knnvc(dev)
    sv = None if args.no_sv else SpeakerEncoder(dev)
    bank = VoiceBank(Path(args.banks) / args.name)
    before = bank.seconds
    print(f"bank {bank.dir}: {before:.1f} s before; {len(files)} file(s) offered", flush=True)
    r = bank.enroll(files, knnvc, sv, min_cos=args.min_cos, ref=Path(args.ref) if args.ref else None,
                    log=lambda m: print(m, flush=True))
    print(f"added {r.get('added_s', 0):.1f} s -> {bank.seconds:.1f} s ({bank.seconds / 60:.2f} min)", flush=True)
    print(f"tier: {tier(bank.seconds)}", flush=True)
    print(f"summary: {bank.dir / 'bank.json'}", flush=True)


if __name__ == "__main__":
    main()
