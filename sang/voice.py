"""Voice cloning for SANG: kNN-VC (Baas et al., arXiv 2305.18975) against a growing voice bank.

The FACE is still driven by the ORIGINAL driving audio (the model never sees converted speech, so
nothing retrains and lip sync is what it was). Only the output audio track is converted:

    driving wav ─► kNN-VC's WavLM-Large, layer 6 (one vector per 20 ms)
                ─► each frame replaced by the mean of its k nearest frames (cosine) in the target's bank
                ─► prematched HiFi-GAN ─► wav on the SAME 20 ms grid as the input ─► muxed onto the video

kNN-VC is frame-synchronous, so the converted voice keeps the driver's timing, pauses and prosody
exactly and the rendered lips stay aligned by construction. It needs no training, and quality grows
with the bank: WER 7.36 / EER 37.15 at ~8 min of target speech (paper Tab. 1), plateauing near
5 min; below ~30 s zero-shot converters beat it (Palindromic VC, arXiv 2606.08843: SIM 0.380 at
3 s, 0.552 at 10 s, 0.617 at 30 s). See reports/SANG jitter emotion and voice cloning.md.

Why kNN-VC's OWN WavLM and not the face model's WavLM pass: its prematched vocoder was trained on
the original unilm WavLM-Large layer-6 features of raw (un-normalised) audio. A second WavLM pass
costs a fraction of a second per clip, and using the exact extractor removes a silent mismatch.

Enrolment keeps only the target's own speech: energy VAD, ~3 s chunks, a speaker-verification
embedding (WavLM-base-plus-sv) per chunk, and chunks too far from the bank's centroid (another
speaker, music, noise) are dropped. Banks are extended file by file, so "more clean audio of the
person" is a rerun of scripts/enroll_voice.py.

Install once on the login node: bash bash_scripts/install_voice.sh
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
KNNVC_DIR = REPO / "third_party" / "knn-vc"
SV_MODEL = "microsoft/wavlm-base-plus-sv"
SR, HOP = 16000, 320                          # kNN-VC: 16 kHz, one WavLM frame per 320 samples (20 ms)
FRAME_S = HOP / SR
AUDIO_EXT = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".mp4", ".mov", ".webm", ".mkv"}


# ---------------------------------------------------------------------- audio helpers
def load_wav(path: str | Path, sr: int = SR) -> torch.Tensor:
    """Any container decord can read (wav, m4a, mp3, mp4, flac) -> [N] float at `sr`, mono."""
    from decord import AudioReader
    return torch.from_numpy(AudioReader(str(path), sample_rate=sr, mono=True)[:].asnumpy()).float()[0]


def save_wav(wav: torch.Tensor, path: str | Path, sr: int = SR) -> Path:
    """[N] float in [-1, 1] -> 16-bit PCM wav (stdlib only)."""
    import wave
    path = Path(path)
    pcm = (wav.detach().float().clamp(-1, 1).cpu().numpy() * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return path


def fit_length(wav: torch.Tensor, n: int) -> torch.Tensor:
    """Pad with zeros or trim to exactly n samples (the vocoder returns ~frames*320, a few ms short)."""
    return F.pad(wav, (0, n - wav.numel())) if wav.numel() < n else wav[:n]


def rms_db(wav: torch.Tensor) -> float:
    return float(10 * torch.log10(wav.float().pow(2).mean() + 1e-10))


def match_level(wav: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Scale `wav` to the RMS level of `ref` (keeps the driver's loudness), clipped to [-1, 1]."""
    gain = 10 ** ((rms_db(ref) - rms_db(wav)) / 20)
    return (wav * gain).clamp(-1, 1)


def speech_segments(wav: torch.Tensor, below_db: float = 35.0, min_s: float = 0.5,
                    bridge_s: float = 0.3, frame: int = HOP) -> list[tuple[int, int]]:
    """Energy VAD -> [(start, end)] sample ranges of speech. A frame is speech when it is within
    `below_db` of the recording's loud level (95th pct); gaps shorter than bridge_s are bridged and
    segments shorter than min_s dropped."""
    n = wav.numel() // frame
    if n == 0:
        return []
    db = 10 * torch.log10(wav[: n * frame].view(n, frame).pow(2).mean(1) + 1e-10).numpy()
    speech = db > np.percentile(db, 95) - below_db
    segs, s = [], None
    for i, v in enumerate(np.append(speech, False)):
        if v and s is None:
            s = i
        elif not v and s is not None:
            segs.append([s, i])
            s = None
    merged = []
    for a, b in segs:
        if merged and (a - merged[-1][1]) * frame / SR < bridge_s:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [(a * frame, b * frame) for a, b in merged if (b - a) * frame / SR >= min_s]


def chunks(wav: torch.Tensor, segs: list[tuple[int, int]], chunk_s: float = 3.0) -> list[torch.Tensor]:
    """Speech segments cut into ~chunk_s pieces (the tail joins its predecessor if short)."""
    L, out = int(chunk_s * SR), []
    for a, b in segs:
        cuts = list(range(a, b, L))
        pieces = [(c, min(c + L, b)) for c in cuts]
        if len(pieces) > 1 and pieces[-1][1] - pieces[-1][0] < L // 2:
            pieces[-2] = (pieces[-2][0], pieces[-1][1])
            pieces.pop()
        out += [wav[c:d] for c, d in pieces]
    return out


# ---------------------------------------------------------------------- kNN regression
def knn_features(query: torch.Tensor, bank: torch.Tensor, k: int, block: int = 1024) -> torch.Tensor:
    """kNN-VC's matching step, memory-bounded: each query frame [T, D] -> the mean of its k nearest
    bank frames [N, D] by cosine similarity (same ranking as kNN-VC's fast_cosine_dist)."""
    k = min(k, bank.shape[0])
    bank = bank.to(query.device)
    bn = F.normalize(bank.float(), dim=-1).to(bank.dtype if bank.is_cuda else torch.float32)
    out = []
    for i in range(0, query.shape[0], block):
        q = F.normalize(query[i:i + block].float(), dim=-1).to(bn.dtype)
        idx = (q @ bn.T).topk(k, dim=-1).indices                  # [b, k]
        out.append(bank[idx].float().mean(1))
    return torch.cat(out) if out else query.new_zeros(0, bank.shape[1])


def auto_k(bank_seconds: float) -> int:
    """k by bank size. kNN-VC's default 4 is tuned on ~8 min; the report's reading of its ablation
    is that larger k helps only with a lot of data (>= ~10 min). A heuristic -- sweep it in eval."""
    return 4 if bank_seconds < 300 else 8 if bank_seconds < 1200 else 16


def tier(bank_seconds: float) -> str:
    if bank_seconds < 10:
        return "too little (<10 s): kNN-VC will be poor; a zero-shot converter is the right tool here"
    if bank_seconds < 30:
        return "short (10-30 s): kNN-VC is usable but zero-shot converters are still better"
    if bank_seconds < 300:
        return "good (30 s-5 min): kNN-VC matches zero-shot similarity and improves as you add audio"
    return "plateau (>5 min): kNN-VC near its best; beyond ~30 min consider a per-speaker fine-tune A/B"


# ---------------------------------------------------------------------- models
def load_knnvc(device: str = "cuda"):
    """kNN-VC with the prematched vocoder, from third_party/knn-vc + torch.hub's weight cache
    (downloaded once on the login node by bash_scripts/install_voice.sh)."""
    if not (KNNVC_DIR / "hubconf.py").exists():
        raise FileNotFoundError(f"kNN-VC not at {KNNVC_DIR}; run bash bash_scripts/install_voice.sh")
    if str(KNNVC_DIR) not in sys.path:
        sys.path.insert(0, str(KNNVC_DIR))
    from hubconf import knn_vc                      # noqa: E402  (third_party/knn-vc)
    return knn_vc(pretrained=True, progress=False, prematched=True, device=device)


class SpeakerEncoder:
    """WavLM-base-plus-sv x-vectors: enrolment filtering and the speaker-similarity metric.
    Same family as kNN-VC's encoder, so report a second verifier before trusting small gaps."""

    def __init__(self, device: str = "cuda"):
        from transformers import AutoFeatureExtractor, WavLMForXVector
        self.fe = AutoFeatureExtractor.from_pretrained(SV_MODEL, local_files_only=True)
        self.m = WavLMForXVector.from_pretrained(SV_MODEL, local_files_only=True).to(device).eval()
        self.device = device

    @torch.no_grad()
    def embed(self, wavs: list[torch.Tensor]) -> torch.Tensor:
        """list of [N] 16 kHz -> [B, 512] L2-normalised."""
        out = []
        for w in wavs:
            x = self.fe(w.numpy(), sampling_rate=SR, return_tensors="pt", padding=True)
            e = self.m(**{k: v.to(self.device) for k, v in x.items()}).embeddings
            out.append(F.normalize(e.float(), dim=-1).cpu())
        return torch.cat(out)


# ---------------------------------------------------------------------- the voice bank
class VoiceBank:
    """One person's matching set on disk: <dir>/bank.pt {feats [N,1024] fp16, centroid [512],
    files {name: stats}} + <dir>/bank.json (human-readable summary). Grows with every enrolment."""

    def __init__(self, path: str | Path):
        self.dir = Path(path)
        self.feats = torch.zeros(0, 1024, dtype=torch.float16)
        self.centroid: torch.Tensor | None = None
        self.files: dict[str, dict] = {}
        if (self.dir / "bank.pt").exists():
            d = torch.load(self.dir / "bank.pt", map_location="cpu", weights_only=False)
            self.feats, self.centroid, self.files = d["feats"], d.get("centroid"), d["files"]

    @property
    def seconds(self) -> float:
        return self.feats.shape[0] * FRAME_S

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        torch.save({"feats": self.feats, "centroid": self.centroid, "files": self.files}, self.dir / "bank.pt")
        (self.dir / "bank.json").write_text(json.dumps(
            {"seconds": round(self.seconds, 1), "minutes": round(self.seconds / 60, 2), "k": auto_k(self.seconds),
             "tier": tier(self.seconds), "files": self.files}, indent=1))

    @staticmethod
    def key(p: Path) -> str:
        st = p.stat()
        return f"{p.resolve()}|{st.st_size}|{int(st.st_mtime)}"

    def enroll(self, files: list[Path], knnvc, sv: SpeakerEncoder | None, min_cos: float = 0.75,
               ref: Path | None = None, log=print) -> dict:
        """Add each new file's clean target speech. Returns a summary.

        min_cos: chunks whose speaker embedding is further than this from the bank centroid are
        dropped (another speaker, music, noise). The centroid is the `ref` recording's embedding if
        given (e.g. a consent/reference clip of the person), else the median of all chunks seen so far
        -- which assumes the target is the majority speaker in what you enrol."""
        new = [p for p in files if self.key(p) not in self.files]
        if not new:
            log(f"nothing new: {len(files)} file(s) already in the bank")
            return {"added_s": 0.0}
        pieces, owner = [], []                                      # owner[i] = bank key of pieces[i]
        for p in new:
            try:
                w = load_wav(p)
                cs = chunks(w, speech_segments(w))
                key = self.key(p)
                pieces += cs
                owner += [key] * len(cs)
                self.files[key] = {"file": p.name, "audio_s": round(w.numel() / SR, 1),
                                   "speech_s": round(sum(c.numel() for c in cs) / SR, 1), "kept_s": 0.0}
            except Exception as e:                                   # one bad file never kills enrolment
                log(f"  skip {p.name}: {type(e).__name__}: {e}")
        if not pieces:
            return {"added_s": 0.0}

        keep = [True] * len(pieces)
        if sv is not None:
            emb = sv.embed(pieces)                                   # [C, 512]
            if ref is not None:
                self.centroid = sv.embed([load_wav(ref)])[0]
            elif self.centroid is None:
                self.centroid = F.normalize(emb.median(0).values, dim=0)
            cos = emb @ self.centroid
            keep = (cos >= min_cos).tolist()
            dropped = sum(p.numel() for p, k in zip(pieces, keep) if not k) / SR
            log(f"  speaker filter: kept {sum(keep)}/{len(pieces)} chunks, dropped {dropped:.1f} s "
                f"(cos to centroid: median {cos.median():.3f}, min {cos.min():.3f})")
        feats = []
        with torch.inference_mode():
            for p, k, o in zip(pieces, keep, owner):
                if k:
                    feats.append(knnvc.get_features(p[None], vad_trigger_level=0).half().cpu())
                    self.files[o]["kept_s"] = round(self.files[o]["kept_s"] + feats[-1].shape[0] * FRAME_S, 1)
        added = torch.cat(feats) if feats else torch.zeros(0, 1024, dtype=torch.float16)
        self.feats = torch.cat([self.feats, added])
        self.save()
        return {"added_s": added.shape[0] * FRAME_S, "total_s": self.seconds}


# ---------------------------------------------------------------------- conversion
@torch.inference_mode()
def convert(wav: torch.Tensor, bank: VoiceBank, knnvc, k: int | None = None) -> torch.Tensor:
    """Driving speech [N] at 16 kHz -> the same speech in the bank's voice, [N] samples exactly,
    on the same 20 ms grid, at the driver's loudness."""
    if bank.feats.shape[0] == 0:
        raise ValueError(f"voice bank {bank.dir} is empty; enrol audio first (scripts/enroll_voice.py)")
    dev = knnvc.device
    q = knnvc.get_features(wav[None].float(), vad_trigger_level=0)               # [T, 1024]
    bank_dev = bank.feats.to(dev)
    out = knn_features(q, bank_dev, k or auto_k(bank.seconds))
    y = knnvc.vocode(out[None].to(dev)).float().cpu().squeeze()
    return match_level(fit_length(y, wav.numel()), wav)


def voice_bank(path: str | Path, knnvc, sv: SpeakerEncoder | None = None, log=print) -> VoiceBank:
    """`path` is a bank dir (has bank.pt) -> load it; otherwise a folder/file of the person's audio
    -> enrol it into voices/<name>/ (persisted, so the next run reuses and only adds new files)."""
    p = Path(path)
    if (p / "bank.pt").exists():
        return VoiceBank(p)
    files = sorted(f for f in (p.rglob("*") if p.is_dir() else [p]) if f.suffix.lower() in AUDIO_EXT)
    if not files:
        raise FileNotFoundError(f"{p}: neither a voice bank (bank.pt) nor audio files ({sorted(AUDIO_EXT)})")
    bank = VoiceBank(REPO / "voices" / (p.stem if p.is_file() else p.name))
    t0 = time.time()
    r = bank.enroll(files, knnvc, sv, log=log)
    log(f"voice bank {bank.dir}: +{r.get('added_s', 0):.1f} s -> {bank.seconds:.1f} s "
        f"({time.time() - t0:.0f} s). {tier(bank.seconds)}")
    return bank



# ---------------------------------------------------------------------- metrics
def levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(hyp: str, ref: str) -> float:
    """Character error rate of hyp against ref, whitespace-normalised, case-folded."""
    h, r = " ".join(hyp.lower().split()), " ".join(ref.lower().split())
    return levenshtein(h, r) / max(1, len(r))
