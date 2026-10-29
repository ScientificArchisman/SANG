"""Audio: loading and saving at 16 kHz, and the frozen WavLM encoder."""
from pathlib import Path

import numpy as np
import torch

SR, FPS = 16000, 25


def load_wav(path: str | Path, start_frame: int = 0, n_frames: int | None = None, sr: int = SR) -> torch.Tensor:
    """Any container decord can read (wav, m4a, mp3, mp4, flac) -> [N] float at `sr`, mono. With n_frames: exactly
    the samples of video frames [start_frame, start_frame + n_frames) at 25 fps, zero-padded at the end."""
    from decord import AudioReader
    wav = torch.from_numpy(AudioReader(str(path), sample_rate=sr, mono=True)[:].asnumpy()).float()[0]
    if n_frames is None:
        return wav[start_frame * sr // FPS:]
    hop = sr // FPS
    wav = wav[start_frame * hop:(start_frame + n_frames) * hop]
    return torch.nn.functional.pad(wav, (0, max(0, n_frames * hop - len(wav))))


def save_wav(wav: torch.Tensor, path: str | Path, sr: int = SR) -> Path:
    """[N] float in [-1, 1] -> 16-bit PCM mono wav (stdlib only)."""
    import wave
    path = Path(path)
    pcm = (wav.detach().float().clamp(-1, 1).cpu().numpy() * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return path


WAVLM = {"wavlm-base": "microsoft/wavlm-base-plus", "wavlm-large": "microsoft/wavlm-large"}


class WavLMEncoder:
    """Frozen WavLM: [B,1,N] @16 kHz -> [B,T,D] features at 50 Hz (1024-d for large)."""

    def __init__(self, name: str = "wavlm-large", device: str = "cpu"):
        from transformers import WavLMModel

        self.sample_rate = 16000
        self.m = WavLMModel.from_pretrained(WAVLM[name], local_files_only=True).to(device).eval()

    @torch.no_grad()
    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        return self.m(wav.squeeze(1)).last_hidden_state


def load_wavlm(name: str = "wavlm-large", device: str = "cpu") -> WavLMEncoder:
    """Load a frozen WavLM from the local HF cache (the cluster has no HF access)."""
    return WavLMEncoder(name, device)

