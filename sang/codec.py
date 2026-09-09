import os
from pathlib import Path
import torch
from moshi.models import loaders

MIMI_WEIGHTS = os.environ.get(
    "SANG_MIMI_WEIGHTS",
    str(Path(__file__).resolve().parents[1] / "checkpoints/mimi_kyutai.safetensors"),
)


def load_mimi(weights: str = MIMI_WEIGHTS, device: str = "cpu"):
    """Load the frozen Mimi codec in eval mode."""
    mimi = loaders.get_mimi(weights, device=device)
    mimi.eval()
    return mimi


WAVLM = {"wavlm-base": "microsoft/wavlm-base-plus", "wavlm-large": "microsoft/wavlm-large"}


class WavLMEncoder:
    """Mimi-compatible wrapper around a frozen WavLM: [B,1,N] @16k -> [B,T,D] continuous features.

    50 Hz, 1024-d (large) features instead of 12.5 Hz discrete codes — the audio representation
    used by Hallo3/VASA-style systems for lip-motion conditioning."""

    def __init__(self, name: str = "wavlm-large", device: str = "cpu"):
        from transformers import WavLMModel  # lazy: keep transformers optional for mimi-only users

        self.sample_rate = 16000
        self.m = WavLMModel.from_pretrained(WAVLM[name], local_files_only=True).to(device).eval()

    def set_num_codebooks(self, k: int) -> None:  # mimi-compat no-op
        pass

    @torch.no_grad()
    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        return self.m(wav.squeeze(1)).last_hidden_state


def load_wavlm(name: str = "wavlm-large", device: str = "cpu") -> WavLMEncoder:
    """Load a frozen WavLM from the local HF cache (cluster has no HF access)."""
    return WavLMEncoder(name, device)


@torch.no_grad()
def reconstruct(mimi, wav: torch.Tensor, num_codebooks: int) -> torch.Tensor:
    """Encode then decode a [1, 1, T] 24 kHz waveform at the given bitrate."""
    mimi.set_num_codebooks(num_codebooks)
    return mimi.decode(mimi.encode(wav))
