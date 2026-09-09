"""Frozen audio encoder: WavLM continuous features."""
import torch

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
