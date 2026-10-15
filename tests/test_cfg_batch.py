"""Batched CFG pair (sang/motion_model.sample, cfg_batch): the conditional and unconditional calls of each
Euler step as one batch give the same trajectory as two calls, up to float rounding. CPU only."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sang.motion import T_DIM
from sang.motion_model import REF_DIM, MotionFlowTransformer, generate, sample


def tiny(seed: int = 0) -> MotionFlowTransformer:
    torch.manual_seed(seed)
    m = MotionFlowTransformer(dim=32, depth=2, heads=2, audio_dim=16).eval()
    with torch.no_grad():                                    # zero-initialised heads: make v non-trivial
        for p in m.parameters():
            if p.ndim >= 1 and p.abs().sum() == 0:
                p.normal_(0, 0.05)
    return m


def test_batched_cfg_matches_two_calls():
    m = tiny()
    g = torch.Generator().manual_seed(1)
    audio, ref, prefix = torch.randn(2, 22, 16, generator=g), torch.randn(2, REF_DIM, generator=g), torch.randn(2, 3, T_DIM, generator=g)
    for kw in ({"cfg_audio": 2.0}, {"cfg_audio": 1.0, "cfg_rescale": 0.7}, {"cfg_audio": 2.0, "g_tmax": 0.5}):
        a = sample(m, audio, ref, 8, prefix=prefix, steps=4, generator=torch.Generator().manual_seed(5), cfg_batch=True, **kw)
        b = sample(m, audio, ref, 8, prefix=prefix, steps=4, generator=torch.Generator().manual_seed(5), cfg_batch=False, **kw)
        assert (a - b).abs().max() < 1e-5, kw


def test_batched_cfg_in_long_clips_with_mouth_average():
    m = tiny()
    g = torch.Generator().manual_seed(0)
    audio, ref = torch.randn(1, 60, 16, generator=g), torch.randn(1, REF_DIM, generator=g)
    a = generate(m, audio, ref, 30, window=8, n_prefix=3, steps=2, cfg_audio=2.0, mouth_avg=2,
                 generator=torch.Generator().manual_seed(0))
    b = generate(m, audio, ref, 30, window=8, n_prefix=3, steps=2, cfg_audio=2.0, mouth_avg=2,
                 generator=torch.Generator().manual_seed(0), cfg_batch=False)
    assert a.shape == (1, 30, T_DIM) and (a - b).abs().max() < 1e-5
