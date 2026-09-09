"""Runnable checks for face-mesh structure conditioning (E2): render shape + model forward/generate with struct."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sang.face import render_structure
from sang.model import TalkingHead


def test_render_shape_and_empty_on_black():
    """No face in a black clip -> same-shape all-zero uint8 render."""
    frames = np.zeros((3, 96, 96, 3), np.uint8)
    out = render_structure(frames)
    assert out.shape == frames.shape and out.dtype == np.uint8
    assert out.sum() == 0


def test_model_struct_forward_and_generate():
    """Struct is optional and shape-preserving; motion positions get conditioning, frame 0 does not."""
    torch.manual_seed(0)
    B, tv, h, w, V = 2, 5, 4, 4, 64
    r, motion = h * w, (tv - 1) * h * w
    model = TalkingHead(V, dim=32, num_heads=2, num_layers=2).eval()
    video = torch.randint(0, V, (B, tv, h, w))
    audio = torch.randint(0, 2048, (B, 32, 3))
    struct = torch.randint(0, V, (B, tv, h, w))

    l0, t0 = model(video, audio)
    l1, t1 = model(video, audio, struct=struct)
    assert l0.shape == l1.shape == (B, motion, V)
    assert t0.shape == t1.shape == (B, motion)
    assert not torch.allclose(l0, l1), "structure must change the motion predictions"

    # frame 0 is identity: its additive conditioning is zeroed
    add = model.struct_add(struct, r)
    assert add[:, :r].abs().sum() == 0 and add[:, r:].abs().sum() > 0

    g = model.generate(audio[:1], video[:1, 0], (tv, h, w), struct=struct[:1])
    assert g.shape == (1, tv, h, w)


if __name__ == "__main__":
    test_render_shape_and_empty_on_black()
    test_model_struct_forward_and_generate()
    print("ok")
