"""Shape + struct checks for StreamingTalkingHead."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sang.streaming_transformer import StreamingTalkingHead


def test_streaming_shapes_and_struct():
    torch.manual_seed(0)
    B, tv, h, w, V = 2, 5, 16, 16, 64
    model = StreamingTalkingHead(V, dim=32, tv=tv, spatial=h, num_heads=2, num_layers=2,
                                 factorized_head=False).eval()
    video = torch.randint(0, V, (B, tv, h, w))
    audio = torch.randint(0, 2048, (B, 32, 9))
    struct = torch.randint(0, V, (B, tv, h, w))

    from sang.masking import per_slice_cosine_mask
    mask = per_slice_cosine_mask(B, tv, h * w, video.device, generator=torch.Generator(device=video.device).manual_seed(0))
    h_masked, tgt = model(video, audio, mask=mask)
    h_struct, tgt_s = model(video, audio, struct=struct, mask=mask)
    assert h_masked.shape == h_struct.shape and tgt.shape == tgt_s.shape
    assert h_masked.shape[0] == tgt.shape[0] > 0

    g = model.generate(audio, video[:, 0], struct=struct, steps=2, gumbel_temp=0.0)
    assert g.shape == (B, tv, h, w)

    model.train()
    hm, tm = model(video, audio)
    assert hm.ndim == 2 and hm.shape[0] == tm.shape[0]


if __name__ == "__main__":
    test_streaming_shapes_and_struct()
    print("ok")
