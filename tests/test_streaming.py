"""StreamingTalkingHead: shapes, conditioning, the bridge prior, and the FSQ head loss."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sang.fsq_codec import FSQIndexCodec
from sang.fsq_head import CoupledFSQHead, FactorizedFSQHead
from sang.masking import per_slice_cosine_mask
from sang.streaming_transformer import StreamingTalkingHead

B, TV, S, V, D = 2, 5, 16, 64, 32


def _model(**kw):
    return StreamingTalkingHead(V, dim=D, tv=TV, spatial=S, num_heads=2, num_layers=2,
                                factorized_head=False, audio_dim=32, **kw)


def _batch(tv=TV):
    return (torch.randint(0, V, (B, tv, S, S)), torch.randn(B, 9, 32),
            torch.randint(0, V, (B, tv, S, S)))


def test_shapes_and_struct():
    torch.manual_seed(0)
    m = _model().eval()
    video, audio, struct = _batch()
    mask = per_slice_cosine_mask(B, TV, S * S, video.device,
                                 generator=torch.Generator(device=video.device).manual_seed(0))
    hm, tgt = m(video, audio, mask=mask)
    hs, tgs = m(video, audio, struct=struct, mask=mask)
    assert hm.shape == hs.shape and tgt.shape == tgs.shape and hm.shape[0] == tgt.shape[0] > 0
    assert m.generate(audio, video[:, 0], struct=struct, steps=2, gumbel_temp=0.0).shape == video.shape


def test_bridge_prev_keeps_slices_distinct():
    """bridge_init="ref" gives every masked content slice slice-0's embedding, so at generation
    time all content slices are identical and the model cannot produce motion (Part VI, D1)."""
    torch.manual_seed(0)
    video, audio, struct = _batch()
    known = torch.zeros(B, TV * S * S, dtype=torch.bool)
    known[:, : S * S] = True  # generation regime: only the ref slice is known

    def spread(mode):
        m = _model(bridge_init=mode).eval()
        with torch.no_grad():
            e = m._embed_grid(video, struct, known).reshape(B, TV, S * S, -1)[:, 1:]
        return (e - e[:, :1]).abs().max().item()

    assert spread("ref") == 0.0
    assert spread("prev") > 0.0


def test_fsq_heads_and_expected_codes():
    """The coupled head must not inherit the uncoupled expected_codes (Part VI, D6)."""
    torch.manual_seed(0)
    codes = torch.stack(torch.meshgrid(*[torch.linspace(-1, 0.75, 8)] * 2, indexing="ij"), -1).reshape(-1, 2)
    codec = FSQIndexCodec.from_codebook(codes)
    h = torch.randn(16, D)
    tgt = torch.randint(0, codes.shape[0], (16,))
    for cls in (FactorizedFSQHead, CoupledFSQHead):
        head = cls(D, codec)
        loss, parts = head.loss(h, tgt)
        assert torch.isfinite(loss) and "acc_token" in parts
        ec = head.expected_codes(h)
        assert ec.shape == (16, 2) and torch.isfinite(ec).all()
    # the coupled head conditions digit d on digits <d, so its expected codes differ
    torch.manual_seed(0); a = FactorizedFSQHead(D, codec).expected_codes(h)
    torch.manual_seed(0); b = CoupledFSQHead(D, codec).expected_codes(h)
    assert not torch.allclose(a, b)


if __name__ == "__main__":
    test_shapes_and_struct()
    test_bridge_prev_keeps_slices_distinct()
    test_fsq_heads_and_expected_codes()
    print("ok")
