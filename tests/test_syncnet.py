"""SyncNet loss geometry: the inner zoom that stands in for a tighter face crop."""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from sang.syncnet import SYNC_ZOOM, StableSyncNet, zoom_frames


def test_zoom_is_a_centred_crop_resized_back():
    """zoom z keeps the central 1/z of each frame at full resolution. With the cache built at
    face_box margin 1.6, z = 1.3 lands the mouth where stable_syncnet expects it: in-sync loss on
    ground truth 0.45 -> 0.42 and the in/out-of-sync gap +0.02 (probe, 2026-09-10), with no rebuild."""
    torch.manual_seed(0)
    x = torch.randn(2, 3, 17, 64, 64)
    assert torch.equal(zoom_frames(x, 1.0), x)
    z = zoom_frames(x, 1.6)
    assert z.shape == x.shape
    c = round(64 / 1.6); t0 = (64 - c) // 2
    ref = F.interpolate(x[:, :, :, t0:t0 + c, t0:t0 + c].permute(0, 2, 1, 3, 4).reshape(-1, 3, c, c),
                        size=(64, 64), mode="bilinear", align_corners=False)
    ref = ref.reshape(2, 17, 3, 64, 64).permute(0, 2, 1, 3, 4)
    assert torch.allclose(z, ref)
    assert SYNC_ZOOM == 1.3


def test_loss_applies_the_zoom_before_the_mouth_band():
    """The band handed to the network is the lower half of the ZOOMED frame."""
    torch.manual_seed(0)
    net = StableSyncNet().eval()            # random init: geometry only, no checkpoint needed
    frames = torch.randn(1, 3, 17, 64, 64)
    mel = torch.randn(1, 80, 52)
    with torch.no_grad():
        a = net.loss(frames, mel, zoom=1.3)
        b = net.loss(zoom_frames(frames, 1.3), mel, zoom=1.0)
    assert torch.allclose(a, b, atol=1e-5), (float(a), float(b))
    assert not torch.allclose(a, net.loss(frames, mel, zoom=1.0), atol=1e-5)
