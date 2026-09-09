"""StableSyncNet matching LatentSync's stable_syncnet.pt (pixel-space, 16-frame)."""
import torch
import torch.nn.functional as F
from torch import nn


class GEGLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.proj(x).chunk(2, dim=-1)
        return a * F.gelu(b)


class FeedForward(nn.Module):
    """diffusers FeedForward(activation_fn='geglu'): net.0.proj, net.1 dropout, net.2 linear."""

    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        inner = dim * mult
        self.net = nn.ModuleList([GEGLU(dim, inner), nn.Dropout(dropout), nn.Linear(inner, dim)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net[0](x)
        x = self.net[1](x)
        return self.net[2](x)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int = 8):
        super().__init__()
        self.heads = heads
        self.to_q = nn.Linear(dim, dim, bias=True)
        self.to_k = nn.Linear(dim, dim, bias=True)
        self.to_v = nn.Linear(dim, dim, bias=True)
        self.to_out = nn.Sequential(nn.Linear(dim, dim, bias=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        h = self.heads
        q = self.to_q(x).reshape(B, N, h, C // h).transpose(1, 2)
        k = self.to_k(x).reshape(B, N, h, C // h).transpose(1, 2)
        v = self.to_v(x).reshape(B, N, h, C // h).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.to_out(out)


class ResnetBlock2D(nn.Module):
    """LatentSync resnet: downsample is pad + 3x3 conv with padding=0."""

    def __init__(self, in_ch: int, out_ch: int, downsample_factor=2, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch, eps=1e-6)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_ch, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.conv_shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else None

        if isinstance(downsample_factor, list):
            downsample_factor = tuple(downsample_factor)
        self.downsample_factor = downsample_factor
        if downsample_factor == 1 or downsample_factor == (1, 1):
            self.downsample_conv = None
            self.pad = None
        else:
            self.downsample_conv = nn.Conv2d(out_ch, out_ch, 3, stride=downsample_factor, padding=0)
            self.pad = (0, 1, 0, 1)
            if isinstance(downsample_factor, tuple):
                if downsample_factor[0] == 1:
                    self.pad = (0, 1, 1, 1)
                elif downsample_factor[1] == 1:
                    self.pad = (1, 1, 0, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        if self.conv_shortcut is not None:
            x = self.conv_shortcut(x)
        h = h + x
        if self.downsample_conv is not None:
            h = F.pad(h, self.pad, mode="constant", value=0)
            h = self.downsample_conv(h)
        return h


class AttentionBlock2D(nn.Module):
    """LatentSync attention block (separate from resnet; no downsample)."""

    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, dropout=dropout)
        self.conv_in = nn.Conv2d(dim, dim, 1)
        self.conv_out = nn.Conv2d(dim, dim, 1)
        self.attn = Attention(dim, heads=8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        B, C, H, W = x.shape
        h = self.conv_in(self.norm1(x)).reshape(B, C, H * W).transpose(1, 2)
        h = self.attn(self.norm2(h)) + h
        h = self.ff(self.norm3(h)) + h
        h = h.transpose(1, 2).reshape(B, C, H, W)
        return self.conv_out(h) + residual


class DownEncoder2D(nn.Module):
    def __init__(self, in_channels, block_out_channels, downsample_factors, attn_blocks, dropout=0.0):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], 3, padding=1)
        self.down_blocks = nn.ModuleList()
        ch = block_out_channels[0]
        for out_ch, ds, attn in zip(block_out_channels, downsample_factors, attn_blocks):
            self.down_blocks.append(ResnetBlock2D(ch, out_ch, downsample_factor=ds, dropout=dropout))
            if attn:
                self.down_blocks.append(AttentionBlock2D(out_ch, dropout=dropout))
            ch = out_ch
        self.norm_out = nn.GroupNorm(32, ch, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        for block in self.down_blocks:
            x = block(x)
        return F.relu(self.norm_out(x))


class StableSyncNet(nn.Module):
    """LatentSync pixel-space StableSyncNet (16 frames, 128x256 face crop)."""

    def __init__(self):
        super().__init__()
        # configs/syncnet/syncnet_16_pixel_attn.yaml
        self.audio_encoder = DownEncoder2D(
            in_channels=1,
            block_out_channels=[32, 64, 128, 256, 512, 1024, 2048],
            downsample_factors=[[2, 1], 2, 2, 1, 2, 2, [2, 3]],
            attn_blocks=[0, 0, 0, 1, 1, 0, 0],
        )
        self.visual_encoder = DownEncoder2D(
            in_channels=48,
            block_out_channels=[64, 128, 256, 256, 512, 1024, 2048, 2048],
            downsample_factors=[[1, 2], 2, 2, 2, 2, 2, 2, 2],
            attn_blocks=[0, 0, 0, 0, 1, 1, 0, 0],
        )

    @classmethod
    def from_checkpoint(cls, path: str, device: str = "cpu"):
        m = cls()
        m.load_state_dict(torch.load(path, map_location="cpu")["state_dict"])
        return m.to(device).eval()

    def forward(self, image_sequences, audio_sequences):
        v = self.visual_encoder(image_sequences).reshape(image_sequences.shape[0], -1)
        a = self.audio_encoder(audio_sequences).reshape(audio_sequences.shape[0], -1)
        return F.normalize(v, p=2, dim=1), F.normalize(a, p=2, dim=1)

    def loss(self, frames: torch.Tensor, audio_mels: torch.Tensor) -> torch.Tensor:
        """frames [B,3,T,H,W] in [-1,1], mels [B,80,T] -> cosine sync loss.

        The 48 channels must be FRAME-MAJOR (ch = t*3 + c). Any other order silently pins the
        loss at its ~0.65 floor: the pretrained conv_in sees a permuted stack."""
        B, C, T, H, W = frames.shape
        n = T // 16
        if n < 1:
            return torch.tensor(0.0, device=frames.device, requires_grad=True)
        # Only a mouth ROI when the frame is a face crop (face_crop: true).
        crop = frames[:, :, : n * 16, H // 2 :, :]
        crop = crop.permute(0, 2, 1, 3, 4).reshape(B * n * 16, C, H // 2, W)  # frame-major
        crop = F.interpolate(crop, size=(128, 256), mode="bilinear", align_corners=False)
        img = crop.reshape(B * n, 16 * C, 128, 256)
        aud = F.interpolate(
            audio_mels.unsqueeze(1), size=(80, 52), mode="bilinear", align_corners=False
        ).repeat_interleave(n, dim=0)
        v, a = self.forward(img, aud)
        return (1 - (v * a).sum(-1)).mean()


if __name__ == "__main__":
    m = StableSyncNet.from_checkpoint("third_party/syncnet/stable_syncnet.pt")
    x = torch.randn(2, 48, 128, 256)
    a = torch.randn(2, 1, 80, 52)
    v, aud = m(x, a)
    print("syncnet shapes:", v.shape, aud.shape)
    loss = m.loss(torch.randn(2, 3, 32, 128, 256), torch.randn(2, 80, 100))
    print("syncnet loss:", loss.item())
