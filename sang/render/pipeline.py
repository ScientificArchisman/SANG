"""Source photo + speech -> frames, with stage timings: the one generation path that scripts/eval_hdtf.py and
infer_motion.py both run, so a demo is exactly what the benchmark scores.

Stages: source (face crop and the photo's motion, detected once), audio (WavLM), motion (the flow model and
the map back to 70-d motion, sang.model.latent.MotionLatent.to_motion), render (LivePortrait, `batch` frames per call, FP16 with
`half`) and paste (back into the photo with LivePortrait's feathered crop mask, GpuPaster).

GpuPaster is LivePortrait's paste-back (cv2.warpAffine of the crop over the full frame, then a float blend)
done with grid_sample inside the bounding box of the crop's footprint; outside it the output IS the
background. cv2.warpAffine maps output pixel (x, y) to crop pixel M^-1 (x, y) with pixel centres at integers,
which is grid_sample's align_corners=True after scaling by 2 / (S - 1). cv2 interpolates in fixed point
(1/32 px) and rounds the warped crop to uint8; the warped crop is rounded here too, so outputs differ from
cv2 by at most a grey level or so (tests/test_fastpath.py).
"""
import time
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F

from sang.render.liveportrait import LP, as3x3

SR, FPS = 16000, 25
HOP = SR // FPS


class Timer:
    """Wall-clock per stage; synchronises CUDA at stage boundaries only (profile mode)."""

    def __init__(self, cuda: bool = True):
        self.cuda = cuda and torch.cuda.is_available()
        self.t: dict[str, float] = {}

    @contextmanager
    def __call__(self, name: str):
        if self.cuda:
            torch.cuda.synchronize()
        t = time.perf_counter()
        yield
        if self.cuda:
            torch.cuda.synchronize()
        self.t[name] = self.t.get(name, 0.0) + time.perf_counter() - t


def crop_mask(S: int) -> np.ndarray:
    """LivePortrait's feathered crop mask at S x S (LP's mask_template.png), else a 16-px linear feather."""
    import cv2
    tpl = LP / "src" / "utils" / "resources" / "mask_template.png"
    if tpl.exists():
        return cv2.resize(cv2.imread(str(tpl), cv2.IMREAD_GRAYSCALE), (S, S)).astype(np.float32) / 255.0
    r = np.minimum(np.arange(S), np.arange(S)[::-1]).astype(np.float32)
    return np.clip(np.minimum(r[:, None], r[None, :]) / 16.0, 0, 1)


class GpuPaster:
    """Paste rendered crops [b, S, S, 3] (uint8, GPU) into `background` with M_c2o (crop -> original)."""

    def __init__(self, M_c2o: np.ndarray, background: np.ndarray, S: int, device: str = "cuda",
                 mask: np.ndarray | None = None):
        H, W = background.shape[:2]
        M = as3x3(M_c2o)
        corners = M @ np.array([[0, 0, 1], [S - 1, 0, 1], [0, S - 1, 1], [S - 1, S - 1, 1]], np.float64).T
        x0, y0 = np.floor(corners[:2].min(1)).astype(int) - 1
        x1, y1 = np.ceil(corners[:2].max(1)).astype(int) + 2
        self.x0, self.y0 = max(0, x0), max(0, y0)
        self.x1, self.y1 = min(W, x1), min(H, y1)
        self.bg_full = background
        self.empty = self.x1 <= self.x0 or self.y1 <= self.y0
        if self.empty:
            return
        ys, xs = np.mgrid[self.y0:self.y1, self.x0:self.x1].astype(np.float64)
        Minv = np.linalg.inv(M)
        sx = Minv[0, 0] * xs + Minv[0, 1] * ys + Minv[0, 2]
        sy = Minv[1, 0] * xs + Minv[1, 1] * ys + Minv[1, 2]
        g = np.stack([sx / (S - 1) * 2 - 1, sy / (S - 1) * 2 - 1], -1).astype(np.float32)
        self.grid = torch.from_numpy(g)[None].to(device)
        m = torch.from_numpy(crop_mask(S) if mask is None else mask.astype(np.float32))[None, None].to(device)
        self.mask = F.grid_sample(m, self.grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        self.bg = torch.from_numpy(np.ascontiguousarray(background[self.y0:self.y1, self.x0:self.x1])).to(device)
        self.bg = self.bg.permute(2, 0, 1)[None].float()
        self.device = device

    @torch.no_grad()
    def __call__(self, crops: torch.Tensor) -> np.ndarray:
        """[b, S, S, 3] uint8 (any device) -> [b, H, W, 3] uint8 numpy."""
        b = len(crops)
        out = np.repeat(self.bg_full[None], b, 0)
        if self.empty:
            return out
        f = crops.to(self.device).permute(0, 3, 1, 2).float()
        w = F.grid_sample(f, self.grid.expand(b, -1, -1, -1), mode="bilinear", padding_mode="zeros",
                          align_corners=True).round()
        roi = (self.mask * w + (1.0 - self.mask) * self.bg).clamp(0, 255).to(torch.uint8)
        out[:, self.y0:self.y1, self.x0:self.x1] = roi.permute(0, 2, 3, 1).cpu().numpy()
        return out


class Talker:
    def __init__(self, ckpt: str, sampler: str | None = None, device: str = "cuda", codec=None, wavlm=None):
        """sampler: a spec such as 'g=2,steps=6' (sang.model.flow.SAMPLER_KEYS); without one, the run's cfg_audio
        and sample_steps."""
        from sang.data.audio import load_wavlm
        from sang.model.flow import Norm, build, parse_spec, sampler_kwargs
        from sang.model.latent import MotionLatent
        from sang.render.liveportrait import MotionCodec
        ck = torch.load(ckpt, map_location=device, weights_only=False)
        spec = parse_spec(sampler)
        self.cfg, self.step, self.dev = ck["cfg"], ck["step"], device
        self.model = build(self.cfg).to(device).eval()
        self.model.load_state_dict(ck["ema"])
        self.norm = Norm(**ck["norm"]).to(device)
        self.latent = MotionLatent.of(ck).to(device)
        self.sampler = sampler_kwargs(spec, self.cfg)
        self.codec = codec or MotionCodec(device=device)
        self.wavlm = wavlm or load_wavlm(self.cfg.get("audio_encoder", "wavlm-large"), device=device)

    @torch.no_grad()
    def run(self, src: np.ndarray, wav: torch.Tensor, n_frames: int, seed: int = 0, stitch: bool = True, paste: bool = True, start: str = "null", batch: int = 16, half: bool = True,
            timer: Timer | None = None, normalize_lip: bool = True):
        """src [H,W,3] uint8, wav [N] 16 kHz -> (frames [n,H,W,3] or, with paste=False, crops [n,512,512,3] uint8,
        motion [n,70], timer). start='source': the first window continues from the photo's own motion after
        `prefix` frames of silence (the HDTF protocol's default); 'null': the dropped-prefix start training uses.
        normalize_lip=False: keep the photo's own lips (training references are raw frames) instead of closing them."""
        from sang.model.flow import generate
        timer = timer or Timer()
        codec = self.codec
        with timer("source"):
            crop = codec.crop_image(np.ascontiguousarray(src))
            sm = codec.source_motion(src, normalize_lip=normalize_lip, crop=crop)
            m_src = sm["m"].to(self.dev)
            ref = self.norm.ref(sm["kp"].to(self.dev), m_src)
        lead = self.cfg["prefix"] if start == "source" else 0
        with timer("audio"):
            audio = self.wavlm.encode(torch.nn.functional.pad(wav, (lead * HOP, 0))[None, None].to(self.dev)).float()
        with timer("motion"):
            first = self.latent.from_motion(m_src, m_src, self.model.t_dim).unsqueeze(1).expand(1, lead, -1) if lead else None
            g = torch.Generator(device=self.dev).manual_seed(seed)
            z = generate(self.model, audio, ref, n_frames, window=self.cfg["frames"], n_prefix=self.cfg["prefix"],
                         generator=g, start=first, **self.sampler)[0]
            m = self.latent.to_motion(z, m_src)
        with timer("render"):
            state = codec.source_state(crop["img_crop_256x256"], precropped=True)
            it = codec.render_iter(state, m.cpu(), relative=False, stitch=stitch, batch=batch, half=half)
        out, paster = [], None
        while True:
            with timer("render"):
                b = next(it, None)
            if b is None:
                break
            if paste:
                with timer("paste"):
                    paster = paster or GpuPaster(crop["M_c2o"], src, b.shape[1], device=self.dev)
                    b = paster(b)
            out.append(b.cpu().numpy() if torch.is_tensor(b) else b)
        return np.concatenate(out), m, timer

