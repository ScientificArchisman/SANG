"""Source photo + speech -> frames, with stage timings: the unit scripts/eval_hdtf.py and
scripts/latency.py both run, so the benchmark measures exactly what the evaluation scores.

Options (all off = the original path, stage for stage):
  cfg_batch    the CFG pair of each Euler step as one batch (sang.motion_model.sample)
  render_batch frames per renderer call (MotionCodec.render_iter); 1 = the per-frame loop
  half         FP16 autocast around the warping module and decoder
  gpu_paste    paste-back with grid_sample inside the face's bounding box (sang.fastpath.GpuPaster)
  reuse_crop   detect the face in the source photo once (the original path detects it three times:
               source motion, renderer source state, paste-back)
"""
from dataclasses import dataclass

import numpy as np
import torch

from sang.fastpath import GpuPaster, Timer

SR, FPS = 16000, 25
HOP = SR // FPS


@dataclass
class Opts:
    cfg_batch: bool = False
    render_batch: int = 1
    half: bool = False
    gpu_paste: bool = False
    reuse_crop: bool = False

    @classmethod
    def fast(cls) -> "Opts":
        return cls(cfg_batch=True, render_batch=16, half=True, gpu_paste=True, reuse_crop=True)


class Talker:
    def __init__(self, ckpt: str, sampler: str | None, device: str = "cuda", codec=None, wavlm=None,
                 ema: str | None = None, cfg_audio=None, steps: int | None = None, guide_ckpt: str | None = None):
        """sampler: a spec ('g=2,mouth=1.25,avg=4'); without one, cfg_audio / steps (default: the run's)."""
        from sang.codec import load_wavlm
        from sang.motion import MotionCodec
        from sang.motion_model import Norm, build, ema_weights, load_ema, parse_spec, sampler_kwargs
        ck = torch.load(ckpt, map_location=device, weights_only=False)
        self.cfg, self.step, self.dev = ck["cfg"], ck["step"], device
        self.model = build(self.cfg).to(device).eval()
        self.model.load_state_dict(ema_weights(ck, ema))
        self.norm = Norm(**ck["norm"]).to(device)
        guide = load_ema(guide_ckpt, device, self.norm) if guide_ckpt else None
        self.sampler = (sampler_kwargs(parse_spec(sampler), self.cfg, guide, steps=steps) if sampler else
                        {"cfg_audio": self.cfg["cfg_audio"] if cfg_audio is None else cfg_audio,
                         "steps": steps or self.cfg["sample_steps"]})
        self.codec = codec or MotionCodec(device=device)
        self.wavlm = wavlm or load_wavlm(self.cfg.get("audio_encoder", "wavlm-large"), device=device)

    @torch.no_grad()
    def run(self, src: np.ndarray, wav: torch.Tensor, n_frames: int, opts: Opts, seed: int = 0,
            held: str = "camera", stitch: bool = True, paste: bool = True, start: str = "null",
            timer: Timer | None = None, sink=None, keep: bool = True):
        """src [H,W,3] uint8, wav [N] 16 kHz -> (frames [n,H,W,3] or crops [n,512,512,3] uint8, motion [n,70],
        timer). Stages: source, audio, motion, render, paste. start='source': the first window continues
        from the photo's own motion after `prefix` frames of silence (the HDTF protocol's default).
        sink: called with each [b,H,W,3] uint8 batch of finished frames (e.g. AsyncWriter.write); with
        gpu_paste the batches stream out as they are rendered, so encoding overlaps rendering.
        keep=False: do not also return the frames (frames is then None)."""
        from sang.motion import from_target, paste_back, to_target
        from sang.naturalness import guided_generate
        timer = timer or Timer()
        codec = self.codec
        with timer("source"):
            crop = codec.crop_image(np.ascontiguousarray(src)) if opts.reuse_crop else None
            sm = codec.source_motion(src, crop=crop)
            m_src = sm["m"].to(self.dev)
            ref = self.norm.ref(sm["kp"].to(self.dev), to_target(m_src))
        lead = self.cfg["prefix"] if start == "source" else 0
        with timer("audio"):
            audio = self.wavlm.encode(torch.nn.functional.pad(wav, (lead * HOP, 0))[None, None].to(self.dev)).float()
        with timer("motion"):
            samp = dict(self.sampler)
            samp["cfg_batch"] = opts.cfg_batch
            first = self.norm.target(to_target(m_src)).unsqueeze(1).expand(1, lead, -1) if lead else None
            y, _ = guided_generate(self.model, audio, ref, n_frames, self.cfg, self.norm, wav, None, "none", seed,
                                   start=first, sampler=samp)
            m = from_target(self.norm.untarget(y[0]), m_src, held=held)
        fast = opts.render_batch > 1 or opts.half or opts.gpu_paste
        if fast and paste and opts.gpu_paste:                # streamed: render a batch, paste it, hand it on
            with timer("render"):
                state = codec.source_state(crop["img_crop_256x256"] if crop is not None else src,
                                           precropped=crop is not None)
            with timer("paste"):
                M = crop["M_c2o"] if crop is not None else codec.crop_image(src)["M_c2o"]
            it = codec.render_iter(state, m.cpu(), relative=False, stitch=stitch, batch=max(1, opts.render_batch),
                                   half=opts.half)
            paster, kept = None, []
            while True:
                with timer("render"):
                    b = next(it, None)
                if b is None:
                    break
                with timer("paste"):
                    paster = paster or GpuPaster(M, src, b.shape[1], device=self.dev)
                    f = paster(b)
                if sink is not None:
                    sink(f)
                if keep:
                    kept.append(f)
            return (np.concatenate(kept) if keep else None), m, timer
        with timer("render"):
            state = None
            if crop is not None:
                state = codec.source_state(crop["img_crop_256x256"], precropped=True)
                state["M_c2o"] = crop["M_c2o"]
            crops = codec.render(src, m.cpu(), relative=False, stitch=stitch, state=state,
                                 batch=max(1, opts.render_batch), half=opts.half,
                                 out="torch" if opts.gpu_paste else "numpy") if fast else \
                codec.render(src, m.cpu(), relative=False, stitch=stitch, state=state)
        if not paste:
            return (crops.cpu().numpy() if torch.is_tensor(crops) else crops), m, timer
        with timer("paste"):
            M = crop["M_c2o"] if crop is not None else codec.crop_image(src)["M_c2o"]
            if opts.gpu_paste:
                c = crops if torch.is_tensor(crops) else torch.from_numpy(crops)
                frames = GpuPaster(M, src, c.shape[1], device=self.dev)(c)
            else:
                frames = paste_back(crops.cpu().numpy() if torch.is_tensor(crops) else crops, M, src)
        if sink is not None:
            sink(frames)
        return (frames if keep else None), m, timer
