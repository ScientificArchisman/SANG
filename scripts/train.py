import argparse
import collections
import gc
import glob
import hashlib
import json
import math
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from sang.data import audio_samples, clip_windows
from sang.masking import per_slice_cosine_mask
from sang.model import build_talking_head
from sang.video import load_vidtok


def load_visual(cfg: dict, device: str):
    """VidTok FSQ (discrete) or frozen Wan2.1 VAE (continuous). Params are frozen explicitly:
    VidTok's fix_decoder defaults to False, so decode_pixels would accumulate decoder grads."""
    if cfg.get("continuous"):
        from sang.vae import load_wan_vae
        model = load_wan_vae(device=device)
    else:
        model = load_vidtok(codebook=cfg["codebook"], device=device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model

from sang.losses import build_losses


def mel_spectrogram(wav: torch.Tensor, sr: int = 16000, n_mels: int = 80) -> torch.Tensor:
    """[1, N] waveform -> [1, n_mels, T] log-mel for SyncNet (16kHz, 80 bins, 25ms/10ms hops)."""
    import torchaudio
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_fft=1024, hop_length=160, win_length=400, n_mels=n_mels,
        f_min=0, f_max=sr // 2, power=2.0,
    )(wav)
    return torch.log(mel.clamp_min(1e-5))


def _mel_audio_ctx(clip_path, cfg: dict):
    """(native_fps, span_frames, sr, n_samples, wav[1,N]) for clip_windows-aligned mel."""
    from decord import AudioReader, VideoReader

    path = Path(clip_path)
    native = VideoReader(str(path)).get_avg_fps()
    fps, frames = cfg["fps"], cfg["frames"]
    span = int(round((frames - 1) * (native / fps))) + 1
    sr = 16000
    n = audio_samples(frames, fps, sr)  # D8: frames-1 intervals, not frames
    wav = torch.from_numpy(AudioReader(str(path.with_suffix(".m4a")), sample_rate=sr, mono=True)[:].asnumpy())
    return native, span, sr, n, wav


def mel_for_window(clip_path, window_idx: int, cfg: dict) -> torch.Tensor:
    native, span, sr, n, wav = _mel_audio_ctx(clip_path, cfg)
    a0 = int(window_idx * span / native * sr)
    win = wav[:, a0:a0 + n]
    if win.shape[-1] < n:
        win = F.pad(win, (0, n - win.shape[-1]))
    return mel_spectrogram(win, sr).squeeze(0).half()


def patch_mel_for_clip(clip_path, cached_files, cfg: dict) -> None:
    """Add mel to cached windows; one audio read per clip."""
    todo = []
    for w, f in enumerate(cached_files):
        d = torch.load(f, weights_only=True)
        if "mel" not in d:
            todo.append((w, f, d))
    if not todo:
        return
    native, span, sr, n, wav = _mel_audio_ctx(clip_path, cfg)
    for w, f, d in todo:
        a0 = int(w * span / native * sr)
        win = wav[:, a0:a0 + n]
        if win.shape[-1] < n:
            win = F.pad(win, (0, n - win.shape[-1]))
        d["mel"] = mel_spectrogram(win, sr).squeeze(0).half()
        torch.save(d, f)


def _clip_cache_complete(cache_dir: Path, key: str, nwin: int) -> list[Path] | None:
    """Return window files if this clip is fully cached, else None.

    A clip is complete if windows 0..k-1 exist with no holes and k==nwin (full) or k>=6
    (short TalkVid clips only fit 6–7 windows). Partial OOM dumps (e.g. 3 of 8) are not complete."""
    files = [cache_dir / f"{key}_{w}.pt" for w in range(nwin)]
    existing = [i for i, f in enumerate(files) if f.exists()]
    if not existing or existing != list(range(len(existing))):
        return None
    k = len(existing)
    if k == nwin or k >= 6:
        return files[:k]
    return None


def build_cache(clips, vidtok, audio_enc, cfg, cache_dir: Path, split: str):
    cache_dir.mkdir(parents=True, exist_ok=True)
    motion_ctx = cfg.get("motion_ctx", False)
    syncnet = cfg.get("syncnet_loss", False)
    continuous = cfg.get("continuous", False)  # v4: store continuous latents, not FSQ indices
    out = []
    for i, c in enumerate(tqdm(clips, desc=f"cache/{split}")):
        key = hashlib.md5(str(c).encode()).hexdigest()
        done = _clip_cache_complete(cache_dir, key, cfg["windows_per_clip"])
        if done is not None:
            if syncnet:
                patch_mel_for_clip(c, done, cfg)
            out.extend(done)
            continue
        try:
            wins = clip_windows(c, vidtok, audio_enc, frames=cfg["frames"], res=cfg["res"],
                                fps=cfg["fps"], max_windows=cfg["windows_per_clip"],
                                face_cond=cfg.get("face_cond", False), continuous=continuous,
                                face_crop=cfg.get("face_crop", False))
        except Exception as e:
            print(f"skip {Path(c).name}: {type(e).__name__} {e}", file=sys.stderr)
            continue
        for w, d in enumerate(wins):
            f = cache_dir / f"{key}_{w}.pt"
            a = d["audio"][0]
            # continuous: video is [1,z_ch,Tv,h,w] float -> store half; discrete: [1,Tv,h,w] -> int16
            v = d["video"][0]
            vstore = v.cpu().half() if continuous else v.to(torch.int16).cpu()
            entry = {"video": vstore,
                     "audio": a.cpu().half() if a.is_floating_point() else a.to(torch.int16).cpu()}
            if "struct" in d:
                entry["struct"] = d["struct"][0].to(torch.int16).cpu()
            if "face_found" in d:
                entry["face_found"] = bool(d["face_found"])
            if motion_ctx:
                # identity anchor = clip frame 0 (matches the static image at inference);
                # motion context = previous window's last slice (window 0: static start = ref).
                # continuous video is [z_ch, Tv, h, w]; discrete is [Tv, h, w]. Slice a frame off dim Tv.
                def frame(vid, t):  # -> [z_ch,h,w] (continuous) | [h,w] (discrete)
                    return vid[:, t] if continuous else vid[t]
                cast = (lambda x: x.half()) if continuous else (lambda x: x.to(torch.int16))
                # Anchor from a DIFFERENT window: window 0 must not get its own content slice 0
                # as ref/ctx, which would hand over a supervised slice verbatim.
                ref_w = 0 if w != 0 else min(1, len(wins) - 1)
                entry["ref"] = cast(frame(wins[ref_w]["video"][0], 0)).cpu()
                prev = (frame(wins[w - 1]["video"][0], -1) if w > 0
                        else frame(wins[ref_w]["video"][0], 0))  # window 0: static start = ref
                entry["ctx"] = cast(prev).cpu()
            if syncnet:
                entry["mel"] = mel_for_window(c, w, cfg)
            torch.save(entry, f)
            out.append(f)
        del wins
        if i % 20 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return out


class TokenSet(Dataset):
    def __init__(self, files, face_cond: bool = False, syncnet: bool = False, continuous: bool = False):
        self.files = files
        self.face_cond = face_cond
        self.syncnet = syncnet
        self.continuous = continuous

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        d = torch.load(self.files[i], weights_only=True)
        if self.continuous:
            # v4: video latents stored [z_ch, Tv, h, w] -> model wants [tv, z_ch, h, w]
            video = d["video"].float().permute(1, 0, 2, 3)
            if "ref" in d:  # motion_ctx: grid = [identity, ctx, content...] on the slice axis
                video = torch.cat([d["ref"][None].float(), d["ctx"][None].float(), video], 0)
        else:
            video = d["video"].long()
            if "ref" in d:
                video = torch.cat([d["ref"][None].long(), d["ctx"][None].long(), video], 0)
        audio = d["audio"]
        audio = audio if audio.is_floating_point() else audio.long()
        out = [video, audio]
        if self.face_cond:
            out.append(d["struct"].long())
        if self.syncnet:
            out.append(d["mel"].float())  # [80, T] log-mel
        return tuple(out)


def optim_groups(model, wd):
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    plain = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    return [{"params": decay, "weight_decay": wd}, {"params": plain, "weight_decay": 0.0}]


def lr_at(step, warmup, total, lr, min_lr):
    if step < warmup:
        return lr * step / max(1, warmup)
    prog = min(1.0, (step - warmup) / max(1, total - warmup))
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * prog))


def unpack(batch, device, face_cond, syncnet=False):
    """batch is (video, audio[, struct][, mel]) — struct omitted when face_cond is false."""
    video, audio = batch[0].to(device), batch[1].to(device)
    i = 2
    struct = None
    if face_cond:
        struct = batch[i].to(device)
        i += 1
    mel = batch[i].to(device) if syncnet else None
    return video, audio, struct, mel


@torch.no_grad()
def token_diagnostic(model, loader, device, face_cond, syncnet=False, n_batches=8):
    model.eval()
    acc = collections.defaultdict(float)
    n = 0
    g = torch.Generator(device=device).manual_seed(1234)
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        video, audio, struct, _ = unpack(batch, device, face_cond, syncnet)
        B, tv, h, w = video.shape
        mask = per_slice_cosine_mask(B, tv, h * w, video.device, generator=g,
                                     ref_slices=getattr(model, "ref_slices", 1))
        hm, tgt = model(video, audio, struct=struct, mask=mask, cond_drop=0.0)
        _, parts = model.compute_loss(hm, tgt)
        for k, v in parts.items():
            if torch.is_tensor(v):
                acc[k] += float(v)
        n += 1
    model.train()
    return {k: v / max(n, 1) for k, v in acc.items()}


@torch.no_grad()
def evaluate_continuous(model, loader, device, face_cond, cfg, vidtok, syncnet=False):
    """v4 eval: generate content latents via flow decode, measure latent MSE + decoded-pixel PSNR.
    Returns (val_loss = latent MSE, val_acc = decoded PSNR dB) — PSNR is the perceptual proxy now."""
    import math as _m
    model.eval()
    mse = psnr_sum = n = 0
    n_cond = getattr(model, "ref_slices", 1)
    steps = cfg.get("decode_steps", 8)
    audio_cfg = cfg.get("audio_cfg", 1.0)
    for batch in loader:
        video, audio, struct, _ = unpack(batch, device, face_cond, syncnet)
        B, tv, z_ch, hh, ww = video.shape
        gen = model.generate_continuous(audio, video[:, 0], struct=struct,
                                        ctx=video[:, 1] if n_cond > 1 else None,
                                        steps=steps, audio_cfg=audio_cfg)  # [B, ctv, z_ch, hh, ww]
        gt = video[:, n_cond:]
        mse += F.mse_loss(gen, gt).item()
        pred_px = vidtok.decode_video(gen.permute(0, 2, 1, 3, 4).contiguous())
        gt_px = vidtok.decode_video(gt.permute(0, 2, 1, 3, 4).contiguous())
        mse_px = F.mse_loss(pred_px, gt_px).item()
        psnr_sum += 10 * _m.log10(4.0 / max(mse_px, 1e-8))  # pixels in [-1,1] -> range 2 -> 4
        n += 1
    model.train()
    return mse / max(1, n), psnr_sum / max(1, n)


@torch.no_grad()
def evaluate(model, loader, device, face_cond, cfg, syncnet=False):
    model.eval()
    loss = acc = n = 0
    gen_kw = dict(
        steps=cfg.get("decode_steps", 8),
        temperature=cfg.get("decode_temperature", 1.0),
        gumbel_temp=cfg.get("decode_gumbel", 4.5),
        schedule=cfg.get("decode_schedule", "cosine"),
        audio_cfg=cfg.get("audio_cfg", 1.0),
    )
    n_cond = getattr(model, "ref_slices", 1)
    # struct is the mesh of the TARGET frames, so it leaks the ground-truth lip shape and is
    # unavailable in audio-driven inference. On only for the reenactment path.
    use_struct = cfg.get("eval_with_struct", False)
    max_batches = cfg.get("eval_max_batches", 0)  # full-val MaskGIT decode is ~30% of wall clock
    for bi, batch in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        video, audio, struct, _ = unpack(batch, device, face_cond, syncnet)
        if not use_struct:
            struct = None
        gen = model.generate(audio, video[:, 0], struct=struct,
                             ctx=video[:, 1] if n_cond > 1 else None, **gen_kw)
        r = video.shape[-2] * video.shape[-1]
        hit = (gen.reshape(video.shape[0], -1)[:, n_cond * r:] ==
               video.reshape(video.shape[0], -1)[:, n_cond * r:]).float().mean().item()
        acc += hit
        loss += 1.0 - hit
        n += 1
    model.train()
    return loss / max(1, n), acc / max(1, n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs/train.yaml"))
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    for kv in args.set:
        k, v = kv.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    cfg["out_dir"] = f"{cfg['out_dir'].rstrip('/')}_{os.environ.get('SLURM_JOB_ID') or datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"checkpoints -> {cfg['out_dir']}")

    device = cfg["device"] if (cfg["device"] == "cpu" or torch.cuda.is_available()) else "cpu"
    torch.manual_seed(cfg["seed"])

    # data_glob may be a glob or a manifest file (one clip path per line, e.g. from
    # scripts/filter_clips.py).
    src = Path(cfg["data_glob"])
    clips = ([l.strip() for l in src.read_text().splitlines() if l.strip()]
             if src.suffix == ".txt" and src.exists() else sorted(glob.glob(cfg["data_glob"])))
    random.Random(cfg["seed"]).shuffle(clips)
    if cfg.get("max_clips"):
        clips = clips[: cfg["max_clips"]]
    # Split by SPEAKER (the per-video directory), not by clip: each speaker owns ~11 clips, so a
    # clip-level split puts the same identity in train and val and val measures memorisation.
    speakers = sorted({Path(c).parent.name for c in clips})
    random.Random(cfg["seed"]).shuffle(speakers)
    n_val_spk = max(1, int(len(speakers) * cfg["val_frac"])) if cfg["val_frac"] > 0 else 0
    val_spk = set(speakers[:n_val_spk])
    clips = [c for c in clips if Path(c).parent.name not in val_spk] \
            + [c for c in clips if Path(c).parent.name in val_spk]
    n_train = sum(1 for c in clips if Path(c).parent.name not in val_spk)
    cache_dir = REPO / cfg["cache_dir"]
    nwin = cfg["windows_per_clip"]
    need_encode = any(
        _clip_cache_complete(cache_dir, hashlib.md5(str(c).encode()).hexdigest(), nwin) is None
        for c in clips
    )
    if need_encode:
        vidtok = load_visual(cfg, device)
        if cfg.get("audio_encoder"):
            from sang.codec import load_wavlm
            audio_enc = load_wavlm(cfg["audio_encoder"], device=device)
    else:
        print("cache exists — patching mel before loading models")
        vidtok = audio_enc = None

    nclips = len(clips)
    val_files = build_cache(clips[n_train:], vidtok, audio_enc, cfg, cache_dir, "val")
    train_files = build_cache(clips[:n_train], vidtok, audio_enc, cfg, cache_dir, "train")
    print(f"{len(speakers) - len(val_spk)} train / {len(val_spk)} val speakers "
          f"({n_train} / {nclips - n_train} clips)")
    print(f"train windows={len(train_files)} val windows={len(val_files)}")
    if not train_files:
        raise ValueError(f"no training windows from {n_train} clips — lower val_frac or check data_glob")

    if vidtok is None:
        vidtok = load_visual(cfg, device)
        if cfg.get("audio_encoder"):
            from sang.codec import load_wavlm
            audio_enc = load_wavlm(cfg["audio_encoder"], device=device)
    fsq_codes = None
    if not cfg.get("continuous"):
        fsq_codes = getattr(vidtok.regularization, "implicit_codebook", None)
        if fsq_codes is not None:
            fsq_codes = fsq_codes.to(device).float()
            if cfg.get("factorized_head", True):
                from sang.fsq_codec import FSQIndexCodec
                FSQIndexCodec.from_codebook(fsq_codes).self_check()
    # Keep vidtok alive if perceptual losses need differentiable decode, or continuous mode needs
    # the decoder for pixel losses + PSNR eval
    keep_vidtok = (cfg.get("syncnet_loss", False)
                   or cfg.get("pixel_weight", 0.0) > 0 or cfg.get("continuous", False))
    if not keep_vidtok:
        del vidtok, audio_enc
        if device == "cuda":
            torch.cuda.empty_cache()
    else:
        vidtok = vidtok.to(device)
        if device == "cuda":
            torch.cuda.empty_cache()

    face_cond = cfg.get("face_cond", False)
    syncnet = cfg.get("syncnet_loss", False)
    continuous = cfg.get("continuous", False)
    cond_drop = cfg.get("cond_dropout", 0.0)
    ls = cfg.get("label_smoothing", 0.0)
    train_loader = DataLoader(TokenSet(train_files, face_cond, syncnet, continuous), batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=cfg["workers"], drop_last=len(train_files) > cfg["batch_size"])
    val_loader = None
    if val_files:
        val_bs = cfg.get("eval_batch_size", cfg["batch_size"])
        val_loader = DataLoader(TokenSet(val_files, face_cond, syncnet, continuous), batch_size=val_bs,
                                num_workers=cfg["workers"])

    model = build_talking_head(cfg, fsq_codes=fsq_codes).to(device)
    losses = build_losses(cfg, device)
    if losses:
        print(f"perceptual losses: {list(losses.keys())}")
    out_dir = REPO / cfg["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    best, best_step, bad = float("inf"), 0, 0
    start = 1
    resumed = None
    if cfg.get("resume"):
        path = Path(cfg["resume"])
        if not path.is_absolute():
            path = REPO / path
        resumed = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(resumed["model"], strict=False)
        start = resumed["step"] + 1
        # best travels inside the checkpoint: out_dir gets a fresh job suffix each launch, so
        # out_dir/best.json never existed on resume.
        best = resumed.get("best", float("inf"))
        best_step = resumed.get("best_step", 0)
        if "best" not in resumed:  # pre-fix checkpoint: fall back to its own directory
            meta = path.parent / "best.json"
            if meta.exists():
                b = json.loads(meta.read_text())
                best, best_step = b["val_loss"], b["step"]
        print(f"resume {path.name}: step {resumed['step']} -> {start}, best {best:.4f}@{best_step}")

    opt = torch.optim.AdamW(optim_groups(model, cfg["weight_decay"]), lr=cfg["lr"], betas=tuple(cfg["betas"]), eps=1e-8)
    if resumed is not None:
        if "opt" in resumed:
            opt.load_state_dict(resumed["opt"])
            print("  restored optimizer state")
        else:
            print("  WARNING: checkpoint predates optimizer-state saving; AdamW moments start at zero")
        del resumed
    print(f"params={sum(p.numel() for p in model.parameters()) / 1e6:.1f}M "
          f"factorized={getattr(model, 'factorized', False)} device={device}")
    print("val_acc = decoded-pixel PSNR (dB)" if getattr(model, "continuous", False)
          else "val_acc = generated motion-token accuracy (MaskGIT decode); val_loss = 1 - val_acc")

    # bf16 autocast: ~2x throughput on H100 and no GradScaler needed (bf16 has fp32 range).
    # Params stay fp32, so the optimizer and grad clipping are unaffected.
    use_amp = bool(cfg.get("bf16", True)) and device == "cuda"
    amp = ((lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if use_amp
           else __import__("contextlib").nullcontext)
    print(f"precision: {'bf16 autocast' if use_amp else 'fp32'}")

    total, accum, warmup = cfg["max_steps"], cfg["grad_accum"], cfg["warmup_steps"]
    log_every = cfg.get("log_every", 50)
    patience = cfg.get("patience", 0)
    meter, meter_n = collections.defaultdict(float), 0

    it = iter(train_loader)
    model.train()
    pbar = tqdm(range(start, total + 1), desc="train", initial=start - 1, total=total)
    for step in pbar:
        for g in opt.param_groups:
            g["lr"] = lr_at(step, warmup, total, cfg["lr"], cfg["min_lr"])
        opt.zero_grad(set_to_none=True)
        parts = {}
        for _ in range(accum):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(train_loader)
                batch = next(it)
            video, audio, struct, mel = unpack(batch, device, face_cond, syncnet)
            with amp():
                continuous = getattr(model, "continuous", False)
                if continuous:
                    loss, parts = model.forward_continuous(video, audio, struct=struct, cond_drop=cond_drop)
                else:
                    # One mask shared with the perceptual losses below, so they see the same masked
                    # regime as the CE loss rather than a fully-visible (copyable) grid.
                    shared_mask = model.sample_mask(video.shape[0], device)
                    h_masked, target = model(video, audio, struct=struct, cond_drop=cond_drop,
                                             mask=shared_mask,
                                             p_corrupt=cfg.get("context_corrupt", 0.0))
                    loss, parts = model.compute_loss(h_masked, target, label_smoothing=ls)

                    if (losses or cfg.get("pixel_weight", 0.0) > 0) and model.factorized:
                        B, tv, h, w = video.shape
                        r = h * w
                        n_cond = model.ref_slices
                        known = ~shared_mask  # NOT ones: that makes pred_px reachable by a copy
                        hs = model.hidden_states(video, audio, known, struct=struct)
                        h_content = hs[:, n_cond:].reshape(B * (tv - n_cond) * r, -1)
                        pred_px = model.decode_pixels(h_content, vidtok, (tv - n_cond, h, w))
                        gt_idx = video[:, n_cond:].reshape(B, (tv - n_cond), h, w)
                        with torch.no_grad():
                            gt_px = vidtok.decode(gt_idx, decode_from_indices=True)
                        if cfg.get("pixel_weight", 0.0) > 0:
                            if cfg.get("face_weight", False):
                                from sang.losses import face_lip_weight
                                wmap = face_lip_weight(pred_px.shape, pred_px.device)
                                p_loss = (wmap * (pred_px - gt_px).abs()).mean()
                            else:
                                p_loss = F.l1_loss(pred_px, gt_px)
                            loss = loss + cfg["pixel_weight"] * p_loss
                            parts["pixel"] = p_loss.detach()
                        if "syncnet" in losses and mel is not None:
                            s_loss = losses["syncnet"](pred_px, mel)
                            loss = loss + cfg.get("syncnet_weight", 0.1) * s_loss
                            parts["syncnet"] = s_loss.detach()

            (loss / accum).backward()
            for k, v in parts.items():
                if torch.is_tensor(v):
                    meter[k] += float(v)
            meter_n += 1

        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        opt.step()

        if step % log_every == 0:
            m = {k: v / max(meter_n, 1) for k, v in meter.items()}
            if getattr(model, "continuous", False):
                pbar.set_postfix(diff=f"{m.get('diff', 0):.3f}", pixel=f"{m.get('pixel', 0):.3f}",
                                 sync=f"{m.get('syncnet', 0):.3f}", lr=f"{opt.param_groups[0]['lr']:.1e}")
                extra = " ".join(f"{k} {m[k]:.4f}" for k in ("pixel", "syncnet") if k in m)
                tqdm.write(f"step {step:7d} lr {opt.param_groups[0]['lr']:.2e} "
                           f"diff {m.get('diff', 0):.4f} (lo {m.get('diff_lo', 0):.4f} hi {m.get('diff_hi', 0):.4f}) "
                           f"gnorm {gnorm:.3f}" + (f" | {extra}" if extra else ""))
            else:
                chance = math.log(cfg["codebook"])
                pbar.set_postfix(ce=f"{m.get('ce', 0):.3f}", acc_d=f"{m.get('acc_dim_mean', 0):.3f}",
                                 pixel=f"{m.get('pixel', 0):.3f}", sync=f"{m.get('syncnet', 0):.3f}",
                                 lr=f"{opt.param_groups[0]['lr']:.1e}")
                extra = " ".join(f"{k} {m[k]:.4f}" for k in ("pixel", "syncnet") if k in m)
                tqdm.write(f"step {step:7d} lr {opt.param_groups[0]['lr']:.2e} "
                           f"ce {m.get('ce', 0):.4f} (chance {chance:.4f}, gap {chance - m.get('ce', 0):+.4f}) "
                           f"acc_dim {m.get('acc_dim_mean', 0):.4f} acc_tok {m.get('acc_token', 0):.5f} "
                           f"gnorm {gnorm:.3f}" + (f" | {extra}" if extra else ""))
            meter.clear()
            meter_n = 0

        if val_loader and (step % cfg["eval_every"] == 0 or step == total):
            if getattr(model, "continuous", False):
                vloss, vacc = evaluate_continuous(model, val_loader, device, face_cond, cfg, vidtok, syncnet)
                diag = {}
            else:
                diag = token_diagnostic(model, val_loader, device, face_cond, syncnet)
                if diag:
                    tqdm.write(f"  [diag] val_ce {diag.get('ce', 0):.4f} acc_dim {diag.get('acc_dim_mean', 0):.4f} "
                               f"acc_tok {diag.get('acc_token', 0):.5f}")
                vloss, vacc = evaluate(model, val_loader, device, face_cond, cfg, syncnet)
            improved = vloss < best - 1e-4
            if improved:
                best, best_step, bad = vloss, step, 0
            else:
                bad += 1
            # optimizer state + best ride along so a resume is lossless
            ckpt = {"model": model.state_dict(), "opt": opt.state_dict(), "cfg": cfg, "step": step,
                    "val_loss": vloss, "val_acc": vacc, "best": best, "best_step": best_step}
            torch.save(ckpt, out_dir / "last.pt")
            if improved:
                torch.save(ckpt, out_dir / "best.pt")
                (out_dir / "best.json").write_text(json.dumps({"step": step, "val_loss": vloss, "val_acc": vacc,
                                                                "diag": diag}, indent=2))
            metric = "val_psnr" if getattr(model, "continuous", False) else "val_acc"
            tqdm.write(f"step {step} val_loss {vloss:.4f} {metric} {vacc:.4f} | best {best:.4f}@{best_step}")
            if patience > 0 and bad >= patience:
                tqdm.write(f"early stop at step {step} (best val_loss {best:.4f}@{best_step})")
                break
        elif step == total:
            ckpt = {"model": model.state_dict(), "opt": opt.state_dict(), "cfg": cfg, "step": step,
                    "best": best, "best_step": best_step}
            torch.save(ckpt, out_dir / "last.pt")
            torch.save(ckpt, out_dir / "best.pt")
    print(f"done. best val_loss {best:.4f}@{best_step} -> {out_dir / 'best.pt'} (best.json)")


if __name__ == "__main__":
    main()
