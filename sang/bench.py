"""Benchmark metrics the repo never had: CSIM, LSE-C/LSE-D, FID, FVD.

Shared by the motion-ceiling gate (M0) and the full evaluation (M1+), so the two report the
same numbers computed the same way. Nothing here is differentiable and nothing is used in a
training loop -- that is deliberate: the SyncNet term that WAS in the loop scored 0.20 against
real video's 0.42 and was being satisfied by mouth-band texture (docs/recovery_plan §2.2).

Every published number in this field is protocol-dependent (SadTalker's HDTF FID is 21.58 in
one paper and 71.95 in another), so nothing is comparable until the baselines are re-run
through THIS module.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SYNCNET = REPO / "third_party" / "syncnet_python"


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """PSNR between two uint8 arrays of the same shape."""
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else float(20 * np.log10(255.0) - 10 * np.log10(mse))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Mean per-frame SSIM over [T,H,W,3] uint8, reusing VidTok's implementation."""
    import torch
    sys.path.insert(0, str(REPO / "third_party"))
    from vidtok.modules.util import compute_ssim

    def prep(x):
        t = torch.from_numpy(np.ascontiguousarray(x)).permute(3, 0, 1, 2).float().div_(255.0)
        return t.unsqueeze(0)
    return float(compute_ssim(prep(a), prep(b)))


# ------------------------------------------------------------------ identity
class ArcFace:
    """InsightFace buffalo_l embeddings. Already a LivePortrait dependency, so no new install.

    ponytail: one detector per process, no batching -- CSIM runs over a few thousand frames per
    job, not millions. Upgrade path is `FaceAnalysis(..., allowed_modules=['recognition'])` plus
    feeding it the crops we already have."""

    def __init__(self, root: str | None = None, device: str = "cuda"):
        """The FULL buffalo_l pack. LivePortrait's HF upload ships only det_10g + 2d106det, no
        recognition model, so CSIM from that folder is nan on every frame. install_motion.sh
        downloads the full pack to third_party/insightface_full on the login node."""
        from insightface.app import FaceAnalysis
        root = root or str(REPO / "third_party" / "insightface_full")
        if not (Path(root) / "models" / "buffalo_l" / "w600k_r50.onnx").exists():
            raise FileNotFoundError(f"no ArcFace recognition model under {root}; run bash_scripts/install_motion.sh")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
        self.app = FaceAnalysis(name="buffalo_l", root=root, providers=providers,
                                allowed_modules=["detection", "recognition"])
        self.app.prepare(ctx_id=0 if device == "cuda" else -1, det_size=(320, 320))

    def embed(self, frame_rgb: np.ndarray) -> np.ndarray | None:
        faces = self.app.get(np.ascontiguousarray(frame_rgb[:, :, ::-1]))   # insightface wants BGR
        if not faces:
            return None
        f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        return f.normed_embedding


def csim(arc: ArcFace, source_rgb: np.ndarray, frames_rgb: np.ndarray, stride: int = 5) -> float:
    """Mean cosine similarity between the source face and every `stride`-th generated frame.

    nan when no face is detectable in the source or in any sampled frame -- report it as nan,
    never as 0.0, or a detector failure reads as an identity failure."""
    ref = arc.embed(source_rgb)
    if ref is None:
        return float("nan")
    sims = [float(np.dot(ref, e)) for e in
            (arc.embed(frames_rgb[i]) for i in range(0, len(frames_rgb), max(1, stride)))
            if e is not None]
    return float(np.mean(sims)) if sims else float("nan")


# ------------------------------------------------------------------ lip sync
def write_mp4(frames_rgb: np.ndarray, path: Path, fps: float = 25.0,
              audio: Path | None = None) -> Path:
    """[T,H,W,3] uint8 -> an mp4 that syncnet_python can ingest (yuv420p, optional audio mux)."""
    import imageio.v2 as imageio

    path = Path(path)
    silent = path.with_suffix(".silent.mp4") if audio else path
    with imageio.get_writer(str(silent), fps=fps, codec="libx264",
                            output_params=["-pix_fmt", "yuv420p", "-crf", "17"]) as w:
        for f in frames_rgb:
            w.append_data(np.ascontiguousarray(f))
    if audio:
        import imageio_ffmpeg                      # ships its own binary: no system ffmpeg needed
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
                        "-i", str(silent), "-i", str(audio),
                        "-c:v", "copy", "-c:a", "aac", "-shortest", str(path)], check=True)
        silent.unlink(missing_ok=True)
    return path


def lse(video: Path, workdir: Path | None = None, ref: str = "sang", verbose: bool = False) -> tuple[float, float, float]:
    """(offset, LSE-D, LSE-C) for one mp4 WITH audio, via joonson/syncnet_python.

    LSE-D is 'Min dist' and LSE-C is 'Confidence' in run_syncnet.py's output -- the two numbers
    every paper in the field reports. Returns nans when the pipeline finds no usable face track
    rather than raising, so one bad clip does not abort a 300-clip sweep.

    NOTE: parsed from stdout + stderr (results moved to stderr upstream in 2026-04). Check the format ONCE against your checkout (`sh download_model.sh`
    then run it on a known clip) before trusting a sweep -- upstream has changed the wording."""
    if not (SYNCNET / "run_syncnet.py").exists():
        raise FileNotFoundError(f"syncnet_python not at {SYNCNET}; see bash_scripts/install_motion.sh")
    tmp = Path(workdir or tempfile.mkdtemp(prefix="lse_"))
    (tmp / "work").mkdir(parents=True, exist_ok=True)
    # absolute: syncnet_python runs with cwd=its own checkout, so a relative path does not resolve
    common = ["--data_dir", str(tmp / "work"), "--reference", ref, "--videofile", str(Path(video).resolve())]
    try:
        subprocess.run([sys.executable, "run_pipeline.py", *common], cwd=SYNCNET,
                       check=True, capture_output=True, text=True, timeout=600)
        r = subprocess.run([sys.executable, "run_syncnet.py", *common], cwd=SYNCNET,
                           check=True, capture_output=True, text=True, timeout=600)
        # syncnet_python PR #78 (2026-04-17) moved print -> logging, so the result lines ("AV offset",
        # "Min dist", "Confidence") now arrive on STDERR; reading stdout alone gave nan every time.
        out = r.stdout + "\n" + r.stderr
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        tail = ((getattr(e, "stderr", "") or "") + (getattr(e, "stdout", "") or ""))[-800:]
        print(f"  [lse] syncnet_python failed on {Path(video).name}: {type(e).__name__}\n{tail}", flush=True)
        return float("nan"), float("nan"), float("nan")

    res, lines = parse_syncnet(out)
    if verbose:
        print("  [lse] matched lines:\n    " + "\n    ".join(lines), flush=True)
    if any(np.isnan(r) for r in res):
        print(f"  [lse] could not parse run_syncnet.py output for {Path(video).name}:\n{out[-800:]}", flush=True)
    return res


def parse_syncnet(out: str) -> tuple[tuple[float, float, float], list[str]]:
    """(AV offset, Min dist, Confidence) from run_syncnet.py's text, plus the lines they came from.

    The number is the one right AFTER the label. Since syncnet_python moved to logging (2026-04), each
    line starts with a timestamp, and taking the line's first number returned the hour -- a run at
    10:xx read (10.0, 10.0, 10.0)."""
    import re
    vals, lines = [], []
    for label in ("AV offset", "Min dist", "Confidence"):
        m = re.search(rf"{re.escape(label)}\s*:?\s*([-+]?\d+(?:\.\d+)?)", out, flags=re.IGNORECASE)
        vals.append(float(m.group(1)) if m else float("nan"))
        if m:
            start = out.rfind("\n", 0, m.start()) + 1
            end = out.find("\n", m.end())
            lines.append(out[start:end if end >= 0 else None].strip())
    return tuple(vals), lines


# ------------------------------------------------------------------ distribution metrics
def fid(real_frames: list[np.ndarray], gen_frames: list[np.ndarray], device: str = "cuda") -> float:
    """Frechet Inception Distance over all frames, pytorch-fid's Inception-v3 pool3 features."""
    import torch
    from pytorch_fid.inception import InceptionV3

    model = InceptionV3([3]).to(device).eval()

    def stats(frames):
        feats = []
        with torch.no_grad():
            for i in range(0, len(frames), 32):
                x = np.stack(frames[i:i + 32]).astype(np.float32) / 255.0
                t = torch.from_numpy(x).permute(0, 3, 1, 2).to(device)
                t = torch.nn.functional.interpolate(t, size=(299, 299), mode="bilinear", align_corners=False)
                feats.append(model(t)[0].squeeze(-1).squeeze(-1).cpu().numpy())
        f = np.concatenate(feats)
        return f.mean(0), np.cov(f, rowvar=False)
    return _frechet(*stats(real_frames), *stats(gen_frames))


def _frechet(mu1, s1, mu2, s2, eps: float = 1e-6) -> float:
    from scipy import linalg
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm((s1 + eps * np.eye(len(s1))).dot(s2 + eps * np.eye(len(s2))), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


# ------------------------------------------------------------------ streaming features (HDTF eval)
I3D_URL = "https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1"   # StyleGAN-V's I3D port
I3D_PATH = REPO / "third_party" / "fvd" / "i3d_torchscript.pt"


class InceptionFeats:
    """pytorch-fid Inception-v3 pool3 features (2048-d), frames [N,H,W,3] uint8 -> [N, 2048].
    Same network and resize as `fid` above, but per batch, so features from many clips can be
    pooled without holding every frame in memory."""

    def __init__(self, device: str = "cuda"):
        from pytorch_fid.inception import InceptionV3
        self.m, self.device = InceptionV3([3]).to(device).eval(), device

    def __call__(self, frames: np.ndarray, batch: int = 32) -> np.ndarray:
        import torch
        out = []
        with torch.no_grad():
            for i in range(0, len(frames), batch):
                t = torch.from_numpy(np.ascontiguousarray(frames[i:i + batch])).permute(0, 3, 1, 2).float().div_(255.0)
                t = torch.nn.functional.interpolate(t.to(self.device), size=(299, 299), mode="bilinear", align_corners=False)
                out.append(self.m(t)[0].squeeze(-1).squeeze(-1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 2048))


class I3DFeats:
    """FVD features: the Kinetics-400 I3D TorchScript port used by StyleGAN-V and
    common_metrics_on_video_quality. A video [T,H,W,3] uint8 is cut into non-overlapping
    `clip_len`-frame clips (FVD-16 by default); each is resized so the short side is 224,
    centre-cropped to 224x224, scaled to [-1, 1], and mapped to a 400-d feature
    (rescale=False, resize=True, return_features=True). Download once on the login node:
    bash bash_scripts/install_eval.sh."""

    def __init__(self, device: str = "cuda", path: Path = I3D_PATH):
        import torch
        if not Path(path).exists():
            raise FileNotFoundError(f"I3D weights not at {path}; run bash bash_scripts/install_eval.sh")
        self.m, self.device = torch.jit.load(str(path)).eval().to(device), device

    def __call__(self, video: np.ndarray, clip_len: int = 16, batch: int = 8) -> np.ndarray:
        import torch
        import torch.nn.functional as F
        n = len(video) // clip_len
        clips, out = [], []
        for k in range(n):
            x = torch.from_numpy(np.ascontiguousarray(video[k * clip_len:(k + 1) * clip_len])).float().div_(255.0)
            x = x.permute(0, 3, 1, 2)                                            # T, C, H, W
            h, w = x.shape[-2:]
            s = 224 / min(h, w)
            x = F.interpolate(x, size=(round(h * s), round(w * s)), mode="bilinear", align_corners=False)
            h, w = x.shape[-2:]
            top, left = (h - 224) // 2, (w - 224) // 2
            x = x[..., top:top + 224, left:left + 224]
            clips.append(((x - 0.5) * 2).permute(1, 0, 2, 3))                 # C, T, H, W in [-1, 1]
        with torch.no_grad():
            for i in range(0, len(clips), batch):
                b = torch.stack(clips[i:i + batch]).to(self.device)
                out.append(self.m(b, rescale=False, resize=True, return_features=True).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 400))


def frechet(a: np.ndarray, b: np.ndarray) -> float:
    """Frechet distance between two feature sets [N, D] (FID / FVD)."""
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    return _frechet(a.mean(0), np.cov(a, rowvar=False), b.mean(0), np.cov(b, rowvar=False))


def fvd(real_videos: list[np.ndarray], gen_videos: list[np.ndarray], device: str = "cuda", clip_len: int = 16) -> float:
    """FVD-16 between two lists of [T,H,W,3] uint8 videos (non-overlapping 16-frame clips)."""
    i3d = I3DFeats(device)
    return frechet(np.concatenate([i3d(v, clip_len) for v in real_videos]),
                   np.concatenate([i3d(v, clip_len) for v in gen_videos]))


def background_motion(real: np.ndarray, gen: np.ndarray, quiet_pct: float = 40.0,
                      texture_pct: float = 70.0, eps: float = 0.05) -> dict:
    """How much the BACKGROUND moves in `gen` compared with `real` ([T,H,W,3] uint8, same size, frames
    aligned). Pixels are chosen where warping is visible and real video is still: off a central head
    box (rows 10-80%, cols 20-80%), among the quietest `quiet_pct`% of the real video's pixels, and
    textured (spatial gradient above the `texture_pct` percentile there) -- warping a flat wall moves
    nothing visible. Motion = mean |frame-to-frame difference| in gray levels on those pixels.
    Returns {bg_real, bg_gen, bg_ratio}; ratio ~1 = as still as the real video (eps floors a
    perfectly static real background, e.g. a studio backdrop, so the ratio stays finite)."""
    w = np.array([0.299, 0.587, 0.114], np.float32)
    n = min(len(real), len(gen))
    g = [x[:n].astype(np.float32) @ w for x in (real, gen)]
    m = [np.abs(np.diff(x, axis=0)).mean(0) for x in g]
    H, W = m[0].shape
    head = np.zeros((H, W), bool)
    head[int(.1 * H):int(.8 * H), int(.2 * W):int(.8 * W)] = True
    static = (m[0] <= np.percentile(m[0], quiet_pct)) & ~head
    gy, gx = np.gradient(g[0][0])
    grad = gx ** 2 + gy ** 2
    if not static.any():
        return {"bg_real": float("nan"), "bg_gen": float("nan"), "bg_ratio": float("nan")}
    tex = static & (grad > np.percentile(grad[static], texture_pct))
    bg_real, bg_gen = float(m[0][tex].mean()), float(m[1][tex].mean())
    return {"bg_real": bg_real, "bg_gen": bg_gen, "bg_ratio": bg_gen / max(bg_real, eps)}


def self_check() -> None:
    a = (np.random.default_rng(0).random((4, 32, 32, 3)) * 255).astype(np.uint8)
    assert psnr(a, a) == float("inf")
    b = a.astype(np.int16) + 8
    got = psnr(a, np.clip(b, 0, 255).astype(np.uint8))
    assert 28.0 < got < 32.0, got                     # 8/255 rms -> ~30 dB
    mu = np.zeros(4); cov = np.eye(4)
    # eps regularisation biases the trace term by -2*d*eps, so 0 is only approached to ~1e-5.
    assert abs(_frechet(mu, cov, mu, cov)) < 1e-4
    assert abs(_frechet(mu, cov, mu + 1, cov) - 4.0) < 1e-4
    rng = np.random.default_rng(1)
    still = np.broadcast_to((rng.random((1, 64, 64, 3)) * 255).astype(np.uint8), (10, 64, 64, 3))
    shaky = np.stack([np.roll(still[0], i % 2, axis=1) for i in range(10)])      # 1-px wobble everywhere
    r = background_motion(still, shaky)
    assert r["bg_real"] == 0.0 and r["bg_gen"] > 10 and background_motion(still, still)["bg_ratio"] == 0.0
    log = ("2026-09-30 10:12:33,512 - INFO - AV offset: \t3\n"
           "2026-09-30 10:12:33,513 - INFO - Min dist: \t5.353\n"
           "2026-09-30 10:12:33,513 - INFO - Confidence: \t10.021\n")
    assert parse_syncnet(log)[0] == (3.0, 5.353, 10.021), parse_syncnet(log)
    assert parse_syncnet("AV offset: \t-2\nMin dist: \t7.1\nConfidence: \t6.5")[0] == (-2.0, 7.1, 6.5)
    print("sang/bench.py self-check ok")


if __name__ == "__main__":
    self_check()
