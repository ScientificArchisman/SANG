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


def lse(video: Path, workdir: Path | None = None, ref: str = "sang") -> tuple[float, float, float]:
    """(offset, LSE-D, LSE-C) for one mp4 WITH audio, via joonson/syncnet_python.

    LSE-D is 'Min dist' and LSE-C is 'Confidence' in run_syncnet.py's output -- the two numbers
    every paper in the field reports. Returns nans when the pipeline finds no usable face track
    rather than raising, so one bad clip does not abort a 300-clip sweep.

    NOTE: parsed from stdout. Check the format ONCE against your checkout (`sh download_model.sh`
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
        out = subprocess.run([sys.executable, "run_syncnet.py", *common], cwd=SYNCNET,
                             check=True, capture_output=True, text=True, timeout=600).stdout
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        tail = ((getattr(e, "stderr", "") or "") + (getattr(e, "stdout", "") or ""))[-800:]
        print(f"  [lse] syncnet_python failed on {Path(video).name}: {type(e).__name__}\n{tail}", flush=True)
        return float("nan"), float("nan"), float("nan")

    def grab(label: str) -> float:
        for line in out.splitlines():
            if label.lower() in line.lower():
                for tok in line.replace(":", " ").split():
                    try:
                        return float(tok)
                    except ValueError:
                        continue
        return float("nan")
    res = grab("AV offset"), grab("Min dist"), grab("Confidence")
    if any(np.isnan(r) for r in res):
        print(f"  [lse] could not parse run_syncnet.py output for {Path(video).name}:\n{out[-800:]}", flush=True)
    return res


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


def fvd(*args, **kwargs) -> float:
    """FVD-16: I3D features over 16-frame sliding windows, as FLOAT computes it.

    ponytail: NOT implemented -- it needs the I3D checkpoint, and the M0 gate does not use it.
    Land it with M1's eval_full, alongside FID above, and report the window/stride used."""
    raise NotImplementedError("FVD lands with M1; see docs/recovery_plan_2026-09-16.md 5.6")


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
    print("sang/bench.py self-check ok")


if __name__ == "__main__":
    self_check()
