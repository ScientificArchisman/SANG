"""LivePortrait as a frozen motion codec: frames -> 70-d motion vectors -> frames.

    m = [scale, yaw, pitch, roll, t_x, t_y, t_z, exp_1..exp_63]      (KDTalker's layout)

This is the only representation SANG-M predicts. Appearance never enters the model: it is
carried by the source frame's `f_s` / `x_s` tensors, which stay inside the frozen renderer.

Upstream symbols follow KwaiVGI/LivePortrait as of 2026-09. Anything that could have moved is
marked `# API:` and checked once by `api_check()` -- run that before a long job, not after.
"""
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
LP = REPO / "third_party" / "LivePortrait"

M_DIM = 70
EXP_DIM = 63          # 21 keypoints x 3
LAYOUT = ("scale", "yaw", "pitch", "roll", "t", "exp")
WEIGHTS = LP / "pretrained_weights"

# Keypoint roles, copied from upstream src/live_portrait_pipeline.py (animation_region branches),
# read 2026-09-23. LivePortrait's own expression transfer only ever drives EXPR_KP in full; the other
# 8 keypoints (0, 3, 4, 5, 7, 8, 9, 10) are held from the source, i.e. they carry face shape, not
# expression. BROW is the remainder of EXPR_KP after the eye and lip sets.
LIP_KP = (6, 12, 14, 17, 19, 20)
EYE_KP = (11, 13, 15, 16, 18)
EXPR_KP = (1, 2, 6, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20)
BROW_KP = tuple(k for k in EXPR_KP if k not in LIP_KP and k not in EYE_KP)   # (1, 2)

# The training target, following AVTR-1 (arXiv:2609.22913 sec. 2.1): head rotation plus the 39
# expression coordinates of EXPR_KP, expressed in the HEAD frame so a mouth shape does not change
# with pose. Scale, translation and the 24 shape coordinates are taken from the source at render.
T_DIM = 3 + 3 * len(EXPR_KP)                                                  # 42


def _span(kps: tuple[int, ...]) -> list[int]:
    """Target-vector columns of the given keypoints (after the 3 rotation columns)."""
    pos = {k: i for i, k in enumerate(EXPR_KP)}
    return [3 + 3 * pos[k] + a for k in kps for a in range(3)]


REGIONS = {"rot": [0, 1, 2], "brow": _span(BROW_KP), "eyes": _span(EYE_KP), "mouth": _span(LIP_KP)}


def rotation_matrix(pitch: torch.Tensor, yaw: torch.Tensor, roll: torch.Tensor) -> torch.Tensor:
    """Degrees [N] -> [N,3,3], LivePortrait's convention: (Rz @ Ry @ Rx) transposed, row vectors.

    Ported from upstream src/utils/camera.py so training never imports the renderer."""
    x, y, z = (a.reshape(-1).float() * (torch.pi / 180.0) for a in (pitch, yaw, roll))
    o, n = torch.ones_like(x), torch.zeros_like(x)
    rx = torch.stack([o, n, n, n, x.cos(), -x.sin(), n, x.sin(), x.cos()], 1).view(-1, 3, 3)
    ry = torch.stack([y.cos(), n, y.sin(), n, o, n, -y.sin(), n, y.cos()], 1).view(-1, 3, 3)
    rz = torch.stack([z.cos(), -z.sin(), n, z.sin(), z.cos(), n, n, n, o], 1).view(-1, 3, 3)
    return (rz @ ry @ rx).transpose(1, 2)


def _rot(m: torch.Tensor) -> torch.Tensor:
    return rotation_matrix(m[:, 2], m[:, 1], m[:, 3])        # layout is (scale, yaw, pitch, roll, ...)


def to_target(m: torch.Tensor) -> torch.Tensor:
    """[N,70] motion -> [N,42] target: (pitch, yaw, roll) + head-frame exp of EXPR_KP."""
    m = m.float()
    exp_head = m[:, 7:M_DIM].reshape(-1, 21, 3) @ _rot(m).transpose(1, 2)   # delta @ R^T
    return torch.cat([m[:, [2, 1, 3]], exp_head[:, list(EXPR_KP)].reshape(-1, 3 * len(EXPR_KP))], 1)


def from_target(y: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    """[N,42] target + [N or 1,70] base motion -> [N,70], ready for MotionCodec.render(relative=False).

    Scale, translation and the 24 shape coordinates come from `base` (the source frame). Those
    shape coordinates are held fixed in the HEAD frame, so they rotate rigidly with the new pose."""
    y, base = y.float(), base.float().expand(y.shape[0], -1)
    exp_head = base[:, 7:M_DIM].reshape(-1, 21, 3) @ _rot(base).transpose(1, 2)
    exp_head = exp_head.clone()
    exp_head[:, list(EXPR_KP)] = y[:, 3:].reshape(-1, len(EXPR_KP), 3)
    out = base.clone()
    out[:, 2], out[:, 1], out[:, 3] = y[:, 0], y[:, 1], y[:, 2]
    out[:, 7:M_DIM] = (exp_head @ _rot(out)).reshape(-1, EXP_DIM)            # back to camera frame
    return out


# Source-image crop convention, used for BOTH the cache and the inference-time reference, so a
# cached motion trajectory and a freshly cropped reference live in the same frame. Mixing the
# source (2.3, -0.125) and driving (2.2, -0.1) conventions shifts every keypoint ~2% of the crop.
CROP = dict(dsize=512, scale=2.3, vx_ratio=0.0, vy_ratio=-0.125)


def _require_repo() -> None:
    if not (LP / "src" / "live_portrait_wrapper.py").exists():
        raise FileNotFoundError(
            f"LivePortrait is not installed at {LP}.\n"
            "Run:  bash bash_scripts/install_motion.sh"
        )
    if str(LP) not in sys.path:
        sys.path.insert(0, str(LP))


def kp_info_to_vec(kp: dict) -> torch.Tensor:
    """LivePortrait kp_info (angles already refined to degrees) -> [B, 70]."""
    return torch.cat([
        kp["scale"].reshape(-1, 1),
        kp["yaw"].reshape(-1, 1),
        kp["pitch"].reshape(-1, 1),
        kp["roll"].reshape(-1, 1),
        kp["t"].reshape(-1, 3),
        kp["exp"].reshape(-1, EXP_DIM),
    ], dim=1)


def vec_to_kp_info(m: torch.Tensor) -> dict:
    """[B, 70] -> kp_info dict. Inverse of kp_info_to_vec (exp reshaped back to [B, 21, 3])."""
    return {"scale": m[:, 0:1], "yaw": m[:, 1:2], "pitch": m[:, 2:3], "roll": m[:, 3:4],
            "t": m[:, 4:7], "exp": m[:, 7:M_DIM].reshape(-1, 21, 3)}


def as3x3(M: np.ndarray) -> np.ndarray:
    """Affine as 3x3. Upstream crop_image returns M_c2o as 3x3; accept 2x3 too."""
    M = np.asarray(M, dtype=np.float64)
    return M if M.shape == (3, 3) else np.vstack([M[:2], [0.0, 0.0, 1.0]])


def o2c(M_c2o: np.ndarray) -> np.ndarray:
    """Crop->original affine -> the original->crop affine cv2.warpAffine wants (2x3)."""
    return np.linalg.inv(as3x3(M_c2o))[:2]


def interp_boxes(boxes: np.ndarray, idx: list[int], n: int) -> np.ndarray:
    """Affine crop matrices sampled at frames `idx` -> one per frame, linearly interpolated.

    Detecting a box per frame makes the crop jitter at ~1 px/frame, which the motion extractor
    reads as head translation; detecting once per clip drifts off the face. One box per second,
    interpolated, is the compromise KDTalker's extractor also uses."""
    if len(idx) != len(boxes):
        raise ValueError(f"{len(idx)} indices vs {len(boxes)} boxes")
    out = np.empty((n,) + boxes.shape[1:], dtype=np.float32)
    if len(idx) == 1:
        out[:] = boxes[0]
        return out
    for a, b, Ma, Mb in zip(idx[:-1], idx[1:], boxes[:-1], boxes[1:]):
        span = max(1, b - a)
        for j in range(a, min(b + 1, n)):
            w = (j - a) / span
            out[j] = (1.0 - w) * Ma + w * Mb
    out[idx[-1]:] = boxes[-1]
    return out


def decode_clip(path: str, max_frames: int, fps: float = 25.0) -> np.ndarray:
    """First `max_frames` frames of a clip resampled to `fps` -> [T,H,W,3] uint8. Frame f sits at
    container time f / fps, which is the clock the cached WavLM ticks use (2 ticks per frame)."""
    from sang.video import open_video
    vr = open_video(path)
    stride = vr.get_avg_fps() / fps
    n = min(max_frames, int(len(vr) / stride))
    if n < 8:
        raise ValueError(f"{len(vr)} frames at {vr.get_avg_fps():.1f} fps is too short")
    return vr.get_batch([int(round(i * stride)) for i in range(n)]).asnumpy()


class MotionCodec:
    """Frozen LivePortrait. `extract` maps frames -> motion; `render` maps source + motion -> frames.

    Nothing here trains. Instantiate once per process: it holds five networks plus an
    InsightFace detector (~1.8 GB of VRAM measured on the 4090 in the upstream repo)."""

    def __init__(self, device: str = "cuda", weights: Path = WEIGHTS, half: bool = False):
        _require_repo()
        from src.config.crop_config import CropConfig            # noqa: E402
        from src.config.inference_config import InferenceConfig  # noqa: E402
        from src.live_portrait_wrapper import LivePortraitWrapper  # noqa: E402
        from src.utils.cropper import Cropper                     # noqa: E402

        base = Path(weights) / "liveportrait"
        cfg = InferenceConfig()
        cfg.checkpoint_F = str(base / "base_models" / "appearance_feature_extractor.pth")
        cfg.checkpoint_M = str(base / "base_models" / "motion_extractor.pth")
        cfg.checkpoint_G = str(base / "base_models" / "spade_generator.pth")
        cfg.checkpoint_W = str(base / "base_models" / "warping_module.pth")
        cfg.checkpoint_S = str(base / "retargeting_models" / "stitching_retargeting_module.pth")
        cfg.flag_use_half_precision = bool(half)
        # LivePortraitWrapper and Cropper choose their device from flag_force_cpu / device_id,
        # not from anything passed in (upstream live_portrait_wrapper.py, cropper.py).
        cfg.flag_force_cpu = device == "cpu"
        cfg.device_id = 0
        for p in (cfg.checkpoint_F, cfg.checkpoint_M, cfg.checkpoint_G, cfg.checkpoint_W, cfg.checkpoint_S):
            if not Path(p).exists():
                raise FileNotFoundError(f"missing renderer weight {p}; see bash_scripts/install_motion.sh")

        self.lp = LivePortraitWrapper(inference_cfg=cfg)
        crop_cfg = CropConfig()
        crop_cfg.insightface_root = str(Path(weights) / "insightface")
        crop_cfg.landmark_ckpt_path = str(base / "landmark.onnx")
        for k, v in CROP.items():
            setattr(crop_cfg, k, v)
        self.crop_cfg = crop_cfg
        crop_cfg.flag_force_cpu = device == "cpu"
        self.cropper = Cropper(crop_cfg=crop_cfg, device_id=0, flag_force_cpu=device == "cpu")
        self.device = device
        api_check(self)

    # -------------------------------------------------------------------- cropping
    def crop_one(self, frame: np.ndarray) -> dict:
        """One RGB uint8 frame -> {'img_crop_256x256', 'M_c2o'} in the SOURCE convention."""
        out = self.cropper.crop_source_image(frame, self.crop_cfg)  # API: name + return keys
        if out is None:
            raise ValueError("no face found")
        return out

    def crop_clip(self, frames: np.ndarray, every: int = 25) -> tuple[np.ndarray, np.ndarray]:
        """[T,H,W,3] uint8 -> ([T,256,256,3] uint8 crops, [T,3,3] crop-to-original affines).

        Boxes are detected every `every` frames (one per second at 25 fps) and interpolated."""
        import cv2
        from src.utils.crop import _transform_img  # API: private helper, used by upstream too

        T = len(frames)
        idx = sorted({*range(0, T, max(1, every)), T - 1})
        boxes, keep = [], []
        for i in idx:
            try:
                boxes.append(as3x3(self.crop_one(np.ascontiguousarray(frames[i]))["M_c2o"]))
                keep.append(i)
            except Exception:
                continue
        if not boxes:
            raise ValueError("no face in any probed frame")
        M = interp_boxes(np.stack(boxes), keep, T)

        crops = np.empty((T, 256, 256, 3), dtype=np.uint8)
        for j in range(T):
            big = _transform_img(frames[j], o2c(M[j]), dsize=CROP["dsize"])
            crops[j] = cv2.resize(big, (256, 256), interpolation=cv2.INTER_AREA)
        return crops, M

    # -------------------------------------------------------------------- extraction
    @torch.no_grad()
    def motion_from_crops(self, crops: np.ndarray, batch: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
        """[T,256,256,3] uint8 -> ([T,70] motion, [T,63] canonical keypoints x_c), fp32.

        x_c is the identity-bearing quantity: Ditto, KDTalker and AVTR-1 all condition on it."""
        x = torch.from_numpy(np.ascontiguousarray(crops)).permute(0, 3, 1, 2).float().div_(255.0)
        ms, kps = [], []
        for i in range(0, len(x), batch):
            kp = self.lp.get_kp_info(x[i:i + batch].to(self.device))  # API: refined -> degrees
            ms.append(kp_info_to_vec(kp).float().cpu())
            kps.append(kp["kp"].reshape(-1, EXP_DIM).float().cpu())
        return torch.cat(ms), torch.cat(kps)

    @torch.no_grad()
    def extract(self, frames: np.ndarray, batch: int = 64, every: int = 25) -> dict:
        """[T,H,W,3] uint8 RGB at 25 fps -> {'m': [T,70], 'kp': [T,63], 'M_c2o', 'crops'}."""
        crops, M = self.crop_clip(frames, every=every)
        m, kp = self.motion_from_crops(crops, batch=batch)
        return {"m": m.half(), "kp": kp.half(), "M_c2o": torch.from_numpy(M), "crops": crops}

    @torch.no_grad()
    def source_motion(self, source: np.ndarray, normalize_lip: bool = True) -> dict:
        """One source image -> {'m': [1,70], 'kp': [1,63]}, optionally with the lips closed.

        LivePortrait's own flag_normalize_lip (upstream live_portrait_pipeline.py): a reference
        photo caught mid-vowel otherwise biases every frame of the animation. The retarget module
        returns a keypoint offset d; since x = s(x_c R + delta) + t, the same offset in expression
        space is d / s, applied in the camera frame where delta lives."""
        c = self.crop_one(np.ascontiguousarray(source))
        m, kp = self.motion_from_crops(c["img_crop_256x256"][None])
        if normalize_lip and c.get("lmk_crop") is not None:
            src = self.lp.prepare_source(c["img_crop_256x256"])
            x_s = self.lp.transform_keypoint(self.lp.get_kp_info(src))
            ratio = self.lp.calc_combined_lip_ratio([0.0], c["lmk_crop"])       # API
            thresh = getattr(self.lp.inference_cfg, "lip_normalize_threshold", 0.03)
            if float(ratio[0][0]) >= thresh:
                d = self.lp.retarget_lip(x_s, ratio).reshape(1, 21, 3).float().cpu()  # API
                m = m.clone()
                m[:, 7:M_DIM] += (d / m[:, 0:1].reshape(1, 1, 1)).reshape(1, EXP_DIM)
        return {"m": m, "kp": kp}

    # -------------------------------------------------------------------- rendering
    @torch.no_grad()
    def source_state(self, source: np.ndarray) -> dict:
        """Source RGB frame -> everything the renderer needs that does not depend on the audio."""
        from src.utils.camera import get_rotation_matrix  # noqa: E402

        c = self.crop_one(np.ascontiguousarray(source))
        src = self.lp.prepare_source(c["img_crop_256x256"])
        info = self.lp.get_kp_info(src)
        return {"info": info,
                "R_s": get_rotation_matrix(info["pitch"], info["yaw"], info["roll"]),
                "f_s": self.lp.extract_feature_3d(src),
                "x_s": self.lp.transform_keypoint(info),
                "x_c": info["kp"],
                "M_c2o": c["M_c2o"]}

    @torch.no_grad()
    def render(self, source: np.ndarray, m: torch.Tensor, relative: bool = True,
               stitch: bool = True, state: dict | None = None) -> np.ndarray:
        """source [H,W,3] uint8 + motion [T,70] -> [T,512,512,3] uint8 crops.

        `relative=True` is KDTalker's / LivePortrait's retargeting: the driving trajectory is
        applied as a delta from its own first frame onto the source pose, so the source keeps its
        identity and rest pose. With `relative=False` the source is forced into the driving pose,
        which is the right setting only for measuring the representation ceiling on the same clip."""
        from src.utils.camera import get_rotation_matrix  # noqa: E402

        s = state or self.source_state(source)
        d0 = vec_to_kp_info(m[:1].float().to(self.device))
        R_d0 = get_rotation_matrix(d0["pitch"], d0["yaw"], d0["roll"])
        out = []
        for i in range(m.shape[0]):
            d = vec_to_kp_info(m[i:i + 1].float().to(self.device))
            R_d = get_rotation_matrix(d["pitch"], d["yaw"], d["roll"])
            if relative:
                R_new = (R_d @ R_d0.permute(0, 2, 1)) @ s["R_s"]
                delta = s["info"]["exp"] + (d["exp"] - d0["exp"])
                scale = s["info"]["scale"] * (d["scale"] / d0["scale"])
                t = s["info"]["t"] + (d["t"] - d0["t"])
            else:
                R_new, delta, scale, t = R_d, d["exp"], d["scale"], d["t"]
            t = t.clone()
            t[:, 2] = 0.0                                     # LivePortrait zeroes t_z
            x_d = scale * (s["x_c"] @ R_new + delta) + t
            if stitch:
                x_d = self.lp.stitching(s["x_s"], x_d)
            frame = self.lp.warp_decode(s["f_s"], s["x_s"], x_d)["out"]
            out.append(self.lp.parse_output(frame)[0])
        return np.stack(out)


NEEDED = {"lp": ("get_kp_info", "extract_feature_3d", "transform_keypoint", "stitching",
                 "warp_decode", "parse_output", "prepare_source",
                 "calc_combined_lip_ratio", "retarget_lip"),
          "cropper": ("crop_source_image",)}


def api_check(codec: "MotionCodec") -> None:
    """Fail at construction, not 9 GPU-hours into a cache build, if upstream renamed something."""
    missing = [f"{obj}.{name}" for obj, names in NEEDED.items()
               for name in names if not hasattr(getattr(codec, obj), name)]
    if missing:
        raise AttributeError(
            "LivePortrait API moved; these are gone: " + ", ".join(missing) +
            "\nFix the names in sang/motion.py (marked '# API:') against the installed src/.")


def self_check() -> None:
    """Runnable without LivePortrait, weights, a GPU or data: the pure-tensor parts only."""
    kp = {"scale": torch.rand(3, 1), "yaw": torch.rand(3, 1), "pitch": torch.rand(3, 1),
          "roll": torch.rand(3, 1), "t": torch.rand(3, 3), "exp": torch.rand(3, 21, 3)}
    v = kp_info_to_vec(kp)
    assert v.shape == (3, M_DIM), v.shape
    back = vec_to_kp_info(v)
    for k in kp:
        assert torch.allclose(kp[k], back[k]), f"round trip broke on {k}"

    assert sorted(i for r in REGIONS.values() for i in r) == list(range(T_DIM)), "regions must partition 42"
    assert [len(REGIONS[k]) for k in ("rot", "brow", "eyes", "mouth")] == [3, 6, 15, 18]

    g = torch.Generator().manual_seed(0)
    m = torch.randn(5, M_DIM, generator=g)
    m[:, 0] = 1.0 + 0.1 * m[:, 0].abs()                   # positive scale
    m[:, 1:4] = 30.0 * torch.rand(5, 3, generator=g) - 15  # degrees
    R = _rot(m)
    assert torch.allclose(R @ R.transpose(1, 2), torch.eye(3).expand(5, 3, 3), atol=1e-5), "not orthonormal"
    assert torch.allclose(torch.linalg.det(R), torch.ones(5), atol=1e-5), "not a rotation"
    assert torch.allclose(from_target(to_target(m), m), m, atol=1e-4), "target round trip broke"
    y = to_target(m)                                      # pose change must not move head-frame exp
    m2 = from_target(torch.cat([y[:, :3] + 10.0, y[:, 3:]], 1), m)
    assert torch.allclose(to_target(m2)[:, 3:], y[:, 3:], atol=1e-4), "exp is not pose-invariant"

    A = np.array([[0.5, 0.1, 30.0], [-0.1, 0.5, 40.0], [0.0, 0.0, 1.0]])   # 3x3, as upstream returns
    assert np.allclose(o2c(A), np.linalg.inv(A)[:2]) and np.allclose(o2c(A[:2]), o2c(A)), "o2c"
    assert interp_boxes(np.stack([A, A]), [0, 4], 5).shape == (5, 3, 3)

    b = np.stack([np.full((2, 3), 0.0, np.float32), np.full((2, 3), 10.0, np.float32)])
    got = interp_boxes(b, [0, 10], 11)
    assert got.shape == (11, 2, 3)
    assert abs(float(got[5, 0, 0]) - 5.0) < 1e-5, got[5, 0, 0]
    assert abs(float(got[10, 0, 0]) - 10.0) < 1e-5
    print("sang/motion.py self-check ok")


if __name__ == "__main__":
    self_check()
