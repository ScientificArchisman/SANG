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
CROP = dict(dsize=512, scale=2.3, vx_ratio=0.0, vy_ratio=-0.125)   # upstream SOURCE convention (unused for clips)

# How EVERY crop in this repo is made -- clip frames and the source image alike. Upstream's DRIVING
# convention (src/utils/cropper.py crop_driving_video, crop_config.py *_crop_driving_video): one
# box for the whole clip, averaged over per-frame landmark boxes, NO rotation.
#
# The first version cropped every frame with crop_source_image instead, which re-centres the box on
# the face and rotates it to level the eyes. M0 (jobs 171503/4, 2026-09-23) showed what that does:
# the crop window chases and counter-rotates with the head ("camera sway"), head translation and
# roll are cancelled out of the extracted motion, and the renderer -- which works in frame 0's
# crop -- disagrees with the ground truth everywhere. PSNR fell from 35 dB at frame 0 to ~20 dB
# within 12 frames; mean 14.7 dB.
DRIVE = dict(dsize=512, scale=2.2, vx_ratio=0.0, vy_ratio=-0.1)

# Shot-cut / runaway-framing test on the sampled landmark boxes, in units of the median box side.
CUT_JUMP = 0.25       # centre moves more than this between consecutive samples -> cut
CUT_SCALE = 1.25      # box side changes by more than this ratio between samples -> cut
MAX_WANDER = 0.35     # centre wanders this far from the clip's mean box -> fixed box loses the face


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


def motion_fidelity(m_in: torch.Tensor, m_out: torch.Tensor) -> dict:
    """How faithfully a render realises the motion it was driven with.

    m_in: the [T,70] driving motion; m_out: motion re-extracted from the rendered frames. Compared
    in the 42-d target space, per region: Pearson correlation over time (does it move when the input
    moves?), amplitude ratio (does it move as MUCH?), and rotation error in degrees. This is the
    representation ceiling for an audio->motion model -- unlike frame PSNR it ignores hands,
    torso, lighting and background, which the motion vector does not control."""
    a, b = to_target(m_in.float()), to_target(m_out.float())
    out = {}
    for name in ("mouth", "eyes", "brow"):
        idx = REGIONS[name]
        x, y = a[:, idx] - a[:, idx].mean(0), b[:, idx] - b[:, idx].mean(0)
        live = x.std(0) > 1e-4                        # a coordinate that never moves has no correlation
        c = (x * y).sum(0) / (x.norm(dim=0) * y.norm(dim=0)).clamp_min(1e-8)
        out[f"{name}_corr"] = float(c[live].mean()) if live.any() else float("nan")
        out[f"{name}_amp"] = float(y.std(0)[live].mean() / x.std(0)[live].mean()) if live.any() else float("nan")
    out["rot_err_deg"] = float((a[:, :3] - b[:, :3]).abs().mean())
    return out


def first_cut(B: np.ndarray) -> int:
    """[K,4] boxes sampled along a clip -> number of leading samples that one fixed box can serve.

    K if there is no shot cut. Stops before the first sample whose centre jumps more than CUT_JUMP
    box sides, whose size changes by more than CUT_SCALE, or whose centre has wandered more than
    MAX_WANDER sides from the running mean."""
    ctr, side = (B[:, :2] + B[:, 2:]) / 2, (B[:, 2] - B[:, 0])
    for k in range(1, len(B)):
        ref = np.median(side[:k + 1])
        if (np.linalg.norm(ctr[k] - ctr[k - 1]) > CUT_JUMP * ref
                or max(side[k], side[k - 1]) / max(1e-6, min(side[k], side[k - 1])) > CUT_SCALE
                or np.linalg.norm(ctr[k] - ctr[:k + 1].mean(0)) > MAX_WANDER * ref):
            return k
    return len(B)


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
    def landmarks(self, frame: np.ndarray) -> np.ndarray | None:
        """RGB uint8 frame -> 203 refined landmarks in image coordinates, largest face; None if none.
        Same two-stage path upstream uses (InsightFace 106 -> LivePortrait landmark.onnx)."""
        f = self.cropper.face_analysis_wrapper.get(                         # API
            np.ascontiguousarray(frame[..., ::-1]), flag_do_landmark_2d_106=True, direction="large-small")
        if len(f) == 0:
            return None
        return self.cropper.human_landmark_runner.run(frame, f[0].landmark_2d_106)   # API

    @staticmethod
    def _box(lmk: np.ndarray) -> np.ndarray:
        """Landmarks -> [x0, y0, x1, y1] under the DRIVE convention (upstream parse_bbox_from_landmark)."""
        from src.utils.crop import parse_bbox_from_landmark
        b = parse_bbox_from_landmark(lmk, scale=DRIVE["scale"], vx_ratio=DRIVE["vx_ratio"],
                                     vy_ratio=DRIVE["vy_ratio"])["bbox"]
        return np.array([b[0, 0], b[0, 1], b[2, 0], b[2, 1]], dtype=np.float64)

    @staticmethod
    def crop_with_box(frames: np.ndarray, box: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """[T,H,W,3] or [H,W,3] + one box -> ([T,256,256,3] crops, 3x3 M_c2o). No rotation."""
        import cv2
        from src.utils.crop import crop_image_by_bbox
        single = frames.ndim == 3
        frames = frames[None] if single else frames
        out, M = np.empty((len(frames), 256, 256, 3), np.uint8), None
        for j, fr in enumerate(frames):
            r = crop_image_by_bbox(fr, box.tolist(), dsize=DRIVE["dsize"], flag_rot=False, borderMode=cv2.BORDER_CONSTANT)
            out[j] = cv2.resize(r["img_crop"], (256, 256), interpolation=cv2.INTER_AREA)
            M = as3x3(r["M_c2o"])
        return (out[0] if single else out), M

    def crop_image(self, frame: np.ndarray) -> dict:
        """A single source photo, cropped by the same convention as the clips -> crop, M_c2o, lmk."""
        lmk = self.landmarks(frame)
        if lmk is None:
            raise ValueError("no face found")
        crop, M = self.crop_with_box(frame, self._box(lmk))
        return {"img_crop_256x256": crop, "M_c2o": M, "lmk": lmk}

    def clip_box(self, frames: np.ndarray, every: int = 5) -> tuple[np.ndarray, int, dict]:
        """One box for the clip. Returns (box, usable_frames, stats).

        Landmark boxes are sampled every `every` frames. At the first shot cut -- or once the face
        wanders so far that one box cannot hold it -- the clip is truncated there and the box is
        re-averaged over what remains. Raises if no face is found."""
        T = len(frames)
        idx = sorted({*range(0, T, max(1, every)), T - 1})
        boxes, keep = [], []
        for i in idx:
            lmk = self.landmarks(frames[i])
            if lmk is not None:
                boxes.append(self._box(lmk))
                keep.append(i)
            elif not keep:                       # no face at the start: skip ahead
                continue
            else:                                # face lost mid-clip: treat as a cut
                break
        if not boxes:
            raise ValueError("no face in any probed frame")
        B = np.stack(boxes)
        end = first_cut(B)
        start = keep[0]
        stop = keep[end] if end < len(keep) else T        # first frame NOT covered by the box
        box = B[:end].mean(0)
        return box, (start, stop), {"cut": end < len(B), "samples": end, "start": start}

    def crop_clip(self, frames: np.ndarray, every: int = 5) -> tuple[np.ndarray, np.ndarray, tuple]:
        """[T,H,W,3] -> ([T',256,256,3] crops, 3x3 M_c2o, (start, stop)) under ONE fixed box."""
        box, (a, b), _ = self.clip_box(frames, every=every)
        crops, M = self.crop_with_box(frames[a:b], box)
        return crops, M, (a, b)

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
    def extract(self, frames: np.ndarray, batch: int = 64, every: int = 5) -> dict:
        """[T,H,W,3] uint8 RGB at 25 fps -> {'m': [T',70], 'kp': [T',63], 'M_c2o', 'crops', 'span'}.

        T' <= T: the clip is truncated at the first shot cut. `span` = (start, stop) frame indices
        into the input, so audio can be cut to match."""
        crops, M, span = self.crop_clip(frames, every=every)
        m, kp = self.motion_from_crops(crops, batch=batch)
        return {"m": m.half(), "kp": kp.half(), "M_c2o": torch.from_numpy(M), "crops": crops, "span": span}

    @torch.no_grad()
    def source_motion(self, source: np.ndarray, normalize_lip: bool = True) -> dict:
        """One source image -> {'m': [1,70], 'kp': [1,63]}, optionally with the lips closed.

        LivePortrait's own flag_normalize_lip (upstream live_portrait_pipeline.py): a reference
        photo caught mid-vowel otherwise biases every frame of the animation. The retarget module
        returns a keypoint offset d; since x = s(x_c R + delta) + t, the same offset in expression
        space is d / s, applied in the camera frame where delta lives."""
        c = self.crop_image(np.ascontiguousarray(source))
        m, kp = self.motion_from_crops(c["img_crop_256x256"][None])
        if normalize_lip:
            src = self.lp.prepare_source(c["img_crop_256x256"])
            x_s = self.lp.transform_keypoint(self.lp.get_kp_info(src))
            ratio = self.lp.calc_combined_lip_ratio([0.0], c["lmk"])            # API; ratio is scale-free
            thresh = getattr(self.lp.inference_cfg, "lip_normalize_threshold", 0.03)
            if float(ratio[0][0]) >= thresh:
                d = self.lp.retarget_lip(x_s, ratio).reshape(1, 21, 3).float().cpu()  # API
                m = m.clone()
                m[:, 7:M_DIM] += (d / m[:, 0:1].reshape(1, 1, 1)).reshape(1, EXP_DIM)
        return {"m": m, "kp": kp}

    # -------------------------------------------------------------------- rendering
    @torch.no_grad()
    def source_state(self, source: np.ndarray, precropped: bool = False) -> dict:
        """Source RGB frame -> everything the renderer needs that does not depend on the audio.
        precropped=True: `source` is already a 256x256 crop made by crop_with_box / crop_clip."""
        from src.utils.camera import get_rotation_matrix  # noqa: E402

        c = ({"img_crop_256x256": source, "M_c2o": None} if precropped
             else self.crop_image(np.ascontiguousarray(source)))
        src = self.lp.prepare_source(c["img_crop_256x256"])
        info = self.lp.get_kp_info(src)
        return {"info": info,
                "R_s": get_rotation_matrix(info["pitch"], info["yaw"], info["roll"]),
                "f_s": self.lp.extract_feature_3d(src),
                "x_s": self.lp.transform_keypoint(info),
                "x_c": info["kp"],
                "M_c2o": c["M_c2o"]}

    @torch.no_grad()
    def render(self, source: np.ndarray | None, m: torch.Tensor, relative: bool = True,
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
          "cropper": ("face_analysis_wrapper", "human_landmark_runner")}


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

    mm = torch.zeros(50, M_DIM); mm[:, 0] = 1.0
    mm[:, 7:] = torch.sin(torch.linspace(0, 6, 50))[:, None] * torch.linspace(0.01, 0.05, EXP_DIM)
    f = motion_fidelity(mm, mm)
    assert abs(f["mouth_corr"] - 1) < 1e-4 and abs(f["mouth_amp"] - 1) < 1e-4 and f["rot_err_deg"] < 1e-5
    half = mm.clone(); half[:, 7:] *= 0.5
    assert abs(motion_fidelity(mm, half)["mouth_amp"] - 0.5) < 1e-3, "amplitude ratio"

    steady = np.array([[100 + i, 100, 300 + i, 300] for i in range(10)], float)   # slow 1 px drift
    assert first_cut(steady) == 10, "slow head motion is not a cut"
    cut = np.vstack([steady[:6], steady[6:] + [150, 0, 150, 0]])                 # jump of 0.75 side
    assert first_cut(cut) == 6, first_cut(cut)
    zoom = np.vstack([steady[:4], [[50, 50, 350, 350]] * 3])                      # 1.5x zoom-out
    assert first_cut(zoom) == 4, first_cut(zoom)
    print("sang/motion.py self-check ok")


if __name__ == "__main__":
    self_check()
