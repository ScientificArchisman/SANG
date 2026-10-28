"""LivePortrait as a frozen motion codec: frames -> 70-d motion vectors -> frames.

    m = [scale, yaw, pitch, roll, t_x, t_y, t_z, exp_1..exp_63]      (KDTalker's layout)

This is the only representation SANG-M predicts. Appearance never enters the model: it is
carried by the source frame's `f_s` / `x_s` tensors, which stay inside the frozen renderer.

Keypoints: LivePortrait's own expression transfer drives EXPR_KP (13 of 21); the other 8 (SHAPE_KP) carry face
shape. The 42-d target is the head rotation plus the head-frame expression of EXPR_KP (AVTR-1, arXiv
2609.22913); `wide` is everything the renderer reads except t_z.

Crops: every clip is cropped with ONE fixed box (upstream's driving convention, DRIVE, averaged over landmark
boxes, no rotation), truncated at the first shot cut (CUT_JUMP / CUT_SCALE / MAX_WANDER, in median box sides).
Per-frame source-style crops chase and counter-rotate with the head and cancel translation and roll out of the
motion (M0: 14.7 dB). The source photo is cropped the same way, so cached motion and the photo share a frame.

Upstream symbols follow KwaiVGI/LivePortrait as of 2026-09; `api_check()` checks them at construction.
"""
import contextlib
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
LP = REPO / "third_party" / "LivePortrait"

M_DIM = 70
EXP_DIM = 63
WEIGHTS = LP / "pretrained_weights"

LIP_KP = (6, 12, 14, 17, 19, 20)
EYE_KP = (11, 13, 15, 16, 18)
EXPR_KP = (1, 2, 6, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20)
BROW_KP = tuple(k for k in EXPR_KP if k not in LIP_KP and k not in EYE_KP)

T_DIM = 3 + 3 * len(EXPR_KP)


def _span(kps: tuple[int, ...]) -> list[int]:
    """Target-vector columns of the given keypoints (after the 3 rotation columns)."""
    pos = {k: i for i, k in enumerate(EXPR_KP)}
    return [3 + 3 * pos[k] + a for k in kps for a in range(3)]


REGIONS = {"rot": [0, 1, 2], "brow": _span(BROW_KP), "eyes": _span(EYE_KP), "mouth": _span(LIP_KP)}

PLACE_DIM = 3
PLACE = list(range(T_DIM, T_DIM + PLACE_DIM))


def regions(t_dim: int = T_DIM) -> dict:
    """The target's regions: REGIONS, plus 'place' for a 45-d (placement) target."""
    if t_dim not in (T_DIM, T_DIM + PLACE_DIM):
        raise ValueError(f"target width must be {T_DIM} or {T_DIM + PLACE_DIM}, got {t_dim}")
    return REGIONS if t_dim == T_DIM else {**REGIONS, "place": PLACE}


SHAPE_KP = tuple(k for k in range(21) if k not in EXPR_KP)
WIDE_DIM = 3 + EXP_DIM + PLACE_DIM
WIDE_REGIONS = {"rot": [0, 1, 2], **{name: [3 + 3 * k + a for k in kps for a in range(3)]
                                     for name, kps in (("brow", BROW_KP), ("eyes", EYE_KP), ("mouth", LIP_KP),
                                                       ("shape", SHAPE_KP))},
                "place": list(range(3 + EXP_DIM, WIDE_DIM))}


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
    return rotation_matrix(m[:, 2], m[:, 1], m[:, 3])


def placement(m: torch.Tensor) -> torch.Tensor:
    """[N,70] motion -> [N,3] absolute head placement (t_x, t_y, log scale)."""
    m = m.float()
    return torch.stack([m[:, 4], m[:, 5], m[:, 0].clamp_min(1e-6).log()], 1)


def to_target(m: torch.Tensor, ref: torch.Tensor | None = None) -> torch.Tensor:
    """[N,70] motion -> [N,42] target: (pitch, yaw, roll) + head-frame exp of EXPR_KP.
    With `ref` ([1 or N, 70], the reference frame's motion): [N,45], the placement relative to it appended."""
    m = m.float()
    exp_head = m[:, 7:M_DIM].reshape(-1, 21, 3) @ _rot(m).transpose(1, 2)
    y = torch.cat([m[:, [2, 1, 3]], exp_head[:, list(EXPR_KP)].reshape(-1, 3 * len(EXPR_KP))], 1)
    return y if ref is None else torch.cat([y, placement(m) - placement(ref)], 1)


def wide(m: torch.Tensor) -> torch.Tensor:
    """[N,70] motion -> [N,69]: everything LivePortrait renders except t_z, which it zeroes. (pitch, yaw, roll),
    the head-frame exp of all 21 keypoints, and the absolute placement (t_x, t_y, log scale). Nothing is held from
    the photo, so a model over this vector (sang.latent) decides for itself which numbers matter."""
    m = m.float()
    exp_head = m[:, 7:M_DIM].reshape(-1, 21, 3) @ _rot(m).transpose(1, 2)
    return torch.cat([m[:, [2, 1, 3]], exp_head.reshape(-1, EXP_DIM), placement(m)], 1)


def from_wide(w: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    """Inverse of `wide`: [N,69] + [N or 1,70] base -> [N,70]. Only t_z comes from `base`."""
    w, out = w.float(), base.float().expand(w.shape[0], -1).clone()
    out[:, 2], out[:, 1], out[:, 3] = w[:, 0], w[:, 1], w[:, 2]
    out[:, 7:M_DIM] = (w[:, 3:3 + EXP_DIM].reshape(-1, 21, 3) @ _rot(out)).reshape(-1, EXP_DIM)
    out[:, 4], out[:, 5], out[:, 0] = w[:, -3], w[:, -2], w[:, -1].exp()
    return out


def smooth_placement(y: torch.Tensor, amp: float = 1.0, k: int = 1) -> torch.Tensor:
    """Render-time control of a RAW [N,45] target's placement columns: scale them by `amp` (0 = hold the
    photo's placement, as a 42-d model does) and take a centred moving mean over `k` frames (Ditto
    smooths scale and translation over 3). A [N,42] target is returned unchanged."""
    if y.shape[1] == T_DIM or (amp == 1.0 and k <= 1):
        return y
    y = y.clone()
    p = y[:, PLACE] * amp
    if k > 1:
        h = k // 2
        pad = torch.cat([p[:1].expand(h, -1), p, p[-1:].expand(h, -1)])
        p = pad.unfold(0, 2 * h + 1, 1).mean(-1)
    y[:, PLACE] = p
    return y


def from_target(y: torch.Tensor, base: torch.Tensor, held: str = "camera") -> torch.Tensor:
    """[N,42 or 45] target + [N or 1,70] base motion -> [N,70], ready for MotionCodec.render(relative=False).

    Scale, translation and the 24 non-driven expression coordinates (keypoints 0,3,4,5,7,8,9,10)
    come from `base` (the source frame); a 45-d target moves t_x, t_y and the scale by its placement
    columns, relative to `base` (t_z stays held). The 39 driven coordinates are the target's head-frame
    expression, rotated into the camera frame by the NEW pose. `held` sets how the non-driven ones
    follow a pose change:
      'camera' (default) -- their delta stays the source's camera-frame delta and only x_c rotates,
               x = s(x_c R + delta) + t. This is upstream LivePortrait's absolute driving, and the
               paper (arXiv 2407.03168 sec. 3.2) reports that rotating delta with the head "leads to
               overly flexible learned expressions delta, causing texture flickering".
      'head'   -- their delta is held in the HEAD frame, so it rotates rigidly with the pose. SANG's
               original choice; its error is 0 at the source pose and grows with every rotation away
               from it -- the suspected cause of the background warping (docs/research_log.md)."""
    if held not in ("camera", "head"):
        raise ValueError(f"held must be 'camera' or 'head', got {held!r}")
    y, base = y.float(), base.float().expand(y.shape[0], -1)
    out = base.clone()
    out[:, 2], out[:, 1], out[:, 3] = y[:, 0], y[:, 1], y[:, 2]
    R_new = _rot(out)
    exp = base[:, 7:M_DIM].reshape(-1, 21, 3)
    exp = exp @ _rot(base).transpose(1, 2) @ R_new if held == "head" else exp.clone()
    exp[:, list(EXPR_KP)] = y[:, 3:T_DIM].reshape(-1, len(EXPR_KP), 3) @ R_new
    out[:, 7:M_DIM] = exp.reshape(-1, EXP_DIM)
    if y.shape[1] > T_DIM:
        regions(y.shape[1])
        d = y[:, PLACE]
        out[:, 4], out[:, 5] = base[:, 4] + d[:, 0], base[:, 5] + d[:, 1]
        out[:, 0] = base[:, 0] * d[:, 2].exp()
    return out


CROP = dict(dsize=512, scale=2.3, vx_ratio=0.0, vy_ratio=-0.125)
DRIVE = dict(dsize=512, scale=2.2, vx_ratio=0.0, vy_ratio=-0.1)

CUT_JUMP = 0.25
CUT_SCALE = 1.25
MAX_WANDER = 0.35


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


def decode_clip(path: str, max_frames: int, fps: float = 25.0, start: int = 0) -> np.ndarray:
    """Up to `max_frames` frames of a clip resampled to `fps`, from frame `start` on -> [T,H,W,3] uint8.
    Frame f sits at container time f / fps, which is the clock the cached WavLM ticks use (2 ticks per
    frame)."""
    from sang.video import open_video
    vr = open_video(path)
    stride = vr.get_avg_fps() / fps
    n = min(max_frames, int(len(vr) / stride) - start)
    if n < min(8, max_frames):
        raise ValueError(f"{len(vr)} frames at {vr.get_avg_fps():.1f} fps is too short"
                         + (f" after frame {start}" if start else ""))
    return vr.get_batch([int(round((start + i) * stride)) for i in range(n)]).asnumpy()


def job_cpus() -> int:
    """CPUs this process may run on (its Slurm allocation), not the machine's count."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


@contextlib.contextmanager
def ort_threads(n: int):
    """ONNX Runtime sessions created inside get an explicit thread count n. By default ONNX Runtime starts one
    pinned thread per core of the whole MACHINE: inside a Slurm CPU allocation the pinning fails
    ('pthread_setaffinity_np failed ... Specify the number of threads explicitly') and the threads
    oversubscribe the job's CPUs (job 2026-10-16: source photo 3.7 s, ~100 s per cached clip)."""
    try:
        import onnxruntime as ort
    except ImportError:
        yield
        return
    orig = ort.InferenceSession.__init__

    def init(self, path_or_bytes, sess_options=None, providers=None, provider_options=None, **kw):
        if sess_options is None:
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = n
            sess_options.inter_op_num_threads = 1
        orig(self, path_or_bytes, sess_options, providers, provider_options, **kw)

    ort.InferenceSession.__init__ = init
    try:
        yield
    finally:
        ort.InferenceSession.__init__ = orig


def ort_report(cropper, threads: int) -> str:
    """Which execution provider the face detector and the landmark model run on."""
    try:
        import onnxruntime as ort
        ver = ort.__version__
    except ImportError:
        return "onnxruntime not installed"
    fa, lm = getattr(cropper, "face_analysis_wrapper", None), getattr(cropper, "human_landmark_runner", None)
    used = {f"insightface {k}": getattr(m, "session", None) for k, m in (getattr(fa, "models", None) or {}).items()}
    used["landmark.onnx"] = getattr(lm, "session", None)
    prov = {k: s.get_providers()[0].replace("ExecutionProvider", "") for k, s in used.items() if s is not None}
    gpu = all(v != "CPU" for v in prov.values())
    return (f"ONNX Runtime {ver}, {threads} CPU threads: " + ", ".join(f"{k} on {v}" for k, v in prov.items())
            + ("" if gpu else "  <- face models on the CPU (slow): install onnxruntime-gpu, bash_scripts/install_motion.sh"))


class MotionCodec:
    """Frozen LivePortrait. `extract` maps frames -> motion; `render` maps source + motion -> frames.

    Nothing here trains. Instantiate once per process: it holds five networks plus an
    InsightFace detector (~1.8 GB of VRAM measured on the 4090 in the upstream repo)."""

    def __init__(self, device: str = "cuda", weights: Path = WEIGHTS, half: bool = False):
        _require_repo()
        from src.config.crop_config import CropConfig
        from src.config.inference_config import InferenceConfig
        from src.live_portrait_wrapper import LivePortraitWrapper
        from src.utils.cropper import Cropper

        base = Path(weights) / "liveportrait"
        cfg = InferenceConfig()
        cfg.checkpoint_F = str(base / "base_models" / "appearance_feature_extractor.pth")
        cfg.checkpoint_M = str(base / "base_models" / "motion_extractor.pth")
        cfg.checkpoint_G = str(base / "base_models" / "spade_generator.pth")
        cfg.checkpoint_W = str(base / "base_models" / "warping_module.pth")
        cfg.checkpoint_S = str(base / "retargeting_models" / "stitching_retargeting_module.pth")
        cfg.flag_use_half_precision = bool(half)
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
        threads = job_cpus()
        with ort_threads(threads):
            self.cropper = Cropper(crop_cfg=crop_cfg, device_id=0, flag_force_cpu=device == "cpu")
        print(f"MotionCodec: {ort_report(self.cropper, threads)}", flush=True)
        self.device = device
        api_check(self)

    def landmarks(self, frame: np.ndarray) -> np.ndarray | None:
        """RGB uint8 frame -> 203 refined landmarks in image coordinates, largest face; None if none.
        Same two-stage path upstream uses (InsightFace 106 -> LivePortrait landmark.onnx)."""
        f = self.cropper.face_analysis_wrapper.get(
            np.ascontiguousarray(frame[..., ::-1]), flag_do_landmark_2d_106=True, direction="large-small")
        if len(f) == 0:
            return None
        return self.cropper.human_landmark_runner.run(frame, f[0].landmark_2d_106)

    @staticmethod
    def _box(lmk: np.ndarray) -> np.ndarray:
        """Landmarks -> [x0, y0, x1, y1] under the DRIVE convention (upstream parse_bbox_from_landmark)."""
        from src.utils.crop import parse_bbox_from_landmark
        b = parse_bbox_from_landmark(lmk, scale=DRIVE["scale"], vx_ratio=DRIVE["vx_ratio"],
                                     vy_ratio=DRIVE["vy_ratio"])["bbox"]
        return np.array([b[0, 0], b[0, 1], b[2, 0], b[2, 1]], dtype=np.float64)

    @staticmethod
    def crop_with_box(frames: np.ndarray, box: np.ndarray, size: int = 256) -> tuple[np.ndarray, np.ndarray]:
        """[T,H,W,3] or [H,W,3] + one box -> ([T,size,size,3] crops, 3x3 M_c2o). No rotation.
        256 is what the motion extractor takes; 512 matches the renderer's output (demos)."""
        import cv2
        from src.utils.crop import crop_image_by_bbox
        single = frames.ndim == 3
        frames = frames[None] if single else frames
        out, M = np.empty((len(frames), size, size, 3), np.uint8), None
        for j, fr in enumerate(frames):
            r = crop_image_by_bbox(fr, box.tolist(), dsize=DRIVE["dsize"], flag_rot=False, borderMode=cv2.BORDER_CONSTANT)
            out[j] = r["img_crop"] if size == DRIVE["dsize"] else cv2.resize(r["img_crop"], (size, size), interpolation=cv2.INTER_AREA)
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
            elif not keep:
                continue
            else:
                break
        if not boxes:
            raise ValueError("no face in any probed frame")
        B = np.stack(boxes)
        end = first_cut(B)
        start = keep[0]
        stop = keep[end] if end < len(keep) else T
        box = B[:end].mean(0)
        return box, (start, stop), {"cut": end < len(B), "samples": end, "start": start}

    def crop_clip(self, frames: np.ndarray, every: int = 5, timings: dict | None = None):
        """[T,H,W,3] -> ([T',256,256,3] crops, 3x3 M_c2o, (start, stop)) under ONE fixed box."""
        import time
        t = time.perf_counter()
        box, (a, b), _ = self.clip_box(frames, every=every)
        t1 = time.perf_counter()
        crops, M = self.crop_with_box(frames[a:b], box)
        if timings is not None:
            timings["landmarks"] = timings.get("landmarks", 0.0) + t1 - t
            timings["crop"] = timings.get("crop", 0.0) + time.perf_counter() - t1
        return crops, M, (a, b)

    @torch.no_grad()
    def motion_from_crops(self, crops: np.ndarray, batch: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
        """[T,256,256,3] uint8 -> ([T,70] motion, [T,63] canonical keypoints x_c), fp32.

        x_c is the identity-bearing quantity: Ditto, KDTalker and AVTR-1 all condition on it."""
        x = torch.from_numpy(np.ascontiguousarray(crops)).permute(0, 3, 1, 2).float().div_(255.0)
        ms, kps = [], []
        for i in range(0, len(x), batch):
            kp = self.lp.get_kp_info(x[i:i + batch].to(self.device))
            ms.append(kp_info_to_vec(kp).float().cpu())
            kps.append(kp["kp"].reshape(-1, EXP_DIM).float().cpu())
        return torch.cat(ms), torch.cat(kps)

    @torch.no_grad()
    def extract(self, frames: np.ndarray, batch: int = 64, every: int = 5,
                timings: dict | None = None) -> dict:
        """[T,H,W,3] uint8 RGB at 25 fps -> {'m': [T',70], 'kp': [T',63], 'M_c2o', 'crops', 'span'}.

        T' <= T: the clip is truncated at the first shot cut. `span` = (start, stop) frame indices
        into the input, so audio can be cut to match."""
        import time
        crops, M, span = self.crop_clip(frames, every=every, timings=timings)
        t = time.perf_counter()
        m, kp = self.motion_from_crops(crops, batch=batch)
        if timings is not None:
            timings["motion"] = timings.get("motion", 0.0) + time.perf_counter() - t
        return {"m": m.half(), "kp": kp.half(), "M_c2o": torch.from_numpy(M), "crops": crops, "span": span}

    @torch.no_grad()
    def source_motion(self, source: np.ndarray, normalize_lip: bool = True, crop: dict | None = None) -> dict:
        """One source image -> {'m': [1,70], 'kp': [1,63]}, optionally with the lips closed.

        LivePortrait's own flag_normalize_lip (upstream live_portrait_pipeline.py): a reference
        photo caught mid-vowel otherwise biases every frame of the animation. The retarget module
        returns a keypoint offset d; since x = s(x_c R + delta) + t, the same offset in expression
        space is d / s, applied in the camera frame where delta lives."""
        c = crop or self.crop_image(np.ascontiguousarray(source))
        m, kp = self.motion_from_crops(c["img_crop_256x256"][None])
        if normalize_lip:
            src = self.lp.prepare_source(c["img_crop_256x256"])
            x_s = self.lp.transform_keypoint(self.lp.get_kp_info(src))
            ratio = self.lp.calc_combined_lip_ratio([0.0], c["lmk"])
            thresh = getattr(self.lp.inference_cfg, "lip_normalize_threshold", 0.03)
            if float(ratio[0][0]) >= thresh:
                d = self.lp.retarget_lip(x_s, ratio).reshape(1, 21, 3).float().cpu()
                m = m.clone()
                m[:, 7:M_DIM] += (d / m[:, 0:1].reshape(1, 1, 1)).reshape(1, EXP_DIM)
        return {"m": m, "kp": kp}

    @torch.no_grad()
    def source_state(self, source: np.ndarray, precropped: bool = False) -> dict:
        """Source RGB frame -> everything the renderer needs that does not depend on the audio.
        precropped=True: `source` is already a 256x256 crop made by crop_with_box / crop_clip."""
        from src.utils.camera import get_rotation_matrix

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
               stitch: bool = True, state: dict | None = None, batch: int = 16, half: bool = False,
               out: str = "numpy"):
        """source [H,W,3] uint8 + motion [T,70] -> [T,512,512,3] uint8 crops (a GPU tensor with out='torch').

        `relative=True` is KDTalker's / LivePortrait's retargeting: the driving trajectory is
        applied as a delta from its own first frame onto the source pose, so the source keeps its
        identity and rest pose. With `relative=False` the source is forced into the driving pose,
        which is what SANG's absolute targets and the representation ceiling need. `batch` frames
        per renderer call; half = FP16 autocast around the warping module and decoder."""
        frames = torch.cat(list(self.render_iter(state or self.source_state(source), m, relative, stitch, batch, half)))
        return frames if out == "torch" else frames.cpu().numpy()

    def driving_keypoints(self, s: dict, m: torch.Tensor, relative: bool = True) -> torch.Tensor:
        """The per-frame loop of `render` for all T frames at once -> x_d [T, 21, 3] (before stitching)."""
        from src.utils.camera import get_rotation_matrix
        mm = m.float().to(self.device)
        d = vec_to_kp_info(mm)
        R_d = get_rotation_matrix(d["pitch"], d["yaw"], d["roll"])
        if relative:
            d0 = vec_to_kp_info(mm[:1])
            R_d0 = get_rotation_matrix(d0["pitch"], d0["yaw"], d0["roll"])
            R_new = (R_d @ R_d0.permute(0, 2, 1)) @ s["R_s"]
            delta = s["info"]["exp"] + (d["exp"] - d0["exp"])
            scale = s["info"]["scale"] * (d["scale"] / d0["scale"])
            t = s["info"]["t"] + (d["t"] - d0["t"])
        else:
            R_new, delta, scale, t = R_d, d["exp"], d["scale"], d["t"]
        t = t.clone()
        t[:, 2] = 0.0
        return scale.reshape(-1, 1, 1) * (s["x_c"] @ R_new + delta) + t.reshape(-1, 1, 3)

    @torch.no_grad()
    def render_iter(self, s: dict, m: torch.Tensor, relative: bool = True, stitch: bool = True,
                    batch: int = 16, half: bool = False):
        """Yields [b, 512, 512, 3] uint8 GPU frames, `batch` at a time (see `render`)."""
        x_d_all = self.driving_keypoints(s, m, relative)
        for i in range(0, len(x_d_all), batch):
            x_d = x_d_all[i:i + batch]
            b = len(x_d)
            x_s = s["x_s"].expand(b, -1, -1)
            if stitch:
                x_d = self.lp.stitching(x_s, x_d)
            with torch.autocast("cuda", dtype=torch.float16, enabled=half and self.device != "cpu"):
                w = self.lp.warping_module(s["f_s"].expand(b, *s["f_s"].shape[1:]), kp_source=x_s, kp_driving=x_d)
                img = self.lp.spade_generator(feature=w["out"])
            yield (img.float().clamp(0, 1) * 255).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)


NEEDED = {"lp": ("get_kp_info", "extract_feature_3d", "transform_keypoint", "stitching", "prepare_source",
                 "calc_combined_lip_ratio", "retarget_lip", "warping_module", "spade_generator"),
          "cropper": ("face_analysis_wrapper", "human_landmark_runner")}


def api_check(codec: "MotionCodec") -> None:
    """Fail at construction, not 9 GPU-hours into a cache build, if upstream renamed something."""
    missing = [f"{obj}.{name}" for obj, names in NEEDED.items()
               for name in names if not hasattr(getattr(codec, obj), name)]
    if missing:
        raise AttributeError(
            "LivePortrait API moved; these are gone: " + ", ".join(missing) +
            "\nFix the names in sang/motion.py (NEEDED) against the installed src/.")


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
    m[:, 0] = 1.0 + 0.1 * m[:, 0].abs()
    m[:, 1:4] = 30.0 * torch.rand(5, 3, generator=g) - 15
    R = _rot(m)
    assert torch.allclose(R @ R.transpose(1, 2), torch.eye(3).expand(5, 3, 3), atol=1e-5), "not orthonormal"
    assert torch.allclose(torch.linalg.det(R), torch.ones(5), atol=1e-5), "not a rotation"
    assert torch.allclose(from_target(to_target(m), m), m, atol=1e-4), "target round trip broke"
    y = to_target(m)
    m2 = from_target(torch.cat([y[:, :3] + 10.0, y[:, 3:]], 1), m)
    assert torch.allclose(to_target(m2)[:, 3:], y[:, 3:], atol=1e-4), "exp is not pose-invariant"
    assert torch.allclose(from_target(y, m, held="head"), m, atol=1e-4), "head-held round trip broke"
    held_kp = [k for k in range(21) if k not in EXPR_KP]
    mh = from_target(torch.cat([y[:, :3] + 10.0, y[:, 3:]], 1), m, held="head")
    mc = from_target(torch.cat([y[:, :3] + 10.0, y[:, 3:]], 1), m, held="camera")
    cam = lambda v: v[:, 7:M_DIM].reshape(-1, 21, 3)
    assert torch.allclose(cam(mc)[:, held_kp], cam(m)[:, held_kp]), "camera-held delta must not move"
    assert torch.allclose(cam(mc)[:, list(EXPR_KP)], cam(mh)[:, list(EXPR_KP)], atol=1e-5), "driven kps differ"
    assert not torch.allclose(cam(mc)[:, held_kp], cam(mh)[:, held_kp], atol=1e-3), "conventions identical?"

    A = np.array([[0.5, 0.1, 30.0], [-0.1, 0.5, 40.0], [0.0, 0.0, 1.0]])
    assert np.allclose(o2c(A), np.linalg.inv(A)[:2]) and np.allclose(o2c(A[:2]), o2c(A)), "o2c"

    w = from_wide(wide(m), m)
    assert torch.allclose(w, m, atol=1e-4), "wide round trip broke"
    assert sorted(i for r in WIDE_REGIONS.values() for i in r) == list(range(WIDE_DIM)), "wide regions must partition 69"

    steady = np.array([[100 + i, 100, 300 + i, 300] for i in range(10)], float)
    assert first_cut(steady) == 10, "slow head motion is not a cut"
    cut = np.vstack([steady[:6], steady[6:] + [150, 0, 150, 0]])
    assert first_cut(cut) == 6, first_cut(cut)
    zoom = np.vstack([steady[:4], [[50, 50, 350, 350]] * 3])
    assert first_cut(zoom) == 4, first_cut(zoom)
    print("sang/motion.py self-check ok")


if __name__ == "__main__":
    self_check()

