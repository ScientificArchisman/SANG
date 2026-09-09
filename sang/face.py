"""MediaPipe FaceMesh -> rich structural render (pose + expression, identity-light) for conditioning.

Renders the 478-point face mesh (dense tessellation + per-feature contours + irises) onto a black
canvas so the signal carries head pose, expression and mouth/eye shape but little appearance/identity.
The render is meant to be VidTok-encoded on the *same* crop as the video, giving a grid aligned 1:1
with the video tokens. Missing face -> empty (all-zero) frame; the model handles that via condition
dropout / a zeroed structure embedding.
"""
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarksConnections as _C

_MODEL = Path(__file__).resolve().parents[1] / "checkpoints" / "face_landmarker.task"
_LANDMARKER = None

# (connection set, RGB colour, thickness); drawn back-to-front: dense mesh first, bright features on top.
# Distinct colours per group pack pose/expression cues into all three channels for the tokenizer.
_LAYERS = [
    (_C.FACE_LANDMARKS_TESSELATION, (60, 90, 70), 1),
    (_C.FACE_LANDMARKS_FACE_OVAL, (180, 180, 180), 1),
    (_C.FACE_LANDMARKS_NOSE, (200, 120, 240), 1),
    (_C.FACE_LANDMARKS_LEFT_EYEBROW, (240, 200, 70), 1),
    (_C.FACE_LANDMARKS_RIGHT_EYEBROW, (240, 200, 70), 1),
    (_C.FACE_LANDMARKS_LEFT_EYE, (70, 160, 240), 1),
    (_C.FACE_LANDMARKS_RIGHT_EYE, (70, 240, 160), 1),
    (_C.FACE_LANDMARKS_LIPS, (240, 70, 90), 1),
    (_C.FACE_LANDMARKS_LEFT_IRIS, (255, 255, 60), 2),
    (_C.FACE_LANDMARKS_RIGHT_IRIS, (255, 255, 60), 2),
]
_LAYERS_P = [([(c.start, c.end) for c in s], col, th) for s, col, th in _LAYERS]


def _landmarker():
    """Lazy singleton FaceLandmarker (IMAGE mode, one face); not picklable, so built per process."""
    global _LANDMARKER
    if _LANDMARKER is None:
        if not _MODEL.exists():
            raise FileNotFoundError(f"missing {_MODEL}; download face_landmarker.task into checkpoints/")
        opts = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(_MODEL)),
            running_mode=vision.RunningMode.IMAGE, num_faces=1,
            output_face_blendshapes=False,
        )
        _LANDMARKER = vision.FaceLandmarker.create_from_options(opts)
    return _LANDMARKER


def _draw(H: int, W: int, pts: np.ndarray) -> np.ndarray:
    """Landmark pixel coords [N, 2] -> [H, W, 3] uint8 mesh on black.

    Always allocates a fresh C-contiguous canvas: cv2 rejects the non-contiguous views that come from
    permuting a tensor to HWC (`to_uint8_frames`), so we never draw straight into such an array."""
    canvas = np.zeros((H, W, 3), np.uint8)
    for pairs, col, th in _LAYERS_P:
        for a, b in pairs:
            if a < len(pts) and b < len(pts):
                cv2.line(canvas, tuple(pts[a]), tuple(pts[b]), col, th, cv2.LINE_AA)
    canvas[pts[:, 1], pts[:, 0]] = (230, 230, 230)
    return canvas


def render_structure(frames: np.ndarray) -> np.ndarray:
    """[T, H, W, 3] uint8 RGB -> [T, H, W, 3] uint8 mesh render on black (empty frame when no face)."""
    lmk = _landmarker()
    T, H, W, _ = frames.shape
    out = np.zeros((T, H, W, 3), np.uint8)
    for t in range(T):
        rgb = np.ascontiguousarray(frames[t])
        res = lmk.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if not res.face_landmarks:
            continue
        pts = np.array([[min(W - 1, max(0, round(l.x * W))), min(H - 1, max(0, round(l.y * H)))]
                        for l in res.face_landmarks[0]], dtype=np.int32)
        out[t] = _draw(H, W, pts)
    return out


def landmarks_px(frame: np.ndarray):
    """[H, W, 3] uint8 RGB -> [478, 2] float32 pixel landmarks, or None if no face."""
    res = _landmarker().detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame)))
    if not res.face_landmarks:
        return None
    h, w = frame.shape[:2]
    return np.array([[lm.x * w, lm.y * h] for lm in res.face_landmarks[0]], np.float32)


if __name__ == "__main__":
    tri = _draw(64, 64, np.array([[10, 10], [50, 15], [30, 55]], np.int32))
    assert tri.shape == (64, 64, 3) and tri.dtype == np.uint8 and tri.sum() > 0, "draw produces a mesh"
    black = np.zeros((3, 96, 96, 3), np.uint8)
    r = render_structure(black)
    assert r.shape == black.shape and r.dtype == np.uint8 and r.sum() == 0, "black -> empty render"
    print("face selfcheck OK")
