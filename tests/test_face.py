"""Face mesh rendering and the face-crop box."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sang.face import render_structure
from sang.video import face_box


def test_render_shape_and_empty_on_black():
    """No face in a black clip -> same-shape all-zero uint8 render."""
    frames = np.zeros((3, 96, 96, 3), np.uint8)
    out = render_structure(frames)
    assert out.shape == frames.shape and out.dtype == np.uint8
    assert out.sum() == 0


def test_face_box_none_without_a_face():
    """face_box returns None so callers can fall back to the centre crop and flag the window."""
    assert face_box(np.zeros((4, 120, 200, 3), np.uint8)) is None


def test_face_box_is_square_and_inside_the_frame():
    """Synthetic frames have no detectable face, so this exercises the geometry contract only
    when a box is produced; it must always be square and fully inside the frame."""
    frames = (np.random.default_rng(0).random((4, 120, 200, 3)) * 255).astype(np.uint8)
    box = face_box(frames)
    if box is None:
        return
    top, left, side = box
    assert top >= 0 and left >= 0 and top + side <= 120 and left + side <= 200


if __name__ == "__main__":
    test_render_shape_and_empty_on_black()
    test_face_box_none_without_a_face()
    test_face_box_is_square_and_inside_the_frame()
    print("ok")
