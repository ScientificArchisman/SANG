"""Background stillness (reports/SANG static background control.md, phase 0).

BG-Drag: how far the background has moved, in px of the output frame, from a fixed anchor to every frame, by OpenCV DIS
flow on pixels that (1) the paste-back covers, (2) the real clip shows still, (3) no head covers in any generated frame,
and (4) have texture. The generated video is anchored to its own zero-motion render and the real clip to its first frame
(the photo), so the same pixels give a pair: ΔBG-Drag = generated - real, in px, never a ratio.

BG-Def / BG-Leak / BG-Occ read the same thing off LivePortrait's dense motion without decoding: the displacement of its
deformation from the anchor's (px at 512), the weight background voxels put on keypoint slots, and the change of the
occlusion map, over the same background cells (sang.render.liveportrait.MotionCodec.probe).
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F

MIN_PX = 2000  # fewer background pixels than this: the clip is not measurable (counted, not dropped)
NEAR = 64      # px from the head envelope: the near band, where drag shows


def gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2GRAY)


def still_mask(real: np.ndarray, ring: int = 32, grow: int = 16) -> np.ndarray:
    """[T,H,W,3] real frames -> [H,W] bool: pixels whose temporal std (3x3 blurred gray) stays under
    max(1.5, 3 x the median std on the outer ring) gray levels, with the moving rest grown by `grow` px."""
    g = np.stack([cv2.blur(gray(f).astype(np.float32), (3, 3)) for f in real])
    s = g.std(0)
    edge = np.ones_like(s, bool)
    edge[ring:-ring, ring:-ring] = False
    moving = (s > max(1.5, 3 * float(np.median(s[edge])))).astype(np.uint8)
    return ~cv2.dilate(moving, np.ones((2 * grow + 1,) * 2, np.uint8)).astype(bool)


def textured(img: np.ndarray, lam: float) -> np.ndarray:
    """Pixels where the smaller structure-tensor eigenvalue (7 px block, sigma 1 blur) reaches lam: flow is reliable.
    shortcut: lam = 5e-5 sits between a JPEG flat wall (p95 9e-6) and band-limited texture (p5 1e-4); calibrate it on
    real HDTF clips (the real far-band drag must stay under 0.1 px)."""
    return cv2.cornerMinEigenVal(cv2.GaussianBlur(gray(img), (0, 0), 1), 7, ksize=3) >= lam


def head_masks(kp_px: np.ndarray, S: int, up: float = 0.5, grow: int = 16) -> np.ndarray:
    """[T, K, 2] keypoints in crop px -> [T, S, S] bool: each frame's keypoint hull, stretched up by `up` x its height
    for the hair, grown by `grow` px."""
    out = np.zeros((len(kp_px), S, S), np.uint8)
    k = np.ones((2 * grow + 1,) * 2, np.uint8)
    for i, p in enumerate(kp_px):
        h = p[:, 1].max() - p[:, 1].min()
        pts = np.concatenate([p, p - [0, up * h]]).astype(np.int32)
        cv2.fillConvexPoly(out[i], cv2.convexHull(pts), 1)
        out[i] = cv2.dilate(out[i], k)
    return out.astype(bool)


def cells(mask: np.ndarray, n: int = 64) -> torch.Tensor:
    """[..., S, S] bool -> [..., n, n] float: a cell counts only if all its pixels do (min-pool)."""
    m = torch.from_numpy(np.ascontiguousarray(mask)).float()
    s = m.shape[-1] // n
    return -F.max_pool2d(-m.reshape(-1, 1, *m.shape[-2:]), s).reshape(*m.shape[:-2], n, n)


def drag(frames: np.ndarray, anchor: np.ndarray, B: np.ndarray, near: np.ndarray) -> dict:
    """DIS flow anchor -> every frame on B (bool [H,W]); pixels failing the forward-backward check (Ruder et al. 2016)
    are rejected. Per frame the median |flow| over the kept pixels (all, and the near band); per clip their mean and the
    95th percentile, the frame-to-frame jitter of the flow and the rejected share."""
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    a = gray(anchor)
    H, W = a.shape
    xs, ys = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    d, dn, j, rej, prev = [], [], [], [], None
    for f in frames:
        g = gray(f)
        w = dis.calc(a, g, None)
        wb = cv2.remap(dis.calc(g, a, None), xs + w[..., 0], ys + w[..., 1], cv2.INTER_LINEAR)
        bad = ((w + wb) ** 2).sum(-1) > 0.01 * ((w ** 2).sum(-1) + (wb ** 2).sum(-1)) + 0.5
        ok = B & ~bad
        rej.append(1 - ok.sum() / B.sum())
        mag = np.linalg.norm(w, axis=-1)
        if ok.any():
            d.append(np.median(mag[ok]))
        if (ok & near).any():
            dn.append(np.median(mag[ok & near]))
        if prev is not None and ok.any():
            j.append(np.median(np.linalg.norm(w - prev, axis=-1)[ok]))
        prev = w
    nan = float("nan")
    return {"drag": float(np.mean(d)) if d else nan, "drag95": float(np.percentile(d, 95)) if d else nan,
            "drag_near": float(np.mean(dn)) if dn else nan, "jitter": float(np.mean(j)) if j else nan,
            "reject": float(np.mean(rej))}


def bg_metrics(gen: np.ndarray, real: np.ndarray, anchor: np.ndarray, footprint: np.ndarray, head: np.ndarray,
               lam: float = 5e-5) -> dict:
    """gen, real [T,H,W,3] in the same frame; anchor the generated video's zero-motion render; footprint and head
    [H,W] bool (paste mask >= 0.5, union of the generated head envelopes). bg_* of the generated video, bg_real_*
    of the real clip on the same pixels, bg_px the pixel count and bg_offset the anchor's PSNR against the photo there
    (the decoder's static colour error, which drag does not count)."""
    n = min(len(gen), len(real))
    B = footprint & ~head & still_mask(real[:n]) & textured(real[0], lam)
    out = {"bg_px": int(B.sum())}
    if B.sum() < MIN_PX:
        return out
    dist = cv2.distanceTransform((~head).astype(np.uint8), cv2.DIST_L2, 5)
    near = dist <= NEAR
    err = (anchor[B].astype(np.float64) - real[0][B]) ** 2
    out["bg_offset"] = float(10 * np.log10(255 ** 2 / max(err.mean(), 1e-10)))
    out |= {f"bg_{k}": v for k, v in drag(gen[:n], anchor, B, near).items()}
    out |= {f"bg_real_{k}": v for k, v in drag(real[:n], real[0], B, near).items()}
    return out


def def_stats(dm: dict, ref: dict, B64: torch.Tensor, w: torch.Tensor) -> dict:
    """Per frame of one dense-motion call: BG-Def (px at 512) = weighted mean |deformation - anchor deformation| (x, y)
    over background cells, BG-Leak = weighted mean keypoint-slot weight there, BG-Occ = mean |occlusion - anchor's|.
    dm / ref: LivePortrait's dense-motion outputs (mask [b,22,D,64,64], deformation [b,D,64,64,3], occlusion_map
    [b,1,64,64]); B64 [64,64] background cells; w [D,64,64] voxel weights (mean |f_s|: depth slices that hold
    features count)."""
    W = (w * B64).float()
    tot = W.sum().clamp_min(1e-8)
    disp = 256 * (dm["deformation"][..., :2].float() - ref["deformation"][..., :2].float()).norm(dim=-1)
    out = {"def": (disp * W).sum((1, 2, 3)) / tot,
           "leak": ((1 - dm["mask"][:, 0].float()) * W).sum((1, 2, 3)) / tot}
    if "occlusion_map" in dm:
        occ = (dm["occlusion_map"].float() - ref["occlusion_map"].float()).abs()[:, 0]
        out["occ"] = (occ * B64).sum((1, 2)) / B64.sum().clamp_min(1)
    return out


def self_check() -> None:
    """A textured background shifted by 3 px from frame 5 on reads ~3 px of drag, the unshifted real clip ~0."""
    rng = np.random.default_rng(0)
    tex = cv2.GaussianBlur(rng.normal(0, 1, (256, 256)).astype(np.float32), (0, 0), 3)
    tex = np.clip(tex / tex.std() * 25 + 128, 0, 255).astype(np.uint8)
    img = np.repeat(tex[..., None], 3, -1)
    real = np.repeat(img[None], 10, 0)
    gen = real.copy()
    gen[5:] = np.roll(img, 3, 1)
    head = np.zeros((256, 256), bool)
    head[96:160, 96:160] = True
    r = bg_metrics(gen, real, img, np.ones((256, 256), bool), head, lam=1e-5)
    assert abs(r["bg_drag"] - 1.5) < 0.2 and r["bg_real_drag"] < 0.05, r  # half the frames moved 3 px
    assert abs(r["bg_drag95"] - 3) < 0.2 and abs(r["bg_jitter"] - 3 / 9) < 0.05, r  # one 3 px jump in 9 steps
    m = np.zeros((2, 128, 128), bool)
    m[:, :64, :] = True
    m[1, 0, 0] = False
    c = cells(m, 64)
    assert c.shape == (2, 64, 64) and c[0, :32].min() == 1 and c[0, 32:].max() == 0 and c[1, 0, 0] == 0
    print("sang/eval/background.py self-check ok")


if __name__ == "__main__":
    self_check()
