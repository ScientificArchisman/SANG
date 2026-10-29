"""Vectors over LivePortrait's 70-d motion. Pure tensors: training never imports the renderer.

    m = [scale, yaw, pitch, roll, t_x, t_y, t_z, exp_1..exp_63]      (KDTalker's layout)

`wide` is everything the renderer reads except t_z: the 69 numbers the motion latent (sang.model.latent) is fitted
on, and so what the model generates. `to_target` is a 42-d description, the head rotation plus the head-frame
expression of the 13 keypoints LivePortrait's expression transfer drives (EXPR_KP; AVTR-1, arXiv 2609.22913). It
is no longer generated; it describes the photo to the model (sang.model.flow.Norm.ref) and is what the lip / eye
openness readouts (sang.eval.naturalness) are fitted on. REGIONS are its mouth / eye / brow / rotation columns.
`describe` picks which of the two describes the photo.
"""
import torch

M_DIM = 70
EXP_DIM = 63

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

SHAPE_KP = tuple(k for k in range(21) if k not in EXPR_KP)
WIDE_DIM = 3 + EXP_DIM + 3
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


def to_target(m: torch.Tensor) -> torch.Tensor:
    """[N,70] motion -> [N,42] target: (pitch, yaw, roll) + head-frame exp of EXPR_KP."""
    m = m.float()
    exp_head = m[:, 7:M_DIM].reshape(-1, 21, 3) @ _rot(m).transpose(1, 2)
    return torch.cat([m[:, [2, 1, 3]], exp_head[:, list(EXPR_KP)].reshape(-1, 3 * len(EXPR_KP))], 1)


def wide(m: torch.Tensor) -> torch.Tensor:
    """[N,70] motion -> [N,69]: everything LivePortrait renders except t_z, which it zeroes. (pitch, yaw, roll),
    the head-frame exp of all 21 keypoints, and the absolute placement (t_x, t_y, log scale). Nothing is held from
    the photo, so a model over this vector (sang.model.latent) decides for itself which numbers matter."""
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


def describe(m: torch.Tensor, width: int) -> torch.Tensor:
    """The photo as the model reads it: `wide` (69, cfg photo=wide) or `to_target` (42, the default)."""
    return {WIDE_DIM: wide, T_DIM: to_target}[width](m)


def self_check() -> None:
    assert sorted(i for r in REGIONS.values() for i in r) == list(range(T_DIM)), "regions must partition 42"
    assert [len(REGIONS[k]) for k in ("rot", "brow", "eyes", "mouth")] == [3, 6, 15, 18]
    g = torch.Generator().manual_seed(0)
    m = torch.randn(5, M_DIM, generator=g)
    m[:, 0] = 1.0 + 0.1 * m[:, 0].abs()
    m[:, 1:4] = 30.0 * torch.rand(5, 3, generator=g) - 15
    R = _rot(m)
    assert torch.allclose(R @ R.transpose(1, 2), torch.eye(3).expand(5, 3, 3), atol=1e-5), "not orthonormal"
    assert torch.allclose(torch.linalg.det(R), torch.ones(5), atol=1e-5), "not a rotation"
    assert torch.allclose(from_wide(wide(m), m), m, atol=1e-4), "wide round trip broke"
    w = wide(m)
    m2 = from_wide(torch.cat([w[:, :3] + 10.0, w[:, 3:]], 1), m)
    assert torch.allclose(to_target(m2)[:, 3:], to_target(m)[:, 3:], atol=1e-4), "exp is not pose-invariant"
    assert sorted(i for r in WIDE_REGIONS.values() for i in r) == list(range(WIDE_DIM)), "wide regions must partition 69"
    print("sang/model/target.py self-check ok")


if __name__ == "__main__":
    self_check()
