"""Grid-native normals — a known plane has a known normal, oriented toward the camera, unit length.

Equivalence-class: a flat fronto-parallel plane (constant z) -> normal is ±z, and the toward-origin
orientation fixes the sign; interior normals are unit; pixels outside `valid` are zeroed.
"""
import torch

from core.geometry import Geometry


def _plane(b=1, h=8, w=8, z=-1.0):
    ys, xs = torch.meshgrid(torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32),
                            indexing="ij")
    xyz = torch.zeros(b, 3, h, w)
    xyz[:, 0], xyz[:, 1], xyz[:, 2] = xs, ys, z          # plane at constant z in front of the origin
    valid = torch.ones(b, 1, h, w)
    return xyz, valid


def test_flat_plane_normal_is_unit_z_toward_origin():
    xyz, valid = _plane(z=-1.0)
    n = Geometry.grid_normals(xyz, valid)
    inner = n[:, :, 1:-1, 1:-1]                           # drop the boundary (one-sided diffs there)
    assert torch.allclose(inner.norm(dim=1), torch.ones_like(inner[:, 0]), atol=1e-4)   # unit
    assert inner[:, 2].abs().mean() > 0.99               # normal is essentially ±z on a fronto-parallel plane
    assert inner[:, :2].abs().mean() < 1e-3             # no tangential component
    # oriented toward the camera at the origin: n · position <= 0 (opposes the view ray)
    assert (inner * xyz[:, :, 1:-1, 1:-1]).sum(1).max() <= 1e-4


def test_normal_zero_outside_valid():
    xyz, valid = _plane()
    valid[:, :, :2, :] = 0.0                              # top rows off-object
    n = Geometry.grid_normals(xyz, valid)
    assert torch.count_nonzero(n[:, :, 0, :]) == 0        # a fully-invalid row has no defined normal
