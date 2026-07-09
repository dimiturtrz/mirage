"""Pure-torch FPS + grouping — the CUDA-free Point-MAE front-end (replaces pointnet2_ops + knn_cuda).

Verified against brute-force references so the reimplementation is provably equivalent to the CUDA
grouping it stands in for: square_distance == cdist², FPS spreads across clusters, and each patch holds
the true k-nearest of its centre.
"""
import torch

from surfscan.models.pointmae_group import (
    farthest_point_sample,
    group,
    index_points,
    square_distance,
)


def test_square_distance_matches_cdist():
    a, b = torch.randn(2, 5, 3), torch.randn(2, 4, 3)
    got = square_distance(a, b)
    assert got.shape == (2, 5, 4)
    assert torch.allclose(got, torch.cdist(a, b) ** 2, atol=1e-4)


def test_index_points_gathers_batchwise():
    pts = torch.arange(2 * 4 * 3).reshape(2, 4, 3).float()
    out = index_points(pts, torch.tensor([[0, 2], [1, 3]]))
    assert out.shape == (2, 2, 3)
    assert torch.equal(out[0, 0], pts[0, 0]) and torch.equal(out[1, 1], pts[1, 3])


def test_fps_spans_two_clusters():
    xyz = torch.cat([torch.zeros(1, 10, 3), torch.full((1, 10, 3), 100.0)], 1)
    picked = index_points(xyz, farthest_point_sample(xyz, 4))[0]
    assert (picked.norm(dim=1) < 1).sum() >= 1 and (picked.norm(dim=1) > 50).sum() >= 1


def test_group_shapes_and_knn_is_true_nearest():
    torch.manual_seed(0)
    xyz = torch.randn(1, 64, 3)
    nb, ctr, nidx, cidx = group(xyz, num_group=8, group_size=6)
    assert nb.shape == (1, 8, 6, 3) and ctr.shape == (1, 8, 3)
    assert nidx.shape == (1, 8, 6) and cidx.shape == (1, 8)

    ref = torch.cdist(ctr, xyz).topk(6, dim=-1, largest=False).indices     # brute-force k-nearest
    assert torch.equal(nidx.sort(-1).values, ref.sort(-1).values)
    assert (nb.norm(dim=-1).min(-1).values < 1e-5).all()                   # centre ∈ its own patch -> a ~0 vector
