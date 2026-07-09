"""M3DM's Point-MAE backbone loads + forward-passes CUDA-free through our shims (no pointnet2_ops/knn_cuda).

Integration of surfscan.models.pointmae (op-shims + loader) with M3DM's checked-out PointTransformer:
proves the real point-transformer runs on our torch>=2.7 sm_120 stack with zero custom CUDA ops. Needs
timm (features extra) + external/M3DM checked out, so it skips where either is absent (CI).
"""
from pathlib import Path

import pytest


def test_pointmae_loads_cuda_free_and_forwards():
    pytest.importorskip("timm")
    if not (Path(__file__).resolve().parents[2] / "external" / "M3DM").exists():
        pytest.skip("external/M3DM not checked out")
    import torch  # noqa: PLC0415

    from surfscan.models.pointmae import load_pointmae, pointmae_features  # noqa: PLC0415

    net = load_pointmae("cpu", num_group=32, group_size=16, ckpt=None)     # random init, small, no CUDA ops
    feats, centers = pointmae_features(net, torch.randn(1, 300, 3))
    assert feats.shape == (1, 32, 1152)                                    # 3 transformer layers x 384
    assert centers.shape == (1, 32, 3)
