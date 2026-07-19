"""Load M3DM's Point-MAE point-transformer backbone, CUDA-free, via pure-torch op shims.

Use-don't-reimplement (CLAUDE.md): we import M3DM's exact `PointTransformer` from the checked-out repo
(external/M3DM @62b7f46) so the ShapeNet weights load unmodified — but M3DM imports two CUDA-compiled
ops (`pointnet2_ops` FPS/gather + `knn_cuda` KNN) pinned to torch 1.9 that will NOT build for Blackwell
(sm_120). We install pure-torch stand-ins (backed by pointmae_group) into sys.modules BEFORE importing
M3DM, so its grouping runs with zero custom CUDA ops on our torch>=2.7 cu130 stack.

Fetch step (dep checked out, never vendored):
    git clone https://github.com/nomewang/M3DM.git external/M3DM
    git -C external/M3DM checkout 62b7f468aa006f6b348f09e02f85655f416c9d76
    # Point-MAE ShapeNet checkpoint (Google Drive, per M3DM README) -> external/M3DM/checkpoints/pointmae_pretrain.pth
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn

from surfscan.models.pointmae_group import PointmaeGroup

_M3DM = Path(__file__).resolve().parents[2] / "external" / "M3DM"


class Pointmae:
    """Loader + pure-torch op-shims for M3DM's Point-MAE backbone (public names kept as staticmethods)."""

    @staticmethod
    def _install_op_shims():
        """Pure-torch stand-ins for the two CUDA deps M3DM imports, matching their call signatures."""
        if "pointnet2_ops" not in sys.modules:
            pu = types.ModuleType("pointnet2_ops.pointnet2_utils")
            pu.furthest_point_sample = lambda xyz, n: PointmaeGroup.farthest_point_sample(xyz, n).int()   # (B,n) idx
            pu.gather_operation = lambda feats, idx: (                                       # (B,C,N),(B,S)->(B,C,S)
                PointmaeGroup.index_points(feats.transpose(1, 2).contiguous(), idx.long()).transpose(1, 2).contiguous())
            pkg = types.ModuleType("pointnet2_ops")
            pkg.pointnet2_utils = pu
            sys.modules["pointnet2_ops"], sys.modules["pointnet2_ops.pointnet2_utils"] = pkg, pu
        if "knn_cuda" not in sys.modules:
            knn = types.ModuleType("knn_cuda")

            class KNN:                                          # M3DM: KNN(k, transpose_mode)(ref, query)->(_, idx)
                def __init__(self, k: int, transpose_mode: bool = True) -> None:  # noqa: FBT002  (mirrors knn_cuda.KNN's signature)
                    self.k = k
                    self.transpose_mode = transpose_mode        # our output is the transpose_mode=True layout

                def __call__(self, ref: Float[Tensor, "b m 3"], query: Float[Tensor, "b n 3"]) -> tuple[Float[Tensor, "b n k"], Int[Tensor, "b n k"]]:
                    dist, idx = PointmaeGroup.square_distance(query, ref).topk(self.k, dim=-1, largest=False)
                    return dist, idx

            knn.KNN = KNN
            sys.modules["knn_cuda"] = knn

    @staticmethod
    def load_pointmae(device: torch.device | str, *, num_group: int = 1024, group_size: int = 128,
                      encoder_dims: int = 384, ckpt: str | None = "pointmae_pretrain.pth") -> nn.Module:
        """M3DM's `PointTransformer` on `device`, CUDA-free, eval mode. `ckpt` (relative to
        external/M3DM/checkpoints/, or absolute) loads ShapeNet weights; None = random init (for shape tests)."""
        Pointmae._install_op_shims()
        if str(_M3DM) not in sys.path:
            sys.path.insert(0, str(_M3DM))
        from models.models import PointTransformer  # noqa: PLC0415  (needs sys.path + shims installed first)

        net = PointTransformer(group_size=group_size, num_group=num_group, encoder_dims=encoder_dims)
        if ckpt is not None:
            path = Path(ckpt) if Path(ckpt).is_absolute() else _M3DM / "checkpoints" / ckpt
            net.load_model_from_ckpt(str(path))                 # M3DM's own loader (strict=False)
        for p in net.parameters():
            p.requires_grad_(requires_grad=False)
        return net.to(device).eval()

    @staticmethod
    @torch.no_grad()
    def pointmae_features(net: nn.Module, xyz: Float[Tensor, "b n 3"]) -> tuple[Float[Tensor, "b g c"], Float[Tensor, "b g 3"]]:
        """xyz (B,N,3) -> (per-group features [B,G,C], centers [B,G,3]). Point-MAE forward wants (B,3,N)."""
        feats, centers, _, _ = net(xyz.transpose(1, 2).contiguous())
        return feats.transpose(1, 2).contiguous(), centers      # (B,G,C), (B,G,3)
