"""Feature-space reconstruction (RD4AD-lite) — reconstruct in FEATURE space, not pixels.

Pixel reconstruction failed because raw-residual tracks geometric complexity, not defect-ness. A
frozen ImageNet backbone maps the image into a feature space where normal complexity is already
normalized; an AE that reconstructs the NORMAL feature map flags anomalies by feature-recon error.
The clean test of "is it the reconstruction idea that's wrong, or the space it runs in?"
"""
from __future__ import annotations

import torch
import torchvision
from jaxtyping import Float

from surfscan.models.feat_ae import FeatAE

__all__ = ["FeatAE", "FeatExtractor"]

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class FeatExtractor:
    def __init__(
        self, backbone: str = "wide_resnet50_2", layer: str = "layer2", device: str | torch.device = "cuda"
    ) -> None:
        net = getattr(torchvision.models, backbone)(weights="DEFAULT").to(device).eval()
        for p in net.parameters():
            p.requires_grad_(requires_grad=False)
        self._feat: torch.Tensor | None = None
        getattr(net, layer).register_forward_hook(lambda _m, _i, o: setattr(self, "_feat", o))
        self.net = net
        self.mean, self.std = _MEAN.to(device), _STD.to(device)

    @torch.no_grad()
    def __call__(self, x: Float[torch.Tensor, "n 3 h w"]) -> Float[torch.Tensor, "n c fh fw"]:
        self.net((x - self.mean) / self.std)
        return self._feat
