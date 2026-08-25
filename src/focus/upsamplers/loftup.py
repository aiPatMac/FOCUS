"""LoftUp coordinate-based feature upsampler (ICCV 2025 oral).

Loads a pretrained LoftUp upsampler via ``torch.hub`` from
`andrehuang/loftup <https://github.com/andrehuang/loftup>`_.

LoftUp checkpoints are *backbone-specific* — each one is trained on the
features of a particular backbone. Pretrained variants exist for DINOv2
(S/14 and B/14, with/without registers), CLIP and SigLIP; there is **no
DINOv3 checkpoint**. The 384-dim ``loftup_dinov2s`` variant matches the
project's ``dinov2_vits`` backbone.

The hub model is a ``UpsamplerwithChannelNorm`` whose forward signature is
``forward(lr_feats, img)`` — identical to this project's
``UpsamplerBase.forward(source, guidance)``.
"""

from __future__ import annotations

import torch

from focus.registry import register
from focus.upsamplers.base import UpsamplerBase

# Map feature dimensionality to the matching pretrained LoftUp variant.
_VARIANT_BY_DIM = {384: "loftup_dinov2s", 768: "loftup_dinov2b"}


@register("upsampler", "loftup")
class LoftUpUpsampler(UpsamplerBase):
    """LoftUp feature upsampler loaded from torch.hub.

    Parameters
    ----------
    feat_dim : int
        Feature channel dimension (384 for DINOv2 ViT-S). Used to pick the
        default pretrained variant when ``variant`` is not given.
    pretrained : bool
        Load pretrained LoftUp weights from the hub checkpoint.
    variant : str | None
        Explicit torch.hub model name (e.g. ``"loftup_dinov2s"``,
        ``"loftup_dinov2b"``, ``"loftup_clip"``, ``"loftup_siglip"``).
        Defaults to the variant matching ``feat_dim``.
    """

    def __init__(
        self,
        feat_dim: int = 384,
        pretrained: bool = True,
        variant: str | None = None,
        **kwargs,
    ):
        super().__init__()
        if variant is None:
            variant = _VARIANT_BY_DIM.get(feat_dim, "loftup_dinov2s")
        self._model = torch.hub.load(
            "andrehuang/loftup", variant, pretrained=pretrained
        )
        for p in self._model.parameters():
            p.requires_grad = False

    def forward(self, source: torch.Tensor, guidance: torch.Tensor) -> torch.Tensor:
        """LoftUp interface: ``upsampler(lr_feats, img) -> hr_feats``."""
        return self._model(source, guidance)
