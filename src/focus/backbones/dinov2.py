"""DINOv2 ViT-S/14 backbone with a pluggable upsampler.

Only the ViT-S/14 variant is kept here — it's the one FOCUS's fused
representation is built on (paired with the LoftUp upsampler; see
`focus.upsamplers.loftup`). The original project also had a ViT-B/14 variant
and several other upsamplers, trimmed here since they play no part in the
final architecture.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from focus.backbones.base import BackboneBase
from focus.registry import register, create

# Ensure upsampler registrations are loaded
import focus.upsamplers  # noqa: F401


@register("backbone", "dinov2_vits")
class DINOv2ViTS(BackboneBase):
    """DINOv2 ViT-Small producing 384-dim patch features + CLS token.

    The upsampler is selected via the registry — all upsamplers (bilinear,
    nearest, featup, …) are treated identically.

    Parameters
    ----------
    pretrained : bool
        Load pretrained DINOv2 weights.
    upsampler_name : str | None
        Registered upsampler name (e.g. ``"featup"``, ``"bilinear"``,
        ``"nearest"``).  ``None`` or ``"none"`` skips upsampling.
    upsampler_kwargs : dict | None
        Extra kwargs forwarded to the upsampler constructor.
    """

    emb_dim = 384

    def __init__(
        self,
        pretrained: bool = True,
        upsampler_name: str | None = "featup",
        upsampler_kwargs: dict | None = None,
    ):
        super().__init__()
        self._dinov2 = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vits14"
        )
        self._upsampler = None
        if upsampler_name and upsampler_name != "none":
            kw = {"feat_dim": self.emb_dim, **(upsampler_kwargs or {})}
            if "pretrained" not in kw:
                kw["pretrained"] = pretrained
            self._upsampler = create("upsampler", upsampler_name, **kw)

        # Freeze all parameters
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, x):
        out = self._dinov2.forward_features(x)
        patch_tokens = out["x_norm_patchtokens"]  # (B, N, 384)
        cls_token = out["x_norm_clstoken"]  # (B, 384)
        B, N, C = patch_tokens.shape
        h = w = int(N ** 0.5)
        features = patch_tokens.permute(0, 2, 1).reshape(B, C, h, w)

        if self._upsampler is not None:
            features = self._upsampler(features, x)

        return {"features": features, "cls_token": cls_token}
