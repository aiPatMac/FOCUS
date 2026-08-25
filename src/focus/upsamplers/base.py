"""Upsampler base class — all upsamplers follow this interface."""

from __future__ import annotations

import abc

import torch.nn as nn


class UpsamplerBase(nn.Module, abc.ABC):
    """Every upsampler maps low-res features to high-res features.

    Interface::

        upsampler(source, guidance) -> upsampled_features

    Parameters
    ----------
    source : Tensor (B, C, H_lo, W_lo)
        Low-resolution feature map (e.g. 16x16 patch tokens).
    guidance : Tensor (B, 3, H_hi, W_hi)
        High-resolution RGB image used as guidance (some upsamplers ignore it).

    Returns
    -------
    Tensor (B, C, H_hi, W_hi)
        Upsampled feature map.
    """

    @abc.abstractmethod
    def forward(self, source, guidance):
        ...
