"""Abstract base class for frozen feature-extraction backbones."""

from __future__ import annotations

import abc

import torch.nn as nn


class BackboneBase(nn.Module, abc.ABC):
    """Every backbone must declare ``emb_dim`` and implement ``forward``.

    The forward method receives a batch of normalised images and returns a dict
    with at least ``features`` and ``cls_token``.
    """

    emb_dim: int  # subclasses must set this

    @abc.abstractmethod
    def forward(self, x):
        """
        Parameters
        ----------
        x : Tensor (B, 3, H, W)
            Batch of ImageNet-normalised images.

        Returns
        -------
        dict with:
            features  : Tensor (B, emb_dim, H_f, W_f)
            cls_token : Tensor (B, emb_dim)
        """
        ...
