"""Model protocol and reproducibility helpers for the experiment harness.

Models are defined by the caller (e.g. a notebook).  A *model factory* trains a
head on a train/val pool and returns a :class:`ScoreModel` — anything exposing
``score(emb, cls) -> Tensor``.  This matches the small ``nn.Module`` heads used
throughout the notebooks (``PointHead``, ``CORAL``, ``CORN`` …) with no
adaptation: the harness derives the pairwise score difference itself.
"""

from __future__ import annotations

import os
import random
from typing import Callable, Protocol, runtime_checkable

import numpy as np
import torch

from .data import PairData


@runtime_checkable
class ScoreModel(Protocol):
    """Anything that maps per-point ``(emb, cls)`` to a scalar score."""

    def score(self, emb: torch.Tensor, cls: torch.Tensor) -> torch.Tensor:
        """Return a ``(M,)`` or ``(M, 1)`` traversability score per point."""
        ...


@runtime_checkable
class PairModel(Protocol):
    """A model that scores a *pair* directly (e.g. RankSVM / forest on diff features).

    Returns a signed per-pair difference score (``> 0`` means B is more
    traversable than A), comparable in scale to a neural ``score(B) - score(A)``
    so the same ``tau`` thresholds apply.
    """

    def pair_diff(self, data: "PairData") -> torch.Tensor:
        ...


# A factory: (train_pool, early_stop_val, seed) -> trained model (point- or pair-scoring).
ModelFactory = Callable[[PairData, PairData, int], object]


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and Torch for reproducible runs (mirrors scripts/train.py)."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


@torch.no_grad()
def score_diff(model: ScoreModel | PairModel, data: PairData) -> torch.Tensor:
    """Per-pair ``score(B) - score(A)`` for every pair in *data*.

    If *model* implements :class:`PairModel` (``pair_diff``), that is used
    directly so pairwise classifiers (SVM, random forest) are first-class.  The
    returned tensor always lives on ``data``'s device for downstream metrics.
    """
    if hasattr(model, "pair_diff"):
        diff = model.pair_diff(data)
        return torch.as_tensor(diff, dtype=torch.float32, device=data.emb.device).reshape(-1)
    sa = model.score(data.emb[:, 0], data.cls[:, 0]).reshape(-1)
    sb = model.score(data.emb[:, 1], data.cls[:, 1]).reshape(-1)
    return sb - sa


__all__ = ["ScoreModel", "PairModel", "ModelFactory", "seed_everything", "score_diff"]
