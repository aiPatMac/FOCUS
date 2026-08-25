"""Pairwise-disagreement metric (HDR@tau / PDR) and threshold helpers.

A model produces a per-pair score difference ``diff = score(B) - score(A)``.
Thresholding ``diff`` at ``±tau`` yields a prediction in {-1, 0, +1}; the PDR is
the fraction of pairs whose prediction disagrees with the human label.  Lower is
better; the trivial all-equal predictor scores ``mean(label != 0)``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch


def predict(diff: torch.Tensor, tau: float) -> torch.Tensor:
    """Threshold score differences into {-1, 0, +1} with a ``±tau`` dead-zone."""
    pred = torch.zeros_like(diff, dtype=torch.long)
    pred[diff > tau] = 1
    pred[diff < -tau] = -1
    return pred


@dataclass(frozen=True)
class PDRResult:
    """PDR (HDR@tau) broken down overall and by within/cross-image pairs."""

    tau: float
    pdr: float
    within: float
    cross: float
    n: int
    n_within: int
    n_cross: int
    trivial: float

    def as_dict(self) -> dict:
        return asdict(self)


def trivial_pdr(label: torch.Tensor) -> float:
    """PDR of the all-equal predictor (predict 0 everywhere)."""
    if label.numel() == 0:
        return float("nan")
    return float((label != 0).float().mean().item())


def pdr(diff: torch.Tensor, label: torch.Tensor, within: torch.Tensor, tau: float) -> PDRResult:
    """Compute overall / within / cross PDR at threshold *tau*."""
    diff = diff.detach().reshape(-1)
    label = label.reshape(-1)
    within = within.reshape(-1).bool()
    err = predict(diff, tau) != label

    def _mean(mask: torch.Tensor) -> float:
        return float(err[mask].float().mean().item()) if mask.any() else float("nan")

    return PDRResult(
        tau=float(tau),
        pdr=float(err.float().mean().item()) if err.numel() else float("nan"),
        within=_mean(within),
        cross=_mean(~within),
        n=int(err.numel()),
        n_within=int(within.sum().item()),
        n_cross=int((~within).sum().item()),
        trivial=trivial_pdr(label),
    )


def select_tau(
    diff: torch.Tensor,
    label: torch.Tensor,
    tau_grid: np.ndarray | list[float],
) -> float:
    """Pick the threshold in *tau_grid* that minimises PDR on this (validation) pool."""
    diff = diff.detach().reshape(-1)
    label = label.reshape(-1)
    grid = np.asarray(tau_grid, dtype=float)
    scores = [float((predict(diff, float(t)) != label).float().mean().item()) for t in grid]
    return float(grid[int(np.argmin(scores))])


__all__ = ["PDRResult", "predict", "pdr", "trivial_pdr", "select_tau"]
