"""TOLRIZZ: confidence-weighted, tolerant-equality pairwise ranking loss.

Built on L-RIZZ (Schreiber & Driggs-Campbell, 2025); see chapters/05-method.tex
(Equations method-tolrizz, method-confidence, method-total) for the derivation.
The best FOCUS-fused configuration found by the margin/epsilon sweep in
chapters/experiments/02-loss-function.tex (Table exp-loss-best-tolrizz-focusfused)
is margin=0.55, eq_margin=0.05 — the module-level defaults below.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from focus.model import FocusHead

MARGIN = 0.55
EQ_MARGIN = 0.05
BETA_RECON = 0.1


def diff(head: FocusHead, emb: torch.Tensor, cls: torch.Tensor) -> torch.Tensor:
    """Score difference p_b - p_a for a batch of pairs.

    `emb`/`cls` have shape (N, 2, D): index 0 is point A, index 1 is point B.
    """
    return (head.score(emb[:, 1], cls[:, 1]) - head.score(emb[:, 0], cls[:, 0]))[:, 0]


def _reconstruction_term(head: FocusHead, emb: torch.Tensor) -> torch.Tensor:
    xa, xb = emb[:, 0], emb[:, 1]
    return F.mse_loss(head.reconstruct(xa), xa) + F.mse_loss(head.reconstruct(xb), xb)


def _tolrizz_pairs(
    diff: torch.Tensor, label: torch.Tensor, margin: float = MARGIN, eq_margin: float = EQ_MARGIN
) -> torch.Tensor:
    """Per-pair TOLRIZZ loss, Equation (method-tolrizz)."""
    ineq = (label != 0).float()
    eq = (label == 0).float()
    return (
        torch.square(F.relu(margin - label * diff)) * ineq
        + torch.square(F.relu(diff.abs() - eq_margin)) * eq
    )


def tolrizz_loss(
    head: FocusHead,
    emb: torch.Tensor,
    cls: torch.Tensor,
    label: torch.Tensor,
    margin: float = MARGIN,
    eq_margin: float = EQ_MARGIN,
    beta_recon: float = BETA_RECON,
) -> torch.Tensor:
    """L_total = L_conf(TOLRIZZ) + beta * R(x_a, x_b), Equation (method-total)."""
    d = diff(head, emb, cls)
    per_pair = _tolrizz_pairs(d, label, margin, eq_margin)
    log_var = 0.5 * (head.log_var(cls[:, 0])[:, 0] + head.log_var(cls[:, 1])[:, 0])
    ranking = (torch.exp(-log_var) * per_pair + 0.5 * log_var).mean()
    return ranking + beta_recon * _reconstruction_term(head, emb)


__all__ = ["MARGIN", "EQ_MARGIN", "BETA_RECON", "diff", "tolrizz_loss"]
