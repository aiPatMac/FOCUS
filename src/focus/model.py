"""FOCUS: Fused Ordinal-CORN with Uncertainty-weighted Scoring.

Two frozen backbones (dinov2_vits__loftup, dinov2_vits__uplift) are fused by a
small trainable encoder, then scored by a CORN ordinal head. Two more
training-only branches — confidence and reconstruction — feed the TOLRIZZ loss
in `focus.loss`; neither is used at inference. See chapters/05-method.tex in
the thesis for the full derivation.
"""

from __future__ import annotations

import torch
import torch.nn as nn

N_RANKS = 5  # K in the thesis; Section 3.3 (subsec:exp-head-k-sweep) shows the result is not sensitive to this


class FusionEncoder(nn.Module):
    """Per-branch LayerNorm->Linear->ReLU, concatenated, then Linear->ReLU."""

    def __init__(self, dims: list[int], hidden: int = 128):
        super().__init__()
        self.dims = list(dims)
        self.branches = nn.ModuleList(
            [nn.Sequential(nn.LayerNorm(d), nn.Linear(d, hidden), nn.ReLU()) for d in self.dims]
        )
        self.fuse = nn.Sequential(nn.Linear(len(self.dims) * hidden, hidden), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts = torch.split(x, self.dims, dim=-1)
        return self.fuse(torch.cat([b(p) for b, p in zip(self.branches, parts)], dim=-1))


class FocusHead(nn.Module):
    """CORN ordinal head over the fused representation, plus the two
    training-only branches (confidence, reconstruction) TOLRIZZ needs.

    Parameters
    ----------
    dims : the per-branch input feature dims to fuse, e.g. [384, 384] for the
        loftup+uplift pair.
    cls_dim : dimensionality of the per-image CLS token the confidence branch
        reads from (see `focus.experiments.data.PairData.cls`).
    n_ranks : K in the CORN head (K-1 independent ordinal nodes).
    hidden : width of the fusion encoder / rank head.
    """

    def __init__(self, dims: list[int], cls_dim: int, n_ranks: int = N_RANKS, hidden: int = 128):
        super().__init__()
        self.dims = list(dims)
        self.n_ranks = n_ranks
        self.encoder = FusionEncoder(dims, hidden)
        self.rank = nn.Linear(hidden, n_ranks - 1)
        self.recon = nn.Linear(hidden, sum(self.dims))
        self.cls_norm = nn.LayerNorm(cls_dim)
        self.conf = nn.Linear(cls_dim, 1)
        with torch.no_grad():
            # bias the K-1 nodes to start near their "expected" cutoff so cumP
            # isn't uniformly saturated at init
            self.rank.bias.copy_(torch.arange(n_ranks - 1, 0, -1).float() / (n_ranks - 1) + 0.5)

    def cumP(self, x: torch.Tensor) -> torch.Tensor:
        """Cumulative probabilities P(y > r_k), Equation (method-corn)."""
        return torch.cumprod(torch.sigmoid(self.rank(self.encoder(x))), dim=-1)

    def score(self, x: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
        """The deployed traversability score p in [0, 1]: mean of cumP."""
        return self.cumP(x).mean(-1, keepdim=True)

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Training-only: fused representation -> reconstructed input."""
        return self.recon(self.encoder(x))

    def log_var(self, c: torch.Tensor) -> torch.Tensor:
        """Training-only: per-image confidence log-variance from the CLS token."""
        return self.conf(self.cls_norm(c))


__all__ = ["N_RANKS", "FusionEncoder", "FocusHead"]
