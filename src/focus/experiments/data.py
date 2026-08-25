"""Embedding pair-data loading for dataset experiments.

A :class:`PairData` is an immutable bundle of the tensors needed to train and
evaluate a pairwise ordinal head on one pool of comparison pairs.  The
:class:`EmbeddingStore` loads the ``{config}__cls.npy`` payloads produced by
``scripts/extract_embeddings_pipeline.py`` and slices them by the
stored ``split`` field, caching raw arrays so repeated settings are cheap.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Literal, Sequence

import numpy as np
import torch

Which = Literal["train", "val", "all"]


def find_repo_root(start: Path | None = None) -> Path:
    """Walk upward from *start* until a folder containing ``data/datasets`` is found."""
    here = (start or Path.cwd()).resolve()
    for cand in [here, *here.parents]:
        if (cand / "data" / "datasets").exists():
            return cand
    raise FileNotFoundError(
        "Could not locate repo root (no 'data/datasets' found upward "
        f"from {here})."
    )


def default_embeddings_root() -> Path:
    return find_repo_root() / "data" / "datasets"


@dataclass(frozen=True)
class PairData:
    """A pool of comparison pairs on a single device.

    Shapes (N = number of pairs):
        emb:    (N, 2, D)   per-point feature embeddings (point A, point B)
        cls:    (N, 2, Dc)  per-point CLS / scene tokens
        label:  (N,)        ordinal label in {-1, 0, +1}  (B vs A)
        within: (N,)        True if the pair is within a single image
    """

    emb: torch.Tensor
    cls: torch.Tensor
    label: torch.Tensor
    within: torch.Tensor

    def __post_init__(self) -> None:
        n = self.emb.shape[0]
        if not (self.cls.shape[0] == self.label.shape[0] == self.within.shape[0] == n):
            raise ValueError("PairData tensors must share the same first dimension.")
        if self.emb.ndim != 3 or self.emb.shape[1] != 2:
            raise ValueError(f"emb must be (N, 2, D); got {tuple(self.emb.shape)}.")

    # ── shape helpers ──────────────────────────────────────────────────────
    @property
    def n(self) -> int:
        return int(self.emb.shape[0])

    @property
    def emb_dim(self) -> int:
        return int(self.emb.shape[-1])

    @property
    def cls_dim(self) -> int:
        return int(self.cls.shape[-1])

    @property
    def device(self) -> torch.device:
        return self.emb.device

    def __len__(self) -> int:
        return self.n

    # ── transforms ─────────────────────────────────────────────────────────
    def to(self, device: torch.device | str) -> "PairData":
        return PairData(
            self.emb.to(device), self.cls.to(device),
            self.label.to(device), self.within.to(device),
        )

    def subset(self, idx: torch.Tensor) -> "PairData":
        return PairData(self.emb[idx], self.cls[idx], self.label[idx], self.within[idx])

    def split_holdout(self, val_frac: float, seed: int) -> tuple["PairData", "PairData"]:
        """Deterministically carve a *val_frac* early-stopping holdout out of this pool.

        Returns ``(train_part, val_part)``, stratified by label
        (``-1``/``0``/``+1``): each class's indices are shuffled and split at
        *val_frac* independently, so train and val carry the same
        equal/unequal-label proportion as the full pool -- mirroring the
        stratified split used when the self-annotated datasets are first
        assembled (``build_combined_dataset.py``). Per-class rounding
        remainders are handed to the largest classes first so the *total*
        val size still matches ``max(1, round(n * val_frac))`` exactly. The
        split depends only on ``(labels, seed)`` so every model sees an
        identical holdout for a given seed.
        """
        if not 0.0 < val_frac < 1.0:
            raise ValueError(f"val_frac must be in (0, 1); got {val_frac}.")
        g = torch.Generator().manual_seed(int(seed))
        n_val_total = max(1, int(round(self.n * val_frac)))

        classes = sorted(torch.unique(self.label).tolist())
        shuffled: dict[int, torch.Tensor] = {}
        n_val_by_class: dict[int, int] = {}
        for cls in classes:
            idx = (self.label == cls).nonzero(as_tuple=True)[0]
            shuffled[cls] = idx[torch.randperm(idx.numel(), generator=g)]
            n_val_by_class[cls] = int(idx.numel() * val_frac)  # floor

        remainder = n_val_total - sum(n_val_by_class.values())
        for cls in sorted(classes, key=lambda c: shuffled[c].numel(), reverse=True):
            if remainder <= 0:
                break
            if n_val_by_class[cls] < shuffled[cls].numel():
                n_val_by_class[cls] += 1
                remainder -= 1

        vi = torch.cat([shuffled[c][:n_val_by_class[c]] for c in classes])
        ti = torch.cat([shuffled[c][n_val_by_class[c]:] for c in classes])
        return self.subset(ti.to(self.device)), self.subset(vi.to(self.device))

    @staticmethod
    def concat(parts: Sequence["PairData"]) -> "PairData":
        parts = [p for p in parts if p.n > 0]
        if not parts:
            raise ValueError("Cannot concat an empty sequence of PairData.")
        return PairData(
            torch.cat([p.emb for p in parts]),
            torch.cat([p.cls for p in parts]),
            torch.cat([p.label for p in parts]),
            torch.cat([p.within for p in parts]),
        )


class EmbeddingStore:
    """Loads and caches ``{config}__cls.npy`` payloads, sliced by split.

    Parameters
    ----------
    config:
        Embedding config name, e.g. ``"dinov2_vits__anyup"``.
    root:
        Folder containing ``{dataset}/embeddings/{config}__cls.npy``.
    device:
        Torch device the returned :class:`PairData` tensors live on.
    """

    def __init__(
        self,
        config: str,
        root: Path | str | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.config = config
        self.root = Path(root) if root is not None else default_embeddings_root()
        self.device = torch.device(device)
        self._raw: dict[str, dict] = {}

    # ── raw loading ────────────────────────────────────────────────────────
    def _path(self, dataset: str) -> Path:
        return self.root / dataset / "embeddings" / f"{self.config}__cls.npy"

    def available(self, datasets: Iterable[str]) -> list[str]:
        """Subset of *datasets* whose embedding file exists on disk."""
        return [d for d in datasets if self._path(d).exists()]

    def _load_raw(self, dataset: str) -> dict:
        if dataset not in self._raw:
            path = self._path(dataset)
            if not path.exists():
                raise FileNotFoundError(f"Missing embeddings for '{dataset}': {path}")
            self._raw[dataset] = np.load(path, allow_pickle=True).item()
        return self._raw[dataset]

    # ── public API ─────────────────────────────────────────────────────────
    def pairs(self, dataset: str, which: Which = "all") -> PairData:
        """Return the pairs of *dataset* restricted to the *which* split."""
        d = self._load_raw(dataset)
        split = np.asarray(d["split"])
        if which == "all":
            mask = np.ones(len(split), dtype=bool)
        elif which in ("train", "val"):
            mask = split == which
        else:  # pragma: no cover - guarded by typing
            raise ValueError(f"Unknown split selector '{which}'.")

        emb = torch.as_tensor(d["point_embeddings"][mask], dtype=torch.float32)
        cls = torch.as_tensor(d["point_cls"][mask], dtype=torch.float32)
        label = torch.as_tensor(np.asarray(d["labels"])[mask], dtype=torch.long)
        within_np = np.array([t != "cross" for t in np.asarray(d["comparison_type"])[mask]])
        within = torch.as_tensor(within_np, dtype=torch.bool)
        return PairData(emb, cls, label, within).to(self.device)

    def pool(self, specs: Sequence[tuple[str, Which]]) -> PairData:
        """Concatenate several ``(dataset, which)`` selections into one pool."""
        return PairData.concat([self.pairs(ds, which) for ds, which in specs])


__all__ = [
    "Which",
    "PairData",
    "EmbeddingStore",
    "find_repo_root",
    "default_embeddings_root",
]
