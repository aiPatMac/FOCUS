"""Build the fused-backbone embedding file FOCUS trains on.

Concatenates the two already-extracted ``{config}__cls.npy`` payloads
(``dinov2_vits__loftup`` and ``dinov2_vits__uplift``, the two branches FOCUS
fuses) along the feature axis, for every self-annotated dataset, and writes a
combined ``dinov2_vits__loftup+dinov2_vits__uplift__cls.npy`` file with the
same schema (row-for-row aligned, same ``labels``/``split``/``pair_uid``/
``comparison_type``), so it loads through the existing
``focus.experiments.data.EmbeddingStore`` unmodified — no new extraction, no
harness changes, just a wider ``point_embeddings``/``point_cls`` tensor
(``D_A + D_B``).

This does *not* re-run the ViT/upsampler pipeline; it only recombines the
per-point embeddings that ``extract_embeddings_pipeline.py`` already produced
for each dataset, which is why it runs in seconds.

Usage
-----
    # the fused pair x all 4 datasets
    python scripts/build_paired_embeddings.py

    # a single dataset
    python scripts/build_paired_embeddings.py --dataset moon_ce3
"""

from __future__ import annotations

import argparse
import itertools
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger("build_paired_embeddings")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DS = PROJECT_ROOT / "data" / "datasets"

DEFAULT_DATASETS = ["moon_ce3", "moon_ce4", "lusnar", "luna_polaris"]
DEFAULT_CONFIGS = [
    "dinov2_vits__loftup",
    "dinov2_vits__uplift",
]

# fields that must be identical across two configs of the same dataset (sanity check)
_ALIGNED_FIELDS = ["sample_id", "labels", "split", "pair_uid", "comparison_type", "domain"]


def pair_config_name(config_a: str, config_b: str) -> str:
    """Canonical (order-independent) combined-config name."""
    a, b = sorted([config_a, config_b])
    return f"{a}+{b}"


def _load(dataset: str, config: str) -> dict:
    path = DS / dataset / "embeddings" / f"{config}__cls.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing embeddings for '{dataset}'/{config}: {path}")
    return np.load(path, allow_pickle=True).item()


def build_pair(dataset: str, config_a: str, config_b: str, overwrite: bool = False) -> Path | None:
    """Concatenate two single-backbone ``__cls.npy`` payloads for *dataset*."""
    combo = pair_config_name(config_a, config_b)
    out_path = DS / dataset / "embeddings" / f"{combo}__cls.npy"
    if out_path.exists() and not overwrite:
        return out_path

    a_name, b_name = sorted([config_a, config_b])
    da, db = _load(dataset, a_name), _load(dataset, b_name)

    for field in _ALIGNED_FIELDS:
        if not np.array_equal(np.asarray(da[field]), np.asarray(db[field])):
            raise ValueError(
                f"'{dataset}': field '{field}' differs between {a_name} and {b_name} — "
                "the two configs are not row-aligned, cannot concatenate."
            )

    point_embeddings = np.concatenate([da["point_embeddings"], db["point_embeddings"]], axis=-1)
    point_cls = np.concatenate([da["point_cls"], db["point_cls"]], axis=-1)

    combined = {
        "point_embeddings": point_embeddings.astype(np.float32),
        "point_cls": point_cls.astype(np.float32),
        "labels": da["labels"],
        "split": da["split"],
        "sample_id": da["sample_id"],
        "pair_uid": da["pair_uid"],
        "comparison_type": da["comparison_type"],
        "domain": da["domain"],
        "image_cls": np.concatenate([da["image_cls"], db["image_cls"]], axis=-1).astype(np.float32),
        "image_cls_names": da["image_cls_names"],
        "settings": {
            "config": combo,
            "kind": "paired",
            "backbone_a": a_name,
            "backbone_b": b_name,
            "emb_dim": int(point_embeddings.shape[-1]),
            "emb_dim_a": int(da["point_embeddings"].shape[-1]),
            "emb_dim_b": int(db["point_embeddings"].shape[-1]),
            "cls_dim": int(point_cls.shape[-1]),
            "dataset": dataset,
            "n_samples": int(point_embeddings.shape[0]),
            "source": "build_paired_embeddings",
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, combined, allow_pickle=True)
    return out_path


def build_all(
    datasets: list[str] = DEFAULT_DATASETS,
    configs: list[str] = DEFAULT_CONFIGS,
    overwrite: bool = False,
) -> list[str]:
    """Build every unordered pair of *configs* for every dataset. Returns combo names."""
    combos = [pair_config_name(a, b) for a, b in itertools.combinations(configs, 2)]
    for dataset in datasets:
        for config_a, config_b in itertools.combinations(configs, 2):
            try:
                path = build_pair(dataset, config_a, config_b, overwrite=overwrite)
                log.info("[%s] %s -> %s", dataset, pair_config_name(config_a, config_b), path)
            except FileNotFoundError as exc:
                log.warning("[%s] skipped %s (%s)", dataset, pair_config_name(config_a, config_b), exc)
    return combos


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="all", help="dataset name or 'all'")
    ap.add_argument("--pair", nargs=2, metavar=("CONFIG_A", "CONFIG_B"), default=None,
                     help="build a single pair only")
    ap.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    datasets = DEFAULT_DATASETS if args.dataset == "all" else [args.dataset]
    if args.pair is not None:
        for dataset in datasets:
            path = build_pair(dataset, args.pair[0], args.pair[1], overwrite=args.overwrite)
            log.info("[%s] -> %s", dataset, path)
        return

    combos = build_all(datasets, args.configs, overwrite=args.overwrite)
    log.info("done: %d pair-configs x %d datasets", len(combos), len(datasets))


if __name__ == "__main__":
    main()
