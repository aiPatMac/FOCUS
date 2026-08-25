"""Point-embedding extraction pipeline for FOCUS's two fused branches.

Generates point embeddings for a dataset with the ``dinov2_vits__loftup`` and
``dinov2_vits__uplift`` configs (the two branches FOCUS fuses; see
``chapters/05-method.tex`` in the thesis), reading the universal index at
``data/datasets/combined/annotations.csv`` and writing one aligned file per
config::

    data/datasets/{dataset}/embeddings/{config}.npy

Each output is row-for-row aligned with the rows of ``annotations.csv`` that
belong to ``{dataset}`` (in CSV order), so a notebook can do::

    df  = pd.read_csv(".../combined/annotations.csv")
    sub = df[df.dataset == "moon_ce3"].reset_index(drop=True)
    emb = np.load(".../moon_ce3/embeddings/dinov2_vits__loftup.npy",
                  allow_pickle=True).item()
    X = emb["point_embeddings"]          # (M, 2, D) aligned with `sub`
    y = emb["labels"]                    # (M,)

Requires your own images under ``data/datasets/{dataset}/images_source/`` (or
``images/``) and a ``data/datasets/combined/annotations.csv`` index — see the
repo README for the expected layout; ``data/annotations/*.csv`` here are only
small per-dataset samples of the final annotation format, not the full index
this script reads.

Design notes
------------
* **ImageNet normalization** is applied to the "native" (loftup) branch — the
  project backbone expects normalised input.
* **UPLiFT** ships its own backbone + preprocessing, so that branch uses the
  upstream extractor directly (``include_extractor=True``, ``iters=4`` — the
  authors' recommended inference path).

Coordinate convention: CSV points are at the *original* image resolution and
are scaled onto a 224x224 feature grid.

Usage
-----
    # both configs for moon_ce3
    python scripts/extract_embeddings_pipeline.py --dataset moon_ce3 --configs all

    # a single config
    python scripts/extract_embeddings_pipeline.py \
        --dataset moon_ce3 --configs dinov2_vits__loftup

    # every dataset, both configs, WITH CLS tokens (writes alongside the
    # existing outputs as '{config}__cls.npy', leaving the old files intact)
    python scripts/extract_embeddings_pipeline.py \
        --dataset all --configs all --with-cls

The CLS-augmented files add ``point_cls (M, 2, D_cls)`` (the source-image CLS
token aligned per point), ``image_cls (n_images, D_cls)`` and the matching
``image_cls_names`` table. Native backbones emit their real class token;
UPLiFT has none, so a mean-pooled feature map is stored as a CLS proxy
(``settings['cls_kind']``).
"""

from __future__ import annotations

import argparse
import csv
import gc
import logging
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm

log = logging.getLogger("extract_pipeline")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DS = PROJECT_ROOT / "data" / "datasets"
COMBINED = DS / "combined"
RESOLUTION = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ── Config matrix ───────────────────────────────────────────────────────────
# kind="native"  → project backbone with a registry upsampler (we normalize).
# kind="uplift"  → upstream UPLiFT extractor (bundles its own backbone).
#
# Trimmed to the two branches FOCUS actually fuses: dinov2_vits__loftup and
# dinov2_vits__uplift (see chapters/05-method.tex in the thesis). The original
# project swept several more backbone/upsampler pairs during architecture
# search; those aren't part of the final model and were dropped here.
CONFIGS: dict[str, dict] = {
    "dinov2_vits__loftup": {
        "kind": "native", "backbone": "dinov2_vits", "upsampler": "loftup",
        "upsampler_kwargs": {"variant": "loftup_dinov2s"},
    },
    "dinov2_vits__uplift": {
        "kind": "uplift", "hub_name": "uplift_dinov2_s14",
    },
}


# ── Helpers ─────────────────────────────────────────────────────────────────
def _force_cpu_torch_load() -> None:
    """Some vendored / hub checkpoints were saved on CUDA and loaded without a
    ``map_location``. Force every ``torch.load`` onto the CPU on CPU-only boxes."""
    if getattr(torch.load, "_cpu_forced", False):
        return
    _orig = torch.load

    def _cpu_load(*a, **k):
        k.setdefault("map_location", "cpu")
        return _orig(*a, **k)

    _cpu_load._cpu_forced = True
    torch.load = _cpu_load


def _images_dir(dataset: str) -> Path:
    d = DS / dataset / "images_source"
    return d if d.exists() else DS / dataset / "images"


def _load_square_rgb(path: Path) -> Image.Image:
    """Open an image and squash it to a square RESOLUTION x RESOLUTION PIL image.

    Squashing (not cropping) keeps the original->grid point mapping exact:
    a point at (px, py) in (w, h) maps to (px*RES/w, py*RES/h).
    """
    img = Image.open(path).convert("RGB")
    return img.resize((RESOLUTION, RESOLUTION), Image.BILINEAR)


# ── Featurizers ─────────────────────────────────────────────────────────────
class NativeFeaturizer:
    """Project backbone + registry upsampler, with ImageNet normalization."""

    def __init__(self, cfg: dict, device: torch.device):
        from focus.registry import create
        import focus.backbones  # noqa: F401
        import focus.upsamplers  # noqa: F401

        self.device = device
        self.backbone = create(
            "backbone", cfg["backbone"],
            pretrained=True,
            upsampler_name=cfg["upsampler"],
            upsampler_kwargs=cfg.get("upsampler_kwargs"),
        ).to(device).eval()
        self._mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    # CLS source: the backbone's own normalised class token.
    cls_kind = "clstoken"

    @torch.no_grad()
    def __call__(self, pil: Image.Image, return_cls: bool = False):
        arr = np.asarray(pil, dtype=np.float32) / 255.0          # (H, W, 3)
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        t = (t.to(self.device) - self._mean) / self._std
        out = self.backbone(t)
        feats = F.interpolate(out["features"], size=(RESOLUTION, RESOLUTION),
                              mode="bilinear", align_corners=False)
        # Return on CPU: the cache may hold many feature maps and a small GPU
        # would otherwise run out of memory. Point sampling is just indexing.
        feats = feats.squeeze(0).detach().cpu()  # (D, RES, RES)
        if not return_cls:
            return feats
        cls = out["cls_token"].squeeze(0).detach().cpu()  # (D,)
        return feats, cls


class UpliftFeaturizer:
    """Upstream UPLiFT extractor (bundles backbone + preprocessing)."""

    def __init__(self, cfg: dict, device: torch.device):
        self.device = device
        self.model = torch.hub.load(
            "mwalmer-umd/UPLiFT", cfg["hub_name"],
            pretrained=True, include_extractor=True,
            iters=4, out_size=RESOLUTION, return_base_feat=False,
            fast=False, silent=True,
        )
        try:
            self.model = self.model.to(device)
        except Exception:  # extractor may not be a plain nn.Module
            pass

    # UPLiFT ships no class token; use a mean-pooled feature map as a CLS proxy.
    cls_kind = "meanpool_proxy"

    @torch.no_grad()
    def __call__(self, pil: Image.Image, return_cls: bool = False):
        feat = self.model(pil)  # (1, D, RES, RES) or (D, RES, RES)
        if isinstance(feat, (tuple, list)):
            feat = feat[0]
        if feat.dim() == 4:
            feat = feat.squeeze(0)
        if feat.shape[-2:] != (RESOLUTION, RESOLUTION):
            feat = F.interpolate(feat.unsqueeze(0), size=(RESOLUTION, RESOLUTION),
                                 mode="bilinear", align_corners=False).squeeze(0)
        feat = feat.detach().cpu()  # (D, RES, RES)
        if not return_cls:
            return feat
        cls = feat.reshape(feat.shape[0], -1).mean(dim=1)  # (D,) mean-pool proxy
        return feat, cls


def build_featurizer(config_name: str, device: torch.device):
    cfg = CONFIGS[config_name]
    if cfg["kind"] == "native":
        return NativeFeaturizer(cfg, device)
    if cfg["kind"] == "uplift":
        return UpliftFeaturizer(cfg, device)
    raise ValueError(f"Unknown config kind: {cfg['kind']}")


# ── Extraction ──────────────────────────────────────────────────────────────
def extract_config(config_name: str, rows: list[dict], dataset: str,
                   device: torch.device, with_cls: bool = False,
                   featurizer=None, max_cache_images: int = 12) -> dict:
    """Run one config over the dataset rows; return the output payload.

    When ``with_cls`` is set, also collect the per-image CLS token (native
    backbones) or a mean-pooled feature proxy (UPLiFT) and expose it both as a
    per-image table and aligned per-point alongside ``point_embeddings``.

    ``featurizer`` may be passed in so a single model is reused across many
    datasets (avoids reloading/swapping the backbone, which is the dominant
    cost and the main OOM risk).
    """
    if featurizer is None:
        log.info("[%s] building featurizer on %s", config_name, device)
        featurizer = build_featurizer(config_name, device)
    images_dir = _images_dir(dataset)

    # Determine emb_dim (and cls_dim) from the first image.
    first_img = _load_square_rgb(images_dir / rows[0]["imageA_name"])
    if with_cls:
        _f0, _c0 = featurizer(first_img, return_cls=True)
        emb_dim = int(_f0.shape[0])
        cls_dim = int(_c0.shape[0])
        log.info("[%s] emb_dim=%d cls_dim=%d (%s)", config_name, emb_dim,
                 cls_dim, getattr(featurizer, "cls_kind", "cls"))
    else:
        emb_dim = int(featurizer(first_img).shape[0])
        cls_dim = 0
        log.info("[%s] emb_dim=%d", config_name, emb_dim)

    n = len(rows)
    point_embeddings = np.zeros((n, 2, emb_dim), dtype=np.float32)
    point_cls = np.zeros((n, 2, cls_dim), dtype=np.float32) if with_cls else None
    image_cls: dict[str, np.ndarray] = {}

    # Reference-count images so feature maps can be evicted as soon as the last
    # pair that needs them is processed.
    img_ref: Counter = Counter()
    for r in rows:
        img_ref[r["imageA_name"]] += 1
        img_ref[r["imageB_name"]] += 1

    cache: OrderedDict[str, torch.Tensor] = OrderedDict()
    cls_cache: OrderedDict[str, torch.Tensor] = OrderedDict()

    def get_feat(name: str) -> torch.Tensor:
        if name not in cache:
            if with_cls:
                feat, cls = featurizer(_load_square_rgb(images_dir / name),
                                       return_cls=True)
                cache[name] = feat
                cls_cache[name] = cls
                if name not in image_cls:
                    image_cls[name] = cls.numpy().astype(np.float32)
            else:
                cache[name] = featurizer(_load_square_rgb(images_dir / name))
            # Bound cache size to keep CPU memory under control.
            while len(cache) > max_cache_images:
                old, _ = cache.popitem(last=False)
                cls_cache.pop(old, None)
        else:
            cache.move_to_end(name)
            if with_cls:
                cls_cache.move_to_end(name)
        return cache[name]

    # Map global sample_id -> local output row index (rows are in CSV order).
    sid_to_local = {int(r["sample_id"]): i for i, r in enumerate(rows)}

    by_pair: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        by_pair[(r["imageA_name"], r["imageB_name"])].append(r)

    with torch.no_grad():
        for (a_name, b_name), prs in tqdm(
            by_pair.items(), total=len(by_pair),
            desc=f"{config_name} [{dataset}]", unit="pair", dynamic_ncols=True,
        ):
            feat_a = get_feat(a_name)
            feat_b = get_feat(b_name)
            for r in prs:
                local = sid_to_local[int(r["sample_id"])]
                w, h = int(r["width"]), int(r["height"])
                for slot, (pi, px, py) in enumerate((
                    (int(r["pt1_img"]), int(r["pt1_x"]), int(r["pt1_y"])),
                    (int(r["pt2_img"]), int(r["pt2_x"]), int(r["pt2_y"])),
                )):
                    feat = feat_a if pi == 0 else feat_b
                    sx = min(int(px * RESOLUTION / w), RESOLUTION - 1)
                    sy = min(int(py * RESOLUTION / h), RESOLUTION - 1)
                    point_embeddings[local, slot] = feat[:, sy, sx].cpu().numpy()
                    if with_cls:
                        src = a_name if pi == 0 else b_name
                        point_cls[local, slot] = cls_cache[src].numpy()

            for nm in (a_name, b_name):
                img_ref[nm] -= 1
                if img_ref[nm] <= 0:
                    cache.pop(nm, None)
                    cls_cache.pop(nm, None)

    cfg = CONFIGS[config_name]
    payload = {
        "settings": {
            "config": config_name,
            "kind": cfg["kind"],
            "backbone": cfg.get("backbone", cfg.get("hub_name")),
            "upsampler": cfg.get("upsampler", "uplift"),
            "emb_dim": emb_dim,
            "normalized": cfg["kind"] == "native",
            "dataset": dataset,
            "n_samples": n,
            "resolution": RESOLUTION,
            "source": "extract_embeddings_pipeline",
            "has_cls": with_cls,
            "cls_dim": cls_dim,
            "cls_kind": getattr(featurizer, "cls_kind", None) if with_cls else None,
        },
        "point_embeddings": point_embeddings,
        "labels": np.array([int(r["label"]) for r in rows], dtype=np.int64),
        "sample_id": np.array([int(r["sample_id"]) for r in rows], dtype=np.int64),
        "split": np.array([r["split"] for r in rows]),
        "domain": np.array([r["domain"] for r in rows]),
        "comparison_type": np.array([r["comparison_type"] for r in rows]),
        "pair_uid": np.array([r["pair_uid"] for r in rows]),
    }
    if with_cls:
        names = sorted(image_cls)
        payload["point_cls"] = point_cls
        payload["image_cls"] = np.stack([image_cls[k] for k in names]) \
            if names else np.zeros((0, cls_dim), dtype=np.float32)
        payload["image_cls_names"] = np.array(names)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="moon_ce3",
                        help="Dataset name, comma-separated list, or 'all' "
                             "(every dataset present in annotations.csv).")
    parser.add_argument("--configs", default="all",
                        help="'all' or a comma-separated list of config names. "
                             f"Available: {', '.join(CONFIGS)}")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Output dir (default: {dataset}/embeddings).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-extract even if an output file already exists.")
    parser.add_argument("--with-cls", action="store_true",
                        help="Also extract CLS tokens (native: class token; "
                             "UPLiFT: mean-pooled proxy). Written to a separate "
                             "'{config}__cls.npy' file so existing outputs are "
                             "preserved for progress tracking.")
    parser.add_argument("--max-cache-images", type=int, default=12,
                        help="Maximum number of image feature maps kept in RAM "
                            "at once during extraction (lower = less RAM, more "
                            "recompute).")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    _force_cpu_torch_load()

    if args.max_cache_images < 2:
        raise SystemExit("--max-cache-images must be >= 2")

    if args.configs == "all":
        config_names = list(CONFIGS)
    else:
        config_names = [c.strip() for c in args.configs.split(",") if c.strip()]
        unknown = [c for c in config_names if c not in CONFIGS]
        if unknown:
            raise SystemExit(f"Unknown configs: {unknown}. Available: {list(CONFIGS)}")

    csv_path = COMBINED / "annotations.csv"
    if not csv_path.exists():
        raise SystemExit(f"Missing {csv_path}. Run build_combined_dataset first.")
    with open(csv_path) as fh:
        all_rows = list(csv.DictReader(fh))

    available = sorted({r["dataset"] for r in all_rows})
    if args.dataset == "all":
        datasets = available
    else:
        datasets = [d.strip() for d in args.dataset.split(",") if d.strip()]
        unknown_ds = [d for d in datasets if d not in available]
        if unknown_ds:
            raise SystemExit(
                f"Unknown datasets: {unknown_ds}. Available: {available}")

    # Pre-resolve rows per dataset (in CSV order) once.
    rows_by_ds = {ds: [r for r in all_rows if r["dataset"] == ds] for ds in datasets}

    # Process datasets SMALLEST-FIRST so the small ones finish (across all
    # configs) early and become usable while the big ones are still running.
    datasets = sorted(datasets, key=lambda ds: len(rows_by_ds[ds]))
    log.info("Datasets to process (smallest-first): %s",
             [(ds, len(rows_by_ds[ds])) for ds in datasets])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    suffix = "__cls" if args.with_cls else ""

    # Loop DATASET-OUTER: finish every config for one dataset before moving to
    # the next, so smaller datasets become fully available first. Within a
    # dataset we build each model once, then tear it down before the next
    # config so only one model is resident on the GPU at a time (avoids OOM).
    n_total = len(config_names) * len([ds for ds in datasets if rows_by_ds[ds]])
    done = 0
    failures: list[tuple[str, str]] = []
    for ds in datasets:
        rows = rows_by_ds[ds]
        if not rows:
            log.warning("No rows for dataset '%s'; skipping.", ds)
            continue
        out_dir = args.out_dir or (DS / ds / "embeddings")
        out_dir.mkdir(parents=True, exist_ok=True)
        log.info("=== DATASET %s: %d rows, %d configs ===",
                 ds, len(rows), len(config_names))

        for name in config_names:
            out_path = out_dir / f"{name}{suffix}.npy"
            if out_path.exists() and not args.overwrite:
                log.info("[%s/%s] exists, skipping (use --overwrite): %s",
                         ds, name, out_path)
                done += 1
                continue

            done += 1
            log.info("[%d/%d] building %s and extracting on '%s' (%d rows)",
                     done, n_total, name, ds, len(rows))
            featurizer = None
            try:
                featurizer = build_featurizer(name, device)
                payload = extract_config(name, rows, ds, device,
                                         with_cls=args.with_cls,
                                         featurizer=featurizer,
                                         max_cache_images=args.max_cache_images)
                np.save(out_path, payload, allow_pickle=True)
                extra = ""
                if args.with_cls:
                    extra = f", point_cls {payload['point_cls'].shape}"
                log.info("[%d/%d] wrote %s  (point_embeddings %s%s)",
                         done, n_total, out_path,
                         payload["point_embeddings"].shape, extra)
            except KeyboardInterrupt:
                raise
            except Exception:
                # Don't let one broken config (e.g. an env/timm issue with
                # UPLiFT) abort the whole run — log and move on.
                failures.append((ds, name))
                log.exception("[%d/%d] FAILED %s on '%s' — skipping",
                              done, n_total, name, ds)
            finally:
                # Tear the model down before loading the next config.
                del featurizer
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    if failures:
        log.warning("Done with %d failure(s): %s", len(failures),
                    ", ".join(f"{ds}/{name}" for ds, name in failures))
    else:
        log.info("Done.")


if __name__ == "__main__":
    main()
