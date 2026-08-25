# FOCUS

**F**used **O**rdinal-**C**ORN with **U**ncertainty-weighted **S**coring — a
frozen-backbone architecture for point-wise lunar terrain traversability
scoring, trained on sparse pairwise human comparisons instead of dense masks.

Two frozen DINOv2 ViT-S/14 backbones — one paired with the LoftUp upsampler,
one with UpLiFT — each produce a dense per-point feature map for an image. A
small trainable fusion encoder combines the two per-point embeddings into one
representation, which a CORN ordinal head scores into a traversability value
in `[0, 1]`. Training reads sparse pairwise comparisons ("point A is more/less/
equally traversable than point B") and optimizes TOLRIZZ, a margin-based
ranking loss with a tolerant equality zone, weighted per-pair by a confidence
branch's estimate of how reliable the comparison's source image is, plus a
small reconstruction term. Only the fusion encoder and CORN head run at
deployment.

## Installation

```bash
pip install -e .
# only needed for ExperimentRunner.run_parallel() and the Wilcoxon test:
pip install -e ".[stats]"
```

## Running it

1. Place your own images under `data/datasets/{dataset}/images_source/` and
   build `data/datasets/combined/annotations.csv` (see `data/annotations/*.csv`
   for the per-dataset schema).
2. Extract and fuse embeddings:
   ```bash
   python scripts/extract_embeddings_pipeline.py --dataset all --configs all --with-cls
   python scripts/build_paired_embeddings.py
   ```
3. Train and evaluate:
   ```bash
   python -m focus.train --only-pooled-all
   ```

UpLiFT and LoftUp load via `torch.hub` on first use (`mwalmer-umd/UPLiFT`,
`andrehuang/loftup`), DINOv2 via `facebookresearch/dinov2` — an internet
connection is needed the first time the extraction script runs.
