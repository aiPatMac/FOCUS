"""Results aggregation, persistence and significance testing.

Every ``(model, setting, seed)`` evaluation is appended as a flat record.  The
store writes a tidy long-form ``results.csv`` plus a ``meta.json`` snapshot of
the config, and offers convenience aggregations (mean ± std per model/setting,
mean rank, and a paired Wilcoxon test of each model against a chosen baseline).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import parse_config


@dataclass
class RunRecord:
    """One ``(config, model, setting, seed)`` evaluation row."""

    model: str
    setting: str
    family: str
    seed: int
    tau: float
    pdr: float
    within: float
    cross: float
    n_test: int
    trivial: float
    config: str = ""                    # embedding backbone config (e.g. dinov2_vits__anyup)
    backbone: str = ""                  # derived: dinov2_vits
    upsampler: str = ""                 # derived: anyup
    train_datasets: str = ""            # e.g. "lusnar+luna_polaris"
    test_datasets: str = ""             # e.g. "moon_ce3+moon_ce4"
    pdr_val_tau: float = float("nan")   # PDR at a holdout-validated tau (optional)
    val_tau: float = float("nan")
    train_pairs: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.config and not (self.backbone and self.upsampler):
            self.backbone, self.upsampler = parse_config(self.config)

    def flat(self) -> dict[str, Any]:
        d = asdict(self)
        extra = d.pop("extra")
        for k, v in extra.items():
            d[f"x_{k}"] = v
        return d


class ResultStore:
    """Collects :class:`RunRecord`s and persists / aggregates them."""

    def __init__(self, name: str, output_dir: Path | str, meta: dict | None = None) -> None:
        self.name = name
        self.run_dir = Path(output_dir) / name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.records: list[RunRecord] = []
        self._meta = {"name": name, "created": datetime.now().isoformat(), **(meta or {})}

    # ── collection ─────────────────────────────────────────────────────────
    def add(self, record: RunRecord) -> None:
        self.records.append(record)

    def frame(self) -> pd.DataFrame:
        if not self.records:
            return pd.DataFrame()
        return pd.DataFrame([r.flat() for r in self.records])

    # ── persistence ────────────────────────────────────────────────────────
    def save(self) -> Path:
        """Write the full long-form results plus standard per-axis aggregations."""
        df = self.frame()
        csv_path = self.run_dir / "results.csv"
        df.to_csv(csv_path, index=False)
        with open(self.run_dir / "meta.json", "w") as f:
            json.dump(self._meta, f, indent=2)

        # always log averages per axis (architecture / model / dataset / family ...)
        if not df.empty:
            agg_dir = self.run_dir / "aggregates"
            agg_dir.mkdir(exist_ok=True)
            self.by_config().to_csv(agg_dir / "by_config.csv")
            self.by_model().to_csv(agg_dir / "by_model.csv")
            self.by_backbone().to_csv(agg_dir / "by_backbone.csv")
            self.by_upsampler().to_csv(agg_dir / "by_upsampler.csv")
            self.by_family().to_csv(agg_dir / "by_family.csv")
            self.by_dataset().to_csv(agg_dir / "by_test_dataset.csv")
            self.leaderboard().to_csv(agg_dir / "leaderboard_config_x_model.csv")
            self.family_leaderboard().to_csv(agg_dir / "family_leaderboard.csv")
            self.mean_rank().to_csv(agg_dir / "mean_rank_by_model.csv")
            self.summary().to_csv(agg_dir / "summary_full.csv", index=False)
        return csv_path

    @staticmethod
    def load(run_dir: Path | str) -> pd.DataFrame:
        return pd.read_csv(Path(run_dir) / "results.csv")

    # ── generic aggregation ────────────────────────────────────────────────
    def mean_by(
        self,
        by: str | list[str],
        metric: str = "pdr",
        extra_metrics: list[str] | None = None,
    ) -> pd.DataFrame:
        """Mean/std/count of *metric* grouped by one or more columns.

        ``extra_metrics`` (e.g. ``["within", "cross", "pdr_val_tau"]``) are added
        as extra mean columns.  Results are sorted by the primary mean ascending.
        """
        df = self.frame()
        if df.empty:
            return df
        keys = [by] if isinstance(by, str) else list(by)
        agg: dict[str, Any] = {
            "mean": (metric, "mean"), "std": (metric, "std"), "count": (metric, "count"),
        }
        for m in (extra_metrics or []):
            if m in df.columns:
                agg[f"{m}_mean"] = (m, "mean")
        out = df.groupby(keys).agg(**agg)
        return out.sort_values("mean")

    # ── named per-axis aggregations ────────────────────────────────────────
    def by_config(self, metric: str = "pdr") -> pd.DataFrame:
        """Average per embedding backbone config (architecture)."""
        return self.mean_by("config", metric, extra_metrics=["within", "cross", "pdr_val_tau"])

    def by_backbone(self, metric: str = "pdr") -> pd.DataFrame:
        return self.mean_by("backbone", metric)

    def by_upsampler(self, metric: str = "pdr") -> pd.DataFrame:
        return self.mean_by("upsampler", metric)

    def by_model(self, metric: str = "pdr") -> pd.DataFrame:
        """Average per model (head)."""
        return self.mean_by("model", metric, extra_metrics=["within", "cross", "pdr_val_tau"])

    def by_family(self, metric: str = "pdr") -> pd.DataFrame:
        return self.mean_by("family", metric)

    def by_dataset(self, metric: str = "pdr") -> pd.DataFrame:
        """Average per test-dataset pool (e.g. each in-domain ds, each transfer target)."""
        return self.mean_by(["family", "test_datasets"], metric)

    def leaderboard(self, metric: str = "pdr") -> pd.DataFrame:
        """Mean *metric* per (config, model) over all settings/seeds — the master table."""
        df = self.frame()
        if df.empty:
            return df
        return (df.groupby(["config", "model"])[metric].mean()
                  .unstack("model").sort_index())

    def family_leaderboard(self, metric: str = "pdr", config: str | None = None) -> pd.DataFrame:
        """Family (rows) x model (cols) table of mean *metric*, plus an ``ALL`` row.

        Each row is the single headline average for one of the four standard
        families (``in_domain``, ``pooled``, ``sim_to_real``, ``leave_one_out``,
        plus any opt-in families present) — the "average of all X" figure
        referenced in the Dataset chapter's evaluation-protocol section. Every
        (setting, seed) cell within a family is weighted equally, i.e. this is
        the mean over settings-then-seeds within that family, not a per-setting
        breakdown; use :meth:`pivot` for the latter.
        """
        df = self.frame()
        if df.empty:
            return df
        if config is not None:
            df = df[df["config"] == config]
        table = df.groupby(["family", "model"])[metric].mean().unstack("model")
        table.loc["ALL"] = df.groupby("model")[metric].mean()
        return table

    # ── existing aggregations ──────────────────────────────────────────────
    def summary(self, metric: str = "pdr") -> pd.DataFrame:
        """Mean ± std of *metric* per (config, setting, model), averaged over seeds."""
        df = self.frame()
        if df.empty:
            return df
        agg = (df.groupby(["config", "setting", "family", "model"])[metric]
                 .agg(["mean", "std", "count"]).reset_index())
        return agg.sort_values(["config", "family", "setting", "mean"])

    def pivot(self, metric: str = "pdr", config: str | None = None) -> pd.DataFrame:
        """Settings (rows) x models (cols) table of seed-mean *metric*, plus a MEAN row.

        If *config* is given, restrict to that backbone; otherwise average over
        all configs.
        """
        df = self.frame()
        if df.empty:
            return df
        if config is not None:
            df = df[df["config"] == config]
        table = df.groupby(["setting", "model"])[metric].mean().unstack("model")
        table.loc["MEAN"] = table.mean(axis=0)
        return table

    def mean_rank(self, metric: str = "pdr") -> pd.DataFrame:
        """Mean rank of each model across (config, setting) cells (1 = best)."""
        df = self.frame()
        if df.empty:
            return df
        cell = df.groupby(["config", "setting", "model"])[metric].mean().reset_index()
        cell["rank"] = cell.groupby(["config", "setting"])[metric].rank(method="min")
        out = (cell.groupby("model")
                   .agg(mean_metric=(metric, "mean"), mean_rank=("rank", "mean"))
                   .sort_values("mean_rank"))
        return out

    def wilcoxon_vs(self, baseline: str, metric: str = "pdr") -> pd.DataFrame:
        """Paired Wilcoxon signed-rank test of every model vs *baseline*.

        Pairs are the per-(config, setting, seed) values.  ``mean_diff < 0`` means
        the model beats the baseline; ``p < 0.05`` flags significance.
        """
        from scipy.stats import wilcoxon

        df = self.frame()
        if df.empty:
            return df
        wide = df.pivot_table(index=["config", "setting", "seed"], columns="model", values=metric)
        if baseline not in wide.columns:
            raise KeyError(f"baseline model '{baseline}' not found in results.")
        rows = []
        base = wide[baseline]
        for model in wide.columns:
            if model == baseline:
                continue
            joined = pd.concat([base, wide[model]], axis=1, keys=["b", "m"]).dropna()
            if len(joined) < 1:
                continue
            diff = (joined["m"] - joined["b"]).values
            wins = int((diff < 0).sum())
            try:
                _, p = wilcoxon(joined["m"].values, joined["b"].values)
            except ValueError:
                p = float("nan")
            rows.append({
                "model": model, "n_pairs": len(joined),
                "wins_vs_baseline": f"{wins}/{len(joined)}",
                "mean_diff": float(np.mean(diff)),
                "wilcoxon_p": float(p),
                "better_sig": bool((p < 0.05) and (np.mean(diff) < 0)),
            })
        return pd.DataFrame(rows).sort_values("mean_diff")


def error_reduction_table(
    df: pd.DataFrame,
    by: str,
    models: list[str] | None = None,
    metric: str = "pdr",
    summary: str | None = None,
) -> pd.DataFrame:
    """Per-``by`` (``"family"`` or ``"setting"``) table of the constant/trivial
    predictor's PDR alongside each model's **mean ± std** PDR and **error
    reduction (%)**: ``100 * (1 - pdr/trivial)`` — the fraction of the
    constant predictor's disagreement rate a model eliminates. 0% = no
    better than guessing, 100% = perfect, negative = worse than guessing,
    higher is always better.

    Raw PDR is not comparable across settings by itself: the trivial baseline
    (``Pr[label != 0]`` on the test pool) swings roughly 0.51-0.73 depending on
    how skewed a dataset's label distribution is toward equality, so the same
    raw PDR means very different things on different pools (see the Dataset
    chapter's evaluation-protocol section). Error reduction rescales every
    setting onto the same 0-100% axis and is the number that should be read
    across settings/datasets, not raw PDR.

    Standard deviation is computed **per row first** (error reduction is
    derived per ``(model, setting, seed)`` record, then grouped), not from
    the grouped means, so it reflects the actual seed-to-seed spread within
    each group — matching this thesis's "mean ± std across five seeds"
    reporting convention (Dataset chapter, evaluation-protocol section).

    ``trivial`` depends only on the test pool's label distribution, not on
    the model, so it is grouped/averaged the same way regardless of which
    model's rows it is read off of. Pass e.g. ``summary="MEAN"`` to append a
    single, uniformly-weighted overall row (do this on a ``by="setting"``
    call, not on a ``by="family"`` one — averaging family means would
    silently give each family equal weight regardless of how many settings
    back it; the summary row's std is the mean of the per-group stds, a
    simple approximation, not a re-pooled standard deviation).
    """
    if df.empty:
        return df
    if models is None:
        models = sorted(df["model"].unique())
    d = df.copy()
    d["_err_reduction"] = 100 * (1 - d[metric] / d["trivial"])
    trivial = d.groupby(by)["trivial"].mean().rename("trivial")
    parts = [trivial]
    for m in models:
        sub = d[d["model"] == m].groupby(by).agg(**{
            f"{m}_pdr": (metric, "mean"),
            f"{m}_pdr_std": (metric, "std"),
            f"{m}_err_reduction%": ("_err_reduction", "mean"),
            f"{m}_err_reduction%_std": ("_err_reduction", "std"),
        })
        parts.append(sub)
    table = pd.concat(parts, axis=1)
    if summary:
        table.loc[summary] = table.mean(axis=0)
    return table


def wilcoxon_vs_frame(
    df: pd.DataFrame,
    baseline: str,
    metric: str = "pdr",
    pair_on: tuple[str, ...] = ("setting", "seed"),
) -> pd.DataFrame:
    """Paired Wilcoxon signed-rank test of every model vs *baseline* on an
    arbitrary long-form results frame — unlike :meth:`ResultStore.wilcoxon_vs`,
    this is not tied to a single store, so it also covers comparisons across
    models trained under *different* embedding configs (e.g. two separate
    ``ExperimentRunner.run()`` calls concatenated into one frame): the default
    ``pair_on=("setting", "seed")`` deliberately excludes ``"config"``, since
    such models share no ``(config, setting, seed)`` triple and pairing on
    config would silently empty the join. Pass
    ``pair_on=("config", "setting", "seed")`` to reproduce
    :meth:`ResultStore.wilcoxon_vs`'s behaviour for same-config comparisons.
    """
    from scipy.stats import wilcoxon

    cols = ["model", "n_pairs", "wins_vs_baseline", "mean_diff", "wilcoxon_p", "better_sig"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    wide = df.pivot_table(index=list(pair_on), columns="model", values=metric)
    if baseline not in wide.columns:
        raise KeyError(f"baseline model '{baseline}' not found in results.")
    rows = []
    base = wide[baseline]
    for model in wide.columns:
        if model == baseline:
            continue
        joined = pd.concat([base, wide[model]], axis=1, keys=["b", "m"]).dropna()
        if len(joined) < 1:
            continue
        diff = (joined["m"] - joined["b"]).values
        wins = int((diff < 0).sum())
        try:
            _, p = wilcoxon(joined["m"].values, joined["b"].values)
        except ValueError:
            p = float("nan")
        rows.append({
            "model": model, "n_pairs": len(joined), "wins_vs_baseline": f"{wins}/{len(joined)}",
            "mean_diff": float(np.mean(diff)), "wilcoxon_p": float(p),
            "better_sig": bool((p < 0.05) and (np.mean(diff) < 0)),
        })
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows).sort_values("mean_diff")


__all__ = ["RunRecord", "ResultStore", "error_reduction_table", "wilcoxon_vs_frame"]
