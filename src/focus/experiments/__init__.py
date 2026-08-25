"""Dataset-experiment harness for pairwise ordinal traversability heads.

This package fixes the **dataset protocol** (which train/test pools, which seeds)
so that arbitrary models — supplied by the caller as small factories — can be
compared on an even footing.  Models themselves are defined externally (e.g. in
a notebook) and only need a ``score(emb, cls)`` method.

Typical use (from a notebook)::

    from focus.experiments import ExperimentConfig, ExperimentRunner
    from focus.train import focus_factory

    cfg = ExperimentConfig.from_yaml("configs/experiment/transfer_default.yaml")
    runner = ExperimentRunner(cfg, device="cuda")

    store = runner.run({"FOCUS": focus_factory})
    store.leaderboard()         # (config x model) master table
    store.by_config()           # average per backbone
    store.by_model()            # average per head
    store.by_dataset()          # average per test-dataset pool
    store.pivot(config="dinov2_vits__loftup+dinov2_vits__uplift")   # settings x models
"""

from __future__ import annotations

from .config import DatasetGroups, ExperimentConfig, Families, parse_config
from .data import EmbeddingStore, PairData, default_embeddings_root, find_repo_root
from .metrics import PDRResult, pdr, predict, select_tau, trivial_pdr
from .protocols import ModelFactory, PairModel, ScoreModel, score_diff, seed_everything
from .results import ResultStore, RunRecord, error_reduction_table, wilcoxon_vs_frame
from .runner import ExperimentRunner
from .settings import ExperimentSetting, Spec, build_settings

__all__ = [
    # config
    "ExperimentConfig", "DatasetGroups", "Families", "parse_config",
    # data
    "EmbeddingStore", "PairData", "find_repo_root", "default_embeddings_root",
    # metrics
    "pdr", "predict", "select_tau", "trivial_pdr", "PDRResult",
    # protocols
    "ScoreModel", "PairModel", "ModelFactory", "seed_everything", "score_diff",
    # settings
    "ExperimentSetting", "Spec", "build_settings",
    # results / runner
    "ResultStore", "RunRecord", "ExperimentRunner",
    "error_reduction_table", "wilcoxon_vs_frame",
]
