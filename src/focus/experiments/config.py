"""Typed experiment configuration, loaded from YAML via OmegaConf.

The config selects *which* dataset transfer protocols to run and with what
seeds / backbone / threshold.  Models are **not** part of the config — they are
supplied in code (e.g. from a notebook) so the harness stays model-agnostic
while keeping the dataset protocol fixed and comparable across models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import OmegaConf


def parse_config(name: str) -> tuple[str, str]:
    """Split an embedding config name into ``(backbone, upsampler)``.

    ``"dinov2_vits__anyup" -> ("dinov2_vits", "anyup")``.  Falls back to
    ``(name, "")`` if the ``__`` separator is absent.
    """
    if "__" in name:
        backbone, upsampler = name.split("__", 1)
        return backbone, upsampler
    return name, ""


@dataclass
class DatasetGroups:
    """Logical roles of the datasets used to build the transfer matrix."""

    real: list[str] = field(default_factory=lambda: ["moon_ce3", "moon_ce4"])
    sim1: list[str] = field(default_factory=lambda: ["lusnar"])         # simulated
    sim2: list[str] = field(default_factory=lambda: ["luna_polaris"])   # real analogue

    def all_datasets(self) -> list[str]:
        seen: list[str] = []
        for d in [*self.sim1, *self.sim2, *self.real]:
            if d not in seen:
                seen.append(d)
        return seen


@dataclass
class Families:
    """Toggles for the experiment families (which settings to generate).

    Four standard families are reported by default, each with a family-level
    average (:meth:`ResultStore.family_leaderboard`): ``in_domain``,
    ``pooled`` (real_pooled + all_pooled + sim_pooled), ``sim_to_real``, and
    ``leave_one_out``. ``reverse_real_to_sim``/``cross_sim`` remain opt-in.
    """

    in_domain: bool = True            # each ds: train split -> val split
    real_pooled: bool = True          # real train -> real val               (family="pooled")
    all_pooled: bool = True           # all train -> all val                 (family="pooled")
    sim_pooled: bool = True           # sim1+sim2 train -> sim1+sim2 val     (family="pooled")
    sim_to_real: bool = True          # sim1/sim2/both -> real (the headline)
    reverse_real_to_sim: bool = False # real -> sim1, real -> sim2
    cross_sim: bool = False           # sim1 -> sim2, sim2 -> sim1
    leave_one_out: bool = True        # train on all-but-one -> held-out ds


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""

    name: str = "transfer_default"
    configs: list[str] = field(                  # embedding backbones to sweep
        default_factory=lambda: [
            "dinov2_vits__anyup",
            "dinov2_vits__featup",
            "dinov2_vits__loftup",
            "dinov2_vits__uplift",
            "dinov3_vits__anyup",
            "dinov3_splus__uplift",
        ]
    )
    tau: float = 0.25                            # fixed decision threshold
    seeds: list[int] = field(default_factory=lambda: [42, 123, 7, 0, 99])
    val_frac: float = 0.15                       # early-stop holdout carved from train pool
    report_val_tau: bool = True                  # also report PDR at a holdout-validated tau
    tau_grid: list[float] = field(
        default_factory=lambda: [round(0.04 + 0.04 * i, 3) for i in range(15)]  # 0.04..0.6
    )
    datasets: DatasetGroups = field(default_factory=DatasetGroups)
    families: Families = field(default_factory=Families)
    embeddings_root: str | None = None           # default: <repo>/active_labeling/datasets
    output_dir: str = "experiments_output"       # results land in <output_dir>/<name>/

    # ── (de)serialisation ──────────────────────────────────────────────────
    @classmethod
    def from_yaml(cls, path: Path | str) -> "ExperimentConfig":
        """Load a config, validating against the dataclass schema."""
        base = OmegaConf.structured(cls)
        loaded = OmegaConf.load(Path(path))
        merged = OmegaConf.merge(base, loaded)
        return OmegaConf.to_object(merged)  # type: ignore[return-value]

    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentConfig":
        base = OmegaConf.structured(cls)
        merged = OmegaConf.merge(base, OmegaConf.create(data))
        return OmegaConf.to_object(merged)  # type: ignore[return-value]

    def to_yaml(self, path: Path | str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(OmegaConf.structured(self), Path(path))


__all__ = ["DatasetGroups", "Families", "ExperimentConfig", "parse_config"]
