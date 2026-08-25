"""Build the standard dataset transfer matrix from an :class:`ExperimentConfig`.

Each :class:`ExperimentSetting` is a named ``(train_specs -> test_specs)`` pair,
where a *spec* is a list of ``(dataset, split)`` selections.  The runner pools
the train specs (carving an early-stopping holdout) and the test specs, so the
exact same splits are seen by every model for a given seed.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import ExperimentConfig
from .data import Which

Spec = list[tuple[str, Which]]


@dataclass(frozen=True)
class ExperimentSetting:
    """One named train -> test protocol."""

    name: str
    family: str
    train: Spec
    test: Spec

    def describe(self) -> str:
        def fmt(spec: Spec) -> str:
            return "+".join(f"{ds}:{w}" for ds, w in spec)
        return f"{fmt(self.train)} -> {fmt(self.test)}"


def _train(datasets: list[str], which: Which = "all") -> Spec:
    return [(d, which) for d in datasets]


def build_settings(cfg: ExperimentConfig, available: list[str] | None = None) -> list[ExperimentSetting]:
    """Construct all enabled settings; drop any that reference unavailable datasets."""
    g = cfg.datasets
    fam = cfg.families
    real, sim1, sim2 = g.real, g.sim1, g.sim2
    all_ds = g.all_datasets()
    settings: list[ExperimentSetting] = []

    if fam.in_domain:
        for ds in all_ds:
            settings.append(ExperimentSetting(
                name=f"in_domain:{ds}", family="in_domain",
                train=[(ds, "train")], test=[(ds, "val")],
            ))

    # "pooled" family: pooled:real, pooled:all, and pooled:sim are siblings,
    # reported together as one family-level average (ResultStore.family_leaderboard).
    if fam.real_pooled and real:
        settings.append(ExperimentSetting(
            name="pooled:real", family="pooled",
            train=_train(real, "train"), test=_train(real, "val"),
        ))

    if fam.all_pooled and all_ds:
        settings.append(ExperimentSetting(
            name="pooled:all", family="pooled",
            train=_train(all_ds, "train"), test=_train(all_ds, "val"),
        ))

    if fam.sim_pooled and (sim1 or sim2):
        sim_all = [*sim1, *(d for d in sim2 if d not in sim1)]
        settings.append(ExperimentSetting(
            name="pooled:sim", family="pooled",
            train=_train(sim_all, "train"), test=_train(sim_all, "val"),
        ))

    # "sim_to_real" family: named after the actual dataset(s) trained on rather
    # than the internal sim1/sim2 config labels, which carry no information of
    # their own beyond "whichever dataset(s) DatasetGroups.sim1/sim2 point to".
    if fam.sim_to_real and real:
        if sim1:
            settings.append(ExperimentSetting(
                name=f"sim_to_real:{'+'.join(sim1)}", family="sim_to_real",
                train=_train(sim1, "all"), test=_train(real, "all"),
            ))
        if sim2:
            settings.append(ExperimentSetting(
                name=f"sim_to_real:{'+'.join(sim2)}", family="sim_to_real",
                train=_train(sim2, "all"), test=_train(real, "all"),
            ))
        if sim1 and sim2:
            settings.append(ExperimentSetting(
                name="sim_to_real:both", family="sim_to_real",
                train=_train(sim1 + sim2, "all"), test=_train(real, "all"),
            ))

    if fam.reverse_real_to_sim and real:
        if sim1:
            settings.append(ExperimentSetting(
                name="real->sim1", family="reverse_real_to_sim",
                train=_train(real, "all"), test=_train(sim1, "all"),
            ))
        if sim2:
            settings.append(ExperimentSetting(
                name="real->sim2", family="reverse_real_to_sim",
                train=_train(real, "all"), test=_train(sim2, "all"),
            ))

    if fam.cross_sim and sim1 and sim2:
        settings.append(ExperimentSetting(
            name="sim1->sim2", family="cross_sim",
            train=_train(sim1, "all"), test=_train(sim2, "all"),
        ))
        settings.append(ExperimentSetting(
            name="sim2->sim1", family="cross_sim",
            train=_train(sim2, "all"), test=_train(sim1, "all"),
        ))

    if fam.leave_one_out and len(all_ds) > 1:
        for held in all_ds:
            others = [d for d in all_ds if d != held]
            settings.append(ExperimentSetting(
                name=f"loo:{held}", family="leave_one_out",
                train=_train(others, "all"), test=[(held, "all")],
            ))

    if available is not None:
        avail = set(available)
        settings = [
            s for s in settings
            if all(ds in avail for ds, _ in s.train)
            and all(ds in avail for ds, _ in s.test)
        ]
    return settings


__all__ = ["Spec", "ExperimentSetting", "build_settings"]
