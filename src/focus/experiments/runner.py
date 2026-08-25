"""The experiment runner: ``(model x setting x seed) -> ResultStore``.

The runner is the single orchestration point.  Given an :class:`ExperimentConfig`
and a dict of model factories, it:

1. builds the dataset transfer matrix (:func:`build_settings`),
2. for each setting, pools the train specs and carves a per-seed early-stopping
   holdout, pools the test specs,
3. trains each model with each seed and scores it on the held-out test pool,
4. records overall / within / cross PDR at the fixed ``tau`` and (optionally) at
   a holdout-validated ``tau``.

Models are intentionally external (see :mod:`.protocols`): the same protocol is
applied to every model so results are directly comparable.

The work is embarrassingly parallel over backbone ``config``: :meth:`run`
executes sequentially, while :meth:`run_parallel` shards the configs across
worker processes (via joblib) and merges the per-config records.  Workers run on
CPU by default — the heads are tiny, this avoids CUDA-in-subprocess issues, and
it keeps results deterministic per ``(config, seed)``.
"""

from __future__ import annotations

from typing import Callable, Iterable, Mapping

import torch

from .config import ExperimentConfig
from .data import EmbeddingStore
from .metrics import pdr, select_tau, trivial_pdr
from .protocols import ModelFactory, score_diff, seed_everything
from .results import ResultStore, RunRecord
from .settings import build_settings

ProgressFn = Callable[[str], None]


def _spec_datasets(spec) -> str:
    seen: list[str] = []
    for ds, _ in spec:
        if ds not in seen:
            seen.append(ds)
    return "+".join(seen)


def _compute_config_records(
    cfg: ExperimentConfig,
    config_name: str,
    models: Mapping[str, ModelFactory],
    device: torch.device | str,
    progress: ProgressFn | None = None,
    limit_threads: bool = False,
    on_step: Callable[[dict], None] | None = None,
    progress_queue=None,
) -> list[RunRecord]:
    """Train/evaluate every ``(setting, seed, model)`` for one backbone config.

    Returns a flat list of :class:`RunRecord`.  This is the unit of work both the
    sequential and the parallel runner use; it is a module-level function so it
    can be shipped to worker processes.  ``on_step`` (sequential use only) is
    called after every single model fit with a small status dict.  ``progress_queue``
    (parallel use) is a multiprocessing queue onto which this worker pushes
    ``("total"|"tick"|"done", config_name, ...)`` messages so the main process can
    drive a per-config progress bar.
    """
    if limit_threads:
        # one BLAS/torch thread per worker so N processes don't oversubscribe cores
        torch.set_num_threads(1)
        try:
            from threadpoolctl import threadpool_limits

            threadpool_limits(1)
        except Exception:
            pass

    log = progress or (lambda _m: None)
    device = torch.device(device)
    emb_store = EmbeddingStore(config_name, cfg.embeddings_root, device)
    all_ds = cfg.datasets.all_datasets()
    available = emb_store.available(all_ds)
    missing = set(all_ds) - set(available)
    if missing:
        log(f"[{config_name}] WARNING: missing embeddings for {sorted(missing)} — skipped those settings.")
    settings = build_settings(cfg, available)
    if not settings:
        log(f"[{config_name}] no runnable settings — skipped.")
        if progress_queue is not None:
            progress_queue.put(("done", config_name, 0))
        return []

    if progress_queue is not None:
        progress_queue.put(("total", config_name, len(settings) * len(cfg.seeds) * len(models)))

    records: list[RunRecord] = []
    for setting in settings:
        train_full = emb_store.pool([(ds, w) for ds, w in setting.train])  # type: ignore[arg-type]
        test_pool = emb_store.pool([(ds, w) for ds, w in setting.test])    # type: ignore[arg-type]
        train_ds = _spec_datasets(setting.train)
        test_ds = _spec_datasets(setting.test)
        log(
            f"[{config_name}][{setting.name}] {setting.describe()}  "
            f"(train={train_full.n} test={test_pool.n} trivial={trivial_pdr(test_pool.label):.3f})"
        )
        for seed in cfg.seeds:
            tr, va = train_full.split_holdout(cfg.val_frac, seed=seed)
            for model_name, factory in models.items():
                seed_everything(seed)
                model = factory(tr, va, seed)
                diff = score_diff(model, test_pool)
                res = pdr(diff, test_pool.label, test_pool.within, cfg.tau)

                val_tau, pdr_val = float("nan"), float("nan")
                if cfg.report_val_tau:
                    val_diff = score_diff(model, va)
                    val_tau = select_tau(val_diff, va.label, cfg.tau_grid)
                    pdr_val = pdr(diff, test_pool.label, test_pool.within, val_tau).pdr

                records.append(RunRecord(
                    model=model_name, setting=setting.name, family=setting.family,
                    seed=seed, tau=cfg.tau, pdr=res.pdr, within=res.within,
                    cross=res.cross, n_test=res.n, trivial=res.trivial,
                    config=config_name, train_datasets=train_ds, test_datasets=test_ds,
                    pdr_val_tau=pdr_val, val_tau=val_tau, train_pairs=tr.n,
                ))
                if on_step is not None:
                    on_step({"config": config_name, "setting": setting.name,
                             "seed": seed, "model": model_name, "pdr": res.pdr})
                if progress_queue is not None:
                    progress_queue.put(("tick", config_name, setting.name, seed, model_name, res.pdr))
    log(f"[{config_name}] done ({len(records)} records).")
    if progress_queue is not None:
        progress_queue.put(("done", config_name, len(records)))
    return records


class ExperimentRunner:
    """Runs a fixed dataset protocol over a set of pluggable model factories.

    Sweeps every embedding backbone in ``cfg.configs``; each ``(config, model,
    setting, seed)`` evaluation is recorded so results can be averaged per
    architecture, per model, per dataset, etc.
    """

    def __init__(
        self,
        cfg: ExperimentConfig,
        device: torch.device | str = "cpu",
        progress: ProgressFn | None = print,
    ) -> None:
        self.cfg = cfg
        self.device = torch.device(device)
        self.progress = progress or (lambda _msg: None)

    def _new_store(self, config_list: list[str], models: Mapping[str, ModelFactory], device: str) -> ResultStore:
        meta = {
            "configs": config_list, "tau": self.cfg.tau, "seeds": self.cfg.seeds,
            "models": list(models), "device": device,
        }
        return ResultStore(self.cfg.name, self.cfg.output_dir, meta=meta)

    # ── planning ───────────────────────────────────────────────────────────
    def _per_config_totals(
        self, models: Mapping[str, ModelFactory], config_list: list[str]
    ) -> dict[str, int]:
        """Number of model fits per config (settings x seeds x models, availability-aware)."""
        all_ds = self.cfg.datasets.all_datasets()
        totals: dict[str, int] = {}
        for config_name in config_list:
            store = EmbeddingStore(config_name, self.cfg.embeddings_root, self.device)
            available = store.available(all_ds)
            n_set = len(build_settings(self.cfg, available))
            totals[config_name] = n_set * len(self.cfg.seeds) * len(models)
        return totals

    def count_trainings(
        self, models: Mapping[str, ModelFactory], configs: Iterable[str] | None = None
    ) -> int:
        """Total number of model fits = sum over configs of settings x seeds x models.

        Counts only settings whose datasets are actually available for each config.
        """
        config_list = list(configs) if configs is not None else list(self.cfg.configs)
        return sum(self._per_config_totals(models, config_list).values())

    @staticmethod
    def _has_widget_tqdm() -> bool:
        """True if ipywidgets-backed tqdm bars will render (no ANSI-position hacks)."""
        import importlib.util

        return importlib.util.find_spec("ipywidgets") is not None

    @staticmethod
    def _make_bar(total: int, desc: str):
        """Return a tqdm bar (notebook-aware) or ``None`` if tqdm is unavailable."""
        try:
            from tqdm.auto import tqdm

            return tqdm(total=total, desc=desc, unit="fit", dynamic_ncols=True)
        except Exception:
            return None

    # ── public API ─────────────────────────────────────────────────────────
    def run(
        self,
        models: Mapping[str, ModelFactory],
        configs: Iterable[str] | None = None,
        save: bool = True,
        progress_bar: bool = True,
    ) -> ResultStore:
        """Run *models* over every backbone config x setting x seed, sequentially.

        With ``progress_bar=True`` a tqdm bar advances per model fit and shows the
        current ``config/setting/seed/model`` plus its PDR in the postfix.
        """
        if not models:
            raise ValueError("Provide at least one model factory.")
        config_list = list(configs) if configs is not None else list(self.cfg.configs)
        if not config_list:
            raise ValueError("No embedding configs to run (cfg.configs is empty).")

        bar = self._make_bar(self.count_trainings(models, config_list), "benchmark") if progress_bar else None

        def _step(info: dict) -> None:
            if bar is not None:
                bar.update(1)
                bar.set_postfix_str(
                    f"{info['config']}|{info['setting']}|s{info['seed']}|{info['model']}={info['pdr']:.3f}"
                )

        store = self._new_store(config_list, models, str(self.device))
        try:
            for config_name in config_list:
                for rec in _compute_config_records(
                    self.cfg, config_name, models, self.device, self.progress, on_step=_step
                ):
                    store.add(rec)
        finally:
            if bar is not None:
                bar.close()

        if not store.records:
            raise RuntimeError("No records produced (check dataset/config availability).")
        if save:
            path = store.save()
            self.progress(f"saved {len(store.records)} records -> {path}")
        return store

    def run_parallel(
        self,
        models: Mapping[str, ModelFactory],
        n_jobs: int = -1,
        configs: Iterable[str] | None = None,
        worker_device: str = "cpu",
        backend: str = "loky",
        save: bool = True,
        progress_bar: bool = True,
        per_worker_bars: bool = True,
    ) -> ResultStore:
        """Run the grid with backbone configs sharded across worker processes.

        Parameters
        ----------
        n_jobs:
            Number of worker processes (``-1`` = all cores).  Effective
            parallelism is capped at the number of configs.
        worker_device:
            Device used *inside* workers.  Defaults to ``"cpu"`` because the heads
            are tiny and CUDA does not survive process forking cleanly.
        backend:
            joblib backend; ``"loky"`` (default) uses cloudpickle so notebook-
            defined model factories serialise correctly.
        progress_bar:
            Show progress while running.
        per_worker_bars:
            If True, give **each backbone config its own live tqdm bar** (advancing
            per model fit) via a shared progress queue; if False, a single bar
            advances once per finished config.

        Results are identical to :meth:`run` (each ``(config, seed)`` is seeded
        independently in its own process), just produced faster.
        """
        if not models:
            raise ValueError("Provide at least one model factory.")
        config_list = list(configs) if configs is not None else list(self.cfg.configs)
        if not config_list:
            raise ValueError("No embedding configs to run (cfg.configs is empty).")

        self.progress(
            f"running {len(config_list)} configs across n_jobs={n_jobs} "
            f"workers on '{worker_device}' (backend={backend})…"
        )
        models = dict(models)
        if progress_bar and per_worker_bars:
            return self._run_parallel_per_worker(
                models, config_list, n_jobs, worker_device, backend, save
            )
        return self._run_parallel_simple(
            models, config_list, n_jobs, worker_device, backend, save, progress_bar
        )

    # ── parallel: one bar per config, driven by a shared queue ──────────────
    def _run_parallel_per_worker(
        self, models, config_list, n_jobs, worker_device, backend, save,
    ) -> ResultStore:
        """Per-config progress.

        With ipywidgets available, each config gets its own **widget** bar (clean,
        stacks as HTML).  Without ipywidgets, text-mode multi-position bars render
        as ANSI garbage in notebooks, so we fall back to a single clean aggregate
        bar over the whole grid.
        """
        import multiprocessing as mp
        import threading

        from joblib import Parallel, delayed

        widgets = self._has_widget_tqdm()
        totals = self._per_config_totals(models, config_list)
        grand_total = sum(totals.values())

        if widgets:
            from tqdm.notebook import tqdm as _tqdm
            # pre-create one bar per config, in order, so they stack deterministically
            bars = {
                c: _tqdm(total=totals[c], desc=c, unit="fit", leave=True)
                for c in config_list
            }
            agg = None
        else:
            self.progress(
                "ipywidgets not installed — using a single aggregate bar. "
                "Run `pip install ipywidgets` for one bar per worker."
            )
            bars = {}
            agg = self._make_bar(grand_total, "benchmark")

        manager = mp.Manager()
        q = manager.Queue()

        def _updater():
            remaining = len(config_list)
            while remaining > 0:
                msg = q.get()
                if msg is None:
                    break
                kind, cfg_name = msg[0], msg[1]
                if kind == "tick":
                    _, _, setting, seed, model_name, pdr_v = msg
                    post = f"{setting}|s{seed}|{model_name}={pdr_v:.3f}"
                    if widgets:
                        b = bars.get(cfg_name)
                        if b is not None:
                            b.update(1)
                            b.set_postfix_str(post)
                    elif agg is not None:
                        agg.update(1)
                        agg.set_postfix_str(f"{cfg_name}|{post}")
                elif kind == "done":
                    remaining -= 1

        thread = threading.Thread(target=_updater, daemon=True)
        thread.start()

        store = self._new_store(config_list, models, worker_device)
        try:
            results = Parallel(n_jobs=n_jobs, backend=backend)(
                delayed(_compute_config_records)(
                    self.cfg, cfg_name, models, worker_device, None, True, None, q
                )
                for cfg_name in config_list
            )
            for recs in results:
                for rec in recs:
                    store.add(rec)
        finally:
            q.put(None)
            thread.join(timeout=5)
            for b in bars.values():
                b.close()
            if agg is not None:
                agg.close()
            manager.shutdown()

        if not store.records:
            raise RuntimeError("No records produced (check dataset/config availability).")
        if save:
            path = store.save()
            self.progress(f"saved {len(store.records)} records -> {path}")
        return store

    # ── parallel: single bar advancing per finished config ─────────────────
    def _run_parallel_simple(
        self, models, config_list, n_jobs, worker_device, backend, save, progress_bar,
    ) -> ResultStore:
        from joblib import Parallel, delayed

        tasks = (
            delayed(_compute_config_records)(
                self.cfg, cfg_name, models, worker_device, None, True
            )
            for cfg_name in config_list
        )
        store = self._new_store(config_list, models, worker_device)
        bar = self._make_bar(len(config_list), "configs") if progress_bar else None
        try:
            try:
                results_iter = Parallel(n_jobs=n_jobs, backend=backend, return_as="generator")(tasks)
            except TypeError:
                results_iter = Parallel(n_jobs=n_jobs, backend=backend)(tasks)
            for recs in results_iter:
                for rec in recs:
                    store.add(rec)
                if bar is not None:
                    bar.update(1)
                    done_cfgs = len({r.config for r in store.records})
                    bar.set_postfix_str(f"{len(store.records)} records, {done_cfgs} configs done")
                else:
                    self.progress(f"  config finished ({len(store.records)} records so far)")
        finally:
            if bar is not None:
                bar.close()

        if not store.records:
            raise RuntimeError("No records produced (check dataset/config availability).")
        if save:
            path = store.save()
            self.progress(f"saved {len(store.records)} records -> {path}")
        return store


__all__ = ["ExperimentRunner"]
