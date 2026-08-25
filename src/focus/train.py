"""Training recipe for FocusHead under TOLRIZZ, and a CLI to run the standard
dataset-transfer benchmark (`focus.experiments`) end to end.

The recipe (optimizer, schedule, early stopping) matches
chapters/05-method.tex / the architecture-search notebook this repo's model
and loss were extracted from: Adam, StepLR(50, gamma=0.1), early stop on
validation PDR@tau=0.25 with patience 15.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from focus.experiments import EmbeddingStore  # noqa: F401  (re-exported for convenience)
from focus.experiments.config import ExperimentConfig
from focus.experiments.data import PairData
from focus.experiments.runner import ExperimentRunner
from focus.loss import diff, tolrizz_loss
from focus.model import N_RANKS, FocusHead

EPOCHS = 100
PATIENCE = 15
LR = 5e-3
WEIGHT_DECAY = 1e-4
STEP_SIZE = 50
GAMMA = 0.1


@torch.no_grad()
def _val_pdr(head: FocusHead, val: PairData, tau: float = 0.25) -> float:
    d = diff(head, val.emb, val.cls)
    pred = torch.zeros_like(val.label)
    pred[d > tau] = 1
    pred[d < -tau] = -1
    return (pred != val.label).float().mean().item()


def focus_factory(train: PairData, val: PairData, seed: int) -> FocusHead:
    """`ModelFactory`-shaped: (train_pool, val_pool, seed) -> trained FocusHead.

    Assumes a two-branch fused embedding of [384, 384] (loftup + uplift); see
    `focus.model.FocusHead` for a differently-shaped fusion.
    """
    torch.manual_seed(seed)
    assert train.emb_dim == 768, f"expected two 384-d branches fused, got emb_dim={train.emb_dim}"
    head = FocusHead([384, 384], train.cls_dim, n_ranks=N_RANKS).to(train.device)
    opt = torch.optim.Adam(head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=STEP_SIZE, gamma=GAMMA)

    best_pdr, best_state, wait = float("inf"), None, 0
    for _ in range(EPOCHS):
        head.train()
        opt.zero_grad()
        tolrizz_loss(head, train.emb, train.cls, train.label).backward()
        opt.step()
        sched.step()

        head.eval()
        v = _val_pdr(head, val)
        if v < best_pdr - 1e-6:
            best_pdr, best_state, wait = v, {k: t.clone() for k, t in head.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break

    if best_state is not None:
        head.load_state_dict(best_state)
    head.eval()
    return head


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config-yaml", type=Path, default=Path("configs/experiment/transfer_default.yaml"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[10, 28, 42, 77, 123],
                     help="thesis's reported seeds (chapters/04-dataset.tex)")
    ap.add_argument("--fused-config", default="dinov2_vits__loftup+dinov2_vits__uplift")
    ap.add_argument("--only-pooled-all", action="store_true",
                     help="restrict to the pooled:all setting only (fast; matches the abstract's headline number)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config_yaml)
    cfg.seeds = args.seeds
    cfg.configs = [args.fused_config]
    if args.only_pooled_all:
        cfg.families.in_domain = False
        cfg.families.real_pooled = False
        cfg.families.sim_pooled = False
        cfg.families.sim_to_real = False
        cfg.families.leave_one_out = False
        cfg.families.all_pooled = True

    runner = ExperimentRunner(cfg, device=args.device, progress=print)
    store = runner.run({"FOCUS": focus_factory})
    df = store.frame()
    print(df.groupby("setting")[["pdr", "trivial"]].mean())


if __name__ == "__main__":
    main()
