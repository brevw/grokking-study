"""Phase A (source) and phase B (transfer) runs.

Directory layout of an experiment ``results/<name>/``::

    config.json                                   the experiment config
    source/p97/s0/        log.csv repr.csv summary.json memorized.pt final.pt
    transfer/p97_q113/warm_grokked/s0/   log.csv repr.csv summary.json

Seeding (all seeds derived with ``utils.stable_seed``):

* phase-A split and init: depend on (p, seed);
* phase-B split and permutation: depend on (q, seed), and are **shared by all
  conditions** of that seed. A paired comparison (C1 vs C0 with the same seed) then
  differs only in the initial weights.
* scratch init of phase B: depends on (q, seed).

Every run is skipped if its ``summary.json`` already exists (unless ``force=True``),
so a grid can be interrupted and resumed at any time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .conditions import CONDITIONS, Condition
from .config import ExperimentConfig
from .data import ModularTask
from .model import GrokTransformer
from .representation import OldTaskInfo
from .train import Trainer
from .utils import read_json, set_seed, stable_seed

# ----------------------------------------------------------------------- paths


def source_dir(cfg: ExperimentConfig, p: int, seed: int) -> Path:
    return cfg.exp_dir / "source" / f"p{p}" / f"s{seed}"


def transfer_dir(cfg: ExperimentConfig, p: int, q: int, condition: str, seed: int) -> Path:
    return cfg.exp_dir / "transfer" / f"p{p}_q{q}" / condition / f"s{seed}"


def is_done(run_dir: Path) -> bool:
    return (run_dir / "summary.json").exists()


# ----------------------------------------------------------------------- tasks


def source_task(cfg: ExperimentConfig, p: int, seed: int) -> ModularTask:
    """The phase-A task ("=p"). Phase B rebuilds exactly the same split to measure forgetting."""
    return ModularTask.build(
        p,
        cfg.model.n_numbers,
        cfg.task.op,
        cfg.task.train_frac,
        split_seed=stable_seed("split", "A", p, seed),
        eq_slot=0,
    )


def target_task(cfg: ExperimentConfig, q: int, seed: int, permute: bool) -> ModularTask:
    """The phase-B task ("=q"), also used from scratch by the C0 baseline."""
    perm_seed = stable_seed("perm", q, seed) if permute else None
    return ModularTask.build(
        q,
        cfg.model.n_numbers,
        cfg.task.op,
        cfg.task.train_frac,
        split_seed=stable_seed("split", "B", q, seed),
        perm_seed=perm_seed,
        eq_slot=1,
    )


def new_model(cfg: ExperimentConfig, init_seed: int) -> GrokTransformer:
    set_seed(init_seed)
    return GrokTransformer(cfg.model)


# --------------------------------------------------------------------- phase A


def run_source(cfg: ExperimentConfig, p: int, seed: int, force: bool = False, verbose: bool = False) -> dict[str, Any]:
    """Train on Z_p from scratch until grokked (+ cleanup). Saves memorized.pt and final.pt."""
    out = source_dir(cfg, p, seed)
    if is_done(out) and not force:
        return read_json(out / "summary.json")

    task = source_task(cfg, p, seed)
    model = new_model(cfg, stable_seed("init", "A", p, seed))
    if verbose:
        print(f"[source] p={p} seed={seed}: {task.describe()}", flush=True)

    trainer = Trainer(
        model,
        train_tasks=[task],
        primary=task,
        cfg=cfg.train_source,
        out_dir=out,
        save_memorized=True,
        save_final=True,
        run_info={"kind": "source", "p": p, "q": p, "condition": "source", "seed": seed},
    )
    return trainer.fit()


# --------------------------------------------------------------------- phase B


def _initial_weights(cfg: ExperimentConfig, cond: Condition, p: int, q: int, seed: int) -> tuple[GrokTransformer, dict]:
    """Build the phase-B starting network for a condition. Returns (model, info for the summary)."""
    info: dict[str, Any] = {}
    model = new_model(cfg, stable_seed("init", "B", q, seed))  # scratch init (also used by S&P)
    if cond.init == "scratch":
        return model, info

    src = source_dir(cfg, p, seed)
    if not is_done(src):
        raise FileNotFoundError(f"Source run {src} missing: run phase A first (`run.py source` or `run.py grid`).")
    src_summary = read_json(src / "summary.json")
    info["source_grokked"] = src_summary["grokked"]
    info["source_T_grok"] = src_summary["T_grok"]

    ckpt = {"grokked": "final.pt", "memorized": "memorized.pt"}[cond.init]
    if not (src / ckpt).exists():
        raise FileNotFoundError(
            f"{src / ckpt} missing: the source run never reached train acc >= threshold "
            f"(T_mem={src_summary['T_mem']}), so there is no memorized checkpoint. Increase train_source.max_steps."
        )
    fresh = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(torch.load(src / ckpt, map_location="cpu"))

    if cond.reinit_embeddings:
        g = torch.Generator().manual_seed(stable_seed("reinit", q, seed))
        model.reset_embeddings(generator=g)
    if cond.shrink_perturb:
        with torch.no_grad():
            for name, param in model.named_parameters():
                param.mul_(cfg.shrink).add_(cfg.perturb * fresh[name])
        info.update(shrink=cfg.shrink, perturb=cfg.perturb)
    return model, info


def run_transfer(
    cfg: ExperimentConfig,
    p: int,
    q: int,
    condition: str,
    seed: int,
    force: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
    """Phase B: train on Z_q under one condition, while monitoring Z_p."""
    out = transfer_dir(cfg, p, q, condition, seed)
    if is_done(out) and not force:
        return read_json(out / "summary.json")

    cond = CONDITIONS[condition]
    new = target_task(cfg, q, seed, permute=cond.permute)
    old = source_task(cfg, p, seed)  # same split as phase A, so its val set was never trained on
    model, info = _initial_weights(cfg, cond, p, q, seed)
    if verbose:
        print(f"[transfer] p={p} -> q={q} {condition} seed={seed}: {new.describe()}", flush=True)

    # Key frequencies of the grokked source network (for the old-frequency retention metric).
    old_info = None
    src_summary_path = source_dir(cfg, p, seed) / "summary.json"
    if src_summary_path.exists():
        freqs = read_json(src_summary_path).get("final_key_freqs") or ""
        old_info = OldTaskInfo(p=p, key_freqs=[int(k) for k in freqs.split(";") if k])

    trainer = Trainer(
        model,
        train_tasks=[old, new] if cond.joint else [new],
        primary=new,
        cfg=cfg.train_transfer,
        out_dir=out,
        eval_tasks={"old": old},
        val_masks=new.overlap_masks(p),
        old=old_info,
        run_info={"kind": "transfer", "p": p, "q": q, "condition": condition, "seed": seed, **info},
    )
    return trainer.fit()
