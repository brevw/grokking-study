"""Phase A (source) and phase B (transfer) runs.

Directory layout of an experiment ``results/<name>/``::

    config.json                                   the experiment config
    source/p97/s0/        log.csv repr.csv summary.json
                          final.pt memorized.pt progress_10.pt ... progress_100.pt
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

import shutil
from pathlib import Path
from typing import Any

import torch

from .conditions import CONDITIONS, PROGRESS, Condition
from .config import ExperimentConfig
from .data import ModularTask, eq_token
from .model import GrokTransformer
from .representation import OldTaskInfo, fourier_power, fourier_sparsity
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
    """Train on Z_p from scratch until grokked (+ clean-up).

    Saves ``final.pt`` and, picked from the periodic checkpoints, the starting points
    of phase B: ``memorized.pt`` and ``progress_XX.pt`` (see ``select_checkpoints``).
    """
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
        save_final=True,
        run_info={"kind": "source", "p": p, "q": p, "condition": "source", "seed": seed},
        on_finish=lambda tr, summary: select_checkpoints(cfg, p, out, tr.log, summary),
    )
    return trainer.fit()


def select_checkpoints(cfg: ExperimentConfig, p: int, out: Path, log: list[dict], summary: dict) -> None:
    """Pick the phase-B starting points among the phase-A checkpoints ``ckpt/step_*.pt``.

    * ``memorized.pt``: end of the memorization plateau = the last checkpoint before
      T_grok with train acc >= grok_threshold and val acc < cfg.mem_val_threshold.
    * ``progress_XX.pt``: the checkpoint closest to XX% of T_grok (only if grokked).

    Adds to ``summary``, for each of them, the step, the val accuracy and the Fourier
    sparsity of the embedding (how much "clock" structure is already there), e.g.
    ``memorized_step``, ``progress_50_val_acc``, ``progress_50_sparsity``. Then deletes
    the periodic checkpoints unless cfg.keep_source_checkpoints.
    """
    ckpt_dir = out / "ckpt"
    saved = sorted(int(f.stem.split("_")[1]) for f in ckpt_dir.glob("step_*.pt"))
    rows = {r["step"]: r for r in log}
    t_grok = summary["T_grok"]

    def keep(name: str, step: int) -> None:
        shutil.copyfile(ckpt_dir / f"step_{step}.pt", out / f"{name}.pt")
        W = torch.load(out / f"{name}.pt", map_location="cpu")["embed.weight"][:p].numpy()
        summary[f"{name}_step"] = step
        summary[f"{name}_val_acc"] = rows[step]["val_acc"]
        summary[f"{name}_sparsity"] = fourier_sparsity(fourier_power(W))

    limit = t_grok if t_grok is not None else saved[-1]
    plateau = [
        s
        for s in saved
        if s <= limit
        and rows[s]["train_acc"] >= cfg.train_source.grok_threshold
        and rows[s]["val_acc"] < cfg.mem_val_threshold
    ]
    if plateau:
        keep("memorized", plateau[-1])
    if t_grok is not None:
        for pct in PROGRESS:
            keep(f"progress_{pct}", min(saved, key=lambda s: abs(s - pct / 100 * t_grok)))
    if not cfg.keep_source_checkpoints:
        shutil.rmtree(ckpt_dir)


# --------------------------------------------------------------------- phase B


def _fresh_unseen_rows(cfg: ExperimentConfig, model: GrokTransformer, fresh: dict, p: int) -> None:
    """Give the tokens never used in phase A (numbers >= p, and "=q") their scratch-init rows.

    In phase A their embedding and unembedding rows get no gradient, so weight decay
    shrinks them to ~0: without this, all new symbols would start identical.
    """
    V = cfg.model.n_numbers
    with torch.no_grad():
        model.embed.weight[p:V] = fresh["embed.weight"][p:V]
        model.embed.weight[eq_token(1, V)] = fresh["embed.weight"][eq_token(1, V)]
        model.unembed.weight[p:V] = fresh["unembed.weight"][p:V]


def _initial_weights(cfg: ExperimentConfig, cond: Condition, p: int, q: int, seed: int) -> tuple[GrokTransformer, dict]:
    """Build the phase-B starting network for a condition. Returns (model, info for the summary)."""
    info: dict[str, Any] = {}
    model = new_model(cfg, stable_seed("init", "B", q, seed))  # scratch init (also used by S&P and fresh rows)
    if not cond.needs_source:
        return model, info

    src = source_dir(cfg, p, seed)
    if not is_done(src):
        raise FileNotFoundError(f"Source run {src} missing: run phase A first (`run.py source` or `run.py grid`).")
    src_summary = read_json(src / "summary.json")
    info["source_grokked"] = src_summary["grokked"]
    info["source_T_grok"] = src_summary["T_grok"]

    stem = "final" if cond.norm_match else cond.init
    if not (src / f"{stem}.pt").exists():
        why = (
            f"no checkpoint with train acc >= {cfg.train_source.grok_threshold} and val acc < {cfg.mem_val_threshold}"
            if stem == "memorized"
            else "the source run did not grok, so it has no progress checkpoints"
        )
        raise FileNotFoundError(f"{src / f'{stem}.pt'} missing: {why}.")
    fresh = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(torch.load(src / f"{stem}.pt", map_location="cpu"))
    info["source_checkpoint"] = stem
    info["source_checkpoint_step"] = src_summary.get(f"{stem}_step", src_summary["steps_run"])

    if cond.reinit_embeddings:
        g = torch.Generator().manual_seed(stable_seed("reinit", q, seed))
        model.reset_embeddings(generator=g)
    if cond.shrink_perturb:
        with torch.no_grad():
            for name, param in model.named_parameters():
                param.mul_(cfg.shrink).add_(cfg.perturb * fresh[name])
        info.update(shrink=cfg.shrink, perturb=cfg.perturb)
    if cfg.fresh_unseen_rows and not cond.reinit_embeddings:  # (reinit already gives fresh rows)
        _fresh_unseen_rows(cfg, model, fresh, p)
        info["fresh_unseen_rows"] = True
    if cond.norm_match:
        # The random init, each weight matrix rescaled to the norm it has in the warm start.
        warm = {k: v.clone() for k, v in model.state_dict().items()}
        with torch.no_grad():
            for name, param in model.named_parameters():
                param.copy_(fresh[name] * (warm[name].norm() / fresh[name].norm()))
        info["norm_matched_to"] = stem
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
        # Old pairs split by what the *new* rule (mod q) answers: right ("agree"),
        # wrong ("disagree": real forgetting shows here), or tokens >= q ("newtok").
        eval_masks={"old": old.overlap_masks(q)},
        old=old_info,
        run_info={"kind": "transfer", "p": p, "q": q, "condition": condition, "seed": seed, **info},
    )
    return trainer.fit()
