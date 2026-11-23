"""Expand an ExperimentConfig into jobs and run them (optionally in parallel).

Order of execution: all phase-A (source) jobs first, then all phase-B jobs, because
warm-started conditions need the source checkpoints.

Parallelism: ``workers > 1`` starts that many processes. Each process uses
``threads`` PyTorch threads. For these tiny models, several single-threaded
processes are usually faster than one multi-threaded process.
"""

from __future__ import annotations

import multiprocessing as mp
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import torch

from .config import ExperimentConfig
from .experiments import is_done, run_source, run_transfer, source_dir, transfer_dir


@dataclass(frozen=True)
class Job:
    kind: str  # "source" | "transfer"
    p: int
    q: int
    condition: str
    seed: int

    @property
    def label(self) -> str:
        if self.kind == "source":
            return f"source   p={self.p:<3d}            seed={self.seed}"
        return f"transfer p={self.p:<3d} q={self.q:<3d} {self.condition:<19s} seed={self.seed}"

    def out_dir(self, cfg: ExperimentConfig) -> Path:
        if self.kind == "source":
            return source_dir(cfg, self.p, self.seed)
        return transfer_dir(cfg, self.p, self.q, self.condition, self.seed)


def expand_jobs(cfg: ExperimentConfig) -> list[Job]:
    jobs = [Job("source", p, p, "source", s) for p in cfg.sources for s in cfg.seeds]
    jobs += [
        Job("transfer", p, q, c, s)
        for p in cfg.sources
        for q in cfg.targets[p]
        for c in cfg.conditions
        for s in cfg.seeds
    ]
    return jobs


# ------------------------------------------------------------------- execution

# Set in each worker process by _init_worker (a process pool cannot pickle closures).
_CFG: ExperimentConfig | None = None


def _init_worker(cfg_dict: dict, threads: int) -> None:
    global _CFG
    _CFG = ExperimentConfig.from_dict(cfg_dict)
    torch.set_num_threads(threads)


def _execute(job: Job, force: bool = False, verbose: bool = False) -> tuple[Job, dict | None, str | None, float]:
    """Run one job; never raises (errors are returned so the grid keeps going)."""
    t0 = time.time()
    try:
        if job.kind == "source":
            s = run_source(_CFG, job.p, job.seed, force=force, verbose=verbose)
        else:
            s = run_transfer(_CFG, job.p, job.q, job.condition, job.seed, force=force, verbose=verbose)
        return job, s, None, time.time() - t0
    except Exception:  # noqa: BLE001 - report and continue with the other jobs
        return job, None, traceback.format_exc(), time.time() - t0


def _report(i: int, n: int, job: Job, summary: dict | None, err: str | None, dt: float) -> None:
    if err is not None:
        print(f"[{i:>4d}/{n}] FAILED {job.label}\n{err}", flush=True)
        return
    tg = summary["T_grok"]
    status = f"T_grok={tg:>6d}" if tg is not None else f"not grokked in {summary['budget']}"
    extra = ""
    if summary.get("forgetting") is not None:
        extra = f"  forgetting={summary['forgetting']:+.2f}"
    print(f"[{i:>4d}/{n}] {job.label}  {status}  T_mem={summary['T_mem']}{extra}  ({dt:.0f}s)", flush=True)


def run_jobs(
    cfg: ExperimentConfig,
    jobs: list[Job],
    workers: int = 1,
    threads: int = 1,
    force: bool = False,
    verbose: bool = False,
) -> list[tuple[Job, str]]:
    """Run jobs (sources first). Returns the list of failed jobs with their tracebacks."""
    cfg.validate()
    cfg.save(cfg.exp_dir / "config.json")
    todo = [j for j in jobs if force or not is_done(j.out_dir(cfg))]
    print(f"{len(jobs)} jobs in grid, {len(jobs) - len(todo)} already done, {len(todo)} to run.", flush=True)

    failures: list[tuple[Job, str]] = []
    counter = 0
    for phase in ("source", "transfer"):
        batch = [j for j in todo if j.kind == phase]
        if not batch:
            continue
        if workers <= 1:
            _init_worker(cfg.to_dict(), threads)
            results = (_execute(j, force, verbose) for j in batch)
            for job, s, err, dt in results:
                counter += 1
                _report(counter, len(todo), job, s, err, dt)
                if err:
                    failures.append((job, err))
        else:
            ctx = mp.get_context("spawn")
            with ctx.Pool(workers, initializer=_init_worker, initargs=(cfg.to_dict(), threads)) as pool:
                args = [(j, force, False) for j in batch]
                # imap_unordered: report each job as soon as it finishes.
                for job, s, err, dt in pool.imap_unordered(_execute_star, args):
                    counter += 1
                    _report(counter, len(todo), job, s, err, dt)
                    if err:
                        failures.append((job, err))
    if failures:
        print(f"\n{len(failures)} job(s) failed; rerun the same command to retry them.", flush=True)
    return failures


def _execute_star(args: tuple) -> tuple:
    return _execute(*args)


# ---------------------------------------------------------------------- status


def status_table(cfg: ExperimentConfig) -> str:
    """Human-readable table of done / total runs per (p, q, condition)."""
    lines = [f"Experiment '{cfg.name}' in {cfg.exp_dir}", ""]
    for p in cfg.sources:
        done = sum(is_done(source_dir(cfg, p, s)) for s in cfg.seeds)
        lines.append(f"source p={p}: {done}/{len(cfg.seeds)} seeds done")
    lines.append("")
    header = f"{'target':>8s} " + " ".join(f"{c[:14]:>14s}" for c in cfg.conditions)
    for p in cfg.sources:
        lines.append(f"p = {p}")
        lines.append(header)
        for q in cfg.targets[p]:
            cells = []
            for c in cfg.conditions:
                done = sum(is_done(transfer_dir(cfg, p, q, c, s)) for s in cfg.seeds)
                cells.append(f"{done}/{len(cfg.seeds)}".rjust(14))
            lines.append(f"{q:>8d} " + " ".join(cells))
        lines.append("")
    return "\n".join(lines)
