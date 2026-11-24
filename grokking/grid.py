"""Expand an ExperimentConfig into jobs and run them (optionally in parallel).

Order of execution: all phase-A (source) jobs first, then all phase-B jobs, because
warm-started conditions need the source checkpoints.

Parallelism: ``workers > 1`` starts that many processes. Each process uses
``threads`` PyTorch threads. For these tiny models, several single-threaded
processes are usually faster than one multi-threaded process. On one GPU, 2 workers
are enough to keep it busy.

Several grid commands may work on the same experiment at the same time (e.g. two
Slurm jobs on two GPUs, see ``cluster/``). Before running a job, a worker *claims*
it by creating ``<run dir>/.claim`` (atomic, so only one worker wins). The others
skip it. While the job runs, the claim file is touched every minute. A claim is
dead, and taken over, if its Slurm job is no longer running (e.g. it hit the time
limit: the next job of a chain redoes its unfinished runs right away), or if it has
not been touched for 10 minutes. Transfer jobs wait for their source runs if another worker
is still training them.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import re
import socket
import subprocess
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import torch

from .conditions import CONDITIONS
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


# ---------------------------------------------------------------------- claims

CLAIM = ".claim"
HEARTBEAT_S = 60  # a running job touches its claim this often
STALE_S = 600  # a claim untouched for this long belongs to a dead process


def _claim_age(run_dir: Path) -> float | None:
    """Seconds since the claim of ``run_dir`` was last touched (None = no claim)."""
    try:
        return time.time() - (run_dir / CLAIM).stat().st_mtime
    except FileNotFoundError:
        return None


def _slurm_job_alive(job_id: str) -> bool | None:
    """Is this Slurm job still running? None if Slurm cannot tell (e.g. no squeue here)."""
    if job_id == os.environ.get("SLURM_JOB_ID"):
        return True
    try:
        out = subprocess.run(["squeue", "-h", "-j", job_id, "-o", "%T"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return False if "Invalid job id" in out.stderr else None
    return out.stdout.strip() in ("RUNNING", "PENDING", "CONFIGURING", "SUSPENDED", "REQUEUED", "RESIZING")


def _claim_alive(run_dir: Path) -> bool:
    """True if the claim of ``run_dir`` belongs to a live worker."""
    age = _claim_age(run_dir)
    if age is None or age >= STALE_S:
        return False
    try:
        m = re.search(r"slurm_job=(\d+)", (run_dir / CLAIM).read_text())
    except FileNotFoundError:
        return False
    if m and _slurm_job_alive(m.group(1)) is False:
        return False  # its Slurm job ended (time limit, cancelled, crashed)
    return True


def is_claimed(run_dir: Path) -> bool:
    """True if a live worker (somewhere) is running this job."""
    return _claim_alive(run_dir)


def try_claim(run_dir: Path) -> bool:
    """Atomically claim a job. False if a live worker already owns it."""
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / CLAIM
    if path.exists() and not _claim_alive(run_dir):
        path.unlink(missing_ok=True)  # the owner died: take over
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    owner = f"host={socket.gethostname()} pid={os.getpid()} slurm_job={os.environ.get('SLURM_JOB_ID', '-')}\n"
    os.write(fd, owner.encode())
    os.close(fd)
    return True


class _Heartbeat:
    """Touches the claim file every HEARTBEAT_S seconds while a job runs; removes it at the end."""

    def __init__(self, run_dir: Path):
        self.path = run_dir / CLAIM
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._beat, daemon=True)

    def _beat(self) -> None:
        while not self.stop.wait(HEARTBEAT_S):
            try:
                self.path.touch()
            except OSError:
                pass

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()
        self.path.unlink(missing_ok=True)


# ------------------------------------------------------------------- execution

# Set in each worker process by _init_worker (a process pool cannot pickle closures).
_CFG: ExperimentConfig | None = None


def _init_worker(cfg_dict: dict, threads: int) -> None:
    global _CFG
    _CFG = ExperimentConfig.from_dict(cfg_dict)
    torch.set_num_threads(threads)


def _execute(job: Job, force: bool = False, verbose: bool = False) -> tuple[Job, dict | None, str | None, float]:
    """Run one job; never raises (errors are returned so the grid keeps going).

    Returns (job, summary, error, seconds). summary and error are both None if the
    job was skipped because another worker owns it.
    """
    t0 = time.time()
    run_dir = job.out_dir(_CFG)
    if (is_done(run_dir) and not force) or not try_claim(run_dir):
        return job, None, None, 0.0
    try:
        with _Heartbeat(run_dir):
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
    if summary is None:
        print(f"[{i:>4d}/{n}] {job.label}  skipped: done or running in another process", flush=True)
        return
    tg = summary["T_grok"]
    status = f"T_grok={tg:>6d}" if tg is not None else f"not grokked in {summary['budget']}"
    extra = ""
    if summary.get("forgetting_disagree") is not None:  # forgetting on the pairs where the new rule is wrong
        extra = f"  forgetting(disagree)={summary['forgetting_disagree']:+.2f}"
    speed = f"{summary['ms_per_step']:.2f} ms/step"
    print(f"[{i:>4d}/{n}] {job.label}  {status}  T_mem={summary['T_mem']}{extra}  ({dt:.0f}s, {speed})", flush=True)


def _wait_for_sources(cfg: ExperimentConfig, jobs: list[Job], poll_s: int = 30) -> None:
    """Block while another worker is still training a source run that ``jobs`` need.

    A source that is neither done nor claimed (e.g. it failed) is not waited for:
    the transfer jobs that need it will fail with a clear message.
    """
    needed = {(j.p, j.seed) for j in jobs if CONDITIONS[j.condition].needs_source}
    waited = False
    while True:
        running = [(p, s) for p, s in sorted(needed) if not is_done(source_dir(cfg, p, s)) and is_claimed(source_dir(cfg, p, s))]
        if not running:
            return
        if not waited:
            print(f"Waiting for source runs trained elsewhere: {running}", flush=True)
            waited = True
        time.sleep(poll_s)


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
        if phase == "transfer":
            _wait_for_sources(cfg, batch)
        if workers <= 1:
            _init_worker(cfg.to_dict(), threads)
            results = (_execute(j, force, verbose) for j in batch)
        else:
            ctx = mp.get_context("spawn")
            pool = ctx.Pool(workers, initializer=_init_worker, initargs=(cfg.to_dict(), threads))
            # imap_unordered: report each job as soon as it finishes. chunksize=1 so
            # that a worker takes a new job only when it is free (load balancing).
            results = pool.imap_unordered(_execute_star, [(j, force, False) for j in batch], chunksize=1)
        for job, s, err, dt in results:
            counter += 1
            _report(counter, len(todo), job, s, err, dt)
            if err:
                failures.append((job, err))
        if workers > 1:
            pool.close()
            pool.join()
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
