"""Command-line interface. Run ``python run.py --help`` for the overview.

Commands
--------
grid       run (or resume) every job of an experiment config
source     run only phase-A jobs (optionally a subset of moduli / seeds)
transfer   run only phase-B jobs (optionally a subset of pairs / conditions / seeds)
status     show which runs are finished
analyze    print and save the result tables
plot       write all figures to <exp_dir>/figures
smoke      tiny end-to-end check: grid + analyze + plot on configs/smoke.json
conditions list the available experimental conditions

Every command that takes a config also accepts ``--set key=value`` overrides, e.g.
``--set train_transfer.max_steps=2000 --set seeds=[0,1]``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .conditions import CONDITIONS
from .config import ExperimentConfig

ROOT = Path(__file__).resolve().parent.parent
SMOKE_CONFIG = ROOT / "configs" / "smoke.json"


def _load(args) -> ExperimentConfig:
    return ExperimentConfig.load(args.config, args.set)


def _add_common(p: argparse.ArgumentParser, config_required: bool = True) -> None:
    if config_required:
        p.add_argument("config", help="experiment config (JSON), e.g. configs/main.json")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a config field")


def _add_run_opts(p: argparse.ArgumentParser) -> None:
    p.add_argument("--workers", type=int, default=1, help="parallel processes (default 1)")
    p.add_argument("--threads", type=int, default=1, help="torch threads per process (default 1)")
    p.add_argument("--force", action="store_true", help="rerun jobs even if already finished")
    p.add_argument("--dry-run", action="store_true", help="only list the jobs that would run")
    p.add_argument("-v", "--verbose", action="store_true", help="print training progress (single worker only)")


# ------------------------------------------------------------------- commands


def cmd_grid(args, kinds=("source", "transfer")) -> int:
    from .experiments import is_done
    from .grid import expand_jobs, run_jobs

    cfg = _load(args)
    jobs = [j for j in expand_jobs(cfg) if j.kind in kinds]
    # Optional filters (used by `source` / `transfer`).
    for attr in ("p", "q", "condition", "seed"):
        values = getattr(args, attr + "s", None)
        if values:
            jobs = [j for j in jobs if getattr(j, attr) in values]
    if args.dry_run:
        for j in jobs:
            print(("done  " if is_done(j.out_dir(cfg)) else "todo  ") + j.label)
        return 0
    if args.verbose:
        cfg.train_source.print_every = cfg.train_source.print_every or 1000
        cfg.train_transfer.print_every = cfg.train_transfer.print_every or 1000
    t0 = time.time()
    failures = run_jobs(cfg, jobs, workers=args.workers, threads=args.threads, force=args.force, verbose=args.verbose)
    print(f"Finished in {time.time() - t0:.0f}s. Results in {cfg.exp_dir}")
    return 1 if failures else 0


def cmd_status(args) -> int:
    from .grid import status_table

    print(status_table(_load(args)))
    return 0


def _exp_dir(args) -> Path:
    if getattr(args, "exp_dir", None):
        return Path(args.exp_dir)
    return _load(args).exp_dir


def cmd_analyze(args) -> int:
    from .analysis import Results

    print(Results(_exp_dir(args)).report(save=True))
    return 0


def cmd_plot(args) -> int:
    from .plotting import make_all_figures

    paths = make_all_figures(_exp_dir(args))
    print(f"{len(paths)} figures written:")
    for p in paths:
        print("  ", p)
    return 0


def cmd_smoke(args) -> int:
    """Run the tiny smoke grid end to end and check the outputs make sense."""
    args.config = str(SMOKE_CONFIG)
    args.set = list(args.set)
    for k in ("dry_run", "force"):
        setattr(args, k, getattr(args, k, False))
    code = cmd_grid(args)
    cfg = _load(args)
    print()
    cmd_analyze(argparse.Namespace(exp_dir=str(cfg.exp_dir)))
    print()
    cmd_plot(argparse.Namespace(exp_dir=str(cfg.exp_dir)))
    return code


def cmd_conditions(_args) -> int:
    for c in CONDITIONS.values():
        print(f"{c.plan_id:<3s} {c.name:<20s} {c.description}")
    return 0


# ----------------------------------------------------------------------- main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run.py", description="Grokking across group sizes (Z_p -> Z_q).",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("grid", help="run or resume the full grid of a config")
    _add_common(p)
    _add_run_opts(p)
    p.set_defaults(func=cmd_grid)

    p = sub.add_parser("source", help="run phase-A jobs only")
    _add_common(p)
    _add_run_opts(p)
    p.add_argument("--p", dest="ps", type=int, nargs="+", help="only these source moduli")
    p.add_argument("--seeds", dest="seeds", type=int, nargs="+", help="only these seeds")
    p.set_defaults(func=lambda a: cmd_grid(a, kinds=("source",)))

    p = sub.add_parser("transfer", help="run phase-B jobs only (sources must exist)")
    _add_common(p)
    _add_run_opts(p)
    p.add_argument("--p", dest="ps", type=int, nargs="+", help="only these source moduli")
    p.add_argument("--q", dest="qs", type=int, nargs="+", help="only these target moduli")
    p.add_argument("--conditions", dest="conditions", nargs="+", choices=sorted(CONDITIONS))
    p.add_argument("--seeds", dest="seeds", type=int, nargs="+", help="only these seeds")
    p.set_defaults(func=lambda a: cmd_grid(a, kinds=("transfer",)))

    p = sub.add_parser("status", help="show finished / missing runs")
    _add_common(p)
    p.set_defaults(func=cmd_status)

    for name, func, help_ in (("analyze", cmd_analyze, "print and save result tables"),
                              ("plot", cmd_plot, "write all figures")):
        p = sub.add_parser(name, help=help_)
        p.add_argument("config", nargs="?", help="experiment config (or use --exp-dir)")
        p.add_argument("--exp-dir", help="results directory, e.g. results/main")
        p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
        p.set_defaults(func=func)

    p = sub.add_parser("smoke", help="tiny end-to-end test (a few minutes on a laptop)")
    _add_common(p, config_required=False)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_smoke, dry_run=False)

    p = sub.add_parser("conditions", help="list experimental conditions")
    p.set_defaults(func=cmd_conditions)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command in ("analyze", "plot") and not (args.config or args.exp_dir):
        print("error: give a config or --exp-dir", file=sys.stderr)
        return 2
    return args.func(args)
