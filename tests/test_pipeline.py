"""End-to-end test of the pipeline on a toy grid (a few seconds).

Short budgets mean nothing has to grok. The test only checks that every condition
runs, writes its files, and that analysis and plotting can read them.
"""

import json

from grokking.analysis import Results
from grokking.conditions import CONDITIONS
from grokking.config import ExperimentConfig
from grokking.grid import expand_jobs, run_jobs
from grokking.plotting import make_all_figures


def tiny_config(tmp_path) -> ExperimentConfig:
    cfg = ExperimentConfig.from_dict(
        {
            "name": "toy",
            "out_root": str(tmp_path),
            "sources": [7],
            "targets": {"7": [5, 11]},
            "conditions": list(CONDITIONS),
            "seeds": [0],
            "model": {"n_numbers": 16, "d_model": 16, "n_heads": 2, "d_mlp": 32},
            "task": {"train_frac": 0.5},
            # lr 1e-2 so the source at least memorizes (needed for warm_memorized)
            "train_source": {"lr": 0.01, "max_steps": 300, "eval_every": 10, "repr_every": 50, "grok_patience": 20},
            "train_transfer": {"max_steps": 60, "eval_every": 10, "repr_every": 20, "grok_patience": 20},
        }
    )
    cfg.validate()
    return cfg


def test_full_pipeline(tmp_path):
    cfg = tiny_config(tmp_path)
    jobs = expand_jobs(cfg)
    assert len(jobs) == 1 + 2 * len(CONDITIONS)
    failures = run_jobs(cfg, jobs)
    assert failures == []

    # every run produced its files; sources also produced the two checkpoints
    for job in jobs:
        d = job.out_dir(cfg)
        for f in ("log.csv", "repr.csv", "summary.json"):
            assert (d / f).exists(), d / f
        if job.kind == "source":
            assert (d / "memorized.pt").exists() and (d / "final.pt").exists()
        else:
            s = json.loads((d / "summary.json").read_text())
            assert "forgetting" in s and "step0_val_acc_agree" in s

    # resuming skips finished jobs
    assert run_jobs(cfg, jobs) == []

    res = Results(cfg.exp_dir)
    assert len(res.summaries) == len(jobs)
    table = res.condition_table()
    assert set(CONDITIONS) <= set(table["condition"])
    ratios = res.transfer_ratios()
    assert len(ratios) == 2 * (len(CONDITIONS) - 1)
    assert "old_freq_r2" in res.repr_logs().columns
    assert "Transfer ratio" in res.report()

    figs = make_all_figures(cfg.exp_dir)
    assert any("transfer_ratios" in str(p) for p in figs)
    assert all(p.exists() for p in figs)


def test_same_init_for_paired_conditions(tmp_path):
    """The phase-B split is shared by all conditions of a seed (permuted or not)."""
    from grokking.experiments import target_task

    cfg = tiny_config(tmp_path)
    a = target_task(cfg, 11, seed=0, permute=False)
    b = target_task(cfg, 11, seed=0, permute=True)
    assert (a.train_idx == b.train_idx).all()
    assert not b.is_identity_map


def test_phase_b_uses_a_new_equals_token(tmp_path):
    """Old task uses "=p", new task "=q", even in the same-task control (q = p)."""
    from grokking.data import eq_token
    from grokking.experiments import source_task, target_task

    cfg = tiny_config(tmp_path)
    V = cfg.model.n_numbers
    old = source_task(cfg, 7, seed=0)
    new = target_task(cfg, 7, seed=0, permute=False)  # q = p
    assert (old.tokens[:, 2] == eq_token(0, V)).all()
    assert (new.tokens[:, 2] == eq_token(1, V)).all()


def test_claims_are_exclusive_and_expire(tmp_path):
    """Two workers never run the same job; the claim of a dead worker is taken over."""
    import os
    import time

    from grokking import grid

    run = tmp_path / "run"
    assert grid.try_claim(run)
    assert not grid.try_claim(run) and grid.is_claimed(run)
    old = time.time() - grid.STALE_S - 1
    os.utime(run / grid.CLAIM, (old, old))  # the owner stopped touching it
    assert not grid.is_claimed(run)
    assert grid.try_claim(run)
