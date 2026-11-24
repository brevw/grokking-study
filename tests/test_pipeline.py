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
            "sources": [23],
            "targets": {"23": [19, 29]},
            "conditions": list(CONDITIONS),
            "seeds": [0],
            "model": {"n_numbers": 40, "d_model": 64, "n_heads": 4, "d_mlp": 256},
            "task": {"train_frac": 0.6},
            # Smallest setup found that groks quickly (~1.4k steps): the progress
            # conditions need a grokked source. Chance level on Z_23 is ~4%, but the toy
            # memorization plateau is short, hence the loose mem_val_threshold.
            "train_source": {"lr": 0.01, "max_steps": 2500, "eval_every": 10, "repr_every": 50,
                             "checkpoint_every": 10, "grok_patience": 50, "stop_after_grok": 50},
            "train_transfer": {"max_steps": 60, "eval_every": 10, "repr_every": 20, "grok_patience": 20},
            "mem_val_threshold": 0.5,
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
            for name in ("final", "memorized", "progress_10", "progress_100"):
                assert (d / f"{name}.pt").exists(), name
            assert not (d / "ckpt").exists()  # periodic checkpoints deleted after selection
        else:
            s = json.loads((d / "summary.json").read_text())
            assert "forgetting_disagree" in s and "step0_val_acc_agree" in s

    # resuming skips finished jobs
    assert run_jobs(cfg, jobs) == []

    res = Results(cfg.exp_dir)
    assert len(res.summaries) == len(jobs)
    table = res.condition_table()
    assert set(CONDITIONS) <= set(table["condition"])
    ratios = res.transfer_ratios()
    assert len(ratios) == 2 * (len(CONDITIONS) - 1)
    assert "old_freq_r2" in res.repr_logs().columns
    assert "old_val_acc_disagree" in res.logs().columns
    assert "Transfer ratio" in res.report()
    assert len(res.progress_table()) == 2 * 6  # 5 progress points + final, per target

    figs = make_all_figures(cfg.exp_dir)
    assert any("transfer_ratios" in str(p) for p in figs)
    assert any("progress_p23" in str(p) for p in figs)
    assert all(p.exists() for p in figs)


def test_same_init_for_paired_conditions(tmp_path):
    """The phase-B split is shared by all conditions of a seed (permuted or not)."""
    from grokking.experiments import target_task

    cfg = tiny_config(tmp_path)
    a = target_task(cfg, 29, seed=0, permute=False)
    b = target_task(cfg, 29, seed=0, permute=True)
    assert (a.train_idx == b.train_idx).all()
    assert not b.is_identity_map


def test_phase_b_uses_a_new_equals_token(tmp_path):
    """Old task uses "=p", new task "=q", even in the same-task control (q = p)."""
    from grokking.data import eq_token
    from grokking.experiments import source_task, target_task

    cfg = tiny_config(tmp_path)
    V = cfg.model.n_numbers
    old = source_task(cfg, 23, seed=0)
    new = target_task(cfg, 23, seed=0, permute=False)  # q = p
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


def test_phase_b_starting_weights(tmp_path):
    """Fresh rows for tokens unused in phase A, and the norm-matched control."""
    import torch

    from grokking.data import eq_token
    from grokking.experiments import _initial_weights, new_model, run_source
    from grokking.utils import stable_seed

    cfg = tiny_config(tmp_path)
    p, q, V = 23, 29, cfg.model.n_numbers
    run_source(cfg, p, seed=0)
    scratch = new_model(cfg, stable_seed("init", "B", q, 0)).state_dict()
    warm, _ = _initial_weights(cfg, CONDITIONS["warm_grokked"], p, q, seed=0)
    E, U = warm.embed.weight, warm.unembed.weight
    # tokens never used in phase A get their scratch-init rows; the others keep phase A
    assert torch.equal(E[p:V], scratch["embed.weight"][p:V])
    assert torch.equal(E[eq_token(1, V)], scratch["embed.weight"][eq_token(1, V)])
    assert torch.equal(U[p:V], scratch["unembed.weight"][p:V])
    assert not torch.equal(E[:p], scratch["embed.weight"][:p])
    # norm-matched control: random directions, per-matrix norms of the warm start
    nm, info = _initial_weights(cfg, CONDITIONS["scratch_norm_matched"], p, q, seed=0)
    for (name, a), b in zip(nm.named_parameters(), warm.parameters()):
        torch.testing.assert_close(a.norm(), b.norm())
        cos = torch.nn.functional.cosine_similarity(a.flatten(), scratch[name].flatten(), dim=0)
        torch.testing.assert_close(cos, torch.tensor(1.0))
    assert info["norm_matched_to"] == "final"


def test_claim_of_a_finished_slurm_job_is_taken_over(tmp_path, monkeypatch):
    """When a job of a chain hits its time limit, the next job redoes its runs at once."""
    from grokking import grid

    run = tmp_path / "run"
    run.mkdir()
    (run / grid.CLAIM).write_text("host=x pid=1 slurm_job=123\n")
    monkeypatch.setattr(grid, "_slurm_job_alive", lambda job_id: True)
    assert grid.is_claimed(run) and not grid.try_claim(run)
    monkeypatch.setattr(grid, "_slurm_job_alive", lambda job_id: False)
    assert not grid.is_claimed(run) and grid.try_claim(run)
