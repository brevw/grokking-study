# Grokking across group sizes

Code for the project described in [`PROJECT_PLAN.md`](PROJECT_PLAN.md). A small transformer first groks
modular addition in **Z_p** (phase A) and is then trained further on **Z_q** (phase B). We measure:

* **speed:** grokking time compared to training from scratch;
* **forgetting:** how much accuracy on the old group Z_p is lost;
* **representation:** Fourier structure, weight norm, effective learning rate, CKA, dead neurons.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q            # 15 tests, ~5 s
```

(The base conda env on this laptop has a broken NumPy/pandas combination, so we use `.venv`.)

## Quick start

```bash
.venv/bin/python run.py smoke --workers 4          # tiny full pipeline, ~2 min on an M1 Pro
.venv/bin/python run.py conditions                 # list conditions C0..C6
.venv/bin/python run.py grid configs/main.json --dry-run      # list the 145 jobs of the main grid
.venv/bin/python run.py grid configs/main.json --workers 6    # run / resume the main grid
.venv/bin/python run.py status  configs/main.json  # what is finished
.venv/bin/python run.py analyze configs/main.json  # tables (printed + CSV)
.venv/bin/python run.py plot    configs/main.json  # figures -> results/main/figures/
```

Useful options:

| Option | Effect |
|---|---|
| `--set key=value` | override any config field, e.g. `--set train_transfer.max_steps=20000 --set seeds=[0,1]` |
| `--workers N --threads T` | N parallel processes with T torch threads each (many single-thread workers is fastest for tiny models) |
| `--force` | rerun finished jobs (otherwise they are skipped, so every command can be resumed) |
| `-v` | print training progress (single worker) |
| `source --p 97 --seeds 0 1` | run only some phase-A jobs |
| `transfer --q 113 91 --conditions scratch warm_grokked` | run only some phase-B jobs |

**Variants of an experiment.** Change the `name`, so the results go to a new folder:
`run.py grid configs/main.json --set name=main_a05 --set task.train_frac=0.5`

## Configs

| File | What |
|---|---|
| `configs/main.json` | main grid: p = 97 → q ∈ {59, 89, 113, 91, 96, 121, 97}, conditions C0–C3, 5 seeds |
| `configs/robustness.json` | second source prime: p = 59 → q ∈ {43, 71, 77, 81}, C0–C1, 5 seeds |
| `configs/smoke.json` | toy version (p = 31, lr = 1e-2, d_model = 64) that groks in ~1k steps. **Only for testing the code.** Its numbers mean nothing. |

**Compute.** Measured on the M1 Pro, the largest task (q = 113) costs ≈ 15–20 ms/step, so the
main grid is an overnight run on a laptop. On the cluster GPUs a step costs ≈ 1 ms (see below).

## Running on the student cluster

`cluster/cluster.sh` does everything from your laptop over ssh. The default host is the ssh alias
`eth-student-cluster`; teammates pass their own with `--host` (or set `GROK_HOST`). The code goes to
`/work/courses/dl/team-20/<login>/grokking` (change with `--dir` / `GROK_DIR`).

```bash
cluster/cluster.sh setup                        # once: copy code, build .venv with CUDA PyTorch (~3 min)
cluster/cluster.sh test                         # unit tests on a GPU node (~1 min, personal "dl" budget)
cluster/cluster.sh submit configs/main.json     # 2 Slurm jobs (5060 Ti + 2080 Ti) sharing the grid
cluster/cluster.sh status configs/main.json     # jobs, GPU hours left, finished runs
cluster/cluster.sh logs                         # follow the latest job
cluster/cluster.sh fetch main                   # copy results/main back, then analyze/plot locally
cluster/cluster.sh --host my-alias status       # a teammate's ssh alias
```

* Jobs use the team account `dl_jobs` (480 GPU hours per team, 12 h max per job, 3 jobs per user).
  `submit` starts one job per GPU type (`--gpus 5060ti,2080ti`). Every job runs `run.py grid`.
  They split the runs between them through claim files (`grokking/grid.py`), so no run is done
  twice, and a resubmitted job resumes where the last one stopped.
* `--workers 2` (default): two runs share each GPU, which is enough to keep it busy.
* GTX 1080 Ti nodes are not used: the CUDA 12.8 PyTorch build has no kernels for them.
  GB10 nodes are not used either: they have ARM CPUs.
* Throughput (q = 113, all logging included): ≈ 900 steps/s per 5060 Ti and ≈ 650 per 2080 Ti.

**GPU speed-ups (and why they are exact).** For this tiny model a step is dominated by overhead,
not by arithmetic. Three changes made it ≈ 6× faster, and none changes the math:
(1) the one-query attention is written out by hand (`model.py`); PyTorch's fused attention kernel is
~4× slower when there are only 3 keys; (2) `torch.compile` fuses the forward pass, the mask and the
loss; (3) after the 10 warm-up steps the whole step (forward, backward, AdamW) is recorded once as a
CUDA graph and replayed (`train.py`). Evaluation is reduced on the GPU and copies a few numbers to the CPU.
Graph and normal training differ only by ordinary GPU floating-point nondeterminism, which is as
large as the difference between two normal runs.

## Code map

```
run.py                  entry point (python run.py --help)
grokking/
  config.py             dataclass configs + JSON loading + --set overrides
  data.py               ModularTask: n² pairs, split, symbol map σ (identity / permuted), "=p"/"=q" tokens
  model.py              1-layer transformer, no LayerNorm (Nanda et al. 2023)
  metrics.py            accuracy/loss, logit masking, SustainedCrossing (T_mem, T_grok, T_forget)
  representation.py     Fourier power, key frequencies, sparsity, old-frequency R², CKA, dead neurons
  train.py              Trainer: full-batch AdamW, logging, stopping, checkpoints
  conditions.py         C0..C6 (how phase B starts)
  experiments.py        run_source (phase A), run_transfer (phase B), seeding, directory layout
  grid.py               expand config → jobs, (parallel) execution, claims, status table
  analysis.py           Results: summaries, per-condition table, paired transfer ratios, survival
  plotting.py           all figures of the plan
  cli.py                command-line interface
cluster/
  cluster.sh            run everything on the Slurm cluster from the laptop (setup/test/submit/status/fetch)
  job.sbatch            the Slurm job: `run.py grid` on one GPU
  setup_env.sh          builds .venv with CUDA PyTorch on the cluster
tests/                  unit tests + an end-to-end pipeline test
```

## How the methodology maps to the code

| Plan (PROJECT_PLAN.md) | Code |
|---|---|
| Symbols ≠ elements, one "=" per task (`=p` = token 128, `=q` = token 129) | `data.ModularTask(eq_slot=…)`, `data.eq_token`, `config.N_EQ_TOKENS` |
| Logits masked to the current group | `metrics.mask_logits` |
| Same split for all conditions of a seed | `experiments.target_task` uses `stable_seed("split", "B", q, seed)` |
| Fresh AdamW at the start of phase B | `train.Trainer.__init__` |
| C1 = grokked + cleanup, C2 = at T_mem | `Trainer(save_memorized=True, save_final=True)`; `train_source.stop_after_grok` |
| T_mem, T_grok (sustained for `grok_patience`), T_95, T_forget | `metrics.SustainedCrossing`, `Trainer._update_trackers` |
| Agree / disagree / new-token split of val acc | `ModularTask.overlap_masks` → columns `val_acc_agree/disagree/newtok` |
| Effective learning rate ‖Δθ‖/‖θ‖ | `Trainer.fit` (column `elr` in `log.csv`) |
| Fourier, old-frequency R², frequency matching, subgroup fraction, CKA, dead neurons | `representation.snapshot` (→ `repr.csv`) |
| Censoring, paired ratio R, bootstrap CI, sign test, survival | `analysis.Results` |

## Output files

Each run directory (`results/<name>/source/p97/s0/`, `results/<name>/transfer/p97_q113/warm_grokked/s0/`)
contains:

* `log.csv`: every `eval_every` steps: train/val loss and accuracy, `val_acc_agree|disagree|newtok`,
  `old_train_acc`, `old_val_acc`, `param_norm`, `elr`, `lr`
* `repr.csv`: every `repr_every` steps: `key_freqs`, `fourier_sparsity`, `subgroup_frac`,
  `old_freq_r2`, `freq_match_dist`, `emb_cka_to_start`, `act_cka_to_start`, `dead_frac`, `rel_dist_from_start`
* `summary.json`: `T_mem`, `T_grok`, `T_95`, `grokked`/`censored`, step-0 accuracies, `forgetting`, `T_forget`, timings
* `memorized.pt`, `final.pt`: source runs only

`run.py analyze` writes `runs.csv` (all summaries), `table_conditions.csv` and `table_ratios.csv`.
In a notebook:

```python
from grokking.analysis import Results
res = Results("results/main")
res.condition_table(); res.transfer_ratios(); res.logs(p=97, q=113); res.repr_logs(condition="warm_grokked")
```

## Things to know when reading the results

* **"Forgetting" includes the agreement set.** Under the identity map, Z_p and Z_q give the same answer
  whenever a + b < min(p, q). A scratch network learning Z_q therefore gets those Z_p pairs right "for free".
  That is why scratch shows *negative* forgetting. Compare conditions to each other, and look at the
  agree/disagree split.
* **Representation metrics have a chance level.** Even random embeddings get a nonzero `old_freq_r2`
  (a least-squares fit with 2|K|+1 parameters). Always compare against the scratch condition.
* **`old_freq_r2` uses all tokens 0..p-1.** The plan said tokens 0..min(p,q)-1. Using all p tokens keeps the Fourier
  basis of Z_p orthogonal, and it also measures whether the unused tokens keep their structure when q < p.
* **Median T_grok includes censored runs at the budget**, so it is a lower bound when `frac_grokked < 1`.
  `n_censored` in the ratio table says how many there were.
* **Shrink-and-perturb parameters** (`shrink=0.5`, `perturb=0.1`) are untuned defaults.
