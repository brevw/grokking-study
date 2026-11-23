#!/bin/bash
# Slurm batch script: run (or resume) an experiment grid on one GPU.
#
#   sbatch --account=dl_jobs --time=12:00:00 --gpus=5060ti:1 cluster/job.sbatch CONFIG WORKERS [run.py grid args]
#
# Normally submitted by cluster/cluster.sh. Several of these jobs can work on the same
# config at once: they split the runs between them (see grokking/grid.py, "claims").
# The CPU count and memory come with the GPU type (5060 Ti: 3 CPUs, 2080 Ti: 4 CPUs).
#SBATCH --job-name=grok
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
CONFIG=$1
WORKERS=$2
shift 2

cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1
# torch.compile caches its generated kernels on the node's local disk ($TMPDIR is
# removed when the job ends).
export TORCHINDUCTOR_CACHE_DIR="${TMPDIR:-/tmp}/inductor"
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/triton"

echo "job $SLURM_JOB_ID on $(hostname): $(nvidia-smi --query-gpu=name --format=csv,noheader), $(nproc) CPUs"
echo "config $CONFIG, $WORKERS workers, extra args: ${*:-none}"
echo "started $(date)"
.venv/bin/python run.py grid "$CONFIG" --workers "$WORKERS" "$@"
echo "finished $(date)"
