#!/usr/bin/env bash
# Run the experiments on the ETH D-INFK student cluster (Slurm) from your laptop.
#
#   cluster/cluster.sh [--host HOST] [--dir DIR] COMMAND [ARGS]
#
#   --host HOST   ssh host alias of the cluster (default: eth-student-cluster, or $GROK_HOST)
#   --dir DIR     project directory on the cluster
#                 (default: /work/courses/dl/team-20/$USER/grokking, or $GROK_DIR;
#                 $USER is your cluster login)
#
# Commands
#   setup                 copy the code and create the Python env (.venv) on the cluster
#   sync                  copy the code (never touches results/, logs/ or .venv/ there)
#   test                  run the unit tests on a GPU node (personal "dl" budget)
#   submit CONFIG [opts] [-- run.py grid args]
#                         copy the code, then start the grid as Slurm jobs (one per GPU
#                         type; they share the work and never run the same job twice)
#       --gpus LIST       GPU types, one job each      (default: 5060ti,2080ti)
#       --workers N       runs sharing each GPU         (default: 2, enough to fill it)
#       --time T          time limit per job, HH:MM:SS  (default: 12:00:00, the maximum)
#       --account NAME    Slurm account                 (default: dl_jobs, the team budget)
#   status [CONFIG]       your Slurm jobs, the GPU hours left, and the finished runs
#   logs [JOBID]          follow the output of a job (default: your latest job)
#   cancel                cancel all your grid jobs (finished runs are kept; resubmit to resume)
#   fetch NAME            copy results/NAME from the cluster to ./results/NAME
#
# Examples
#   cluster/cluster.sh setup                                  # once
#   cluster/cluster.sh test
#   cluster/cluster.sh submit configs/main.json
#   cluster/cluster.sh submit configs/main.json --gpus 5060ti -- --set seeds=[0]
#   cluster/cluster.sh status configs/main.json
#   cluster/cluster.sh fetch main && python run.py analyze --exp-dir results/main
#   cluster/cluster.sh --host my-eth-alias status             # a teammate's ssh alias

set -euo pipefail

HOST=${GROK_HOST:-eth-student-cluster}
DIR=${GROK_DIR:-'/work/courses/dl/team-20/$USER/grokking'}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

usage() { sed -n '2,/^$/p' "$0" | sed -E 's/^# ?//'; exit "${1:-0}"; }
die() { echo "error: $*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
    case $1 in
        --host) HOST=$2; shift 2 ;;
        --dir) DIR=$2; shift 2 ;;
        -h | --help) usage ;;
        *) break ;;
    esac
done
[[ $# -ge 1 ]] || usage 1
CMD=$1
shift

# Resolve $USER etc. on the cluster side, once.
RDIR=$(ssh "$HOST" "echo $DIR") || die "cannot reach '$HOST' with ssh"

# Run a shell command on the login node, inside the project directory.
remote() { ssh "$HOST" "cd '$RDIR' && $1"; }

sync_code() {
    ssh "$HOST" "mkdir -p '$RDIR'"
    rsync -az --delete \
        --exclude .git --exclude .venv --exclude results --exclude logs \
        --exclude __pycache__ --exclude .pytest_cache --exclude .DS_Store \
        --exclude papers --exclude '*.pdf' --exclude 'scratch_*' \
        "$ROOT/" "$HOST:$RDIR/"
    echo "code synced to $HOST:$RDIR"
}

case $CMD in
    setup)
        sync_code
        remote "bash cluster/setup_env.sh"
        ;;

    sync)
        sync_code
        ;;

    test)
        sync_code
        remote "srun --account=dl --time=15 --gpus=5060ti:1 .venv/bin/python -m pytest -q"
        ;;

    submit)
        [[ $# -ge 1 ]] || die "submit needs a config, e.g. configs/main.json"
        CONFIG=$1
        shift
        GPUS=5060ti,2080ti WORKERS=2 TIME=12:00:00 ACCOUNT=dl_jobs
        EXTRA=()
        while [[ $# -gt 0 ]]; do
            case $1 in
                --gpus) GPUS=$2; shift 2 ;;
                --workers) WORKERS=$2; shift 2 ;;
                --time) TIME=$2; shift 2 ;;
                --account) ACCOUNT=$2; shift 2 ;;
                --) shift; EXTRA=("$@"); break ;;
                *) die "unknown submit option '$1'" ;;
            esac
        done
        [[ -f "$ROOT/$CONFIG" ]] || die "no such config: $CONFIG"
        sync_code
        extra_q=$(printf '%q ' ${EXTRA[@]+"${EXTRA[@]}"})
        name="grok-$(basename "$CONFIG" .json)"
        for gpu in ${GPUS//,/ }; do
            id=$(remote "mkdir -p logs && sbatch --parsable --account=$ACCOUNT --time=$TIME \
                --gpus=$gpu:1 --job-name=$name cluster/job.sbatch $CONFIG $WORKERS $extra_q")
            echo "submitted job $id ($name on a $gpu, $WORKERS workers, limit $TIME)"
        done
        echo "follow with: cluster/cluster.sh logs   |   progress: cluster/cluster.sh status $CONFIG"
        ;;

    status)
        remote "squeue -u \$USER -o '%.10i %.22j %.9T %.10M %.11l %.18b %R'; echo; courses | grep -E 'Course tag|dl_jobs'"
        if [[ $# -ge 1 ]]; then
            echo
            remote ".venv/bin/python run.py status $1"
        fi
        ;;

    logs)
        if [[ $# -ge 1 ]]; then
            remote "tail -n 40 -f logs/*-$1.out"
        else
            remote "f=\$(ls -t logs/*.out 2>/dev/null | head -1); [ -n \"\$f\" ] || { echo 'no logs yet'; exit 1; }; echo \"==> \$f\"; tail -n 40 -f \"\$f\""
        fi
        ;;

    cancel)
        remote "squeue -u \$USER -h -o '%i %j' | awk '\$2 ~ /^grok/ {print \$1}' | xargs -r scancel; squeue -u \$USER"
        ;;

    fetch)
        [[ $# -ge 1 ]] || die "fetch needs an experiment name, e.g. main"
        mkdir -p "$ROOT/results/$1"
        rsync -az --exclude .claim --exclude '*.tmp' "$HOST:$RDIR/results/$1/" "$ROOT/results/$1/"
        echo "results copied to results/$1"
        ;;

    *)
        die "unknown command '$CMD' (see cluster/cluster.sh --help)"
        ;;
esac
