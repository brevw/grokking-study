#!/bin/bash
# Create the Python environment (.venv) on the cluster. Run on a login node, from the
# project directory (cluster/cluster.sh setup does this). Takes a few minutes.
#
# PyTorch build: CUDA 12.8 wheels. They support the RTX 2080 Ti (sm_75) and the
# RTX 5060 Ti (sm_120). They do not support the GTX 1080 Ti (sm_61), so we do not
# use those nodes. No CUDA module is needed: the wheels bring their own CUDA runtime.
set -euo pipefail
cd "$(dirname "$0")/.."

TORCH_VERSION=${TORCH_VERSION:-2.11.0}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}

[ -x .venv/bin/python ] || python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q --no-cache-dir "torch==$TORCH_VERSION" --index-url "$TORCH_INDEX"
.venv/bin/pip install -q -r requirements.txt
.venv/bin/python -c "import torch; print('torch', torch.__version__, '| CUDA', torch.version.cuda)"
