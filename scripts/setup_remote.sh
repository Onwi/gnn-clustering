#!/bin/bash
# One-time environment setup for a fresh machine (e.g. before an overnight
# training run). Run this once after `git clone`/`git pull`; run
# scripts/run_dmon_full.sh separately afterwards to actually launch training.
#
# Prerequisites on the remote machine before running this:
#   1. This repo already cloned/pulled (this script assumes it's being run
#      from inside the repo, or you pass the repo path as $1).
#   2. Copy data.zip into the repo root (scp/rsync from wherever it lives --
#      it is NOT tracked in git, .gitignore excludes /data, so `git pull`
#      alone will not bring the dataset over).
#   3. conda (or miniconda) already installed. NVIDIA driver + GPU present
#      (checked below).
#
# Usage:
#   ./scripts/setup_remote.sh [repo_root]
#
# What it does:
#   - Extracts data.zip into data/ if data/ doesn't already exist.
#   - Creates the `pooling_genomic` conda env (python 3.10) if it doesn't
#     already exist, matching the exact package versions verified working
#     in this session (torch 2.1.0+cu121 / torch_geometric 2.7.0), all via
#     pip wheels -- NOT the older conda-channel torch 1.12.1 recipe in
#     README.md/AGENTS.md, which is unverified against current hardware.
#   - Runs a CUDA sanity check.
#
# Re-running this script is safe: it skips extraction/env-creation/installs
# that already look done.

set -euo pipefail

REPO_ROOT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
echo "Repo root: $REPO_ROOT"

if [ ! -d data ]; then
  if [ ! -f data.zip ]; then
    echo "ERROR: no data/ directory and no data.zip found in $REPO_ROOT." >&2
    echo "Copy data.zip here first (it is not tracked in git)." >&2
    exit 1
  fi
  echo "Extracting data.zip ..."
  unzip -q -o data.zip
else
  echo "data/ already present, skipping extraction."
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found on PATH. Install miniconda/anaconda first." >&2
  exit 1
fi
source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | grep -q "^pooling_genomic "; then
  echo "Creating conda env 'pooling_genomic' (python 3.10) ..."
  conda create -n pooling_genomic python=3.10 -y
fi
conda activate pooling_genomic

echo "Checking NVIDIA GPU visibility ..."
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || {
  echo "ERROR: nvidia-smi failed -- no GPU visible to this shell." >&2
  exit 1
}

if ! python -c "import torch" 2>/dev/null; then
  echo "Installing torch 2.1.0+cu121 / torchvision / torchaudio ..."
  pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121
  echo "Installing torch_geometric + torch-scatter/sparse/cluster ..."
  pip install torch_geometric==2.7.0
  pip install torch-scatter==2.1.2+pt21cu121 torch-sparse==0.6.18+pt21cu121 torch-cluster==1.6.3+pt21cu121 \
    -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
  pip install -r requirements.txt
  pip install -e .
else
  echo "torch already importable in this env, skipping package install (assuming it's already set up)."
fi

echo "CUDA sanity check ..."
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available to torch'; print('CUDA OK:', torch.cuda.get_device_name(0))"

echo
echo "Setup complete. Next: ./scripts/run_dmon_full.sh"
