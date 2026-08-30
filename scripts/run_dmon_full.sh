#!/bin/bash
# Launch the full DMoN sweep (compared against the existing
# outputs/diffpool_hybrid2_rep0 70.3% DiffPool record -- this only runs
# DMoN) in the background so it survives an SSH disconnect. Run
# scripts/setup_remote.sh first if you haven't already set up the env/data.
#
# Usage:
#   ./scripts/run_dmon_full.sh [repo_root]
#
# Config: num_samples=6, n_cycles=7 (63-epoch tuning trials + 127-epoch
# final retrain), hybrid n_hybrid=2 -- matches the architecture behind the
# existing 70.3% DiffPool baseline. batch_size=24 was validated on a 12GB
# RTX 3060: if this machine's GPU has meaningfully less VRAM, lower
# --batch-size below before trusting this; if it has much more, this is a
# conservative (not optimal) choice.
#
# Launches via nohup + disown, logging to dmon_full_sweep.log and writing
# the process's PID to dmon_full_sweep.pid, both in the repo root.
#
# To check on it later (after reconnecting):
#   tail -f dmon_full_sweep.log
#   kill -0 $(cat dmon_full_sweep.pid) && echo "still running" || echo "done or dead"

set -euo pipefail

REPO_ROOT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
echo "Repo root: $REPO_ROOT"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found on PATH." >&2
  exit 1
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pooling_genomic

python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available -- run scripts/setup_remote.sh first'" || exit 1

DATASET="$REPO_ROOT/data/string_data/data/tcga_cohorts_and_tumor_classification"
LEVELS="$REPO_ROOT/data/string_data/data/networks/levels"
NETWORK="$REPO_ROOT/data/string_data/data/networks/stringdb_top100pc.csv"
OUTDIR="$REPO_ROOT/outputs/dmon_full"

for f in "$DATASET" "$LEVELS" "$NETWORK"; do
  if [ ! -e "$f" ]; then
    echo "ERROR: expected path missing: $f (did you run scripts/setup_remote.sh?)" >&2
    exit 1
  fi
done

echo "Launching full DMoN sweep in the background (nohup + disown) ..."
echo "batch_size=24 was validated on a 12GB RTX 3060 -- see header comment above."

nohup python scripts/experiments/diffpool_experiment.py \
  "$DATASET" "$LEVELS" \
  --path-network "$NETWORK" \
  --pooling-type dmon \
  --n-hybrid 2 --n-hybrid-start 2 \
  --tune --num-samples 6 --n-cycles 7 \
  --batch-size 24 --device cuda --gpu-per-trial 1 --cpu-per-trial 4 \
  --n-holdouts 1 \
  --path-output "$OUTDIR" \
  > "$REPO_ROOT/dmon_full_sweep.log" 2>&1 &

PID=$!
disown
echo "$PID" > "$REPO_ROOT/dmon_full_sweep.pid"
echo "Launched. PID=$PID"
echo "Log:    $REPO_ROOT/dmon_full_sweep.log"
echo "Output: $OUTDIR/dmon_hybrid2_rep0/final_model_results/ (once finished)"
echo "Check progress any time with: tail -f $REPO_ROOT/dmon_full_sweep.log"
