#!/bin/bash
# Re-run of the shrunk-scope Full DMoN experiment (outputs/full_dmon_shrunk, 71.16%)
# on the CURRENT code, to validate the fixes made since it ran -- and nothing else.
#
# Identical to scripts/run_full_dmon_shrunk.sh in every experiment flag
# (3 levels, --sparsify-density 0.04, batch 8, 6 samples, --n-cycles 5 -> 31-epoch
# retrain, 1 holdout). What differs is the code underneath:
#   - fix #7: Ray Tune checkpoints now hold real weights and the final retrain
#     warm-starts from the best trial (the old run re-initialised from scratch)
#   - fix #5: assignment-head dropout 0.5 (default), removal of collapse_regularization
#   - fix #4: dead-parameter cleanup / DiffPool link-pred (DiffPool only; no effect on DMoN)
#   - fix #8: deterministic val/test order (does not change accuracy)
# The split is unchanged: n_holdouts=1 draws the same first seed from
# default_rng(123), i.e. the same test patients as the old run.
#
# Expected: ~2.2 days (the old run took 51h57m at ~50 min/epoch, batch 8).
# No pre-flight probe: batch 8 / 3 levels was re-measured on 2026-09-25 at
# 4.46 s/step, 11.6 GB peak.
#
# Usage:  ./scripts/run_full_dmon_rerun.sh [repo_root]
# Check:  tail -f full_dmon_rerun.log ; cat outputs/full_dmon_rerun/results.md
set -euo pipefail
REPO_ROOT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
if ! command -v conda >/dev/null 2>&1; then echo "ERROR: conda not found on PATH." >&2; exit 1; fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pooling_genomic
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" || exit 1

DATASET="$REPO_ROOT/data/string_data/data/tcga_cohorts_and_tumor_classification"
LEVELS="$REPO_ROOT/data/string_data/data/networks/levels"
NETWORK="$REPO_ROOT/data/string_data/data/networks/stringdb_top100pc.csv"
OUTDIR="$REPO_ROOT/outputs/full_dmon_rerun"
for f in "$DATASET" "$LEVELS" "$NETWORK"; do
  if [ ! -e "$f" ]; then echo "ERROR: expected path missing: $f" >&2; exit 1; fi
done

nohup bash -c '
  echo "FULL_DMON_TIMER start $(date +%s)"
  python scripts/experiments/diffpool_experiment.py \
    "'"$DATASET"'" "'"$LEVELS"'" \
    --path-network "'"$NETWORK"'" \
    --pooling-type dmon \
    --full-mode --n-hybrid 3 --n-hybrid-start 3 \
    --sparsify-density 0.04 \
    --tune --num-samples 6 --n-cycles 5 \
    --batch-size 8 --device cuda --gpu-per-trial 1 --cpu-per-trial 8 \
    --n-holdouts 1 --use-train-set-weights \
    --path-output "'"$OUTDIR"'"
  echo "FULL_DMON_TIMER end $(date +%s)"
  echo "=== Writing results report: $(date) ==="
  python scripts/analysis/analyze_full_dmon_results.py \
    --path-output "'"$OUTDIR"'" --n-hybrid 3 --sparsify-density 0.04 \
    --num-samples 6 --n-cycles 5 --batch-size 8 --n-holdouts 1 \
    --log "'"$REPO_ROOT"'/full_dmon_rerun.log" \
    --out "'"$OUTDIR"'/results.md"
' > "$REPO_ROOT/full_dmon_rerun.log" 2>&1 &
PID=$!
disown
echo "$PID" > "$REPO_ROOT/full_dmon_rerun.pid"
echo "Launched. PID=$PID  Log: $REPO_ROOT/full_dmon_rerun.log  Output: $OUTDIR"
