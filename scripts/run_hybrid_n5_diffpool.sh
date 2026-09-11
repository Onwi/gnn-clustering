#!/bin/bash
# Quick (~6.5h target) exploratory re-run of Hybrid DiffPool at --n-hybrid 5
# instead of the --n-hybrid 2 used everywhere else in this repo so far.
#
# Why: DiffPoolGNN's dense_threshold parameter (documented as "node count
# below which hybrid mode switches to full mode") was found to be dead code
# -- never actually read anywhere (see changes-from-claude.md fix #4b). The
# real switch point is n_hybrid, a *level count* matched against precomputed
# HEM files, not a node-count threshold. At n_hybrid=2 (used throughout every
# Hybrid run so far, including outputs/hybrid_rerun2), the trailing learned
# layer actually receives ~3,534 nodes -- not the "<=500" the rest of the
# docs (plan.md, ARCHITECTURE.md) assume. n_hybrid=5 lands at ~442 nodes,
# close to that long-documented design point, and was benchmarked on this
# hardware at ~30s/epoch -- slightly *cheaper* than n_hybrid=2 (~32-33s),
# since 3 extra cheap sparse HEM levels more than pay for handing the
# expensive trailing layer a much smaller graph.
#
# Scope: DiffPool only (DMoN to follow later), num_samples=14 (trimmed from
# the rigorous protocol's 16 to fit the time budget), n_cycles=7 (full
# 127-epoch final retrain -- kept at full depth rather than trimmed, since
# the existing n_hybrid=2 Hybrid DiffPool run's own metrics show accuracy
# still climbing through epoch 126, so a shallow final retrain would risk
# understating what n_hybrid=5 can reach), n_holdouts=1 (vs. the rigorous
# protocol's 3 -- the piece traded away for time; a fuller, matched
# DiffPool+DMoN, 3-rep comparison is intended as a follow-up run once more
# GPU time is available).
#
# Estimated ~6.5h: ASHA (grace_period=31, max_t=63) averages ~47
# epoch-equivalents/trial * 14 samples + 127-epoch final retrain ~= 785
# epochs, at ~30s/epoch.
#
# Usage:
#   ./scripts/run_hybrid_n5_diffpool.sh [repo_root]
#
# To check on it later (after reconnecting):
#   tail -f hybrid_n5_diffpool.log
#   kill -0 $(cat hybrid_n5_diffpool.pid) && echo "still running" || echo "done or dead"
#   cat outputs/hybrid_n5_diffpool/comparison.md   # once finished

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
OUTDIR="$REPO_ROOT/outputs/hybrid_n5_diffpool"

for f in "$DATASET" "$LEVELS" "$NETWORK"; do
  if [ ! -e "$f" ]; then
    echo "ERROR: expected path missing: $f (did you run scripts/setup_remote.sh?)" >&2
    exit 1
  fi
done

echo "Launching Hybrid DiffPool at n_hybrid=5 in the background (nohup + disown) ..."

nohup bash -c '
  echo "HYBRID_RERUN_TIMER diffpool start $(date +%s)"
  python scripts/experiments/diffpool_experiment.py \
    "'"$DATASET"'" "'"$LEVELS"'" \
    --path-network "'"$NETWORK"'" \
    --pooling-type diffpool \
    --n-hybrid 5 --n-hybrid-start 5 \
    --tune --num-samples 14 --n-cycles 7 \
    --batch-size 96 --device cuda --gpu-per-trial 1 --cpu-per-trial 8 \
    --n-holdouts 1 --use-train-set-weights \
    --path-output "'"$OUTDIR"'"
  echo "HYBRID_RERUN_TIMER diffpool end $(date +%s)"
  echo "=== Writing comparison report: $(date) ==="
  python scripts/analysis/compare_hybrid_results.py \
    --path-output "'"$OUTDIR"'" --n-hybrid 5 --n-holdouts 1 \
    --log "'"$REPO_ROOT"'/hybrid_n5_diffpool.log" \
    --out "'"$OUTDIR"'/comparison.md"
' > "$REPO_ROOT/hybrid_n5_diffpool.log" 2>&1 &

PID=$!
disown
echo "$PID" > "$REPO_ROOT/hybrid_n5_diffpool.pid"
echo "Launched. PID=$PID"
echo "Log:    $REPO_ROOT/hybrid_n5_diffpool.log"
echo "Output: $OUTDIR/diffpool_hybrid5_rep0/final_model_results/ (once finished)"
echo "Report: $OUTDIR/comparison.md (once finished)"
echo "Check progress any time with: tail -f $REPO_ROOT/hybrid_n5_diffpool.log"
