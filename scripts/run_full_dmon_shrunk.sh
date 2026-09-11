#!/bin/bash
# Launch the shrunk-scope Full DMoN tuning run in the background (nohup +
# disown) so it survives an SSH disconnect, then write a markdown results
# report once it finishes.
#
# Background: Full DMoN (--full-mode, no HEM pre-coarsening) hits the
# 440:1 single-hop compression problem and the dense-pooled-adjacency
# corruption problem documented in analysis-approaches.MD/plan.md for Full
# DiffPool. changes-from-claude.md fixes #1 (progressive cluster schedule,
# --n-hybrid 3 --n-hybrid-start 3 spreads 14133->32 across 3 hops instead of
# one) and #3 (--sparsify-density 0.04, matching stringdb_top100pc.csv's
# actual measured density, prunes each level's pooled adjacency instead of
# leaving it fully connected) both apply here.
#
# Scope: measured on this hardware at ~43 min/epoch (batch=8 is the largest
# size that survives sustained training without OOM/fragmentation -- see
# conversation), a Hybrid-comparable search (num_samples=16, n_cycles=7,
# n_holdouts=3) would take ~11 weeks. This run is deliberately shrunk
# (num_samples=6, n_cycles=5 -> 31-epoch final retrain, n_holdouts=1) to fit
# in ~2.9 days, at the cost of being exploratory rather than a rigorous,
# variance-characterized comparison like outputs/hybrid_rerun2/comparison.md.
# A 30-min smoke test (untuned defaults, outputs/full_dmon_smoke) already
# confirmed the model learns (accuracy climbed from ~0% to ~17.5% within a
# partial first epoch) rather than collapsing to random, before this was
# launched.
#
# Usage:
#   ./scripts/run_full_dmon_shrunk.sh [repo_root]
#
# To check on it later (after reconnecting):
#   tail -f full_dmon_shrunk.log
#   kill -0 $(cat full_dmon_shrunk.pid) && echo "still running" || echo "done or dead"
#   cat outputs/full_dmon_shrunk/results.md   # once finished

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
OUTDIR="$REPO_ROOT/outputs/full_dmon_shrunk"

for f in "$DATASET" "$LEVELS" "$NETWORK"; do
  if [ ! -e "$f" ]; then
    echo "ERROR: expected path missing: $f (did you run scripts/setup_remote.sh?)" >&2
    exit 1
  fi
done

echo "Launching Full DMoN shrunk-scope run in the background (nohup + disown) ..."
echo "batch-size=8 was empirically probed on this machine -- see header comment above."

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
    --log "'"$REPO_ROOT"'/full_dmon_shrunk.log" \
    --out "'"$OUTDIR"'/results.md"
' > "$REPO_ROOT/full_dmon_shrunk.log" 2>&1 &

PID=$!
disown
echo "$PID" > "$REPO_ROOT/full_dmon_shrunk.pid"
echo "Launched. PID=$PID"
echo "Log:    $REPO_ROOT/full_dmon_shrunk.log"
echo "Output: $OUTDIR/dmon_full3_rep0/final_model_results/ (once finished)"
echo "Report: $OUTDIR/results.md (once finished)"
echo "Check progress any time with: tail -f $REPO_ROOT/full_dmon_shrunk.log"
