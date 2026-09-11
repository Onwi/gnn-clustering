#!/bin/bash
# Re-run Hybrid DiffPool and Hybrid DMoN back-to-back with IDENTICAL config
# on the actual hardware available (24GB RTX 3090 Ti, confirmed idle via
# nvidia-smi), to investigate whether the previously-recorded 70.3% Hybrid
# DiffPool number reproduces, and to get a fair DMoN comparison against it.
#
# The prior DMoN sweep (outputs/dmon_full) used --batch-size 24,
# --num-samples 6, --cpu-per-trial 4 -- all sized for a 12GB RTX 3060 per
# that script's own comment, which this machine is not. This run instead
# uses:
#   --batch-size 96      empirically probed on this machine's real hybrid
#                         pipeline: 18.5/24.6 GB peak (128 hit 23.9/24.6 GB,
#                         too tight for an unattended multi-hour run)
#   --num-samples 16      raised from 8: the tuned space is 4 continuous
#                         dims (lr, weight_decay, + 2 pooling-type-specific
#                         lambdas) per pooling type -- 8 random samples was
#                         sparse coverage, several trials in the prior run
#                         were clear bad draws (diverging lr).
#   --cpu-per-trial 8     16 CPUs idle on this box; only 1 trial runs at a
#                         time anyway (--gpu-per-trial 1, single GPU)
#   --n-holdouts 3         raised from 1: a single holdout rep can't
#                         distinguish a real DiffPool-vs-DMoN effect from
#                         run-to-run variance (the prior single-rep 16pp
#                         "DiffPool underperformance" turned out to be a
#                         tuning-selection bug, not signal). Report now
#                         aggregates mean +/- std across reps.
#   --use-train-set-weights  both runs showed a large accuracy vs. balanced-
#                         accuracy gap (real class imbalance) -- weighted
#                         CE loss should help both pooling types similarly.
# --n-hybrid 2 --n-hybrid-start 2 and --n-cycles 7 are unchanged from the
# prior run and match the documented 70.3% baseline's architecture
# (n_hybrid=2, 127-epoch final retrain).
#
# Output goes to outputs/hybrid_rerun2 (not outputs/hybrid_rerun) --
# diffpool_experiment.py's run_holdout() skips any n_hybrid/rep directory
# that already exists, so reusing the old output dir would silently skip
# rep0 (built with the pre-fix, buggy tuning-selection code) while only
# regenerating rep1/rep2 with the fixed code, mixing stale and corrected
# results in the same average. A fresh directory avoids that.
#
# Runs DiffPool first, then DMoN (sequentially -- both want the whole GPU),
# then writes a markdown comparison report to
# outputs/hybrid_rerun2/comparison.md (scripts/analysis/compare_hybrid_results.py)
# summarizing test accuracy/balanced accuracy/loss, tuned hyperparameters, and
# wall-clock time for both runs side by side, aggregated (mean +/- std) across
# all holdout reps. Launched via nohup + disown so it survives an SSH
# disconnect.
#
# Usage:
#   ./scripts/run_hybrid_rerun.sh [repo_root]
#
# To check on it later (after reconnecting):
#   tail -f hybrid_rerun.log
#   kill -0 $(cat hybrid_rerun.pid) && echo "still running" || echo "done or dead"
#   cat outputs/hybrid_rerun2/comparison.md   # once both runs finish

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
OUTDIR="$REPO_ROOT/outputs/hybrid_rerun2"

for f in "$DATASET" "$LEVELS" "$NETWORK"; do
  if [ ! -e "$f" ]; then
    echo "ERROR: expected path missing: $f (did you run scripts/setup_remote.sh?)" >&2
    exit 1
  fi
done

run_one () {
  local pooling_type="$1"
  echo "=== Starting Hybrid $pooling_type: $(date) ==="
  echo "HYBRID_RERUN_TIMER $pooling_type start $(date +%s)"
  python scripts/experiments/diffpool_experiment.py \
    "$DATASET" "$LEVELS" \
    --path-network "$NETWORK" \
    --pooling-type "$pooling_type" \
    --n-hybrid 2 --n-hybrid-start 2 \
    --tune --num-samples 16 --n-cycles 7 \
    --batch-size 96 --device cuda --gpu-per-trial 1 --cpu-per-trial 8 \
    --n-holdouts 3 --use-train-set-weights \
    --path-output "$OUTDIR"
  echo "HYBRID_RERUN_TIMER $pooling_type end $(date +%s)"
  echo "=== Finished Hybrid $pooling_type: $(date) ==="
}

export -f run_one
export DATASET LEVELS NETWORK OUTDIR

echo "Launching Hybrid DiffPool -> Hybrid DMoN sequentially in the background (nohup + disown) ..."
echo "batch-size=96 was empirically probed on this machine -- see header comment above."

nohup bash -c '
  run_one diffpool && run_one dmon
  echo "=== Writing comparison report: $(date) ==="
  python scripts/analysis/compare_hybrid_results.py \
    --path-output "'"$OUTDIR"'" --n-holdouts 3 --log "'"$REPO_ROOT"'/hybrid_rerun.log" \
    --out "'"$OUTDIR"'/comparison.md"
' > "$REPO_ROOT/hybrid_rerun.log" 2>&1 &

PID=$!
disown
echo "$PID" > "$REPO_ROOT/hybrid_rerun.pid"
echo "Launched. PID=$PID"
echo "Log:    $REPO_ROOT/hybrid_rerun.log"
echo "Output: $OUTDIR/{diffpool,dmon}_hybrid2_rep0/final_model_results/ (once each finishes)"
echo "Check progress any time with: tail -f $REPO_ROOT/hybrid_rerun.log"
