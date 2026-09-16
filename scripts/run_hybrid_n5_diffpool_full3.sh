#!/bin/bash
# Full 3-rep protocol re-run of Hybrid DiffPool at --n-hybrid 5, on this
# machine (RTX 3060, 12GB) after the lab203 run was interrupted by a reboot
# mid-final-retrain (see scripts/recover_hybrid_n5_diffpool.py, which was
# written for that machine's paths and is now moot -- redoing the tuning
# search from scratch here instead).
#
# Why n_hybrid=5: DiffPoolGNN's dense_threshold parameter (documented as
# "node count below which hybrid mode switches to full mode") was found to
# be dead code -- never read anywhere (changes-from-claude.md fix #4b). The
# real switch point is n_hybrid, a *level count* matched against precomputed
# HEM files, not a node-count threshold. At n_hybrid=2 (used throughout every
# Hybrid run so far, including outputs/hybrid_rerun2), the trailing learned
# layer actually receives ~3,534 nodes -- not the "<=500" the rest of the
# docs (plan.md, ARCHITECTURE.md) assume. n_hybrid=5 lands at ~442 nodes,
# close to that long-documented design point.
#
# Hardware differs from the original lab203 run (RTX 3090 Ti, 24GB,
# batch=96, ~30s/epoch): this machine's RTX 3060 (12GB) was empirically
# probed to top out around batch=48 for a single step, but batch=32 is what
# survived 60 sustained iterations with flat peak memory (4.88GB, no
# fragmentation growth) -- batch=64 OOMs outright. At batch=32, epochs run
# ~64s -- about 2x lab203's rate (smaller batch + weaker GPU).
#
# Scope: full rigorous protocol, matching outputs/hybrid_rerun2's original
# design -- num_samples=16, n_cycles=7 (127-epoch final retrain), n_holdouts=3.
# Estimated ~47h (~2 days): ASHA (grace_period=31, max_t=63) averages ~47
# epoch-equivalents/trial * 16 samples + 127-epoch final retrain ~= 879
# epochs/rep * 3 reps ~= 2637 epochs, at ~64s/epoch.
#
# --cpu-per-trial 4 (vs. 8 on lab203): this is an actively-used desktop
# (Discord/Chrome/JetBrains running), not a dedicated compute box -- leaves
# 12 of 16 threads free for normal use during the run.
#
# Usage:
#   ./scripts/run_hybrid_n5_diffpool_full3.sh [repo_root]
#
# To check on it later:
#   tail -f hybrid_n5_diffpool_full3.log
#   kill -0 $(cat hybrid_n5_diffpool_full3.pid) && echo "still running" || echo "done or dead"
#   cat outputs/hybrid_n5_diffpool_full3/comparison.md   # once finished

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

python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" || exit 1

DATASET="$REPO_ROOT/data/string_data/data/tcga_cohorts_and_tumor_classification"
LEVELS="$REPO_ROOT/data/string_data/data/networks/levels"
NETWORK="$REPO_ROOT/data/string_data/data/networks/stringdb_top100pc.csv"
OUTDIR="$REPO_ROOT/outputs/hybrid_n5_diffpool_full3"

for f in "$DATASET" "$LEVELS" "$NETWORK"; do
  if [ ! -e "$f" ]; then
    echo "ERROR: expected path missing: $f" >&2
    exit 1
  fi
done

echo "Launching Hybrid DiffPool n_hybrid=5 full 3-rep protocol in the background (nohup + disown) ..."
echo "batch-size=32 was empirically probed on this machine's RTX 3060 -- see header comment above."

nohup bash -c '
  echo "HYBRID_RERUN_TIMER diffpool start $(date +%s)"
  python scripts/experiments/diffpool_experiment.py \
    "'"$DATASET"'" "'"$LEVELS"'" \
    --path-network "'"$NETWORK"'" \
    --pooling-type diffpool \
    --n-hybrid 5 --n-hybrid-start 5 \
    --tune --num-samples 16 --n-cycles 7 \
    --batch-size 32 --device cuda --gpu-per-trial 1 --cpu-per-trial 4 \
    --n-holdouts 3 --use-train-set-weights \
    --path-output "'"$OUTDIR"'"
  echo "HYBRID_RERUN_TIMER diffpool end $(date +%s)"
  echo "=== Writing comparison report: $(date) ==="
  python scripts/analysis/compare_hybrid_results.py \
    --path-output "'"$OUTDIR"'" --n-hybrid 5 --n-holdouts 3 \
    --log "'"$REPO_ROOT"'/hybrid_n5_diffpool_full3.log" \
    --out "'"$OUTDIR"'/comparison.md" \
    --notes "batch-size=32, RTX 3060 12GB, cpu-per-trial=4 (see this script header)"
' > "$REPO_ROOT/hybrid_n5_diffpool_full3.log" 2>&1 &

PID=$!
disown
echo "$PID" > "$REPO_ROOT/hybrid_n5_diffpool_full3.pid"
echo "Launched. PID=$PID"
echo "Log:    $REPO_ROOT/hybrid_n5_diffpool_full3.log"
echo "Output: $OUTDIR/diffpool_hybrid5_rep{0,1,2}/final_model_results/ (once finished)"
echo "Report: $OUTDIR/comparison.md (once finished)"
echo "Check progress any time with: tail -f $REPO_ROOT/hybrid_n5_diffpool_full3.log"
