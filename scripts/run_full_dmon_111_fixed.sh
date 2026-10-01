#!/bin/bash
# Full DMoN at the same bottleneck/test-split as scripts/run_full_dmon_111.sh
# (max_clusters=111, matching outputs/hem_vs_dmon111_n6's 3,552-feature
# width and shared test split), PLUS the four improvement ideas from
# changes-from-claude.md #13:
#   --pool-gnn-layers 3     deeper assignment head (wider receptive field
#                           when deciding each node's cluster)
#   --encoder-channels 32 --encoder-layers 3   wider pre-pooling encoder
#   (collapse-loss normalization and per-patient pooled graphs are
#   unconditional code fixes now, not flags)
#
# batch-size=2, not 6: the wider encoder + deeper assignment head cost real
# extra memory, and the per-patient-graph fix's scatter-based rewrite (see
# #13 -- torch.sparse.mm's backward crashed with an illegal-memory-access
# when asked to differentiate through gradient-requiring sparse values,
# reproducibly on 5/5 seeds; replaced with torch_scatter.scatter) costs
# more still (a dense (num_edges, k+1) gather tensor instead of an internal
# sparse representation). Batch 3/4/6 OOM; batch 2 verified stable over
# 3 seeds x 20 iterations on the real graph before this script was written.
#
# Estimated ~72 min/epoch pure-step (vs ~65 min for the same bottleneck
# without these four changes) at the same shrunk-scope budget (6 tuning
# samples capped at 15 epochs, 31-epoch warm-started retrain, 1 rep) --
# roughly ~3.5-4 days total, not yet run as a real multi-day job.
#
# Usage:  ./scripts/run_full_dmon_111_fixed.sh [repo_root]
# Check:  tail -f full_dmon_111_fixed.log ; cat outputs/full_dmon_111_fixed/results.md
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
OUTDIR="$REPO_ROOT/outputs/full_dmon_111_fixed"
for f in "$DATASET" "$LEVELS" "$NETWORK"; do
  if [ ! -e "$f" ]; then echo "ERROR: expected path missing: $f" >&2; exit 1; fi
done

nohup bash -c '
  echo "FULL_DMON_TIMER start $(date +%s)"
  python scripts/experiments/diffpool_experiment.py \
    "'"$DATASET"'" "'"$LEVELS"'" \
    --path-network "'"$NETWORK"'" \
    --pooling-type dmon --max-clusters 111 \
    --pool-gnn-layers 3 --encoder-channels 32 --encoder-layers 3 \
    --full-mode --n-hybrid 3 --n-hybrid-start 3 \
    --sparsify-density 0.04 \
    --tune --num-samples 6 --n-cycles 5 \
    --batch-size 2 --device cuda --gpu-per-trial 1 --cpu-per-trial 8 \
    --n-holdouts 1 --shared-test-split --use-train-set-weights \
    --path-output "'"$OUTDIR"'"
  echo "FULL_DMON_TIMER end $(date +%s)"
  echo "=== Writing results report: $(date) ==="
  python scripts/analysis/analyze_full_dmon_results.py \
    --path-output "'"$OUTDIR"'" --n-hybrid 3 --sparsify-density 0.04 \
    --num-samples 6 --n-cycles 5 --batch-size 2 --n-holdouts 1 \
    --log "'"$REPO_ROOT"'/full_dmon_111_fixed.log" \
    --out "'"$OUTDIR"'/results.md"
' > "$REPO_ROOT/full_dmon_111_fixed.log" 2>&1 &
PID=$!
disown
echo "$PID" > "$REPO_ROOT/full_dmon_111_fixed.pid"
echo "Launched. PID=$PID  Log: $REPO_ROOT/full_dmon_111_fixed.log  Output: $OUTDIR"
