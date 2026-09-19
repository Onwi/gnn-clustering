#!/bin/bash
# Full 3-rep protocol run of Hybrid DiffPool + Hybrid DMoN at --n-hybrid 6
# (one step past the --n-hybrid 5 that took Hybrid DiffPool from 65.82% to
# 83.71% -- see changes-from-claude.md fix #4b/HYBRID_DMON.md Sec 5.1), with
# two additions on top of that comparison:
#
#   --shared-test-split   all 3 reps share one train/val/test patient split
#                         (default protocol gives each rep a different test
#                         set, which is good for a variance estimate but
#                         means there's nothing to average across reps) --
#                         needed for the ensemble step below to be valid.
#                         See changes-from-claude.md fix #8.
#   (implicit)            the final retrain now warm-starts from the tuning
#                         phase's winning checkpoint instead of a fresh
#                         random init -- fixes the failure mode that
#                         collapsed outputs/hybrid_n5_dmon_full3's rep0 to
#                         4.34% (frozen uniform-class prediction) despite its
#                         hyperparameters reaching 83.2% val accuracy during
#                         tuning. See changes-from-claude.md fix #7.
#
# Batch size (96) and cpu-per-trial (8) already validated safe and fast on
# this machine (MARCS, RTX 3090 Ti) for n_hybrid=5; n_hybrid=6 benchmarked
# slightly cheaper per-epoch (~28.5s vs ~30s at n_hybrid=5, consistent with
# the trend of a smaller trailing-layer graph). Still runs its own pre-flight
# probe before committing to the multi-day run, in case this differs from
# the isolated benchmark once real Ray Tune worker overhead is added.
#
# Scope: full rigorous protocol -- num_samples=16, n_cycles=7 (127-epoch
# final retrain), n_holdouts=3, both pooling types, run sequentially.
# Estimated ~37h/pooling-type (~74h / ~3.1 days total), extrapolated from
# outputs/hybrid_n5_dmon_full3's actual 39h11m at n_hybrid=5 on this machine.
#
# Usage:
#   ./scripts/run_hybrid_n6_full3.sh [repo_root]
#
# To check on it later:
#   tail -f hybrid_n6_full3.log
#   kill -0 $(cat hybrid_n6_full3.pid) && echo "still running" || echo "done or dead"
#   cat outputs/hybrid_n6_full3/comparison.md   # DiffPool vs DMoN, once both finish
#   cat outputs/hybrid_n6_full3/diffpool_ensemble.md
#   cat outputs/hybrid_n6_full3/dmon_ensemble.md

set -euo pipefail

REPO_ROOT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
echo "Repo root: $REPO_ROOT"

BATCH_SIZE=96
CPU_PER_TRIAL=8

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
OUTDIR="$REPO_ROOT/outputs/hybrid_n6_full3"

for f in "$DATASET" "$LEVELS" "$NETWORK"; do
  if [ ! -e "$f" ]; then
    echo "ERROR: expected path missing: $f" >&2
    exit 1
  fi
done

echo "Pre-flight: sustained forward/backward probe at batch-size=$BATCH_SIZE, n_hybrid=6 ..."
python - "$DATASET" "$NETWORK" "$LEVELS" "$BATCH_SIZE" <<'PYEOF'
import sys, torch
from pooling_genomic.datasets import get_genomic_classification_dataset
from pooling_genomic.models import build_diffpool_model
from pooling_genomic.networks import get_pyg_data, load_coarse_edges_for_diffpool
from torch.optim import AdamW

path_dataset, path_network, path_levels, batch_size = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
device = "cuda"

train_set, val_set, test_set, dataset = get_genomic_classification_dataset(
    path_dataset=path_dataset, return_original_set=True, random_state=0
)
output_dims = dataset.get_n_classes()
genes = dataset.get_genes()
base_graph = get_pyg_data(genes=genes, path_to_csv=path_network).to(device)
coarse_edges, parents_list = load_coarse_edges_for_diffpool(path_levels=path_levels, n_levels=8, device=device)

for pooling_type in ("diffpool", "dmon"):
    model = build_diffpool_model(
        base_graph=base_graph, output_dims=output_dims,
        n_hybrid=6, coarse_edges=coarse_edges, parents_list=parents_list,
        pooling_type=pooling_type,
    ).to(device)
    opt = AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    X = torch.randn(batch_size, base_graph.num_nodes, device=device)
    y = torch.randint(0, output_dims, (batch_size,), device=device)
    try:
        for i in range(15):
            opt.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            opt.step()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"Pre-flight OK ({pooling_type}): batch_size={batch_size} survived 15 sustained iterations, peak {peak:.2f}GB.")
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"Pre-flight FAILED ({pooling_type}): batch_size={batch_size} OOM'd. Lower BATCH_SIZE in this script and retry.")
            sys.exit(1)
        raise
    del model, opt
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
PYEOF

echo "Pre-flight passed. Launching Hybrid DiffPool -> Hybrid DMoN at n_hybrid=6 in the background (nohup + disown) ..."

run_one () {
  local pooling_type="$1"
  echo "=== Starting Hybrid $pooling_type (n_hybrid=6): $(date) ==="
  echo "HYBRID_RERUN_TIMER $pooling_type start $(date +%s)"
  python scripts/experiments/diffpool_experiment.py \
    "$DATASET" "$LEVELS" \
    --path-network "$NETWORK" \
    --pooling-type "$pooling_type" \
    --n-hybrid 6 --n-hybrid-start 6 \
    --tune --num-samples 16 --n-cycles 7 \
    --batch-size "$BATCH_SIZE" --device cuda --gpu-per-trial 1 --cpu-per-trial "$CPU_PER_TRIAL" \
    --n-holdouts 3 --shared-test-split --use-train-set-weights \
    --path-output "$OUTDIR"
  echo "HYBRID_RERUN_TIMER $pooling_type end $(date +%s)"
  echo "=== Finished Hybrid $pooling_type (n_hybrid=6): $(date) ==="
  echo "=== Writing $pooling_type ensemble report: $(date) ==="
  python scripts/analysis/ensemble_predictions.py \
    --path-output "$OUTDIR" --pooling-type "$pooling_type" --n-hybrid 6 --n-holdouts 3 \
    --out "$OUTDIR/${pooling_type}_ensemble.md"
}

export -f run_one
export DATASET LEVELS NETWORK OUTDIR BATCH_SIZE CPU_PER_TRIAL

nohup bash -c '
  run_one diffpool && run_one dmon
  echo "=== Writing comparison report: $(date) ==="
  python scripts/analysis/compare_hybrid_results.py \
    --path-output "'"$OUTDIR"'" --n-hybrid 6 --n-holdouts 3 \
    --log "'"$REPO_ROOT"'/hybrid_n6_full3.log" \
    --out "'"$OUTDIR"'/comparison.md" \
    --notes "batch-size='"$BATCH_SIZE"', cpu-per-trial='"$CPU_PER_TRIAL"', shared-test-split=true, hardware=RTX 3090 Ti (MARCS)"
' > "$REPO_ROOT/hybrid_n6_full3.log" 2>&1 &

PID=$!
disown
echo "$PID" > "$REPO_ROOT/hybrid_n6_full3.pid"
echo "Launched. PID=$PID"
echo "Log:    $REPO_ROOT/hybrid_n6_full3.log"
echo "Output: $OUTDIR/{diffpool,dmon}_hybrid6_rep{0,1,2}/final_model_results/ (once each finishes)"
echo "Reports: $OUTDIR/{diffpool,dmon}_ensemble.md, $OUTDIR/comparison.md (once both finish)"
echo "Check progress any time with: tail -f $REPO_ROOT/hybrid_n6_full3.log"
