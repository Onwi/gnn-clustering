#!/bin/bash
# Full 3-rep protocol run of Hybrid DMoN at --n-hybrid 5 -- the DMoN
# counterpart to scripts/run_hybrid_n5_diffpool_full3.sh, which measured
# Hybrid DiffPool at 83.71% +/- 7.68pp (vs. 65.82% at the old --n-hybrid 2)
# once the trailing learned layer got the ~442-node graph the docs always
# assumed, instead of the ~3,534 nodes --n-hybrid 2 actually produces (see
# changes-from-claude.md fix #4b). This run completes the matched
# DiffPool-vs-DMoN comparison at the corrected n_hybrid.
#
# PREPARED FOR A DIFFERENT (REMOTE) MACHINE THAN IT WAS WRITTEN ON.
# --batch-size 32 below is NOT a portable default -- it was empirically
# calibrated for this machine's RTX 3060 (12GB): batch=32 survived 60
# sustained iterations at 4.88GB peak with no fragmentation growth, batch=48
# fit a single step but wasn't tested sustained, batch=64 OOM'd outright.
# On a different GPU this number means nothing. This script therefore runs
# its own quick pre-flight probe (a handful of real forward/backward
# iterations at the configured batch size, on the real graph) before
# committing to the multi-day run, and aborts with a clear message if it
# OOMs -- rerun with a smaller --batch-size (edit BATCH_SIZE below) if it
# does. This catches an outright-too-large batch size; it does NOT rule out
# the slower fragmentation-related OOM that only shows up after many
# iterations (see the DiffPool script's header) -- if the run OOMs a few
# hours in despite passing the pre-flight check, that's what happened, and
# BATCH_SIZE should be lowered further.
#
# Scope: full rigorous protocol, matching outputs/hybrid_rerun2's original
# design and this DiffPool run -- num_samples=16, n_cycles=7 (127-epoch
# final retrain), n_holdouts=3. Timing will depend entirely on the target
# machine -- on this machine's RTX 3060 at batch=32, Hybrid DiffPool took
# ~25h/rep (~75h/3 reps) in practice, well above the ~21h/rep theoretical
# estimate; budget accordingly and expect DMoN's per-epoch cost to be
# similar to DiffPool's at the same n_hybrid (both go through the same
# DiffPoolGNN scaffolding, only the auxiliary loss differs).
#
# Usage:
#   ./scripts/run_hybrid_n5_dmon_full3.sh [repo_root]
#
# To check on it later:
#   tail -f hybrid_n5_dmon_full3.log
#   kill -0 $(cat hybrid_n5_dmon_full3.pid) && echo "still running" || echo "done or dead"
#   cat outputs/hybrid_n5_dmon_full3/comparison.md   # once finished

set -euo pipefail

REPO_ROOT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
echo "Repo root: $REPO_ROOT"

# Edit this if the pre-flight probe below OOMs on the target machine.
BATCH_SIZE=32
CPU_PER_TRIAL=4

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
OUTDIR="$REPO_ROOT/outputs/hybrid_n5_dmon_full3"

for f in "$DATASET" "$LEVELS" "$NETWORK"; do
  if [ ! -e "$f" ]; then
    echo "ERROR: expected path missing: $f" >&2
    exit 1
  fi
done

echo "Pre-flight: sustained forward/backward probe at batch-size=$BATCH_SIZE on this machine's GPU ..."
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

model = build_diffpool_model(
    base_graph=base_graph, output_dims=output_dims,
    n_hybrid=5, coarse_edges=coarse_edges, parents_list=parents_list,
    pooling_type="dmon",
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
    print(f"Pre-flight OK: batch_size={batch_size} survived 15 sustained iterations, peak {peak:.2f}GB.")
except RuntimeError as e:
    if "out of memory" in str(e).lower():
        print(f"Pre-flight FAILED: batch_size={batch_size} OOM'd. Lower BATCH_SIZE in this script and retry.")
        sys.exit(1)
    raise
PYEOF

echo "Pre-flight passed. Launching Hybrid DMoN n_hybrid=5 full 3-rep protocol in the background (nohup + disown) ..."

nohup bash -c '
  echo "HYBRID_RERUN_TIMER dmon start $(date +%s)"
  python scripts/experiments/diffpool_experiment.py \
    "'"$DATASET"'" "'"$LEVELS"'" \
    --path-network "'"$NETWORK"'" \
    --pooling-type dmon \
    --n-hybrid 5 --n-hybrid-start 5 \
    --tune --num-samples 16 --n-cycles 7 \
    --batch-size '"$BATCH_SIZE"' --device cuda --gpu-per-trial 1 --cpu-per-trial '"$CPU_PER_TRIAL"' \
    --n-holdouts 3 --use-train-set-weights \
    --path-output "'"$OUTDIR"'"
  echo "HYBRID_RERUN_TIMER dmon end $(date +%s)"
  echo "=== Writing comparison report: $(date) ==="
  python scripts/analysis/compare_hybrid_results.py \
    --path-output "'"$OUTDIR"'" --n-hybrid 5 --n-holdouts 3 \
    --log "'"$REPO_ROOT"'/hybrid_n5_dmon_full3.log" \
    --out "'"$OUTDIR"'/comparison.md" \
    --notes "batch-size='"$BATCH_SIZE"', cpu-per-trial='"$CPU_PER_TRIAL"', hardware not recorded -- see this script header"
' > "$REPO_ROOT/hybrid_n5_dmon_full3.log" 2>&1 &

PID=$!
disown
echo "$PID" > "$REPO_ROOT/hybrid_n5_dmon_full3.pid"
echo "Launched. PID=$PID"
echo "Log:    $REPO_ROOT/hybrid_n5_dmon_full3.log"
echo "Output: $OUTDIR/dmon_hybrid5_rep{0,1,2}/final_model_results/ (once finished)"
echo "Report: $OUTDIR/comparison.md (once finished)"
echo "Check progress any time with: tail -f $REPO_ROOT/hybrid_n5_dmon_full3.log"
