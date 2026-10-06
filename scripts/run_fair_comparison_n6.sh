#!/bin/bash
# Controls for the Hybrid n=6 DiffPool/DMoN results (outputs/hybrid_n6_full3),
# so "learned vs. fixed clustering" can be separated from "how narrow the
# bottleneck into the classifier is" (changes-from-claude.md #11).
#
# Every run uses exactly the protocol of scripts/run_hybrid_n6_full3.sh --
# same script, 6 HEM levels, 16 tuning samples, n_cycles=7 (127-epoch
# warm-started retrain), 3 reps, --shared-test-split (same 1,543 test
# patients, since the shared random_state is the first draw of
# default_rng(123)), --use-train-set-weights -- and writes into the SAME
# output directory, so ensemble reports sit next to the learned runs'.
# Only the trailing pooling step differs:
#
#   spectral   221 -> 32 clusters, fixed spectral clustering  (1,024 features, same as learned)
#   random     221 -> 32 clusters, fixed random partition     (1,024 features, same as learned)
#   hem        221 -> 111 nodes, one more fixed HEM level     (3,552 features; = Fixed HEM, 7 levels)
#   hem0       Fixed HEM, 1 level: 14,133 -> 7,067 nodes      (14,134 features; the 96.05% architecture,
#              mean instead of sum pooling, no weighted pooling)
#   dmon111    learned DMoN with --max-clusters 111           (3,552 features: same depth, node
#              count and width as `hem` -- the head-to-head HEM vs. DMoN test for the last level)
#   dmon_wide  learned DMoN with --max-clusters 221           (7,072 features)
#   (dmon111/dmon_wide write under $WIDE_OUTDIR/k111 and /k221, since their dir names would
#   collide with the existing dmon_hybrid6_* runs)
#
# Each run is ~40-50h on the RTX 3090 Ti at batch 96 (the learned n=6 runs took
# 45-53h). Pick which to run; they execute sequentially.
#
# Overridable via environment (defaults = the n=6 protocol on the RTX 3090 Ti):
#   BATCH_SIZE=96 CPU_PER_TRIAL=8 NUM_SAMPLES=16 OUTDIR=... WIDE_OUTDIR=...
#   METADATA_COLUMN=sample_type   (label; default: cohort -- tumour vs normal with sample_type)
#   RUN_TAG=fair_comparison_n6    (name of the .log/.pid files in the repo root)
# Changing BATCH_SIZE or NUM_SAMPLES makes the runs comparable only with each
# other, not with outputs/hybrid_n6_full3.
#
# Usage:
#   ./scripts/run_fair_comparison_n6.sh [run ...]      # default: spectral random hem
#   ./scripts/run_fair_comparison_n6.sh hem0 dmon_wide
#   # HEM vs DMoN head-to-head on a 12 GB RTX 3060, reduced tuning budget:
#   BATCH_SIZE=32 CPU_PER_TRIAL=4 NUM_SAMPLES=8 \
#     OUTDIR=outputs/hem_vs_dmon111_n6 WIDE_OUTDIR=outputs/hem_vs_dmon111_n6 \
#     ./scripts/run_fair_comparison_n6.sh hem dmon111
#
# Check:
#   tail -f fair_comparison_n6.log
#   cat outputs/hybrid_n6_full3/{spectral,random,hem}_ensemble.md

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
RUNS=("$@")
if [ ${#RUNS[@]} -eq 0 ]; then RUNS=(spectral random hem); fi
for r in "${RUNS[@]}"; do
  case "$r" in spectral|random|hem|hem0|dmon111|dmon_wide) ;; *) echo "ERROR: unknown run '$r'" >&2; exit 1 ;; esac
done

BATCH_SIZE="${BATCH_SIZE:-96}"
CPU_PER_TRIAL="${CPU_PER_TRIAL:-8}"
NUM_SAMPLES="${NUM_SAMPLES:-16}"
METADATA_COLUMN="${METADATA_COLUMN:-}"
RUN_TAG="${RUN_TAG:-fair_comparison_n6}"

if ! command -v conda >/dev/null 2>&1; then echo "ERROR: conda not found on PATH." >&2; exit 1; fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pooling_genomic
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" || exit 1

DATASET="$REPO_ROOT/data/string_data/data/tcga_cohorts_and_tumor_classification"
LEVELS="$REPO_ROOT/data/string_data/data/networks/levels"
NETWORK="$REPO_ROOT/data/string_data/data/networks/stringdb_top100pc.csv"
OUTDIR="$(realpath -m "${OUTDIR:-$REPO_ROOT/outputs/hybrid_n6_full3}")"
WIDE_OUTDIR="$(realpath -m "${WIDE_OUTDIR:-$REPO_ROOT/outputs/hybrid_n6_wide}")"
for f in "$DATASET" "$LEVELS" "$NETWORK"; do
  if [ ! -e "$f" ]; then echo "ERROR: expected path missing: $f" >&2; exit 1; fi
done

echo "Pre-flight: batch-size=$BATCH_SIZE forward/backward probe for: ${RUNS[*]} ..."
python - "$DATASET" "$NETWORK" "$LEVELS" "$BATCH_SIZE" "${METADATA_COLUMN:-cohort}" "${RUNS[@]}" <<'PYEOF'
import sys, torch
from pooling_genomic.datasets import get_genomic_classification_dataset
from pooling_genomic.models import build_diffpool_model
from pooling_genomic.networks import get_pyg_data, load_coarse_edges_for_diffpool
from torch.optim import AdamW

path_dataset, path_network, path_levels, batch_size, metadata_column, *runs = sys.argv[1:]
batch_size = int(batch_size)
_, _, _, dataset = get_genomic_classification_dataset(
    path_dataset=path_dataset, return_original_set=True, random_state=0, metadata_column=metadata_column)
print(f"Label '{metadata_column}': {dataset.get_n_classes()} classes {list(dataset.label_encoder.classes_)}")
base_graph = get_pyg_data(genes=dataset.get_genes(), path_to_csv=path_network).to("cuda")
coarse_edges, parents_list = load_coarse_edges_for_diffpool(path_levels=path_levels, n_levels=8, device="cuda")
spec = {
    "spectral": ("spectral", 6, 32), "random": ("random", 6, 32), "hem": ("hem", 6, 32),
    "hem0": ("hem", 0, 32), "dmon111": ("dmon", 6, 111), "dmon_wide": ("dmon", 6, 221),
}
for run in runs:
    pooling_type, n_hybrid, max_clusters = spec[run]
    model = build_diffpool_model(
        base_graph=base_graph, output_dims=dataset.get_n_classes(), n_hybrid=n_hybrid,
        coarse_edges=coarse_edges, parents_list=parents_list,
        pooling_type=pooling_type, max_clusters=max_clusters,
    ).cuda()
    opt = AdamW(model.parameters(), lr=1e-3)
    X = torch.randn(batch_size, base_graph.num_nodes, device="cuda")
    y = torch.randint(0, dataset.get_n_classes(), (batch_size,), device="cuda")
    try:
        for _ in range(15):
            opt.zero_grad()
            torch.nn.functional.cross_entropy(model(X), y).backward()
            opt.step()
        torch.cuda.synchronize()
        print(f"Pre-flight OK ({run}): classifier input {model[1].fcs[0].in_features}, "
              f"peak {torch.cuda.max_memory_allocated() / 1e9:.2f}GB.")
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"Pre-flight FAILED ({run}): batch_size={batch_size} OOM'd. Lower BATCH_SIZE and retry.")
            sys.exit(1)
        raise
    del model, opt
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
PYEOF

run_one () {
  local run="$1" pooling_type n_hybrid outdir extra=""
  case "$run" in
    spectral|random|hem) pooling_type="$run"; n_hybrid=6; outdir="$OUTDIR" ;;
    hem0)      pooling_type=hem;  n_hybrid=0; outdir="$OUTDIR" ;;
    dmon111)   pooling_type=dmon; n_hybrid=6; outdir="$WIDE_OUTDIR/k111"; extra="--max-clusters 111" ;;
    dmon_wide) pooling_type=dmon; n_hybrid=6; outdir="$WIDE_OUTDIR/k221"; extra="--max-clusters 221" ;;
  esac
  echo "=== Starting $run ($pooling_type, n_hybrid=$n_hybrid $extra): $(date) ==="
  echo "FAIR_COMPARISON_TIMER $run start $(date +%s)"
  # shellcheck disable=SC2086
  python scripts/experiments/diffpool_experiment.py \
    "$DATASET" "$LEVELS" \
    --path-network "$NETWORK" \
    --pooling-type "$pooling_type" $extra \
    --n-hybrid "$n_hybrid" --n-hybrid-start "$n_hybrid" \
    --tune --num-samples "$NUM_SAMPLES" --n-cycles 7 \
    --batch-size "$BATCH_SIZE" --device cuda --gpu-per-trial 1 --cpu-per-trial "$CPU_PER_TRIAL" \
    --n-holdouts 3 --shared-test-split --use-train-set-weights \
    ${METADATA_COLUMN:+--metadata-column "$METADATA_COLUMN"} \
    --path-output "$outdir"
  echo "FAIR_COMPARISON_TIMER $run end $(date +%s)"
  python scripts/analysis/ensemble_predictions.py \
    --path-output "$outdir" --pooling-type "$pooling_type" --n-hybrid "$n_hybrid" --n-holdouts 3 \
    --out "$outdir/${run}_ensemble.md"
}

export -f run_one
export DATASET LEVELS NETWORK OUTDIR WIDE_OUTDIR BATCH_SIZE CPU_PER_TRIAL NUM_SAMPLES METADATA_COLUMN

echo "Pre-flight passed. Launching in the background: ${RUNS[*]} (batch=$BATCH_SIZE, samples=$NUM_SAMPLES, label=${METADATA_COLUMN:-cohort}, out=$OUTDIR)"
nohup bash -c 'for r in "$@"; do run_one "$r"; done' _ "${RUNS[@]}" \
  > "$REPO_ROOT/$RUN_TAG.log" 2>&1 &
PID=$!
disown
echo "$PID" > "$REPO_ROOT/$RUN_TAG.pid"
echo "Launched. PID=$PID  Log: $REPO_ROOT/$RUN_TAG.log"
