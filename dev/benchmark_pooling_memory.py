"""Benchmark Full DiffPool's pooling step: peak GPU memory + wall time.

Exercises the real `DiffPoolGNN` (full_mode=True) directly on a synthetic
graph sized like the real STRING-DB / TCGA setup, so no dataset is needed.

This does NOT compare old-vs-new by itself. Run it once on the baseline
(`git stash` to undo the working-tree fix, or `git checkout <commit-before-fix>`)
and once with the fix applied (working tree / current HEAD), and diff the
printed numbers. See changes-from-claude.md, fixes #1 and #2, for the local
(12 GB) measurements this is meant to reproduce/extend on real hardware.

Usage:
    python dev/benchmark_pooling_memory.py \
        --n-nodes 14133 --n-edges 8248194 --batch-size 32 --levels 3

    # sweep a few configs in one go:
    for levels in 1 2 3 4; do
        python dev/benchmark_pooling_memory.py --levels $levels
    done
"""
import argparse
import time

import torch

from pooling_genomic.models import DiffPoolGNN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-nodes", type=int, default=14133,
                         help="default matches the project's STRING-DB gene count")
    parser.add_argument("--n-edges", type=int, default=8_248_194,
                         help="default matches the project's STRING-DB edge count")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--levels", type=int, default=3,
                         help="full-mode DiffPool levels (--n-hybrid in diffpool_experiment.py)")
    parser.add_argument("--max-clusters", type=int, default=32)
    parser.add_argument("--max-filters", type=int, default=32)
    parser.add_argument("--K", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--pooling-type", choices=["diffpool", "dmon"], default="diffpool",
                         help="DMoN skips DiffPool's dense (n,n) to_dense_adj() call used for "
                              "the link-prediction loss, so it should show a lower level-0 peak.")
    args = parser.parse_args()

    if args.device == "cuda":
        assert torch.cuda.is_available(), "no CUDA device visible -- pass --device cpu"
    device = args.device
    torch.manual_seed(args.seed)

    print(f"config: pooling_type={args.pooling_type} n_nodes={args.n_nodes} n_edges={args.n_edges} "
          f"batch_size={args.batch_size} levels={args.levels} "
          f"max_clusters={args.max_clusters} max_filters={args.max_filters} K={args.K}")

    edge_index = torch.randint(0, args.n_nodes, (2, args.n_edges), device=device)
    edge_weight = torch.rand(args.n_edges, device=device)

    model = DiffPoolGNN(
        base_edge_index=edge_index,
        base_edge_weight=edge_weight,
        full_mode=True,
        n_levels=args.levels,
        max_filters=args.max_filters,
        max_clusters=args.max_clusters,
        K=args.K,
        n_nodes=args.n_nodes,
        pooling_type=args.pooling_type,
    ).to(device)

    print(f"cluster_schedule (per-level pooled node counts): {model.cluster_schedule}")

    x = torch.randn(args.batch_size, args.n_nodes, device=device)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    out = model(x)
    t_fwd = time.time() - t0
    fwd_mem = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else float("nan")
    print(f"forward:  {t_fwd:6.2f}s  peak_mem={fwd_mem:6.2f} GB  output_shape={tuple(out.shape)}")

    t0 = time.time()
    out.sum().backward()
    t_bwd = time.time() - t0
    total_mem = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else float("nan")
    print(f"backward: {t_bwd:6.2f}s  peak_mem(fwd+bwd)={total_mem:6.2f} GB")


if __name__ == "__main__":
    main()
