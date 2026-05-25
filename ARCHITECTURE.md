# Architecture

## Overview

Hierarchical graph pooling for genomic (gene expression) classification using STRING-DB protein interaction networks. Three approaches:

- **Fixed HEM** (Heavy Edge Matching): pre-computed hierarchical coarsening, no learned pooling
- **Hybrid DiffPool**: differentiable pooling with hybrid early levels (HEM scatter) + full learned last layer
- **Full DiffPool**: all layers use learned soft-assignment pooling (`S^T Z`, `S^T A S`)

All operate on the same input: RNA-seq gene expression profiles mapped to ~14K genes in a STRING-DB PPI graph. Goal: classify 16 TCGA cancer types from expression alone.

---

## Data Pipeline

### Raw Data
- `data/string_data/data/tcga_cohorts_and_tumor_classification/` — gene expression matrix (7709 samples × ~14K genes), metadata with `cohort` column
- 16 cancer types: blca, brca, coad, esca, hnsc, kich, kirc, kirp, lihc, luad, lusc, prad, read, stad, thca, ucec

### Dataset Loading (`src/pooling_genomic/datasets.py`)
- `get_genomic_classification_dataset()` — dispatches by path string to dataset class
- `get_tcga_classification_datasets()` — loads expression data, encodes labels via `LabelEncoder`, returns `train/val/test` subsets (80/10/10 default with `random_split`)

### Graph Construction (`src/pooling_genomic/networks.py`)
- `get_pyg_data()` — loads STRING-DB edge list CSV, filters to genes present in the dataset, builds `torch_geometric.data.Data` with `edge_index` and `edge_weight` (combined_score / 1000)
- STRING-DB top100pc: ~14K nodes, ~8M edges (after filtering to expressed genes)

### Graph Levels (Pre-computation, `scripts/generate_graph_levels.py`)
- `coarsening.HEM()` — produces 8 hierarchical levels (14K → 7K → 3.5K → 1.8K → 884 → 442 → 221 → 111 nodes)
- Saves per level: `edge_index_lvl{N}.pt`, `edge_weight_lvl{N}.pt`, `parents_lvl{N}.pt`
- Loaded via `load_graph_levels()` (for HEM) or `load_coarse_edges_for_diffpool()` (for DiffPool)

---

## Fixed HEM Approach (`coarsening_levels.py`)

### Model Architecture (`build_coarsening_model` in `models.py`)

```
Input (batch, 14000 genes)
    │
    ▼ reshape→(batch, 14000, 1)
    │
    ┌──────────────────────────────────────────────┐
    │           GNNPooling (Sequential)            │
    │                                              │
    │   Level 0:  (14000 nodes, 1→1 channels)     │
    │     Optional: ChebConv(K=2) + ReLU            │
    │     Optional: weighted_pooling (learned node  │
    │               importance per level)           │
    │     Pool: scatter(parents, sum)  → 7000 nodes │
    │                                              │
    │   Level 1:  (7000 nodes, 1→2 channels)       │
    │     ChebConv + ReLU + scatter → 3500 nodes   │
    │                                              │
    │   Level 2:  (3500 nodes, 2→4 channels)       │
    │     ChebConv + ReLU + scatter → 1800 nodes   │
    │              ...                              │
    │   Last Level: (N_last nodes, C_last channels) │
    │     ChebConv + ReLU + scatter → final nodes  │
    └──────────────────────────────────────────────┘
    │
    ▼ Flatten: (batch, final_nodes × C_last)
    │
    ▼ FCModel: Linear(256) → BN → ReLU → Dropout → Linear(n_classes)
```

**Key details:**
- Hierarchical coarse graph structure is **fixed** (from HEM pre-computation)
- `cluster_indices` = HEM parent mapping, used with `scatter(..., reduce='sum')`
- Progressive channel doubling: 1 → 2 → 4 → ... → 32 (max_filters)
- Chebyshev convolution (ChebConv K=2) on each level, or no convs at all (`use_convs=False`)
- Optional weighted pooling: learnable scalar per node, multiplied before scatter
- Final flatten gives input dimension = `num_super_nodes × 32` for the MLP

### Training Loop (`coarsening_levels.py`)
- AdamW optimizer, CosineAnnealingWarmRestarts scheduler (T_0=1, T_mult=2)
- Cross-entropy loss
- Optional L1 regularization on node importances (`lambda_l1`)
- Hyperparameter tuning via Ray Tune (optional `--tune` flag)
- Iterates all `n_levels=0..7`, `weighted_pooling=False/True`, `use_convs=False/True`

### Best Result
- **92% test accuracy** with `n_levels=0` (no coarsening, just MLP on raw 14K features)

---

## Learned DiffPool Approach (`diffpool_experiment.py`)

### Model Architecture (`DiffPoolGNN` + `DiffPoolLayer` in `models.py`)

```
Input (batch, 14000 genes)
    │
    ▼ reshape→(batch, 14000, 1)
    │
    ┌──────────────────────────────────────────────────┐
    │            DiffPoolGNN (Sequential)              │
    │                                                  │
    │  Hybrid Level 0 (14000 nodes, 1→2 channels):    │
    │     ChebConv embed + ReLU                        │
    │     Scatter(parents[0], mean)  → 7000 nodes      │
    │     (learned assignment S skipped, uses HEM      │
    │      parents for efficient scatter pooling)      │
    │     ✓ Pre-computed coarse edges from level 1     │
    │                                                  │
    │  Hybrid Level 1 (7000 nodes, 2→4 channels):     │
    │     ChebConv embed + ReLU                        │
    │     Scatter(parents[1], mean)  → 3500 nodes      │
    │     ✓ Pre-computed coarse edges from level 2     │
    │                                                  │
    │  ... Hybrid levels continue until ...            │
    │                                                  │
    │  Transition: when n ≤ dense_threshold (500)      │
    │     → Clear coarse edges → switch to full mode   │
    │                                                  │
    │  Full Level N (≤500 nodes, C_{N-1}→C_N):        │
    │     ChebConv embed: Z = ReLU(embed(x))           │
    │     Pool GNN: S = softmax(pool_gnn(x))           │
    │     X' = S^T Z  (learned assignment)             │
    │     A' = S^T A S  (pooled adjacency)             │
    │     Aux losses: link prediction + entropy        │
    │     Extract sparse edges from A' for next level  │
    │     Output: k ≤ max_clusters (32) nodes          │
    │                                                  │
    │  Total levels = n_hybrid + 1                     │
    │  Last level is always full mode                   │
    └──────────────────────────────────────────────────┘
    │
    ▼ Flatten: (batch, max_clusters × last_channels)
    │
    ▼ FCModel: Linear(256) → BN → ReLU → Dropout → Linear(n_classes)
```

**Total levels = `n_hybrid + 1`** — e.g., `n_hybrid=2` = 3 levels total (2 hybrid + 1 full).

### Hybrid vs Full Mode

| Aspect | Hybrid | Full |
|---|---|---|
| Assignment | Fixed HEM parents | Learned `S = softmax(pool_gnn(x))` |
| Pooling | `scatter(z, parents, mean)` | `S^T @ Z` |
| Adjacency | Pre-computed coarse edges | `S^T @ A @ S` → dense_to_sparse |
| Auxiliary loss | None (0) | Link prediction MSE + Entropy |
| Computation | O(n) sparse | O(n²) dense (only when n ≤ 500) |
| Edge count | Fixed from HEM | Grows/dynamic |

### Training Loop (`diffpool_experiment.py`)
- AdamW optimizer with `weight_decay`
- CosineAnnealingWarmRestarts (T_0=1, T_mult=2)
- Loss = CrossEntropy + `λ_link_pred × link_pred_loss` + `λ_entropy × entropy_loss`
- `get_diffpool_aux_losses()` extracts aux losses from `DiffPoolGNN._aux_records`
- Hyperparameter tuning via Ray Tune (grid over lr, weight_decay, lambdas)
- Iterates all `n_hybrid=0..min(max_n_levels, args.n_hybrid)`

### Best Result
- **70.3% test accuracy** with tuned HPs (lr=0.0099, wd=0.0344, λ_lp=0.000166, λ_ent=1.98e-5) and 127 epochs (n_cycles=7)

---

## Full Learned DiffPool (`diffpool_experiment.py --full-mode`)

### Model Architecture (`DiffPoolGNN` with `full_mode=True` in `models.py`)

All layers use the full DiffPool path from `DiffPoolLayer` (no hybrid early levels, no HEM parents/coarse edges). The graph is dynamically pooled at every level via learned soft assignments.

```
Input (batch, 14000 genes)
    │
    ▼ reshape→(batch, 14000, 1)
    │
    ┌──────────────────────────────────────────────────────┐
    │               DiffPoolGNN (full_mode=True)           │
    │                                                      │
    │  Level 0:  (14000 nodes, 1→2 channels)              │
    │     Z = ReLU(ChebConv(x, edge_index))                │
    │     S = softmax(ChebConv(x, edge_index))             │
    │     X' = S^T @ Z           (learned clustering)      │
    │     A_dense = to_dense_adj(edge_index)               │
    │     A' = S^T @ A_dense @ S  (pooled adjacency)       │
    │     edge_index' = dense_to_sparse(A')                │
    │     k = ceil(n * sigmoid(logit_pool_ratio))          │
    │     k = clamp(k, 2, max_clusters=32)                 │
    │     aux = link_pred_loss + entropy_loss              │
    │                                                      │
    │  Level 1:  (k0 nodes, 2→4 channels)                 │
    │     Same full DiffPool step                          │
    │     k1 = ceil(k0 * sigmoid(logit_pool_ratio))        │
    │                                                      │
    │  Level 2:  (k1 nodes, 4→8 channels)                 │
    │     Same full DiffPool step                          │
    │     ...                                              │
    │                                                      │
    │  Last Level:  (k_{N-1} nodes, C_last channels)      │
    │     Pools to ≤ max_clusters (32) nodes               │
    └──────────────────────────────────────────────────────┘
    │
    ▼ Flatten: (batch, max_clusters × last_channels)
    │
    ▼ FCModel: Linear(256) → BN → ReLU → Dropout → Linear(n_classes)
```

**Number of levels** = `n_levels` (`--n-hybrid` CLI argument reused as the level count).

**Progressive channels** (same as hybrid): 1→2→4→...→32. Each level gets `(in_ch=2^i, out_ch=2^{i+1})`, capped at `max_filters=32`.

**Pool ratio**: each level has a learned `logit_pool_ratio` parameter (initialized 0 → sigmoid = 0.5). The output cluster count is `k = ceil(n * sigmoid(logit_pool_ratio))`, bounded by `[2, max_clusters]`.

### Key Differences from Hybrid Mode

| Aspect | Hybrid | Full |
|---|---|---|
| Level 0 pooling | `scatter(parents[0], mean)` — O(n) | `S^T @ Z` — O(n²) dense |
| Adjacency at level 1+ | Pre-computed HEM coarse edges | `S^T @ A @ S` → sparse |
| Memory at level 0 | `(batch, 14K, 1)` features only | `(batch, 14K, k)` assignment S + `(14K, 14K)` dense A |
| Auxiliary losses | None on hybrid levels | Link prediction + entropy on **every** level |
| HEM dependency | Requires pre-computed levels | Only the base graph is needed |
| Structural prior | Strong (HEM hierarchy) | None (everything learned) |

### Training Loop
- Same as hybrid mode: AdamW, CosineAnnealingWarmRestarts, cross-entropy + aux losses
- `--full-mode` flag toggles off `load_coarse_edges_for_diffpool()` and HEM parent usage
- Iterates `n_levels=0..N` (same loop as hybrid, reusing `n_hybrid` as the level count)
- Output dir: `diffpool_full{N}_rep{R}/`

### Current Status
- Not yet evaluated (requires GPU with ≥24 GiB memory for dense `(14K, 14K)` operations at level 0)

---

## Comparison

| Aspect | Fixed HEM | Hybrid DiffPool | Full DiffPool |
|---|---|---|---|
| Pooling strategy | `scatter(parents, sum)` | Hybrid: `scatter(parents, mean)`; Last: learned `S^T Z` | Learned `S^T Z` at every level |
| Adjacency | Fixed HEM per level | Fixed HEM coarse edges (hybrid) + dynamic (last) | Dynamic `S^T A S` at every level |
| Memory (level 0) | O(n) sparse | O(n) sparse (scatter) | O(n²) dense (14K×14K) |
| Architecture | 1→1 or progressive channels 1→2→4→...→32 | Same progressive channels | Same progressive channels |
| Hyperparameters | lr, wd, optional λ_l1 | lr, wd, λ_link_pred, λ_entropy | lr, wd, λ_link_pred, λ_entropy |
| Best test accuracy | **92%** (n_levels=0, MLP only) | **70.3%** (n_hybrid=2, tuned, 127 epochs) | Not yet evaluated |
| Training speed | 1 epoch to convergence | 63–127 epochs | Unknown (likely >127) |
| MLP input size | 14K × 1 or N_last × C_last | max_clusters (32) × last_channels | max_clusters (32) × last_channels |
| Structural prior | Very strong (HEM) | Strong (HEM early, learned late) | None (fully learned) |
| GNN at level 0 | No (bare scatter) | ChebConv (always) | ChebConv (always) |
| Pre-computation required | HEM levels (8 files) | HEM levels (8 files) | No (base graph only) |

---

## Experiment Scripts

### `scripts/experiments/coarsening_levels.py`
- Iterates n_levels=[0..7], wpool=[F,T], convs=[F,T] — 47 configs total
- Tuning and reporting via path-based `--path-output`
- Output dir pattern: `nlevels{N}_rep{R}_wpool{B}_convs{B}/`
- Existing output dirs are skipped (resume-safe)

### `scripts/experiments/diffpool_experiment.py`
- Two modes controlled by `--full-mode` flag:
  - **Default (hybrid)**: iterates n_hybrid=[0..N], uses pre-computed HEM coarse edges + parents for early levels. Output dir: `diffpool_hybrid{N}_rep{R}/`
  - **`--full-mode`**: all levels are full learned DiffPool, no HEM pre-computed data needed beyond the base graph. Output dir: `diffpool_full{N}_rep{R}/`
- Same Ray Tune integration pattern
- OOM handling: catches CUDA OOM errors and skips to next config

### `scripts/analysis_v2/`
- Post-hoc comparison analysis: parser → confusion matrices → error analysis → training curves → hyperparameter sensitivity
- Run via `python -m scripts.analysis_v2.main --path-output outputs`

---

## Pre-computation (`scripts/generate_graph_levels.py`)

1. Load TCGA cohort dataset → extract gene list
2. Load STRING-DB network → filter to overlapping genes
3. Convert to NetworkX → scipy sparse adjacency
4. Run `coarsening.HEM(adj, levels=8)` — produces 8 coarsened graphs + parent mappings
5. Save each level's `edge_index`, `edge_weight`, `parents`

**HEM Algorithm** (`src/pooling_genomic/coarsening.py`, adapted from xbresson):
- Heavy Edge Matching: pairs vertices with heaviest edge weight
- At each level, 2:1 reduction (approximately halving node count)
- Parent mapping: each coarsened node corresponds to 1-2 original nodes

---

## Package Structure (`src/pooling_genomic/`)

| File | Responsibility |
|---|---|
| `models.py` | All model definitions: `DiffPoolLayer`, `DiffPoolGNN` (hybrid + full_mode), `GNNPooling`, `FCModel`, `build_coarsening_model`, `build_diffpool_model`, `CohortAndTumorLoss` |
| `networks.py` | Graph construction (`get_pyg_data`), level loading (`load_graph_levels`, `load_coarse_edges_for_diffpool`) |
| `coarsening.py` | Heavy Edge Matching algorithm (`HEM`, `HEM_one_level`, `compute_perm`) |
| `datasets.py` | Dataset wrappers (`get_genomic_classification_dataset`, `TCGACohorts`, `WrapperDataset`) |
| `engines.py` | Training/eval loops (`train_epoch_clf`, `evaluate_clf`, `train_cohort_tumor_clf`) |
| `settings.py` | Pydantic BaseSettings (env prefix `POOLING_GENOMIC_`, reads `.env`) |
| `utils.py` | Confusion matrix plotting, JSON utilities |
