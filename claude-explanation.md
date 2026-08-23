# Pooling GNNs for Genomic Data — Project Explanation

## What this project is

This is the research codebase for a master's dissertation comparing **fixed vs. learned
hierarchical graph pooling** in Graph Neural Networks (GNNs), applied to **cancer type
classification from RNA-seq gene expression data**.

The central question: given a gene expression profile mapped onto a protein-protein
interaction (PPI) network, does letting the model *learn* how to cluster/coarsen the graph
(via DiffPool) beat a *fixed*, pre-computed hierarchical clustering (Heavy Edge Matching, HEM)
inherited from prior thesis work (Fontanari)?

The short answer the experiments arrive at: **no** — the fixed HEM baseline (~96% accuracy)
consistently outperforms learned DiffPool approaches (~70–72% best case), and the dissertation
is largely an analysis of *why* that happens (see `plan.md` for the full narrative and
`ARCHITECTURE.md` for architecture-level detail).

## The task

- **Input**: RNA-seq expression vector per patient, ~14,133 genes (log2(TPM+1) normalized).
- **Graph**: STRING-DB protein-protein interaction network restricted to those genes
  (~14K nodes, ~8.2M edges, same topology for every sample — only node features vary).
- **Output**: classification into 16 TCGA pan-cancer tumor types (blca, brca, coad, esca,
  hnsc, kich, kirc, kirp, lihc, luad, lusc, prad, read, stad, thca, ucec).
- **Dataset**: 7,709 patients (TCGA pan-cancer cohort).

## Three modeling approaches compared

| Approach | Pooling | Adjacency | Structural prior |
|---|---|---|---|
| **Fixed HEM** | `scatter(parents, sum)` using pre-computed Heavy Edge Matching | Fixed per level | Strong (biological/topological) |
| **Hybrid DiffPool** | HEM scatter pooling for large early levels, learned `Sᵀ Z` once the graph shrinks below `dense_threshold` (default 500 nodes) | Fixed HEM edges early, `Sᵀ A S` late | Strong early, learned late |
| **Full DiffPool** | Learned soft-assignment `Sᵀ Z` at *every* level | Dynamic `Sᵀ A S` → `dense_to_sparse` at every level | None — everything learned |

All three feed into the same downstream classifier head: flatten → `Linear(256) → BatchNorm →
ReLU → Dropout → Linear(n_classes)`.

Key empirical results (see `ARCHITECTURE.md` §Comparison and `plan.md` Chapter 5):

- MLP on raw expression only: **95.46%** — expression alone is highly discriminative.
- Fixed HEM (best config, `n_levels=0`, conv+weighted pooling): **~96%**.
- Full DiffPool, untuned: collapses to **~27.7%** (near random for 16 classes).
- Full DiffPool, heavily tuned (pre-pooling encoder, low LR, low aux-loss weights, gradient
  clipping): **71.68%** — still ~24 points below the HEM baseline.
- Hybrid DiffPool, tuned: **~70.3%**.

Root cause analysis (Chapter 6 of `plan.md`): at 14K nodes, learning a soft assignment matrix
from a single scalar feature per node is ill-posed; the dense `Sᵀ A S` operation destroys
graph sparsity/structure; and the auxiliary losses (link-prediction + entropy) dominate
gradients early in training. The conclusion is that for RNA-seq data, the PPI graph functions
as a *regularizer*, not a *primary signal source* — so a strong fixed structural prior (HEM)
beats a flexible but under-constrained learned one.

## Repository layout

```
src/pooling_genomic/        Installable library (pip install -e .)
  models.py                 All model definitions (see below)
  networks.py                Graph construction & pre-computed level loading
  coarsening.py              Heavy Edge Matching (HEM) algorithm
  datasets.py                Dataset loading/wrapping (path-based dispatch)
  engines.py                 Train/eval loops
  saliency.py                Guided backprop saliency analysis, feature-ablation utilities
  settings.py                Pydantic BaseSettings (env prefix POOLING_GENOMIC_, reads .env)
  utils.py                   Confusion-matrix plotting, JSON helpers

scripts/
  generate_graph_levels.py           Pre-computes the 8 HEM hierarchy levels from STRING-DB
  experiments/
    coarsening_levels.py             Fixed-HEM experiment entrypoint (Ray Tune sweep)
    diffpool_experiment.py           Hybrid/Full DiffPool entrypoint (--full-mode toggles)
    fixed_supernodes_coarsening.py   Variant experiment
    tcga_cohort_and_tumor_classification.py / perf_*  Multi-task (cohort + tumor) variants
    dataset_information.py, random_features.py, rerun_models.py, run.sh
  analysis/                 Older post-hoc analysis scripts
  analysis_v2/              Newer, modular analysis pipeline (parsers, confusion matrices,
                             error analysis, training curves, hyperparameter sensitivity) —
                             run via `python -m scripts.analysis_v2.main --path-output outputs`
  utils/                    Pan-cancer cohort index helpers

data/
  string_data/               STRING-DB network generation notebooks/scripts
  example_data/               Example datasets (networks + TCGA BRCA subtype task)
  (`.gitignore` excludes the real `/data`, `/results`, `/outputs`, `/tests`, `/artifacts` dirs)

dev/                        Scratch/dev scripts, not part of the installable package

outputs/                    Experiment run outputs (metrics, confusion matrices, model
                             checkpoints), directory-per-config, resume-safe

Documentation:
  README.md                 Install + quickstart instructions
  ARCHITECTURE.md            Detailed architecture reference for all 3 approaches (primary
                             technical doc — read this for implementation-level detail)
  AGENTS.md                  Contributor/agent quick-reference (setup, layout, gotchas)
  plan.md / plan.MD          Full dissertation outline (motivation, methodology, results,
                             discussion) — the "why" behind the code
  hybrid_levels.md            Deep-dive on how a "hybrid level" works inside DiffPoolLayer
  analysis-approaches.MD, apresentacao_arquitetura.md, presentation.md   Supporting/auxiliary
                             notes and presentation material
```

## Core library components (`src/pooling_genomic/`)

- **`coarsening.py`** — Heavy Edge Matching (HEM), adapted from xbresson's spectral graph
  pooling code. Greedily matches vertex pairs by heaviest edge weight and contracts them,
  producing roughly a 2:1 node reduction per level. Used to pre-compute 8 hierarchy levels
  (14K → 7K → 3.5K → 1.8K → 884 → 442 → 221 → 111 nodes).
- **`networks.py`** — Builds the base PyG `Data` graph from the STRING-DB edge list
  (`get_pyg_data()`), and loads the pre-computed HEM levels from disk
  (`load_graph_levels()`, `load_coarse_edges_for_diffpool()`).
- **`datasets.py`** — Path-based dataset dispatch (`get_genomic_classification_dataset()`),
  TCGA expression loading with `LabelEncoder` labels and train/val/test splitting.
- **`models.py`** (largest file, ~800 lines) — All model architectures:
  - `GNNPooling` / `build_coarsening_model()` — the Fixed-HEM model (ChebConv + scatter pooling
    per level, optional learnable per-node importance weights).
  - `DiffPoolLayer` / `DiffPoolGNN` / `build_diffpool_model()` — the Hybrid/Full DiffPool model.
    A `DiffPoolLayer` operates in **hybrid mode** when given HEM parents (learned ChebConv
    embedding + fixed HEM scatter pooling, no aux losses) or **full mode** (learned soft
    assignment `S`, `Sᵀ Z` pooling, `Sᵀ A S` adjacency pooling, link-prediction + entropy aux
    losses).
  - `FCModel` — shared MLP classifier head used by all approaches.
  - `CohortAndTumorLoss` — multi-task loss variant (cohort-of-origin + tumor/normal).
- **`engines.py`** — `train_epoch_clf` / `evaluate_clf` (single-task), `train_cohort_tumor_clf`
  (multi-task); handles optional L1 regularization on node importances and DiffPool auxiliary
  losses.
- **`saliency.py`** — Guided backpropagation saliency maps for interpretability, and a
  KNN-based feature-ablation comparison (ranked genes vs. random genes) used for explainability
  analysis.
- **`settings.py`** — Minimal Pydantic settings object (`path_data`, `path_results`), configurable
  via `POOLING_GENOMIC_*` env vars or a `.env` file.

## Why Hybrid DiffPool switches at 500 nodes

Inside `DiffPoolLayer` (`src/pooling_genomic/models.py`), hybrid mode stays active as long as
pre-computed HEM coarse edges are attached to the layer. Each forward pass checks the current
node count and drops the HEM structure once it falls at or below `dense_threshold`:

```python
if n_nodes <= self.dense_threshold and layer._coarse_edge_index is not None:
    layer._coarse_edge_index = None
    layer._parents = None
```

Once cleared, the next forward pass falls through to full DiffPool for that (and all
subsequent) levels: a learned soft assignment `S = softmax(pool_gnn(x))`, with **dense**
adjacency pooling `A' = Sᵀ A S`.

**Why the switch has to happen at all** — `Sᵀ A S` requires materializing a dense `n × n`
adjacency matrix. At the full graph size (~14K nodes) that's ~196M elements per sample; the
project's own analysis (`plan.md` §6.3, `hybrid_levels.md`) measured this as ~23 GiB for the
assignment tensor alone at level 0, which OOMs. Dense DiffPool operations are only tractable
once the graph has already been shrunk to a "small" size — hence hybrid mode uses cheap HEM
scatter pooling (O(n) sparse) for the large early levels, and only switches to the expensive
learned dense pooling once it's safe to do so.

**Why specifically 500** — at n=500, the dense adjacency is 500×500 = 250,000 elements, trivial
even batched on a normal GPU (the project's setup: RTX 3090 Ti). It's a comfortably-safe cutoff
picked to make dense operations tractable while still leaving the last level(s) of the
hierarchy free to use learned assignment instead of the fixed HEM parent mapping. It is exposed
as a CLI default (`--dense-threshold`, default 500 in `diffpool_experiment.py`) rather than
derived from a formula, and it was **not** part of the dissertation's hyperparameter sweeps
(only `n_hybrid`, `lr`, `weight_decay`, and the auxiliary-loss weights were tuned; `dense_threshold`
stayed fixed at 500 throughout). In short: the *purpose* of the threshold is to bound the
hybrid→full transition so dense adjacency pooling only ever runs on graphs small enough to fit
in memory; 500 itself is an engineering safety margin below the observed OOM boundary near 14K
nodes, not a value optimized for accuracy.

## How experiments are run

Both experiment entrypoints follow the same pattern: iterate over a hyperparameter grid,
optionally tune with Ray Tune (`--tune`), train a final model with the best config, and write
results to a resume-safe, per-config output directory (existing dirs are skipped).

```sh
# Fixed HEM
python scripts/experiments/coarsening_levels.py <data_dir> <levels_dir> \
  --tune --max-n-levels 7 --n-cycles 5 --num-samples 8 --path-output outputs --n-holdouts 5

# Hybrid DiffPool
python scripts/experiments/diffpool_experiment.py <data_dir> <levels_dir> ...

# Full DiffPool (no HEM pre-computation needed beyond the base graph)
python scripts/experiments/diffpool_experiment.py <data_dir> <levels_dir> --full-mode ...
```

Ray Tune requires **absolute paths**. Training length is controlled by `--n-cycles` via the
cosine-annealing-warm-restarts schedule (`T_0 * (1 - T_mult^n_cycles) / (1 - T_mult)`).
Multiple holdout repetitions use seeds derived from `np.random.default_rng(123)` for
reproducibility.

## Setup

```sh
conda create -n pooling_genomic python=3.10
conda install pytorch==1.12.1 torchvision torchaudio cudatoolkit=11.3 -c pytorch  # match your CUDA
# install PyTorch Geometric per its own instructions
conda install pytorch-scatter -c pyg
pip install -r requirements.txt
pip install -e .   # required — package lives under src/
```

Verify with `python scripts/experiments/coarsening_levels.py --help`. There is no formal
automated test suite; this help invocation is the smoke test.

## Where to look for more detail

- **`ARCHITECTURE.md`** — the authoritative, detailed technical reference (model diagrams,
  hybrid-vs-full comparison tables, package responsibilities). Read this first for
  implementation questions.
- **`plan.md`** — the full dissertation narrative: motivation, related work, methodology,
  hyperparameter grids, results tables, and discussion of *why* learned pooling underperforms
  on this task.
- **`hybrid_levels.md`** — focused explanation of the hybrid-mode mechanics inside
  `DiffPoolLayer.forward()`.
- **`AGENTS.md`** — terse contributor cheat-sheet (setup, commands, gotchas) aimed at coding
  agents working in this repo.
- **`changes-from-claude.md`** — running log of code changes made in collaboration with Claude
  (what changed, why, and how it was verified). Check here before assuming the code still
  matches the "as originally written" description above.
