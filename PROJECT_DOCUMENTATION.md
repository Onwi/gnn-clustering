# pooling_genomic — Project Documentation

This document is derived entirely from reading the source code, the CLI scripts, the
`data/` directory layout, and empirical runs performed directly on this machine. It
deliberately does not summarize or rely on any of the repository's existing `.md`
files (`README.md`, `ARCHITECTURE.md`, `plan.md`/`plan.MD`, `changes-from-claude.md`,
`claude-explanation.md`, `AGENTS.md`, `hybrid_levels.md`) — where those documents make
claims not verifiable from code or from a run on this machine, this document either
omits the claim or states it as unverified.

## 1. Overview

This is a research codebase for classifying TCGA (The Cancer Genome Atlas) gene-expression
samples using graph neural networks, where the graph is a fixed STRING-DB protein-protein
interaction network shared across all samples and each sample's per-gene expression value
is the node feature. Because the graph has 14,133 nodes, the models rely on hierarchical
graph *pooling* (coarsening) to make classification over such a large graph tractable.

The codebase implements and compares three pooling strategies, all built on the same
`ChebConv` (Chebyshev spectral graph convolution, K-hop) building block:

- **Fixed HEM** (`src/pooling_genomic/coarsening.py`, `models.py:build_coarsening_model`) —
  graph coarsening is precomputed once, offline, via Heavy Edge Matching (HEM), producing a
  fixed hierarchy of increasingly coarse graphs (8 levels). Pooling at train time is a
  parameter-free `scatter(reduce="sum")` over the precomputed parent assignment. This is
  attributed in code comments/paths to a prior author ("Thomas"/`tvfontanari`) and is
  treated as a baseline the newer work compares against.
- **Full DiffPool** (`models.py:DiffPoolGNN`, `full_mode=True`) — cluster assignments are
  *learned* end-to-end via a differentiable soft-assignment matrix (Ying et al.-style
  DiffPool), with no precomputed structural prior.
- **Hybrid DiffPool** (`models.py:DiffPoolGNN`, `full_mode=False`) — early levels reuse the
  precomputed HEM coarsening (structural prior, cheap), and only the levels below a node-count
  threshold switch to learned DiffPool assignments.

The classification tasks seen in the data/scripts are: TCGA molecular subtype
classification (BRCA-only example dataset), tumor cohort classification (which of 16 TCGA
cancer types a sample belongs to), and tumor-vs-normal (`sample_type`) classification,
including a multi-task variant that predicts cohort and tumor/normal simultaneously.

## 2. Repository layout

```
src/pooling_genomic/       importable package (setup.py: packages found under src/)
  models.py                 all model architectures (912 lines) — see §4
  engines.py                 train_epoch_clf / evaluate_clf / train_cohort_tumor_clf / evaluate_cohort_tumor_clf
  datasets.py                 Dataset classes + get_genomic_classification_dataset dispatcher — see §3
  networks.py                 STRING-DB CSV -> PyG Data conversion, coarsening-level loading
  coarsening.py                Heavy Edge Matching (HEM) graph coarsening (adapted from a public ChebNet lab: see file header credit to xbresson/CE7454_2019)
  saliency.py                  guided-backprop saliency analysis utilities (standalone, not wired into diffpool_experiment.py)
  settings.py                  pydantic BaseSettings: path_data / path_results, overridable via POOLING_GENOMIC_* env vars or .env
  utils.py                     get_lr, plot_confusion_matrix, savefig, write_json, build_data_loaders

scripts/experiments/        CLI entry points, one process per invocation
  diffpool_experiment.py       Full/Hybrid DiffPool training + Ray Tune — the actively used script this session; see §5
  coarsening_levels.py         Fixed HEM training + Ray Tune, sweeping n_levels x weighted_pooling x use_convs
  fixed_supernodes_coarsening.py  Fixed HEM variant that always coarsens to the full fixed hierarchy but sweeps which level learned ChebConv filters start at (first_level)
  tcga_cohort_and_tumor_classification.py       multi-task (cohort + tumor/normal) Fixed-HEM-style training
  perf_tcga_cohort_and_tumor_classification.py  same multi-task setup, but dispatches to either the "coarsening" or "fixed" supernode sweep via --experiment
  dataset_information.py       builds a CSV table of class counts (single- or multi-task metadata)
  random_features.py           KNN cross-validation baseline over random/ranked gene subsets (hardcoded /home/thomas/... paths — not runnable as-is)
  rerun_models.py               reloads a saved Fixed-HEM final_model.pt and re-evaluates it on a test set (hardcoded paths)
  run.sh                        example invocation of fixed_supernodes_coarsening.py / coarsening_levels.py with hardcoded /scratch/tvfontanari/... paths

scripts/analysis/           legacy (v1) result-analysis scripts, all with hardcoded /home/thomas/... or /scratch/tvfontanari/... paths — not runnable as-is on this machine
  analyze_performance_results.py   builds comparison tables/plots from Fixed-HEM and coarsening-sweep output dirs
  compare_fixed_vs_learned.py       parses output dir names via regex (nlevelsN_repR_wpoolW_convsC) to compare model families
  pan_cancer_vs_specific.py          pan-cancer vs. cohort-specific model comparison, dataset-size/imbalance influence
  single_task_vs_multi_task.py       imports from analyze_performance_results.py; compares single- vs multi-task results

scripts/analysis_v2/        newer, portable analysis package (relative imports, module docstrings, no hardcoded paths)
  main.py                       entry point: `python -m scripts.analysis_v2.main --path-output outputs`
  parsers.py                     scans any output dir for */final_model_results/{metrics.csv,model_configs.json,predictions.csv,outputs.csv}
  confusion_matrix.py             side-by-side confusion matrices, per-class precision/recall/F1
  error_analysis.py                prediction-agreement / disagreement heatmaps between two models
  hyperparam_sensitivity.py         accuracy vs. hyperparameter scatter plots
  training_curves.py                per-epoch train/test loss & accuracy curves (needs multi-epoch metrics.csv)

scripts/utils/
  pan_cancer_cohort_indices.py, unseen_cohorts_tumor_prediction_indices.py — generate fixed
  train/val/test index CSVs per holdout/cohort so different experiments can reuse identical
  splits (consumed via --path-indices / PCRunIndicesLoader)
  run_nn_cs_with_pc_indices.sh, run_nn_unseen_cohorts_with_indices.sh — example invocations,
  hardcoded /home/thomas/... paths

scripts/generate_graph_levels.py   one-off script: builds the 8-level HEM hierarchy from the
                                     raw STRING-DB CSV + a dataset's gene list, writes
                                     edge_index_lvl{N}.pt / edge_weight_lvl{N}.pt / parents_lvl{N}.pt

dev/
  benchmark_pooling_memory.py   standalone GPU memory/wall-time micro-benchmark of DiffPoolGNN
                                  (full_mode) on a synthetic graph sized like the real STRING-DB
                                  graph — no dataset needed; added/used this session (see §7, §9)
  gnn_cluster_pool.py            early exploratory script that hand-builds a fixed 4-level
                                   coarsening model directly (predates the general
                                   build_coarsening_model abstraction) — not used by any CLI script
  dev.py                          7-line settings sanity check

data/
  example_data/tcga_brca_subtypes_classification/    small BRCA-only dataset (1,206 samples per
    sample_metadata.csv), used for fast smoke tests
  example_data/networks/levels/                        matching 8-level HEM hierarchy for the
    example dataset's gene set
  string_data/data/tcga_cohorts_and_tumor_classification/  the full pan-cancer dataset:
    sample_metadata.csv with columns `cohort` (16 unique values) and `sample_type` (2 values:
    "Primary Tumor" / "Solid Tissue Normal"), 7,709 samples total (verified this session)
  string_data/data/networks/                            raw STRING-DB v11.5 files
    (9606.protein.links/.info/.aliases.v11.5.txt), plus pre-filtered edge lists
    stringdb_top100pc.csv / top10pc.csv / top1pc.csv, and two HEM hierarchies:
    levels/ and levels_new/ (each with edge_index/edge_weight/parents_lvl0..7.pt)
  string_data/network_generation/coarsening.py           a second copy of the HEM coarsening
    code (identical algorithm) kept alongside the raw STRING data, separate from
    src/pooling_genomic/coarsening.py

setup.py, requirements.txt   package metadata + pinned pip dependencies (torch/torch_geometric
                               are conda-only and not listed here — see §7)
remote-test-commands.txt      env-setup + validation command reference (updated this session)
```

## 3. Data

### 3.1 Datasets and the `get_genomic_classification_dataset` dispatcher

`src/pooling_genomic/datasets.py:13-40` dispatches on substrings in `path_dataset` to one of
four dataset constructors:

| path substring | function | default `metadata_column` | notes |
|---|---|---|---|
| `tcga_cohort_classification` | `get_tcga_cohort_classification_datasets` | n/a (`TCGACohorts` always uses `cohort`) | |
| `tcga_brca_subtypes_classification` | `get_tcga_classification_datasets` | hardcoded `'Subtype_mRNA'` | |
| `tcga` + `tumor_prediction` | `get_tcga_classification_datasets` | hardcoded `'sample_type'` | |
| `tcga_cohorts_and_tumor_classification` | `get_tcga_classification_datasets` | **`'cohort'`** (fixed this session — see §8.2) | 16 classes, 7,709 samples |

Each per-sample file is a per-gene expression CSV (`sample_metadata.csv` maps a TCGA sample ID
to metadata columns; `samples/TCGA/.../<sample_id>.csv` holds the actual per-gene values, one
column). `TCGADataset.__getitem__` (`datasets.py:249-262`) reads that CSV, converts to numeric,
and returns a `(n_genes,)` float tensor plus an integer-encoded label — there is no
normalization step visible in this code path (any log/TPM normalization, if applied, must have
happened when the CSV files were generated, not at load time).

Splits are computed with `random_split`/index slicing at `train_proportion=0.6`,
`validation_proportion=0.2` (i.e. 60/20/20), seeded by a `random_state` that
`diffpool_experiment.py:main()` derives per-holdout-rep from `np.random.default_rng(seed=123)`.

`PCRunIndicesLoader` (`datasets.py`, used via `--path-indices`) allows loading a fixed,
externally-generated train/val/test index split (produced by `scripts/utils/*_indices.py`)
instead of a fresh random split — used to guarantee identical splits across different
experiment scripts/cohort-specific runs.

### 3.2 The STRING-DB graph

`networks.get_pyg_data` (`networks.py:11-44`) reads a STRING-DB edge-list CSV
(`protein1,protein2,combined_score`), restricts nodes to the gene list of a given dataset,
remaps gene names to contiguous integer node indices in the order of the input `genes` list,
and returns a PyG `Data` object with `edge_weight = combined_score / 1000`.

Verified this session with the full pan-cancer network at `stringdb_top100pc.csv`:
**14,133 nodes, 8,248,194 edges**.

### 3.3 "Levels" — the precomputed HEM hierarchy

`scripts/generate_graph_levels.py` runs `coarsening.HEM(adj, levels=8)` once on the full graph
and writes, per level `i` (0..7), three tensors: `edge_index_lvl{i}.pt`, `edge_weight_lvl{i}.pt`
(min-max scaled), and `parents_lvl{i}.pt` (a length-`N_i` int vector mapping each node at level
`i` to its parent's index at level `i+1`). `HEM` (`coarsening.py:74-157`) is a greedy Heavy Edge
Matching graph coarsening: it repeatedly pairs each unmatched node with its highest-weight
unmatched neighbor and merges the pair into one supernode, roughly halving the node count each
level. There are two precomputed hierarchies for the full graph, `data/string_data/data/networks/levels/`
and `.../levels_new/` (both 8 levels each), and a matching one for the BRCA example dataset's gene
set (`data/example_data/networks/levels/`).

`networks.load_coarse_edges_for_diffpool` / `load_graph_levels` (`networks.py:61-116`) load these
`.pt` files back for use by Fixed HEM (`GNNPooling`) and Hybrid DiffPool (`DiffPoolLayer`'s
`set_coarse_edges`/`_parents` path).

## 4. Model architecture

All three architectures ultimately produce `nn.Sequential(<pooling GNN stack>, FCModel(...))`
— a stack of graph-pooling layers that reduces `(batch, 14133 nodes, 1 feature)` down to a
small flattened vector, followed by a plain fully-connected classifier head (`FCModel`,
`models.py:167-238`: linear -> batchnorm -> ReLU -> dropout per hidden layer, plain linear last
layer).

### 4.1 Fixed HEM (`GNNPooling`, `models.py:66-165`)

For each precomputed level `i`: optionally apply a `ChebConv(in_ch, out_ch, K)` (channels grow
`1, 2, 4, ..., max_filters=32`; `use_convs=False` skips the conv entirely, i.e. identity), then
optionally multiply by a learned per-node importance weight (`weighted_pooling=True`,
`node_importances[i]`), ReLU, then `scatter(reduce="sum")` by the precomputed `parents_lvl{i}`
to merge each matched pair into its supernode. `n_levels=0` is handled specially in
`build_coarsening_model` (`models.py:356-369`): it skips the GNN entirely and feeds the raw
14,133-dim input straight into `FCModel` — this degenerate case is handled correctly (unlike
Full DiffPool's equivalent case before this session's fix — see §8.1).

`fixed_supernodes_coarsening.py`'s model variant (`get_fixed_supernodes_convs_list`,
`models.py:245-267`) always coarsens through the *entire* fixed 8-level hierarchy, but only
attaches learned `ChebConv` filters starting at a configurable `first_level`; levels before that
use `conv=None` (identity + scatter only).

### 4.2 Full / Hybrid DiffPool (`DiffPoolLayer`, `DiffPoolGNN`, `models.py:394-780`)

`DiffPoolLayer.forward` (`models.py:429-505`):
1. `z = ReLU(embed_gnn(x, edge_index, edge_weight))` — a `ChebConv(in_ch, hidden_ch, K)` embedding.
2. **Hybrid-mode branch** (`self._parents is not None`, i.e. an early Hybrid-mode level with
   precomputed HEM parents attached via `set_coarse_edges`): pool via
   `scatter(z, parents, reduce='mean')` — no learned assignment, identical mechanism to Fixed
   HEM's scatter step. `aux = {'link_pred_loss': 0.0, 'entropy_loss': 0.0}`.
3. **Full-mode / learned branch**: a second ChebConv, `pool_gnn = ChebConv(in_ch, max_clusters, K)`,
   produces raw assignment logits; `k = clamp(ceil(n * sigmoid(logit_pool_ratio)), 2, max_clusters)`
   determines how many of the `max_clusters` output columns are actually used this layer (`ratio`
   is a learned scalar per layer, initialized so `sigmoid(0)=0.5`); `S = softmax(logits[:, :, :k])`.
   Then `X' = S^T Z` (`torch.bmm`), and the coarsened adjacency `A' = S^T A S` is computed via a
   per-sample sparse-dense matmul (`torch.sparse.mm`) rather than materializing a dense `(n, n)`
   adjacency per sample — this sparse-matmul approach and a geometric per-level cluster-count
   schedule (`_compute_cluster_schedule`, `models.py:518-547`, so a 3-level full-mode stack goes
   e.g. `14133 -> 1856 -> 244 -> 32` instead of `14133 -> 32 -> 32`) were both added in the commit
   immediately preceding this session (`9108f7f`, "Fix Full DiffPool cluster collapse and dense
   adjacency pooling") and are described in the pre-existing `changes-from-claude.md` (not
   otherwise used as a source here).
4. Auxiliary losses: `link_pred_loss = MSE(normalized dense A, normalized S S^T)` (reconstruction of
   the adjacency from the assignment matrix), `entropy_loss = mean(-sum(S * log(S)))` (encourages
   near-one-hot assignments).

`DiffPoolGNN.__init__` (`models.py:571-668`) builds `levels` `DiffPoolLayer`s (Full mode:
`levels = n_levels`, default 3 if unspecified; Hybrid mode: `levels = n_hybrid + 1`), with
per-level channel widths from `_compute_channel_list` (`models.py:508-515`, geometric doubling
`start_channels, 2*start_channels, ..., max_filters`). It raises a `ValueError` if the resulting
`levels < 1` (added this session — see §8.1).

**Pre-pooling encoder (`PrePoolingEncoder`, `models.py:518-544`, added this session — see §8.3):**
in full mode only, before the first `DiffPoolLayer`, raw per-node scalar features
`(batch, 14133, 1)` are passed through a pointwise `Conv1d(1, encoder_channels, kernel_size=1)`
(a per-node linear projection, no cross-node mixing) followed by `encoder_layers - 1` further
`ChebConv(encoder_channels, encoder_channels, K)` layers (with cross-node message passing). This
enriches the otherwise 1-dimensional raw input before cluster-assignment learning begins. Default
`encoder_channels=16, encoder_layers=2`. Hybrid mode does not use this encoder (`self.encoder =
None` when `full_mode=False`) since its early levels already have a structural prior from HEM.

`build_diffpool_model` (`models.py:692-...`) wires a `DiffPoolGNN` to an `FCModel`, with the
classifier's input dimension computed as `max_clusters * last_channels`, where `last_channels =
min(start_channels * 2**levels, max_filters)` and `start_channels = encoder_channels` in full
mode (`1` in hybrid mode) — this line had to be updated in step with the encoder addition (see
§8.3), since it assumes the final `DiffPoolLayer` always pools down to exactly `max_clusters`
nodes, which only holds when the incoming node count at that layer is large relative to
`max_clusters` (empirically true at the real 14,133-node scale for `n_hybrid<=3`, not
necessarily true for small synthetic graphs — verified this session, see §8.1 and §9).

### 4.3 Multi-task cohort+tumor model (`CohortAndTumorLoss`, `build_gnn_pooling_tumor_and_cohort_clf`, `models.py:800-...`)

A separate architecture (Fixed-HEM-style pooling trunk only, not wired to DiffPool) with a
shared `FCModel` trunk feeding two heads: a cohort classifier (`CrossEntropyLoss`) and a
tumor/normal binary head (`BCEWithLogitsLoss`, `pos_weight` computed from class imbalance).
Combined loss is the unweighted sum. Used only by `train_cohort_tumor_clf`/
`evaluate_cohort_tumor_clf` (`engines.py`) and the two `*cohort_and_tumor_classification.py`
scripts — not used by `diffpool_experiment.py`.

## 5. Training pipeline / CLI

### 5.1 `scripts/experiments/diffpool_experiment.py` (the actively-used script this session)

```
python scripts/experiments/diffpool_experiment.py <path_dataset> <path_levels> [options]
```

Key arguments (full list in `parse_args`, `diffpool_experiment.py:219-275`):

| flag | default | meaning |
|---|---|---|
| `path_dataset`, `path_levels` | required | dataset dir; precomputed HEM levels dir (only read in hybrid mode) |
| `--path-network` | derived from `PoolingGenomicSettings` | STRING-DB edge-list CSV |
| `--full-mode` | off | Full DiffPool (no HEM priors) vs. Hybrid DiffPool |
| `--n-hybrid` | 2 | max depth to sweep (see below) |
| `--n-hybrid-start` | 0 | start of the depth sweep; **now clamped to >=1 automatically in full mode** (fixed this session, §8.1) |
| `--max-filters`, `--max-clusters` | 32, 32 | channel/cluster-count ceilings |
| `--encoder-channels`, `--encoder-layers` | 16, 2 | full-mode pre-pooling encoder width/depth (added this session, §8.3) |
| `--dense-threshold` | 500 | hybrid mode: node count below which it switches to learned DiffPool |
| `--device` | cpu | `cpu`/`cuda` |
| `--tune` | off | Ray Tune hyperparameter search vs. a single fixed-hyperparameter run |
| `--lr`, `--weight-decay`, `--lambda-link-pred`, `--lambda-entropy` | non-tune-mode overrides | see §5.2 for defaults |
| `--cpu-per-trial`, `--gpu-per-trial` | 1, 0.1 | Ray Tune per-trial resource request — `--gpu-per-trial 0.1` implies Ray will try to pack ~10 trials per GPU; **empirically this OOMs** at this dataset's scale (verified this session) unless raised toward 1.0 |
| `--num-samples` | 1 | Ray Tune trial count |
| `--batch-size` | 64 | **empirically must be much lower on a 24GB GPU** — see §7.4 |
| `--max-epochs` | 50 | upper bound passed to `ASHAScheduler`; actual epoch count per run is computed from `--n-cycles`, not this flag, in the code paths actually exercised (see below) |
| `--n-cycles` | 5 | warm-restart cycle count; epoch count derived as `T_0*(1 - T_mult**n_cycles)/(1 - T_mult)` with `T_0=1, T_mult=2` (i.e. `2**n_cycles - 1` epochs, using the tuning-loop's `n_cycles-1` adjustment where applicable — see the epoch-formula inconsistency noted in §10) |
| `--path-output` | `./outputs` | output root |
| `--n-holdouts` | 5 | independent holdout repetitions |
| `--path-indices`, `--cohort-indices` | none | use a precomputed fixed split (`PCRunIndicesLoader`) instead of a fresh random split |
| `--use-train-set-weights` | off | class-weighting for `CrossEntropyLoss` |

`main()` (`diffpool_experiment.py`) loops `n_holdouts` times with a per-rep `random_state` drawn
from `np.random.default_rng(seed=123)`; `run_holdout` then loops `n_hybrid` from
`n_hybrid_start` to `min(max_n_levels, n_hybrid+1)`, skipping any output directory
(`<path-output>/diffpool_{full|hybrid}{n_hybrid}_rep{rep}/`) that already exists.

For each `(rep, n_hybrid)`: if `--tune` is off, `train_and_validate_model` runs a single
fixed-hyperparameter training+test pass directly. If `--tune` is on, an `ASHAScheduler`
(`max_t=grace_period=args.max_epochs`, i.e. effectively disabled since the trainable's own loop
always finishes before that budget) drives a Ray Tune `Tuner` (`metric="loss", mode="min"`) over
`train_and_validate_model` as the trainable, then `test_tuned_model` re-trains
`results.get_best_result(scope="last")`'s config on train+val combined and evaluates once on the
held-out test set. Ray's live status display can show a diverged (`loss=nan`) trial as "current
best" during the run — verified this session that the final `get_best_result()` call (backed by
pandas' `idxmin()`, which skips NaN by default) correctly ignores NaN trials in the actual
selection.

Output per `(rep, n_hybrid)`, under `final_model_results/`: `metrics.csv`, `predictions.csv`,
`outputs.csv` (raw per-class scores), `confusion_matrix.{jpg,pdf}`, `model_configs.json`,
`final_model.pt` (`analyze_final_model_results`, `diffpool_experiment.py`). Ray Tune trial
artifacts (per-trial checkpoints, `progress.csv`, TensorBoard event files) live under
`<path-output>/.../ray_results/`.

### 5.2 Hyperparameter defaults (fixed this session, §8.4)

Non-tuning-mode defaults (`build_hp_config`, `diffpool_experiment.py:40-63`):
`lr=0.0008` (was `0.05`), `weight_decay=0.01`, `lambda_link_pred=0.001`, `lambda_entropy=0.001`,
`eta_min=1e-5`, `T_0=1`, `T_mult=2`.

`--tune` search space: `lr, weight_decay ~ loguniform(1e-4, 1e-1)`; `lambda_link_pred,
lambda_entropy ~ loguniform(1e-5, 1e-3)` (upper bound narrowed from `1e-1` this session).

### 5.3 `scripts/experiments/coarsening_levels.py` (Fixed HEM)

Structurally near-identical to `diffpool_experiment.py` (same `build_hp_config` /
`train_and_validate_model` / Ray Tune / holdout pattern), but sweeps
`n_levels x weighted_pooling in {False, True} x use_convs in {False, True}` (skipping
`weighted_pooling`/`use_convs=True` when `n_levels==0`, since there is no GNN to weight/convolve),
calling `build_coarsening_model` instead of `build_diffpool_model`. Output directory naming:
`nlevels{N}_rep{R}_wpool{W}_convs{C}`.

## 6. Analysis tooling

`scripts/analysis_v2/` is the current, portable analysis package: `parsers.parse_all_results`
scans any `<path-output>/*/final_model_results/` tree, reads `metrics.csv` +
`model_configs.json` per run into one DataFrame, and `main.py` drives confusion matrices,
per-class precision/recall/F1, training curves, prediction-disagreement heatmaps, and
hyperparameter-sensitivity scatter plots from that DataFrame plus the saved `predictions.csv`.

`scripts/analysis/` (v1) covers similar ground (Fixed-HEM vs. learned-model comparison tables,
pan-cancer vs. cohort-specific comparisons, single- vs. multi-task comparisons) but every file
hardcodes absolute paths under `/home/thomas/...` or `/scratch/tvfontanari/...` — these are not
runnable on this machine without editing the path constants first.

## 7. Environment setup (verified working on this machine this session)

```bash
conda create -n pooling_genomic python=3.10 -y
conda activate pooling_genomic
conda install pytorch==1.12.1 torchvision torchaudio cudatoolkit=11.3 -c pytorch -y
conda install "mkl=2021.4.0" -y          # see §7.1
conda install pytorch-scatter pyg -c pyg -y   # torch_geometric ("pyg") — see §7.3
pip install -r requirements.txt
pip install "setuptools<81"              # see §7.2
pip install -e .
```

### 7.1 mkl/numpy ABI mismatch

A fresh `conda install pytorch==1.12.1 ... -c pytorch` on this machine resolves `mkl` from the
`defaults` channel to a version (`2025.0.0` at the time) that is ABI-incompatible with the
PyTorch 1.12.1 build: `import torch` fails with `undefined symbol: iJIT_NotifyEvent`. Pinning
`mkl=2021.4.0` (which also pulls `numpy` back to `1.24.3`) resolves it. This also matters for
`pandas`: a numpy that's too new for the installed pandas build raises `ValueError: numpy.dtype
size changed, may indicate binary incompatibility`.

### 7.2 `setuptools` / `pkg_resources`

`ray==2.2.0`'s `ray.air._internal.remote_storage` does `from pkg_resources import packaging` at
import time. `setuptools>=81` removed `pkg_resources`. Fix: `pip install "setuptools<81"`
(verified working at `80.10.2`, with a deprecation warning that is otherwise harmless for this
use).

### 7.3 `torch_geometric` is a required, undocumented dependency

`src/pooling_genomic/models.py` and `networks.py` import `torch_geometric` (`ChebConv`,
`global_mean_pool`, `Data`, `from_networkx`, `to_dense_adj`, `dense_to_sparse`,
`from_scipy_sparse_matrix`), but it is listed in neither `requirements.txt` nor (originally)
the environment-setup instructions — only `pytorch-scatter` was. Fix: install `pyg` (the
`torch_geometric` conda package) from the `pyg` channel, matched to the installed torch/cuda
build (verified: `pyg=2.5.2` against `pytorch=1.12.1` / `cuda 11.3`).

### 7.4 GPU memory / batch size

Measured on an RTX 3090 Ti (24GB) via `dev/benchmark_pooling_memory.py --levels 3` (full-mode,
`n_hybrid=3`, the real 14,133-node/8.2M-edge graph):

| config | batch size | forward+backward peak |
|---|---|---|
| pre-encoder (before §8.3) | 32 | OOM |
| pre-encoder | 16 | 15.6 GB (safe) |
| pre-encoder | 8 | 9.7 GB (safe) |
| post-encoder (after §8.3) | 8 | OOM |
| post-encoder | 4 | 15.3 GB (safe) |
| post-encoder | 2 | 10.0 GB |
| post-encoder | 1 | 7.4 GB |

The pre-pooling encoder (§8.3, §4.2) raises the first `DiffPoolLayer`'s ChebConv input width
from 1 to `encoder_channels` (16 by default); since that layer still operates on the full
unpooled 14,133-node/8.2M-edge graph, its message-passing memory cost scales with that channel
count, roughly quartering the safe batch size on this GPU (16 -> 4).

## 8. Known issues and fixes applied this session

All four fixes below are in the current working tree (uncommitted at the time of writing this
document) on top of `HEAD` (`9108f7f`, "Fix Full DiffPool cluster collapse and dense adjacency
pooling", which pre-dates this session).

### 8.1 Full-mode `n_hybrid=0` crash

**Symptom:** `RuntimeError: mat1 and mat2 shapes cannot be multiplied (16x14133 and 32x256)`,
deep inside `FCModel.forward` (`models.py:236`), whenever `run_holdout`'s depth sweep reached
`n_hybrid=0` in full mode (the default `--n-hybrid-start=0` always includes it).

**Root cause:** in full mode, `n_levels=n_hybrid=0` makes `DiffPoolGNN.__init__` build zero
`DiffPoolLayer`s (`for i in range(levels)` with `levels=0`), so the model does no pooling at
all and `forward()` returns the raw, un-pooled 14,133-dim feature vector — but
`build_diffpool_model` sizes `FCModel`'s input assuming at least one pooling layer always ran
(`last_channels = min(2**levels, max_filters)`, which degenerates to `1` at `levels=0`, giving a
32-dim expected input against the actual 14,133-dim tensor).

**Fix:** `DiffPoolGNN.__init__` now raises a clear `ValueError` if the computed `levels < 1`
(`models.py`), and `run_holdout` (`diffpool_experiment.py`) clamps
`n_hybrid_start = max(args.n_hybrid_start, 1)` when `--full-mode` is set, so the invalid depth is
skipped automatically rather than reached. Verified: `n_levels=0` now raises immediately with a
descriptive message; `n_levels=1` and `n_levels=3` construct identically to before the fix
(unchanged cluster schedules); an end-to-end run with `--n-hybrid 1` and no `--n-hybrid-start`
override goes straight to `n_hybrid=1` and completes with exit code 0.

### 8.2 Missing `metadata_column` default for the full pan-cancer dataset

**Symptom:** `TypeError: get_tcga_classification_datasets() missing 1 required positional
argument: 'metadata_column'` whenever `tcga_cohorts_and_tumor_classification` was used without
an explicit `--metadata-column` CLI flag (as in `remote-test-commands.txt`'s section 3/4
commands).

**Root cause:** `get_genomic_classification_dataset`'s dispatcher (`datasets.py:36-40`) is the
only one of its four dataset branches that does not set a default `metadata_column` in `kwargs`
before calling `get_tcga_classification_datasets` (the BRCA and tumor-prediction branches both
do, at lines 23 and 30).

**Fix:** added `kwargs['metadata_column'] = 'cohort'` to that branch. Verified: `cohort` has
exactly 16 unique values across 7,709 samples in `sample_metadata.csv` (matching the "16
classes" scale this dataset is used at elsewhere), vs. `sample_type`'s 2 values — `'cohort'` is
the only column consistent with a 16-class task. Confirmed post-fix: `dataset.get_n_classes()
== 16`, train/val/test sizes `4625/1541/1543` (sums to 7,709, matching the 60/20/20 split).

### 8.3 Missing pre-pooling encoder (Full DiffPool)

See §4.2 for the mechanism. Added `PrePoolingEncoder` and wired it into `DiffPoolGNN`/
`build_diffpool_model` (full mode only). This required also generalizing
`_compute_channel_list` to accept a `start_channels` parameter (previously hardcoded to start
channel-doubling at `1`) and updating `build_diffpool_model`'s `last_channels`/`mlp_input_dim`
formula to account for the encoder's output width. Verified at real (14,133-node) scale:
`output_shape=(batch, 1024)` matches the expected `max_clusters(32) * last_channels(32)`.
Verified end-to-end on the BRCA smoke test (`n_hybrid=3`, 1 epoch): test accuracy rose from the
42.5-49.6% range (pre-fix, across several single-epoch holdout runs) to 57.9%. Real memory cost
of this change is documented in §7.4.

### 8.4 Gradient clipping and hyperparameter defaults/search space

`train_epoch_clf` (`engines.py`) had no gradient clipping at all. Added
`torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_max_norm)` (new parameter,
default `5.0`) between `loss.backward()` and `optimizer.step()`. Separately, the non-tuning-mode
default `lr` was `0.05` and the `--tune` search space for `lambda_link_pred`/`lambda_entropy`
was `loguniform(1e-5, 1e-1)`; both were changed as described in §5.2. (Note: `lr=0.05` is still
reachable via `--tune`'s `lr ~ loguniform(1e-4, 1e-1)` search space, and was empirically observed
this session to cause `loss=nan` divergence at a sampled `lr=0.0258` — see §9. Gradient clipping
bounds the gradient norm per step but does not by itself prevent divergence at too large a
learning rate.)

`coarsening_levels.py`'s Fixed-HEM training path (`train_and_validate_model`,
`coarsening_levels.py:79-221`) was not touched — it still defaults to `lr=0.05` (its own
`build_hp_config`, `coarsening_levels.py:44-76`) and does not call gradient clipping, since it
was out of scope for this session's Full-DiffPool-focused fixes.

## 9. Current empirical status

All numbers below were produced on this machine this session, after the fixes in §8, using
`--full-mode --n-hybrid 3 --n-hybrid-start 3` (i.e. exactly 3 learned pooling levels,
`14133 -> ~1856 -> ~244 -> 32`, matching the file-naming convention `diffpool_full3_rep*` used
throughout `remote-test-commands.txt`).

- **BRCA smoke test** (`data/example_data/tcga_brca_subtypes_classification`, 1,206 samples,
  `--n-cycles 1` = 1 epoch, `--batch-size 16`, 5 holdouts): test accuracy 42.5-49.6% range
  before the §8.3/§8.4 fixes; 57.9% (single holdout, 1 epoch) after.
- **Full pan-cancer, `--tune`** (`tcga_cohorts_and_tumor_classification`, 7,709 samples, 16
  classes, `--n-cycles 3`, `--batch-size 4`, `--gpu-per-trial 1`, 2 Ray Tune trials, 1 holdout):
  one trial diverged to `loss=nan` (`lr=0.0258`, `accuracy=5.6%` — essentially random for 16
  classes); the other reached `loss=2.05`, `accuracy=33.6%` (`lr=0.00169`). The final
  retrain-and-test step (7 epochs on train+val combined, using the non-diverged trial's config
  as selected by `get_best_result`) reached **test accuracy 43.2%, balanced accuracy 33.9%,
  loss 1.79**.

This is with far fewer epochs (7) than a fully converged run would need, and the `--tune` search
space currently only covers `lr, weight_decay, lambda_link_pred, lambda_entropy` — it does not
search over `encoder_channels`, `encoder_layers`, `max_clusters`, or `K`, all of which are fixed
CLI defaults (`16, 2, 32, 2`) rather than tuned per-trial. A leftover ad-hoc log file at
`compare.txt` (not otherwise referenced by any script) shows one historical run at 92%
validation accuracy alongside another at 24% with no labels distinguishing which configuration
produced which — suggestive that some configuration on this codebase reaches much higher
accuracy than what was reproduced this session, but not independently verifiable from that file
alone.

## 10. Open questions / observations

These are things noticed while reading the code, not confirmed as bugs — flagged for someone
with fuller project context to judge:

- **Inconsistent epoch-count formula between tuning and final-retrain.** Inside
  `train_and_validate_model` (used as the Ray Tune trainable), `n_cycles = max(args.n_cycles -
  1, 1)` (`diffpool_experiment.py:79`). Inside `train_and_test_model` (the final retrain on the
  tuner's best config), `n_cycles = args.n_cycles` directly (`diffpool_experiment.py:350`,
  approx.). For `--n-cycles 3` this gives 3 epochs during tuning trials but 7 epochs for the
  final retrain — same flag, two different epoch counts, in two functions that otherwise mirror
  each other closely. This same `-1` discrepancy exists identically in `coarsening_levels.py`
  (lines 98 vs. 371), so it's not specific to the DiffPool script.
- **`train_cohort_tumor_clf`/`evaluate_cohort_tumor_clf`, `TCGACohortsAndTumor`,
  `CohortAndTumorLoss`, `build_gnn_pooling_tumor_and_cohort_clf`** form a complete, separate
  multi-task pipeline that is not used by `diffpool_experiment.py` or `coarsening_levels.py` —
  only by the two `*cohort_and_tumor_classification.py` scripts, which use Fixed-HEM-style
  pooling only (no DiffPool integration exists for the multi-task loss).
- **`dev/gnn_cluster_pool.py`** hand-builds a fixed 4-level model by calling
  `networks.load_graph_level` four times individually; this predates (and is now redundant with)
  `build_coarsening_model`'s more general, parametrized construction. Not imported by anything
  else.
- **Most of `scripts/analysis/*.py`, `scripts/utils/*.sh`, `scripts/experiments/run.sh`,
  `scripts/experiments/random_features.py`, and `scripts/experiments/rerun_models.py`** hardcode
  absolute paths under `/home/thomas/...` or `/scratch/tvfontanari/...` from what appears to be a
  different machine/user than this one — none of these are directly runnable here without
  editing path constants first.
- **`src/pooling_genomic/coarsening.py` and `data/string_data/network_generation/coarsening.py`**
  are two separate copies of the same HEM algorithm in different locations; it's unclear from the
  code alone whether one is stale/superseded or they're intentionally kept in sync manually.
- **`--gpu-per-trial` default is `0.1`** in both `diffpool_experiment.py` and
  `coarsening_levels.py`, which tells Ray Tune it can schedule ~10 concurrent trials per GPU.
  Given a single trial's real memory footprint at this dataset's scale (§7.4), this default will
  reliably cause CUDA OOM once a second trial starts on a single-GPU machine unless
  `--gpu-per-trial` is raised (to `~1.0` to fully serialize) at the command line.
