# AGENTS.md

## Project

Pooling GNNs for Genomic Data Classification — research code comparing hierarchical GNN pooling approaches on TCGA genomic data.

## Setup

```sh
conda create -n pooling_genomic python=3.10
conda install pytorch==1.12.1 torchvision torchaudio cudatoolkit=11.3 -c pytorch  # adjust CUDA
conda install pytorch-scatter -c pyg
pip install -r requirements.txt
pip install -e .
```

Follow PyTorch Geometric install at https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html.

## Layout

| Path | Purpose |
|---|---|
| `src/pooling_genomic/` | Library package: models, coarsening, datasets, training engines, saliency, network loading |
| `scripts/experiments/` | Experiment entrypoints (coarsening_levels.py, fixed_supernodes_coarsening.py, etc.) |
| `scripts/generate_graph_levels.py` | Pre-compute hierarchical graph levels from STRING-DB |
| `scripts/analysis/` | Post-hoc performance analysis |
| `scripts/utils/` | Pan-cancer cohort index utilities |
| `dev/` | Scratch / dev scripts (not part of package) |

## Key architecture

- Models are built by `build_coarsening_model()` / `build_fixed_supernodes_coarsening_model()` in `src/pooling_genomic/models.py` — compose a `GNNPooling` module with an `FCModel` classifier.
- Coarsening via Heavy Edge Matching (HEM) in `coarsening.py` (adapted from xbresson).
- Graph levels are **pre-computed** and loaded from disk via `load_graph_levels()` in `networks.py`.
- Dataset selection is path-based: `get_genomic_classification_dataset()` inspects the path string to choose the dataset class.
- Settings via pydantic `BaseSettings`: env prefix `POOLING_GENOMIC_`, reads `.env` file.

## Commands

```sh
# Run experiment (no tuning, quick test)
python scripts/experiments/coarsening_levels.py <path_to_data> <path_to_levels> --max-n-levels 2 --n-cycles 1 --n-holdouts 1

# With hyperparameter tuning via Ray Tune
python scripts/experiments/coarsening_levels.py <path_to_data> <path_to_levels> --tune --max-n-levels 7 --n-cycles 5 --num-samples 8 --path-output outputs --n-holdouts 5

# Use --cuda for GPU, --device cuda also accepted
```

**Ray Tune requires absolute paths** (noted in README). Experiment output dirs use pattern: `nlevels{N}_rep{R}_wpool{bool}_convs{bool}`. Existing output dirs are **skipped** (resume-safe).

## Testing

No formal test framework. Verify setup with:
```sh
python scripts/experiments/coarsening_levels.py --help
```

## Gotchas

- `n-cycles` determines total epochs automatically via cosine annealing warm restarts formula: `T_0 * (1 - T_mult^n_cycles) / (1 - T_mult)`. During tuning, `n_cycles` is reduced by 1.
- `pip install -e .` required before running scripts (package `pooling_genomic` is under `src/`).
- `.gitignore` excludes `/data`, `/results`, `/outputs`, `/tests`, `/artifacts`.
- Graph levels must be pre-generated (run `generate_graph_levels.py` first or download pre-computed data).
- Multiple holdout repetitions iterate with `n_holdouts`, each with a random seed from `np.random.default_rng(123)`.
