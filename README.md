# GNN Clustering

A PyTorch Geometric-based project for building and training Graph Neural Networks for clustering and classification tasks.

## Setup

### Prerequisites
- Python 3.9+
- Conda (recommended)

### Installation

1. **Create conda environment:**
```bash
conda env create -f environment.yml
conda activate gnn-clustering
```

2. **Install the package in development mode:**
```bash
pip install -e .
```

3. **Verify installation:**
```bash
python -c "import torch; import torch_geometric; print('✓ Installation successful')"
```

## Project Structure

```
.
├── src/gnn_clustering/          # Main package
│   ├── models.py               # GNN model definitions
│   ├── networks.py             # Network utilities and graph processing
│   ├── datasets.py             # Dataset loading and preparation
│   ├── engines.py              # Training and evaluation loops
│   └── utils.py                # Helper functions
├── scripts/
│   └── train.py                # Example training script
├── requirements.txt            # Python dependencies
├── environment.yml             # Conda environment specification
├── setup.py                    # Package setup configuration
└── README.md                   # This file
```

## Quick Start

```bash
# Activate the environment
conda activate gnn-clustering

# Run the example training script
python scripts/train.py
```

## Development

To add new models:
1. Edit `src/gnn_clustering/models.py`
2. Add corresponding utilities in other modules as needed
3. Create test scripts in `scripts/`

## Dependencies

- **torch**: Deep learning framework
- **torch_geometric**: GNN layers and utilities
- **networkx**: Graph manipulation
- **pandas**: Data handling
- **scikit-learn**: ML utilities

## License

MIT
