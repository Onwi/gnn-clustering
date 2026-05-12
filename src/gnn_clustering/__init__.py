"""GNN Clustering - Graph Neural Networks for clustering and classification."""

__version__ = '0.1.0'

from .models import GCNModel
from .networks import get_graph_data
from .datasets import GraphDataset

__all__ = [
    'GCNModel',
    'get_graph_data',
    'GraphDataset',
]
