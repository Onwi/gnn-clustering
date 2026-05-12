"""Graph network utilities and processing functions."""

from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx


def get_graph_data(
    genes: list,
    edge_list_path: Path,
    features: Optional[np.ndarray] = None,
    edge_weight_col: str = 'weight',
    source_col: str = 'source',
    target_col: str = 'target',
) -> Data:
    """Create PyTorch Geometric Data object from graph information.
    
    Parameters
    ----------
    genes : list
        List of node identifiers
    edge_list_path : Path
        Path to CSV file with edges
    features : np.ndarray, optional
        Node feature matrix [num_nodes, num_features]
    edge_weight_col : str
        Column name for edge weights
    source_col : str
        Column name for source nodes
    target_col : str
        Column name for target nodes
        
    Returns
    -------
    Data
        PyTorch Geometric Data object
    """
    # Read edges
    df_edges = pd.read_csv(edge_list_path)
    
    # Create node mapping
    mapping = {gene: idx for idx, gene in enumerate(genes)}
    
    # Filter edges to keep only those with both nodes in genes list
    keep_mask = (
        df_edges[source_col].isin(genes) & 
        df_edges[target_col].isin(genes)
    )
    df_edges = df_edges.loc[keep_mask, :].copy()
    
    # Map nodes to indices
    df_edges[source_col] = df_edges[source_col].map(mapping)
    df_edges[target_col] = df_edges[target_col].map(mapping)
    df_edges = df_edges.dropna(subset=[source_col, target_col])
    
    # Create edge index
    edge_index = torch.from_numpy(
        df_edges[[source_col, target_col]].values.T
    ).long()
    
    # Create edge weights if available
    edge_attr = None
    if edge_weight_col in df_edges.columns:
        edge_attr = torch.from_numpy(
            df_edges[edge_weight_col].values
        ).float().unsqueeze(1)
    
    # Create node features
    if features is None:
        x = torch.eye(len(genes), dtype=torch.float)
    else:
        x = torch.from_numpy(features).float()
    
    # Create Data object
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=len(genes)
    )
    
    return data


def networkx_to_pyg(G: nx.Graph, node_features: Optional[np.ndarray] = None) -> Data:
    """Convert NetworkX graph to PyTorch Geometric Data.
    
    Parameters
    ----------
    G : nx.Graph
        NetworkX graph
    node_features : np.ndarray, optional
        Node feature matrix
        
    Returns
    -------
    Data
        PyTorch Geometric Data object
    """
    data = from_networkx(G)
    
    if node_features is not None:
        data.x = torch.from_numpy(node_features).float()
    else:
        # Use one-hot encoding if no features provided
        data.x = torch.eye(data.num_nodes, dtype=torch.float)
    
    return data


def add_node_features(
    data: Data,
    features: np.ndarray
) -> Data:
    """Add or update node features in Data object.
    
    Parameters
    ----------
    data : Data
        PyTorch Geometric Data object
    features : np.ndarray
        Node feature matrix [num_nodes, num_features]
        
    Returns
    -------
    Data
        Updated Data object
    """
    data.x = torch.from_numpy(features).float()
    return data


def graph_statistics(data: Data) -> dict:
    """Compute basic graph statistics.
    
    Parameters
    ----------
    data : Data
        PyTorch Geometric Data object
        
    Returns
    -------
    dict
        Dictionary with graph statistics
    """
    return {
        'num_nodes': data.num_nodes,
        'num_edges': data.edge_index.shape[1],
        'num_features': data.x.shape[1] if data.x is not None else 0,
        'node_degree_mean': (data.edge_index[0].bincount().float().mean().item()),
        'density': data.edge_index.shape[1] / (data.num_nodes * (data.num_nodes - 1) / 2),
    }
