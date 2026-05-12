"""Dataset and data loading utilities."""

from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, InMemoryDataset
from sklearn.model_selection import train_test_split


class GraphDataset(Dataset):
    """Simple graph dataset for loading multiple graphs.
    
    Parameters
    ----------
    graphs : list
        List of PyTorch Geometric Data objects
    labels : np.ndarray
        Node or graph labels
    """
    
    def __init__(self, graphs: List[Data], labels: Optional[np.ndarray] = None):
        self.graphs = graphs
        self.labels = labels
    
    def __len__(self) -> int:
        return len(self.graphs)
    
    def __getitem__(self, idx: int) -> Tuple[Data, Optional[int]]:
        if self.labels is not None:
            return self.graphs[idx], self.labels[idx]
        return self.graphs[idx]


class NodeClassificationDataset(InMemoryDataset):
    """Dataset for node classification tasks.
    
    Parameters
    ----------
    name : str
        Dataset name
    root : str
        Root directory for data storage
    """
    
    def __init__(self, name: str = 'node_data', root: str = './data'):
        self.name = name
        super().__init__(root=root)
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def raw_file_names(self):
        return []
    
    @property
    def processed_file_names(self):
        return ['data.pt']
    
    def download(self):
        pass
    
    def process(self):
        pass


def create_data_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data into train, validation, and test sets.
    
    Parameters
    ----------
    X : np.ndarray
        Features
    y : np.ndarray
        Labels
    test_size : float
        Proportion for test set
    val_size : float
        Proportion for validation set (from training set)
    random_state : int
        Random seed
        
    Returns
    -------
    tuple
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=random_state
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def collate_fn(batch: List[Tuple[Data, int]]) -> Tuple[Data, torch.Tensor]:
    """Custom collate function for DataLoader with graphs.
    
    Parameters
    ----------
    batch : list
        List of (data, label) tuples
        
    Returns
    -------
    tuple
        Batched data and labels
    """
    data_list, labels = zip(*batch)
    
    # Stack data
    from torch_geometric.data import Batch
    batched_data = Batch.from_data_list(data_list)
    labels = torch.tensor(labels, dtype=torch.long)
    
    return batched_data, labels
