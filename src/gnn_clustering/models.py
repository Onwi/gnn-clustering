"""GNN model definitions."""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class GCNModel(nn.Module):
    """Graph Convolutional Network for node and graph-level tasks.
    
    Parameters
    ----------
    input_dim : int
        Input node feature dimension
    hidden_dim : int or list
        Hidden layer dimensions. If list, multiple layers will be created.
    output_dim : int
        Output dimension
    dropout : float, optional
        Dropout rate (default: 0.5)
    task : str, optional
        'node' for node classification or 'graph' for graph classification (default: 'graph')
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.5,
        task: str = 'graph'
    ):
        super().__init__()
        self.task = task
        self.dropout_rate = dropout
        
        # Build layers
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
        # Output layer
        if task == 'graph':
            self.fc = nn.Linear(hidden_dim, output_dim)
        else:
            self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Node feature tensor [num_nodes, input_dim]
        edge_index : torch.Tensor
            Edge indices [2, num_edges]
        batch : torch.Tensor, optional
            Batch indicators for graph-level tasks
            
        Returns
        -------
        torch.Tensor
            Output predictions
        """
        # Graph convolution layers
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        
        # Pooling for graph-level tasks
        if self.task == 'graph' and batch is not None:
            x = global_mean_pool(x, batch)
        
        # Output layer
        x = self.fc(x)
        
        return x


class MLPClassifier(nn.Module):
    """Simple MLP classifier.
    
    Parameters
    ----------
    input_dim : int
        Input dimension
    hidden_dims : list
        List of hidden layer dimensions
    output_dim : int
        Output dimension
    dropout : float, optional
        Dropout rate (default: 0.5)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list,
        output_dim: int,
        dropout: float = 0.5
    ):
        super().__init__()
        self.dropout_rate = dropout
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)
