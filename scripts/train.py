"""Example training script for GNN model."""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric.data import Data

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from gnn_clustering.models import GCNModel
from gnn_clustering.engines import Trainer
from gnn_clustering.utils import set_seed, get_device, count_parameters


def create_synthetic_graph(num_nodes: int = 100, num_edges: int = 300, num_features: int = 10):
    """Create a synthetic graph for demonstration.
    
    Parameters
    ----------
    num_nodes : int
        Number of nodes
    num_edges : int
        Number of edges
    num_features : int
        Number of node features
        
    Returns
    -------
    Data
        PyTorch Geometric Data object
    """
    # Create random features
    x = torch.randn(num_nodes, num_features)
    
    # Create random edges
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
    # Create random labels for demonstration
    y = torch.randint(0, 2, (num_nodes,))
    
    # Create batch tensor (single graph)
    batch = torch.zeros(num_nodes, dtype=torch.long)
    
    data = Data(x=x, edge_index=edge_index, y=y, batch=batch)
    
    return data


def main():
    """Run example training."""
    
    # Configuration
    config = {
        'seed': 42,
        'device': get_device(),
        'num_nodes': 100,
        'num_edges': 300,
        'input_dim': 10,
        'hidden_dim': 64,
        'output_dim': 2,
        'epochs': 50,
        'batch_size': 32,
        'learning_rate': 0.001,
        'dropout': 0.5,
    }
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Set seed
    set_seed(config['seed'])
    
    # Create synthetic data
    print("\nCreating synthetic graph...")
    data = create_synthetic_graph(
        num_nodes=config['num_nodes'],
        num_edges=config['num_edges'],
        num_features=config['input_dim']
    )
    print(f"Graph: {data}")
    
    # Create model
    print("\nCreating model...")
    model = GCNModel(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        output_dim=config['output_dim'],
        dropout=config['dropout'],
        task='node'
    )
    
    num_params = count_parameters(model)
    print(f"Model parameters: {num_params:,}")
    
    # Create optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    
    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=config['device']
    )
    
    # Create dummy data loaders (in practice, you'd use real data)
    print("\nPreparing data...")
    
    # Split data for train/val
    num_train = int(0.7 * data.x.shape[0])
    num_val = int(0.15 * data.x.shape[0])
    
    indices = torch.randperm(data.x.shape[0])
    train_indices = indices[:num_train]
    val_indices = indices[num_train:num_train + num_val]
    test_indices = indices[num_train + num_val:]
    
    # Create mask tensors
    train_mask = torch.zeros(data.x.shape[0], dtype=torch.bool)
    val_mask = torch.zeros(data.x.shape[0], dtype=torch.bool)
    test_mask = torch.zeros(data.x.shape[0], dtype=torch.bool)
    
    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True
    
    print(f"Train: {int(train_mask.sum())}, Val: {int(val_mask.sum())}, Test: {int(test_mask.sum())}")
    
    # Simple data loaders using the same graph with different masks
    class GraphDataLoader:
        def __init__(self, data, mask, batch_size=32):
            self.data = data
            self.mask = mask
            self.batch_size = batch_size
        
        def __iter__(self):
            yield self.data
        
        def __len__(self):
            return 1
    
    train_loader = GraphDataLoader(data, train_mask, config['batch_size'])
    val_loader = GraphDataLoader(data, val_mask, config['batch_size'])
    
    # Train
    print("\nTraining...")
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config['epochs'],
        patience=10,
        save_path='best_model.pt'
    )
    
    print("\nTraining completed!")
    print(f"Final train loss: {history['train_loss'][-1]:.4f}")
    if history['val_loss']:
        print(f"Final val loss: {history['val_loss'][-1]:.4f}")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loader = GraphDataLoader(data, test_mask, config['batch_size'])
    test_metrics = trainer.evaluate(test_loader)
    
    print(f"Test loss: {test_metrics['loss']:.4f}")
    if 'accuracy' in test_metrics:
        print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    
    # Save model
    print("\nSaving model...")
    trainer.save_model('final_model.pt')
    print("Model saved to 'final_model.pt'")


if __name__ == '__main__':
    main()
