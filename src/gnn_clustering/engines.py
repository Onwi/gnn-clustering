"""Training and evaluation engines."""

from typing import Optional, Dict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric.data import Data
import numpy as np
from pathlib import Path


class Trainer:
    """Training engine for GNN models.
    
    Parameters
    ----------
    model : nn.Module
        Neural network model
    optimizer : torch.optim.Optimizer
        Optimizer
    criterion : nn.Module
        Loss function
    device : str
        Device to use ('cpu' or 'cuda')
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: str = 'cpu'
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.model.to(device)
    
    def train_epoch(self, loader: DataLoader) -> float:
        """Train for one epoch.
        
        Parameters
        ----------
        loader : DataLoader
            Training data loader
            
        Returns
        -------
        float
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0
        
        for batch in loader:
            if isinstance(batch, tuple):
                data, labels = batch
            else:
                data = batch
                labels = data.y if hasattr(data, 'y') else None
            
            # Move to device
            if isinstance(data, Data):
                data = data.to(self.device)
            else:
                data = torch.tensor(data, device=self.device, dtype=torch.float)
            
            if labels is not None:
                labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            out = self.model(data.x, data.edge_index, data.batch if hasattr(data, 'batch') else None)
            
            if labels is not None:
                loss = self.criterion(out, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
        
        return total_loss / len(loader)
    
    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on a dataset.
        
        Parameters
        ----------
        loader : DataLoader
            Data loader
            
        Returns
        -------
        dict
            Evaluation metrics
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in loader:
            if isinstance(batch, tuple):
                data, labels = batch
            else:
                data = batch
                labels = data.y if hasattr(data, 'y') else None
            
            if isinstance(data, Data):
                data = data.to(self.device)
            else:
                data = torch.tensor(data, device=self.device, dtype=torch.float)
            
            if labels is not None:
                labels = labels.to(self.device)
            
            out = self.model(data.x, data.edge_index, data.batch if hasattr(data, 'batch') else None)
            
            if labels is not None:
                loss = self.criterion(out, labels)
                total_loss += loss.item()
                
                pred = out.argmax(dim=1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
        
        metrics = {'loss': total_loss / len(loader)}
        if total > 0:
            metrics['accuracy'] = correct / total
        
        return metrics
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        patience: int = 10,
        save_path: Optional[str] = None
    ) -> Dict:
        """Full training loop with early stopping.
        
        Parameters
        ----------
        train_loader : DataLoader
            Training data loader
        val_loader : DataLoader, optional
            Validation data loader
        epochs : int
            Number of epochs
        patience : int
            Early stopping patience
        save_path : str, optional
            Path to save best model
            
        Returns
        -------
        dict
            Training history
        """
        history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            history['train_loss'].append(train_loss)
            
            # Validate
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                history['val_loss'].append(val_metrics['loss'])
                if 'accuracy' in val_metrics:
                    history['val_accuracy'].append(val_metrics['accuracy'])
                
                # Early stopping
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    patience_counter = 0
                    if save_path:
                        torch.save(self.model.state_dict(), save_path)
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}")
        
        return history
    
    def save_model(self, path: str):
        """Save model weights.
        
        Parameters
        ----------
        path : str
            Path to save model
        """
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path: str):
        """Load model weights.
        
        Parameters
        ----------
        path : str
            Path to load model
        """
        self.model.load_state_dict(torch.load(path, map_location=self.device))
