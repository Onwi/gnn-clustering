"""Utility functions."""

import json
from pathlib import Path
from typing import Any, Dict
import numpy as np
import torch
import random


def set_seed(seed: int):
    """Set random seed for reproducibility.
    
    Parameters
    ----------
    seed : int
        Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def save_config(config: Dict[str, Any], path: str):
    """Save configuration to JSON file.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
    path : str
        Path to save config
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(config, f, indent=4)


def load_config(path: str) -> Dict[str, Any]:
    """Load configuration from JSON file.
    
    Parameters
    ----------
    path : str
        Path to config file
        
    Returns
    -------
    dict
        Configuration dictionary
    """
    with open(path, 'r') as f:
        return json.load(f)


def get_device() -> str:
    """Get available device.
    
    Returns
    -------
    str
        'cuda' if available, 'cpu' otherwise
    """
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters.
    
    Parameters
    ----------
    model : nn.Module
        PyTorch model
        
    Returns
    -------
    int
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class EarlyStopping:
    """Early stopping handler.
    
    Parameters
    ----------
    patience : int
        Number of checks with no improvement after which training will be stopped
    delta : float
        Minimum change to qualify as an improvement
    save_path : str, optional
        Path to save the best model
    """
    
    def __init__(self, patience: int = 10, delta: float = 0.0, save_path: str = None):
        self.patience = patience
        self.delta = delta
        self.save_path = save_path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, val_loss: float, model: torch.nn.Module):
        """Check if should stop training.
        
        Parameters
        ----------
        val_loss : float
            Validation loss
        model : nn.Module
            Model to potentially save
        """
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            if self.save_path:
                torch.save(model.state_dict(), self.save_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            if self.save_path:
                torch.save(model.state_dict(), self.save_path)


def normalize_features(features: np.ndarray) -> np.ndarray:
    """Normalize features to zero mean and unit variance.
    
    Parameters
    ----------
    features : np.ndarray
        Feature matrix
        
    Returns
    -------
    np.ndarray
        Normalized features
    """
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std == 0] = 1.0  # Avoid division by zero
    return (features - mean) / std
