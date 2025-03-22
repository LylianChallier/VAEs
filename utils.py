"""The functions to handle with data.

Contains functions to:
- generate an id to save model to .pth file
- load the data, normalize it and make dataloader
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from typing import Tuple, Dict, List, Optional, Union, Callable
import streamlit as st
import io
from PIL import Image
import time
import pandas as pd
import json
import hashlib

# Fonction pour créer un identifiant unique de modèle basé sur ses paramètres
def generate_model_id(params):
    param_str = json.dumps(params, sort_keys=True)
    return hashlib.md5(param_str.encode()).hexdigest()

def get_dataset(dataset_name: str, batch_size: int = 128, test_split: float = 0.2, device: torch.device = None) -> Tuple[DataLoader, DataLoader, Tuple[int, int, int]]:
    """
    Load and prepare the dataset for training.
    
    Parameters
    ----------
    dataset_name : str
        Name of the dataset ('mnist', 'cifar10', or 'freyfaces')
    batch_size : int
        Batch size for the data loaders
    test_split : float
        Proportion of the dataset to use for testing
    device : torch.device, optional
        Device to put the data on (not used directly in this function but passed to the data loaders)
        
    Returns
    -------
    tuple
        A tuple containing the train and test DataLoaders, and input dimensions
    """
    # Define transformations
    if dataset_name.lower() == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
        input_channels, input_height, input_width = 1, 28, 28
        
    elif dataset_name.lower() == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
        input_channels, input_height, input_width = 3, 32, 32
        
    elif dataset_name.lower() == 'freyfaces':
        # FreyFaces dataset is not available in torchvision, so we need to load it manually
        # This is a placeholder - you might need to adjust based on how the dataset is stored
        # The expected shape is (1, 28, 20)
        raise NotImplementedError("FreyFaces dataset loading not implemented in this example")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Split into train and test sets
    train_size = int((1 - test_split) * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, (input_channels, input_height, input_width)

