"""
Data handling utilities.

This module contains functions to:
- Generate a unique ID for saving model files
- Load, normalize and prepare dataloaders for common datasets
"""

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Tuple, Dict
import json
import hashlib


def generate_model_id(params: Dict) -> str:
    """
    Create a unique model identifier based on its parameters.

    Parameters
    ----------
    params : Dict
        Dictionary containing model parameters.

    Returns
    -------
    str
        MD5 hash of the sorted parameters, to be used as model ID.
    """
    param_str = json.dumps(params, sort_keys=True)
    return hashlib.md5(param_str.encode()).hexdigest()


def get_dataset(
    dataset_name: str,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader, Tuple[int, int, int]]:
    """
    Load and prepare the dataset for training.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset ('mnist' or 'cifar10').
    batch_size : int
        Batch size for the data loaders.

    Returns
    -------
    Tuple[DataLoader, DataLoader, Tuple[int, int, int]]
        A tuple containing:
        - train_loader: DataLoader for training data
        - test_loader: DataLoader for testing data
        - dimensions: (channels, height, width) of input images

    Raises
    ------
    ValueError
        If dataset_name is neither 'mnist' nor 'cifar10'.
    """
    # Define transformations based on dataset
    if dataset_name.lower() == "mnist":
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        train_dataset = datasets.MNIST(
            root="./data", train=True, transform=transform, download=True
        )
        test_dataset = datasets.MNIST(
            root="./data", train=False, transform=transform, download=True
        )
        input_channels, input_height, input_width = 1, 28, 28
    elif dataset_name.lower() == "cifar10":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
                ),
            ]
        )
        train_dataset = datasets.CIFAR10(
            root="./data", train=True, transform=transform, download=True
        )
        test_dataset = datasets.CIFAR10(
            root="./data", train=False, transform=transform, download=True
        )
        input_channels, input_height, input_width = 3, 32, 32
    else:
        raise ValueError("Unknown dataset, please select 'mnist' or 'cifar10'")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, (input_channels, input_height, input_width)
