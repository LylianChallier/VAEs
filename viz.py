"""
Visualization functions for VAE models.

This module contains functions to:
- Plot training and test losses
- Visualize input images vs. reconstructed images
- Generate and visualize images from Gaussian random samples
- Create VAE architecture diagrams
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, List, Tuple, Optional
import graphviz


def plot_loss(
    train_losses: List[float],
    test_losses: List[float],
    placeholder=None,
    fig=None,
):
    """
    Plot the training and test loss during the training.

    Parameters
    ----------
    train_losses : List[float]
        List of training losses.
    test_losses : List[float]
        List of test losses.
    placeholder : st.empty, optional
        Streamlit placeholder to update with the plot.
    fig : matplotlib.figure.Figure, optional
        Figure to use for plotting.

    Returns
    -------
    matplotlib.figure.Figure
        The figure with loss plots.
    """
    epochs = list(range(1, len(train_losses) + 1))

    # Create figure if not provided
    if fig is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig.clf()
        ax = fig.add_subplot(111)

    # Plot losses with both lines and points
    ax.plot(epochs, train_losses, "b-", label="Training Loss")
    ax.scatter(epochs, train_losses, color="blue")
    ax.plot(epochs, test_losses, "r-", label="Testing Loss")
    ax.scatter(epochs, test_losses, color="red")

    # Add labels and title
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Testing Loss vs. Epochs")

    # Add grid and legend
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend()

    if placeholder is not None:
        # Update the placeholder
        placeholder.pyplot(fig)

    return fig


def visualize_reconstructions(
    model: nn.Module,
    images: Union[torch.Tensor, DataLoader],
    placeholder=None,
    epoch: Optional[int] = None,
    fig=None,
    num_images: int = 4,
):
    """
    Visualize original images and their reconstructions.

    Parameters
    ----------
    model : nn.Module
        The trained VAE model.
    images : Union[torch.Tensor, DataLoader]
        Batch of images to reconstruct or a DataLoader to get images from.
    placeholder : st.empty, optional
        Streamlit placeholder to update with the plot.
    epoch : int, optional
        Current epoch number (for title).
    fig : matplotlib.figure.Figure, optional
        Figure to use for plotting.
    num_images : int, optional
        Number of images to visualize, by default 4.

    Returns
    -------
    matplotlib.figure.Figure
        The figure with reconstructions.
    """
    # Make sure the model is in evaluation mode
    model.eval()

    # Get images if a DataLoader is provided
    if isinstance(images, DataLoader):
        batch = next(iter(images))[0][:num_images].to(next(model.parameters()).device)
    else:
        batch = images[:num_images]

    # Get reconstructions
    with torch.no_grad():
        reconstructions, _, _ = model(batch)

    # Convert to numpy arrays for plotting
    images_np = batch.cpu().numpy()
    reconstructions_np = reconstructions.cpu().numpy()

    # Create or clear figure
    if fig is None:
        fig = plt.figure(figsize=(10, 4))
    else:
        fig.clf()

    # Plot original images
    for i in range(num_images):
        ax = fig.add_subplot(2, num_images, i + 1)
        if images_np.shape[1] == 1:  # Grayscale
            ax.imshow(images_np[i, 0], cmap="gray")
        else:  # RGB
            # Transpose from [C, H, W] to [H, W, C] and normalize
            img = np.transpose(images_np[i], (1, 2, 0))
            img = (img + 1) / 2  # Denormalize from [-1, 1] to [0, 1]
            ax.imshow(img)
        ax.set_title(f"Original {i + 1}")
        ax.axis("off")

    # Plot reconstructions
    for i in range(num_images):
        ax = fig.add_subplot(2, num_images, num_images + i + 1)
        if reconstructions_np.shape[1] == 1:  # Grayscale
            ax.imshow(reconstructions_np[i, 0], cmap="gray")
        else:  # RGB
            # Transpose from [C, H, W] to [H, W, C] and normalize
            img = np.transpose(reconstructions_np[i], (1, 2, 0))
            img = (img + 1) / 2  # Denormalize from [-1, 1] to [0, 1]
            ax.imshow(img)
        ax.set_title(f"Reconstructed {i + 1}")
        ax.axis("off")

    # Add a title to the figure
    if epoch is not None:
        plt.suptitle(f"Epoch {epoch + 1} Reconstructions")
    else:
        plt.suptitle("VAE Reconstructions")

    plt.tight_layout()

    # Update the placeholder if provided
    if placeholder is not None:
        placeholder.pyplot(fig)

    return fig


def generate_samples(
    model: nn.Module,
    num_samples: int,
    device: Optional[torch.device] = None,
    fig=None,
):
    """
    Generate and visualize samples from the latent space.

    Parameters
    ----------
    model : nn.Module
        The trained VAE model.
    num_samples : int
        Number of samples to generate.
    device : torch.device, optional
        Device to run the generation on. If None, uses the device of the model.
    fig : matplotlib.figure.Figure, optional
        Figure to use for plotting.

    Returns
    -------
    matplotlib.figure.Figure
        Figure with generated samples.
    """
    # Make sure the model is in evaluation mode
    model.eval()

    # If device is not specified, use the device of the model
    if device is None:
        device = next(model.parameters()).device

    # Generate samples
    with torch.no_grad():
        samples = model.sample(num_samples, device)

    # Convert to numpy arrays for plotting
    samples_np = samples.cpu().numpy()

    # Create/clear figure
    if fig is None:
        fig = plt.figure(figsize=(10, 10))
    else:
        fig.clf()

    # Determine grid size
    grid_size = int(np.ceil(np.sqrt(num_samples)))

    # Plot samples
    for i in range(num_samples):
        ax = fig.add_subplot(grid_size, grid_size, i + 1)
        if samples_np.shape[1] == 1:  # Grayscale
            ax.imshow(samples_np[i, 0], cmap="gray")
        else:  # RGB
            # Transpose from [C, H, W] to [H, W, C] and normalize
            img = np.transpose(samples_np[i], (1, 2, 0))
            img = (img + 1) / 2  # Denormalize from [-1, 1] to [0, 1]
            ax.imshow(img)
        ax.axis("off")

    plt.suptitle("Generated Samples")
    plt.tight_layout()

    return fig


def create_vae_diagram(
    input_dim: Tuple[int, int, int],
    latent_dim: int,
    hidden_dims: List[int],
):
    """
    Create a visualization diagram of the VAE architecture.
    Handles the case with a single hidden dimension correctly.

    Parameters
    ----------
    input_dim : Tuple[int, int, int]
        Input dimensions (channels, height, width).
    latent_dim : int
        Dimension of the latent space.
    hidden_dims : List[int]
        List of hidden dimensions.

    Returns
    -------
    graphviz.Digraph
        A graphviz diagram of the VAE architecture.
    """
    graph = graphviz.Digraph(format="png")
    graph.attr(rankdir="TB")  # Top to bottom layout
    graph.attr("node", shape="egg", style="filled", color="lightblue")

    # Input node
    input_channels, input_height, input_width = input_dim
    graph.node("Input", f"Input Image\n{input_channels}×{input_height}×{input_width}")

    # Track dimensions through the network
    current_channels = input_channels
    current_height = input_height
    current_width = input_width

    # Encoder convolutional layers
    for i, h_dim in enumerate(hidden_dims):
        # Simulate convolution to get output dimensions
        # Conv2d with kernel=3, stride=2, padding=1
        next_height = (current_height + 2 * 1 - 3) // 2 + 1
        next_width = (current_width + 2 * 1 - 3) // 2 + 1

        graph.node(
            f"Enc{i}",
            f"2D Convolution\n{current_channels} → {h_dim}\n{current_height}×{current_width} → {next_height}×{next_width}",
        )

        if i > 0:
            graph.edge(f"Enc{i - 1}", f"Enc{i}")
        else:
            graph.edge("Input", f"Enc{i}")

        current_channels = h_dim
        current_height = next_height
        current_width = next_width

    # Flattened size
    flat_size = current_channels * current_height * current_width
    graph.node(
        "Flatten",
        f"Flatten\n{current_channels}×{current_height}×{current_width} → {flat_size}",
    )
    graph.edge(f"Enc{len(hidden_dims) - 1}", "Flatten")

    # Latent space
    graph.node("FC_mu", f"FC μ\n{flat_size} → {latent_dim}")
    graph.node("FC_var", f"FC log(σ)\n{flat_size} → {latent_dim}")
    graph.edge("Flatten", "FC_mu")
    graph.edge("Flatten", "FC_var")

    # Reparameterization
    graph.node("Z", f"Latent Space\n{latent_dim}\nz = μ + σ·ε")
    graph.edge("FC_mu", "Z", label="reparameterization with μ and σ")
    graph.edge("FC_var", "Z")

    # Decoder - FC and reshape
    graph.node("Dec_FC", f"FC Latent Output\n{latent_dim} → {flat_size}")
    graph.edge("Z", "Dec_FC")

    graph.node(
        "Reshape",
        f"Reshape\n{flat_size} → {current_channels}×{current_height}×{current_width}",
    )
    graph.edge("Dec_FC", "Reshape")

    # Decoder - Transposed convolutional layers
    hidden_dims_reversed = hidden_dims.copy()
    hidden_dims_reversed.reverse()

    # Handle case with single hidden dimension
    if len(hidden_dims_reversed) == 1:
        # First transposed convolution directly to output channels
        next_height = current_height * 2
        next_width = current_width * 2

        graph.node(
            "Dec0",
            f"Transposed Convolution\n{current_channels} → {current_channels}\n{current_height}×{current_width} → {next_height}×{next_width}",
        )
        graph.edge("Reshape", "Dec0")

        current_height = next_height
        current_width = next_width

        # Final convolution to match input channels
        graph.node(
            "Dec_final",
            f"2D Convolution\n{current_channels} → {input_channels}\n{current_height}×{current_width} → {current_height}×{current_width}",
        )
        graph.edge("Dec0", "Dec_final")

        # Output
        graph.node(
            "Output", f"Output Image\n{input_channels}×{input_height}×{input_width}"
        )
        graph.edge("Dec_final", "Output")
    else:
        # Multiple hidden dimensions case
        for i in range(len(hidden_dims_reversed) - 1):
            in_channels = hidden_dims_reversed[i]
            out_channels = hidden_dims_reversed[i + 1]

            # Simulate transposed convolution
            next_height = current_height * 2
            next_width = current_width * 2

            node_name = f"Dec{i}"
            graph.node(
                node_name,
                f"Transposed 2D Conv\n{in_channels} → {out_channels}\n{current_height}×{current_width} → {next_height}×{next_width}",
            )

            if i == 0:
                graph.edge("Reshape", node_name)
            else:
                graph.edge(f"Dec{i - 1}", node_name)

            current_channels = out_channels
            current_height = next_height
            current_width = next_width

        # Final transposed convolution
        next_height = current_height * 2
        next_width = current_width * 2

        graph.node(
            "Dec_final1",
            f"Transposed 2D Conv\n{current_channels} → {current_channels}\n{current_height}×{current_width} → {next_height}×{next_width}",
        )
        graph.edge(f"Dec{len(hidden_dims_reversed) - 2}", "Dec_final1")

        current_height = next_height
        current_width = next_width

        # Final convolutional layer
        graph.node(
            "Dec_final2",
            f"2D Convolution\n{current_channels} → {input_channels}\n{current_height}×{current_width} → {current_height}×{current_width}",
        )
        graph.edge("Dec_final1", "Dec_final2")

        # Output
        graph.node(
            "Output", f"Output Image\n{input_channels}×{input_height}×{input_width}"
        )
        graph.edge("Dec_final2", "Output")

    return graph
