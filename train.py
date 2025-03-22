"""The training function.

Contains training loop with evaluation on each epoch
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List
import time

# Import des modules personnalisés
from viz import plot_loss_interactive, visualize_reconstructions


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_epochs: int = 10,
    learning_rate: float = 1e-3,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    progress_bar=None,
    status_text=None,
    loss_placeholder=None,
    images_placeholder=None,
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Train the VAE model with Streamlit visualization.

    Parameters
    ----------
    model : nn.Module
        The VAE model to train
    train_loader : DataLoader
        DataLoader for the training set
    test_loader : DataLoader
        DataLoader for the test set
    num_epochs : int
        Number of epochs to train for
    learning_rate : float
        Learning rate for the optimizer
    device : torch.device
        Device to run the training on
    progress_bar : st.empty, optional
        Streamlit progress bar
    status_text : st.empty, optional
        Streamlit text element to show status
    loss_placeholder : st.empty, optional
        Streamlit placeholder for loss plot
    images_placeholder : st.empty, optional
        Streamlit placeholder for image reconstructions

    Returns
    -------
    tuple
        A tuple containing the trained model and dictionary of training/test losses
    """
    # Move the model to the device
    model = model.to(device)

    # Initialize the optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize lists to store losses
    train_losses = []
    test_losses = []

    # Get a batch of test images for visualization
    test_batch = next(iter(test_loader))[0][:4].to(device)

    # Figure for loss curves
    loss_fig = plt.figure(figsize=(10, 6))

    # Figure for reconstructions
    recon_fig = plt.figure(figsize=(10, 4))

    # Training loop
    for epoch in range(num_epochs):
        # Update progress bar if available
        if progress_bar is not None:
            progress_bar.progress((epoch) / num_epochs)

        # Training
        model.train()
        train_loss = 0

        for batch_idx, (data, _) in enumerate(train_loader):
            # Move data to the correct device
            data = data.to(device)

            # Reset gradients
            optimizer.zero_grad()

            # Forward pass
            recon_batch, mu, log_var = model(data)

            # Calculate loss
            loss_dict = model.loss_function(recon_batch, data, mu, log_var)
            loss = loss_dict["loss"]

            # Backward pass
            loss.backward()

            # Update parameters
            optimizer.step()

            # Accumulate loss
            train_loss += loss.item()

            # Update status with batch progress (optional)
            if status_text is not None and batch_idx % 10 == 0:
                status_text.text(
                    f"Epoch {epoch+1}/{num_epochs} - Batch {batch_idx}/{len(train_loader)}"
                )

        # Average training loss
        avg_train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)

        # Evaluation
        model.eval()
        test_loss = 0

        with torch.no_grad():
            for data, _ in test_loader:
                # Move data to the correct device
                data = data.to(device)

                # Forward pass
                recon_batch, mu, log_var = model(data)

                # Calculate loss
                loss_dict = model.loss_function(recon_batch, data, mu, log_var)

                # Accumulate loss
                test_loss += loss_dict["loss"].item()

        # Average test loss
        avg_test_loss = test_loss / len(test_loader.dataset)
        test_losses.append(avg_test_loss)

        # Print the loss for each epoch
        print(
            f"Epoch: {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}"
        )

        # Update loss curves plot if placeholder is available
        if loss_placeholder is not None:
            plot_loss_interactive(train_losses, test_losses, loss_placeholder, loss_fig)

        # Update reconstructions plot if placeholder is available
        if images_placeholder is not None and (
            epoch % 2 == 0 or epoch == num_epochs - 1
        ):
            visualize_reconstructions(
                model, test_batch, images_placeholder, epoch, recon_fig
            )

        # Brief pause to allow UI to update
        time.sleep(0.1)

    # Final progress
    if progress_bar is not None:
        progress_bar.progress(1.0)
    if status_text is not None:
        status_text.text(f"Entaînement terminé après {num_epochs} epochs!")

    # Return the trained model and the losses
    return model, {"train_losses": train_losses, "test_losses": test_losses}
