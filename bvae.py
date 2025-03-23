"""Beta gaussian VAE module.

This module contains the implementation of a beta gaussian VAE architecture.
Includes encoder, decoder, reparameterization and loss function components.
Beta VAE is a classic VAE with a coefficient (beta) applied to the KL Divergence.
When beta = 1, it's equivalent to the standard VAE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List
import numpy as np


class BetaVAE(nn.Module):
    """Beta Variational Autoencoder implementation.

    This implementation is based on the original VAE paper by Kingma and Welling,
    with the beta modification from Higgins et al., to control the disentanglement in the latent space.

    The architecture includes:
    - Convolutional layers for the encoder
    - Transposed convolutional layers for the decoder
    - Gaussian MLP encoder and decoder
    - Reparameterization trick
    - Customizable loss function with beta coefficient

    Parameters
    ----------
    input_channels : int
        Number of input channels (1 for grayscale, 3 for RGB)
    input_height : int
        Height of the input images
    input_width : int
        Width of the input images
    latent_dim : int
        Dimension of the latent space
    hidden_dims : List[int], optional
        List of hidden dimensions for the encoder and decoder networks.
        If None, defaults to [32, 64, 128]
    recon_loss_type : str, default="mse"
        Type of reconstruction loss ('mse' or 'l1')
    beta : float, default=1.0
        Coefficient for the KL divergence in the loss function
    """

    def __init__(
        self,
        input_channels: int,
        input_height: int,
        input_width: int,
        latent_dim: int,
        hidden_dims: List[int] = None,
        recon_loss_type: str = "mse",
        beta: float = 1.0,
    ):
        super(BetaVAE, self).__init__()

        self.input_channels = input_channels
        self.input_height = input_height
        self.input_width = input_width
        self.latent_dim = latent_dim
        self.recon_loss_type = recon_loss_type.lower()
        self.beta = beta

        # Default architecture if not specified
        if hidden_dims is None:
            hidden_dims = [32, 64]
        self.hidden_dims = hidden_dims

        # Encoder convolutional network
        modules = []
        in_channels = input_channels

        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels=h_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(),
                )
            )
            in_channels = h_dim

        self.encoder_conv = nn.Sequential(*modules)

        # Calculate flattened size after convolutions
        with torch.no_grad():
            dummy_input = torch.zeros(
                1, self.input_channels, self.input_height, self.input_width
            )
            conv_output = self.encoder_conv(dummy_input)
            self.conv_shape = conv_output.shape
            self.flat_size = np.prod(self.conv_shape[1:])

        # Latent space projection
        self.fc_mu = nn.Linear(self.flat_size, latent_dim)
        self.fc_var = nn.Linear(self.flat_size, latent_dim)

        # Decoder initial projection
        self.decoder_input = nn.Linear(latent_dim, self.flat_size)

        # Decoder transposed convolutional network
        modules = []

        # Reverse hidden dimensions for decoder architecture
        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims[i],
                        hidden_dims[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU(),
                )
            )

        self.decoder_conv = nn.Sequential(*modules)

        # Final layer to reconstruct the original image dimensions
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_dims[-1],
                hidden_dims[-1],
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(
                hidden_dims[-1], out_channels=input_channels, kernel_size=3, padding=1
            ),
            nn.Tanh(),
        )

        # Restore hidden_dims order
        hidden_dims.reverse()

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode the input into latent space parameters.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [batch_size, channels, height, width]

        Returns
        -------
        mu : torch.Tensor
            Mean of the latent Gaussian distribution
        log_var : torch.Tensor
            Log variance of the latent Gaussian distribution
        """
        # Extract features with convolutional network
        x = self.encoder_conv(x)

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)

        # Project to latent parameters
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)

        return mu, log_var

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Apply the reparameterization trick.

        Sample from N(mu, var) using N(0,1) to enable backpropagation
        through the sampling process.

        Parameters
        ----------
        mu : torch.Tensor
            Mean of the latent Gaussian
        log_var : torch.Tensor
            Log variance of the latent Gaussian

        Returns
        -------
        z : torch.Tensor
            Sampled latent vector
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vectors into reconstructed images.

        Parameters
        ----------
        z : torch.Tensor
            Latent vectors of shape [batch_size, latent_dim]

        Returns
        -------
        reconstruction : torch.Tensor
            Reconstructed images
        """
        # Project latent vector to decoder input size
        h = self.decoder_input(z)
        batch_size = z.size(0)

        # Reshape to match the expected spatial dimensions
        h = h.view(
            batch_size, self.conv_shape[1], self.conv_shape[2], self.conv_shape[3]
        )

        # Apply transposed convolutions
        h = self.decoder_conv(h)

        # Generate final reconstruction
        return self.final_layer(h)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the VAE.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [batch_size, channels, height, width]

        Returns
        -------
        reconstruction : torch.Tensor
            Reconstructed input tensor
        mu : torch.Tensor
            Mean of the latent Gaussian
        log_var : torch.Tensor
            Log variance of the latent Gaussian
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

    def loss_function(
        self,
        recon_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Calculate the VAE loss function.

        Computes the combined loss consisting of:
        - Reconstruction loss (MSE or L1)
        - KL divergence between the encoded distribution and standard normal prior,
          weighted by beta

        Parameters
        ----------
        recon_x : torch.Tensor
            Reconstructed input tensor
        x : torch.Tensor
            Original input tensor
        mu : torch.Tensor
            Mean of the latent Gaussian
        log_var : torch.Tensor
            Log variance of the latent Gaussian

        Returns
        -------
        losses : Dict[str, torch.Tensor]
            Dictionary containing the total loss and individual components:
            - 'loss': Combined weighted loss
            - 'recon_loss': Reconstruction loss component
            - 'kl_loss': KL divergence component

        Raises
        ------
        ValueError
            If unknown reconstruction loss type is specified
        """
        batch_size = x.size(0)

        # Calculate reconstruction loss
        if self.recon_loss_type == "mse":
            recon_loss = F.mse_loss(recon_x, x, reduction="sum")
        elif self.recon_loss_type == "l1":
            recon_loss = F.l1_loss(recon_x, x, reduction="sum")
        else:
            raise ValueError(
                f"Unknown reconstruction loss type: {self.recon_loss_type}, please use MSE or L1"
            )

        # KL divergence with standard normal prior
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        # Combined loss, normalized by batch size
        total_loss = (recon_loss + self.beta * kl_loss) / batch_size

        return {
            "loss": total_loss,
            "recon_loss": recon_loss / batch_size,
            "kl_loss": kl_loss / batch_size,
        }

    def sample(self, num_samples: int, device: torch.device = None) -> torch.Tensor:
        """Generate new samples by sampling from the latent space.

        Parameters
        ----------
        num_samples : int
            Number of samples to generate
        device : torch.device, optional
            Device to run the sampling on. If None, uses the model's device.

        Returns
        -------
        samples : torch.Tensor
            Generated image samples
        """
        # Use model's device if none specified
        if device is None:
            device = next(self.parameters()).device

        # Sample from standard normal distribution
        z = torch.randn(num_samples, self.latent_dim, device=device)

        # Generate images from latent samples
        samples = self.decode(z)

        return samples
