"""The Sigma VAE module.

Contains the class for the architecture of a Sigma VAE.
With encoder, decoder, reparametrization and loss function.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, List
import math


class SigmaVAE(nn.Module):
    """
    Variational Autoencoder implementation based on Kingma and Welling's paper.

    This VAE implementation includes:
    - Convolutional layers before the encoder
    - Transposed convolutional layers after the decoder
    - Gaussian MLP encoder
    - Gaussian MLP decoder
    - Reparameterization method
    - Loss function with KL divergence and reconstruction error

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
    hidden_dims : List[int]
        List of hidden dimensions for the encoder and decoder
    recon_loss_type : str
        Type of reconstruction loss ('gaussian' or 'laplace' log-likelyhood)
    """

    def __init__(
        self,
        input_channels: int = 1,
        input_height: int = 28,
        input_width: int = 28,
        latent_dim: int = 10,
        hidden_dims: List[int] = None,
        recon_loss_type: str = "gaussian",
        beta: float = 1,
    ):
        super(SigmaVAE, self).__init__()

        self.input_channels = input_channels
        self.input_height = input_height
        self.input_width = input_width
        self.latent_dim = latent_dim
        self.recon_loss_type = recon_loss_type.lower()
        self.beta = beta

        # Set default hidden dimensions if not provided
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]
        self.hidden_dims = hidden_dims

        # Build encoder convolutional layers
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

        # Calculate size after convolutions and store the shape
        with torch.no_grad():
            dummy_input = torch.zeros(
                1, self.input_channels, self.input_height, self.input_width
            )
            conv_output = self.encoder_conv(dummy_input)
            self.conv_shape = conv_output.shape
            self.flat_size = np.prod(self.conv_shape[1:])

        # MLP Gaussian encoder
        self.fc_mu = nn.Linear(self.flat_size, latent_dim)
        self.fc_var = nn.Linear(self.flat_size, latent_dim)

        # MLP Gaussian decoder - first part
        self.decoder_input = nn.Linear(latent_dim, self.flat_size)

        # Build decoder transposed convolutional layers
        modules = []

        # Reverse the hidden dimensions for the decoder
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

        # Final layer to output the reconstructed image
        self.decoder_conv = nn.Sequential(*modules)

        # Final transposed convolution to get back to original channels
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

        # Reset hidden_dims to original order
        hidden_dims.reverse()

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode the input into the latent space.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [batch_size, channels, height, width]

        Returns
        -------
        tuple
            A tuple containing the mean and log variance of the latent space
        """
        # Pass through conv layers
        x = self.encoder_conv(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Get mean and log variance
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)

        return mu, log_var

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from N(mu, var) using N(0,1).

        Parameters
        ----------
        mu : torch.Tensor
            Mean of the latent Gaussian
        log_var : torch.Tensor
            Log variance of the latent Gaussian

        Returns
        -------
        torch.Tensor
            Sampled latent vector
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent samples into reconstructed images.

        Parameters
        ----------
        z : torch.Tensor
            Latent samples

        Returns
        -------
        torch.Tensor
            Reconstructed images
        """
        # Project and reshape
        h = self.decoder_input(z)

        # Calculate the number of channels and spatial dimensions
        batch_size = z.size(0)

        # Reshape to match the expected input for transposed convolutions
        h = h.view(
            batch_size, self.conv_shape[1], self.conv_shape[2], self.conv_shape[3]
        )

        # Apply transposed convolutions
        h = self.decoder_conv(h)

        # Final layer to get reconstructed image
        return self.final_layer(h)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the VAE.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [batch_size, channels, height, width]

        Returns
        -------
        tuple
            A tuple containing the reconstructed image, mean, and log variance
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
        """
        Calculate the VAE loss function.

        The loss consists of:
        - Reconstruction loss (either Gaussian or Laplace log-likelihood)
        - KL divergence between the encoded distribution and the prior

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
        dict
            A dictionary containing the total loss and its components
        """
        batch_size = x.size(0)

        # Calculate reconstruction loss based on the specified type
        if self.recon_loss_type == "gaussian":
            # Gaussian log-likelihood
            # Calculate the scale (variance) based on MSE
            scale = torch.mean((x - recon_x) ** 2)
            # Log-likelihood formula
            recon_loss = -0.5 * torch.sum(
                ((x - recon_x) ** 2) / scale + torch.log(scale) + math.log(2 * math.pi)
            )
        elif self.recon_loss_type == "laplace":
            # Laplace log-likelihood
            # Calculate scale based on absolute error
            scale = torch.mean(torch.abs(x - recon_x))
            # Log-likelihood formula
            recon_loss = -torch.sum(
                torch.abs(x - recon_x) / scale + torch.log(2 * scale)
            )
        else:
            raise ValueError(
                f"Unknown reconstruction loss type: {self.recon_loss_type}"
            )

        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        # Total loss: negative ELBO (Evidence Lower BOund)
        total_loss = (-recon_loss + self.beta * kl_loss) / batch_size

        return {
            "loss": total_loss,
            "recon_loss": -recon_loss / batch_size,
            "kl_loss": kl_loss / batch_size,
        }

    def sample(self, num_samples: int, device: torch.device = None) -> torch.Tensor:
        """
        Sample from the latent space and generate new images.

        Parameters
        ----------
        num_samples : int
            Number of samples to generate
        device : torch.device, optional
            Device to run the sampling on. If None, uses the device of the model.

        Returns
        -------
        torch.Tensor
            Generated samples
        """
        # If device is not specified, use the device of the model
        if device is None:
            device = next(self.parameters()).device

        # Sample from the latent space
        z = torch.randn(num_samples, self.latent_dim, device=device)

        # Decode the samples
        samples = self.decode(z)

        return samples
