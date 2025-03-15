import torch
import torch.nn as nn
import torch.nn.functional as F


# VAE classique
class VAE(nn.Module):
    def __init__(self, image_channels=1, image_size=28, hidden_dim=400, latent_dim=20):
        """
        Initialise un Variational Autoencoder (VAE) non convolutionnel

        Args:
            input_dim (int): Dimension de l'entrée (par exemple, 28*28 pour MNIST)
            hidden_dim (int): Dimension des couches cachées
            latent_dim (int): Dimension de l'espace latent
        """
        super(VAE, self).__init__()

        self.image_channels = image_channels
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.input_dim = self.image_channels * self.image_size**2

        # Encodeur
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.image_channels * self.image_size**2, hidden_dim),
            nn.ReLU(),
        )

        # Projection vers l'espace latent
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)

        # De l'espace latent vers les features pour le décodeur
        self.fc_decoder = nn.Linear(latent_dim, hidden_dim)

        # Décodeur
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, self.input_dim),
            nn.Sigmoid(),  # Pour normaliser les valeurs entre 0 et 1
        )

    def encode(self, x):
        """
        Encode les données d'entrée vers l'espace latent

        Args:
            x (torch.Tensor): Images d'entrée [B, input_dim]

        Returns:
            tuple: (mu, log_var) - les paramètres de la distribution latente
        """
        # Passer à travers l'encodeur
        x = self.encoder(x)

        # Projeter vers les paramètres de la distribution latente
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)

        return mu, log_var

    def reparameterize(self, mu, log_var):
        """
        Astuce de reparamétrisation pour permettre la rétropropagation

        Args:
            mu (torch.Tensor): Moyenne de la distribution latente
            log_var (torch.Tensor): Log-variance de la distribution latente

        Returns:
            torch.Tensor: Échantillon tiré de la distribution latente
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z):
        """
        Décode un point de l'espace latent vers l'espace d'origine

        Args:
            z (torch.Tensor): Point dans l'espace latent [B, latent_dim]

        Returns:
            torch.Tensor: Images reconstruites [B, input_dim]
        """
        # Projeter vers les features pour le décodeur
        z = self.fc_decoder(z)

        # Passer à travers le décodeur
        reconstruction = self.decoder(z)
        reconstruction = reconstruction.view(
            z.shape[0], self.image_channels, self.image_size, self.image_size
        )

        return reconstruction

    def forward(self, x):
        """
        Passe avant complète: encode, échantillonne, et décode

        Args:
            x (torch.Tensor): Images d'entrée [B, input_dim]

        Returns:
            tuple: (reconstruction, mu, log_var)
        """

        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decode(z)
        return reconstruction, mu, log_var

    def loss_function(self, recon_x, x, mu, log_var, beta=1.0):
        """
        Calcule la fonction de perte du VAE: reconstruction + β * KL divergence

        Args:
            recon_x (torch.Tensor): Images reconstruites [B, input_dim]
            x (torch.Tensor): Images originales [B, input_dim]
            mu (torch.Tensor): Moyenne de la distribution latente
            log_var (torch.Tensor): Log-variance de la distribution latente
            beta (float): Coefficient pour le terme KL

        Returns:
            torch.Tensor: Valeur de la perte totale
        """

        # BCE pour les images
        BCE = F.binary_cross_entropy(recon_x, x, reduction="sum")

        # KL divergence
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        return BCE + beta * KLD

    def generate_samples(self, num_samples, device):
        """
        Génère des échantillons à partir du modèle

        Args:
            num_samples (int): Nombre d'échantillons à générer
            device (torch.device): Device sur lequel générer les échantillons

        Returns:
            torch.Tensor: Images générées [num_samples, input_dim]
        """
        with torch.no_grad():
            # Échantillonner de l'espace latent
            z = torch.randn(num_samples, self.latent_dim).to(device)

            # Décoder les échantillons
            samples = self.decode(z)

        return samples

    def reconstruct(self, x, device):
        """
        Reconstruit les images d'entrée

        Args:
            x (torch.Tensor): Images d'entrée [num_images, C, H, W]
            device (torch.device): Device sur lequel générer les images

        Returns:
            torch.Tensor: Images reconstruites [num_images, C, H, W]
        """
        return self.forward(x)[0]


# VAE convolutif
class ConvVAE(nn.Module):
    def __init__(self, image_channels=1, image_size=28, latent_dim=20):
        """
        Initialise un Variational Autoencoder Convolutionnel (ConvVAE)

        Args:
            image_channels (int): Nombre de canaux de l'image d'entrée (1 pour MNIST)
            image_size (int): Taille de l'image d'entrée (28 pour MNIST)
            latent_dim (int): Dimension de l'espace latent
        """
        super(ConvVAE, self).__init__()

        self.image_channels = image_channels
        self.image_size = image_size
        self.latent_dim = latent_dim

        # Calculer la taille des features après les convolutions
        self.feature_size = (
            image_size // 4
        )  # Réduction x4 des dimensions avec 2 maxpools
        self.final_features = 64 * (self.feature_size**2)  # 64 canaux, taille réduite

        # Encodeur
        self.encoder = nn.Sequential(
            # Première couche convolutionnelle: image_channels -> 32 canaux
            nn.Conv2d(image_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Réduit la taille par 2
            # Deuxième couche convolutionnelle: 32 -> 64 canaux
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Réduit la taille par 2 encore
            nn.Flatten(),  # Aplatir pour les couches fully connected
        )

        # Projection vers l'espace latent
        self.fc_mu = nn.Linear(self.final_features, latent_dim)
        self.fc_var = nn.Linear(self.final_features, latent_dim)

        # De l'espace latent vers les features pour le décodeur
        self.fc_decoder = nn.Linear(latent_dim, self.final_features)

        # Décodeur
        self.decoder = nn.Sequential(
            # Première couche de déconvolution: 64 -> 32 canaux
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # Deuxième couche de déconvolution: 32 -> image_channels
            nn.ConvTranspose2d(
                32, image_channels, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.Sigmoid(),  # Pour normaliser les valeurs entre 0 et 1
        )

    def encode(self, x):
        """
        Encode les données d'entrée vers l'espace latent

        Args:
            x (torch.Tensor): Images d'entrée [B, C, H, W]

        Returns:
            tuple: (mu, log_var) - les paramètres de la distribution latente
        """
        # Passer à travers l'encodeur convolutionnel
        x = self.encoder(x)

        # Projeter vers les paramètres de la distribution latente
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)

        return mu, log_var

    def reparameterize(self, mu, log_var):
        """
        Astuce de reparamétrisation pour permettre la rétropropagation

        Args:
            mu (torch.Tensor): Moyenne de la distribution latente
            log_var (torch.Tensor): Log-variance de la distribution latente

        Returns:
            torch.Tensor: Échantillon tiré de la distribution latente
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z):
        """
        Décode un point de l'espace latent vers l'espace d'origine

        Args:
            z (torch.Tensor): Point dans l'espace latent [B, latent_dim]

        Returns:
            torch.Tensor: Images reconstruites [B, C, H, W]
        """
        # Projeter vers les features pour le décodeur
        z = self.fc_decoder(z)

        # Reshape pour les couches de convolution transposées
        z = z.view(-1, 64, self.feature_size, self.feature_size)

        # Passer à travers le décodeur
        reconstruction = self.decoder(z)

        return reconstruction

    def forward(self, x):
        """
        Passe avant complète: encode, échantillonne, et décode

        Args:
            x (torch.Tensor): Images d'entrée [B, C, H, W]

        Returns:
            tuple: (reconstruction, mu, log_var)
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decode(z)
        return reconstruction, mu, log_var

    def loss_function(self, recon_x, x, mu, log_var, beta=1.0):
        """
        Calcule la fonction de perte du VAE: reconstruction + β * KL divergence

        Args:
            recon_x (torch.Tensor): Images reconstruites [B, C, H, W]
            x (torch.Tensor): Images originales [B, C, H, W]
            mu (torch.Tensor): Moyenne de la distribution latente
            log_var (torch.Tensor): Log-variance de la distribution latente
            beta (float): Coefficient pour le terme KL

        Returns:
            torch.Tensor: Valeur de la perte totale
        """
        # BCE pour les images
        BCE = F.binary_cross_entropy(recon_x, x, reduction="sum")

        # KL divergence
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        return BCE + beta * KLD

    def generate_samples(self, num_samples, device):
        """
        Génère des échantillons à partir du modèle

        Args:
            num_samples (int): Nombre d'échantillons à générer
            device (torch.device): Device sur lequel générer les échantillons

        Returns:
            torch.Tensor: Images générées [num_samples, C, H, W]
        """
        with torch.no_grad():
            # Échantillonner de l'espace latent
            z = torch.randn(num_samples, self.latent_dim).to(device)

            # Décoder les échantillons
            samples = self.decode(z)

        return samples

    def reconstruct(self, x, device):
        """
        Reconstruit les images d'entrée

        Args:
            x (torch.Tensor): Images d'entrée [num_images, C, H, W]
            device (torch.device): Device sur lequel générer les images

        Returns:
            torch.Tensor: Images reconstruites [num_images, C, H, W]
        """
        return self.forward(x)[0]
