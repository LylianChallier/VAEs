"""Main module of the Variational Autoencoder Explorer app.

Contains the core components of the application and the main function `modelisation`
that trains the model, plots losses and displays reconstructed images.
"""

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import torch
import os

# Import modules
from bvae import BetaVAE
from svae import SigmaVAE
from train import train_model
from utils import get_dataset, generate_model_id
from viz import (
    visualize_reconstructions,
    generate_samples,
    create_vae_diagram,
    plot_loss,
)

# Page configuration
st.set_page_config(
    page_title="Variational Autoencoder Explorer", page_icon="🖼", layout="wide"
)

# Ensure models directory exists
if not os.path.exists("models"):
    os.makedirs("models")

# Main application structure with two columns
col1, col2 = st.columns([3, 1])


def modelisation(
    model_name,
    dataset,
    latent_dim,
    hidden_layers,
    reconstruction_error,
    beta,
    batch_size,
    epochs,
):
    """Create, train or load a VAE model with the specified parameters.

    Parameters
    ----------
    model_name : str
        Type of VAE model to use ("VAE original", "β-VAE" or "σ-VAE")
    dataset : str
        Dataset to use for training and testing
    latent_dim : int
        Dimension of the latent space
    hidden_layers : list
        List of integers specifying the number of filters in each convolutional layer
    reconstruction_error : str
        Type of reconstruction error to use ("MSE", "L1", "gaussian", or "laplace")
    beta : float
        Coefficient for the KL divergence term
    batch_size : int
        Batch size for training
    epochs : int
        Number of training epochs

    Returns
    -------
    torch.nn.Module
        Trained or loaded VAE model
    """
    # Create dataloaders
    train_loader, test_loader, dim = get_dataset(dataset, batch_size)

    # Model parameters to identify previously trained models
    model_params = {
        "model_name": model_name,
        "dataset": dataset,
        "latent_dim": latent_dim,
        "hidden_layers": hidden_layers,
        "reconstruction_error": reconstruction_error,
        "beta": beta,
        "epochs": epochs,
    }

    model_id = generate_model_id(model_params)
    model_path = f"models/vae_{model_id}.pth"
    losses_path = f"models/vae_{model_id}_losses.pth"

    # Create appropriate VAE model based on user selection
    c, w, h = dim
    if model_name == "VAE original":
        model = BetaVAE(
            c,
            w,
            h,
            latent_dim=latent_dim,
            hidden_dims=hidden_layers,
            recon_loss_type=reconstruction_error,
            beta=1,
        )
    elif model_name == "β-VAE":
        model = BetaVAE(
            c,
            w,
            h,
            latent_dim=latent_dim,
            hidden_dims=hidden_layers,
            recon_loss_type=reconstruction_error,
            beta=beta,
        )
    elif model_name == "σ-VAE":
        model = SigmaVAE(
            c,
            w,
            h,
            latent_dim=latent_dim,
            hidden_dims=hidden_layers,
            recon_loss_type=reconstruction_error,
            beta=beta,
        )

    # Check if model already exists
    if os.path.exists(model_path) and os.path.exists(losses_path):
        st.info("Chargement d'un modèle existant avec les mêmes paramètres...")

        # Load model and loss history
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        losses = torch.load(losses_path, map_location=torch.device("cpu"))

        # Display loss summary
        st.subheader("Fonction de perte et reconstruction")
        loss_fig = plt.figure(figsize=(10, 6))
        plot_loss(losses["train_losses"], losses["test_losses"], fig=loss_fig)
        st.pyplot(loss_fig)
        plt.close(loss_fig)

        recon_fig = plt.figure(figsize=(10, 4))
        visualize_reconstructions(model, test_loader, fig=recon_fig, num_images=7)
        st.pyplot(recon_fig)
        plt.close(recon_fig)

        # Display confirmation message
        st.success(f"Modèle chargé avec succès! (Nombre d'epochs: {epochs})")

        # Display generated images
        st.subheader("Images générées")
        fig_gen = plt.figure(figsize=(10, 10))
        generate_samples(model, num_samples=49, fig=fig_gen)
        st.pyplot(fig_gen)
        plt.close(fig_gen)
    else:
        # Prepare display elements for training
        st.subheader("Entraînement du modèle")
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Placeholders for losses and reconstructions during training
        loss_placeholder = st.empty()
        images_placeholder = st.empty()

        # Train the model
        trained_model, losses = train_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            num_epochs=epochs,
            progress_bar=progress_bar,
            status_text=status_text,
            loss_placeholder=loss_placeholder,
            images_placeholder=images_placeholder,
        )

        # Save trained model and loss history
        torch.save(model.state_dict(), model_path)
        torch.save(losses, losses_path)

        # Display end of training message
        st.success(
            f"Entrainement terminé après {epochs} epochs! Modèle sauvegardé dans {model_path}"
        )

        # Display generated images after training
        st.subheader("Images générées")
        fig_gen = plt.figure(figsize=(10, 10))
        generate_samples(model, num_samples=49, fig=fig_gen)
        st.pyplot(fig_gen)
        plt.close(fig_gen)

    return model


# Sidebar for hyperparameters
st.sidebar.title("Paramètres du VAE")

# Model selection
model_name = st.sidebar.selectbox("Type de VAE", ["VAE original", "β-VAE", "σ-VAE"])

# Dataset selection
dataset = st.sidebar.selectbox("Dataset", ["MNIST", "CIFAR10"])

# Latent space dimension
latent_dim = st.sidebar.slider(
    "Dimension de l'espace latent", min_value=2, max_value=400, value=128, step=2
)

# Hidden convolutional layers
hidden_layers = st.sidebar.multiselect(
    "Couches convolutionnelles cachées",
    options=[8, 16, 32, 64, 128, 256, 512],
    default=[32, 64],
)
# Sort layers and ensure they are integers
hidden_layers = sorted(hidden_layers)
hidden_layers = [int(item) for item in hidden_layers]

# Check if the number of selections exceeds the maximum allowed
if len(hidden_layers) > 5:
    st.sidebar.error(
        "Vous ne pouvez sélectionner qu'un maximum de 5 couches de convolution."
    )
    # Limit selection to the first 5 layers
    hidden_layers = hidden_layers[:5]

# Check if the number of selections is at least 1
if len(hidden_layers) == 0:
    st.sidebar.error("Vous devez choisir au moins une couche de convolution.")
    # Set default layer if none selected
    hidden_layers = [32]

# Reconstruction error type
if model_name == "σ-VAE":
    reconstruction_error = st.sidebar.selectbox(
        "Erreur de reconstruction (log-vraissemblance)", ["gaussian", "laplace"]
    )
else:
    reconstruction_error = st.sidebar.selectbox(
        "Erreur de reconstruction", ["MSE", "L1"]
    )

# Beta parameter for β-VAE and σ-VAE
if model_name != "VAE original":
    beta = st.sidebar.slider(
        "Coefficient de la KL divergence β",
        min_value=0.1,
        max_value=2.0,
        value=0.5,
        step=0.1,
    )
else:
    beta = 1

# Batch size selection
batch_size = st.sidebar.selectbox("Taille du batch", options=[64, 128, 256], index=0)

# Number of training epochs
epochs = st.sidebar.slider(
    "Nombre d'epochs", min_value=1, max_value=50, value=5, step=1
)

# Dataset information
if dataset.lower() == "mnist":
    st.sidebar.info(
        """
    **MNIST** : 60 000 images d'entraînement et 10 000 images de test
    de chiffres manuscrits (0-9) en noir et blanc de dimensions 1 canaux et 28x28 pixels.
    """
    )
if dataset.lower() == "cifar10":
    st.sidebar.info(
        """
    **Cifar10** : 50 000 images d'entraînement et 10 000 images de test
    reparti en 10 classes (avion, automobile, oiseau, chat, cerf, chien, grenouille, bateau, camion) en couleur
    de dimensions 3 canaux et 32x32 pixels.
    """
    )

# Column 1: Main content
with col1:
    st.title("Explorez les Variational Autoencoders (VAEs)")

    st.markdown(
        """
    <style>
    .justified {
        text-align: justify;
    }
    </style>
    <div class="justified">
        Bienvenue dans notre application Streamlit dédiée aux Variational Autoencoders (VAEs) !
        Découvrez comment ces modèles génèrent de nouvelles données en ajustant les paramètres de trois types de VAEs :
        <strong>l'original, le β-VAE et le σ-VAE</strong>. Plongez dans l'univers fascinant de l'apprentissage génératif et expérimentez par vous-même les différences entre ces modèles.
        Prêt à jouer avec les VAEs ? C'est parti !
    </div>
    <br>
    """,
        unsafe_allow_html=True,
    )

    st.subheader("Qu'est-ce qu'un VAE ?")

    st.markdown(
        """
    <style>
    .justified {
        text-align: justify;
    }
    </style>
    <div class="justified">
       Un Variational Autoencoder (VAE) est un <strong>modèle génératif</strong>, c'est-à-dire qu'il permet de créer des images, du texte, de la musique, etc.
       Nous nous concentrerons uniquement sur la génération d'images, bien que les VAEs aient d'autres applications.
       Voici une frise chronologique des modèles que nous explorerons, ainsi que des principaux modèles génératifs de la période 2014-2020.
    </div>
    <br>
    """,
        unsafe_allow_html=True,
    )

    st.image(
        image="img/frise.png",
        caption="Frise chronologique des modèles génératifs, 2014-2020",
        use_container_width=True,
    )

    st.markdown(
        """
    <style>
    .justified {
        text-align: justify;
    }
    </style>
    <div class="justified">
        Les VAEs sont dérivés des auto-encodeurs, qui compressent des images pour les représenter dans un espace de plus basse dimension : c'est le rôle de l'encodeur.
        Ensuite, cette représentation latente est décompressée pour reconstituer l'image originale : c'est le rôle du décodeur.
        Les auto-encodeurs sont utilisés quotidiennement pour envoyer des images et du son, avec pour objectif de récupérer exactement ce qui a été envoyé.
        <br><br>
        C'est ici que les VAEs se distinguent. Contrairement aux auto-encodeurs classiques, les VAEs ne sont pas déterministes.
        Ils permettent, à partir d'une même représentation latente, de générer des images variées.
        L'encodeur ne fournit plus une représentation latente fixe, mais une distribution probabiliste de cette représentation.
    </div>
    <br>
    """,
        unsafe_allow_html=True,
    )

    # Section "Learn more about VAE"
    with st.expander("ℹ️ En savoir plus sur le VAE et ses variantes"):

        st.write(
            """
        ### Architecture

        Un VAE se compose de deux parties principales :

        1. **Encodeur**: Transforme les données d'entrée en distributions dans l'espace latent caractérisées par une moyenne μ et une variance σ².
        2. **Décodeur**: Reconstruit les données à partir d'échantillons de l'espace latent.
        """
        )

        st.image(
            image="img/vae_illustration.png",
            caption="Schéma des VAEs",
            use_container_width=True,
        )

        st.write(
            """

        ### Fonction de perte

        La fonction de perte d'un VAE comprend deux termes :

        - **Erreur de reconstruction**: Mesure la différence entre les données d'entrée et leur reconstruction, souvent une erreur quadratique (MSE).
        - **Divergence Kullback-Leibler**: Force la distribution latente a posteriori à se rapprocher de la distribution a priori, souvent une distribution normale.

        La fonction de perte totale est : L = Reconstruction_Loss + KL_Divergence

        ### La reparamétrisation

        Étant donné que le VAE n'est pas déterministe, il y a une part d'aléatoire dans la construction de l'espace latent z.
        Cela pose problème lors de la rétropropagation. Pour contourner ce problème, on utilise une technique de reparamétrisation de z.
        En pratique, comme on utilise une distribution gaussienne, la reparamétrisation est donnée par : z = μ + σ ⋅ ε, où ε suit une loi normale standard.

        """
        )

        st.image(
            image="img/reparam.png",
            caption="Reparametrization Trick",
            use_container_width=True,
        )

        st.write(
            """
        ### Le β-VAE

        Le β-VAE est une variante du VAE original qui introduit un poids β dans la fonction de perte pour ajuster l'importance relative des deux termes.

        La nouvelle fonction de perte est : L(β) = Reconstruction_Loss + β ⋅ KL_Divergence

        Dans l'article de Higgins et al., β est souvent inférieur à 1,
        ce qui donne plus de poids à la reconstruction au détriment de la proximité des distributions a priori et a posteriori.
        Il est crucial d'optimiser β pour répondre aux attentes et objectifs spécifiques du β-VAE.

        ### Le σ-VAE

        Le σ-VAE est une variante du β-VAE qui calcule analytiquement la variance des données σ et l'utilise dans la perte de reconstruction, qui est cette fois-ci une log-vraisemblance.
        Cela permet au modèle d'être plus robuste face aux données aberrantes et de produire des reconstructions plus fidèles des images d'entrée.
        """
        )

        st.image(
            image="img/vae_results.png",
            caption="Génération d'images par des VAEs",
            use_container_width=True,
        )

    # Button to start training
    if st.button("Entrainer le modèle"):
        with st.spinner("Entrainement/chargement du modèle en cours..."):
            trained_model = modelisation(
                model_name=model_name,
                dataset=dataset,
                latent_dim=latent_dim,
                hidden_layers=hidden_layers,
                reconstruction_error=reconstruction_error,
                beta=beta,
                batch_size=batch_size,
                epochs=epochs,
            )

# Column 2: VAE diagram (always visible)
with col2:
    st.markdown(4 * "<br>", unsafe_allow_html=True)
    st.subheader("Architecture")

    # Determine number of channels based on selected dataset
    dim = (1, 28, 28) if dataset.lower() == "mnist" else (3, 32, 32)

    # Create and display the interactive VAE diagram
    vae_diagram = create_vae_diagram(
        input_dim=dim, latent_dim=latent_dim, hidden_dims=hidden_layers
    )
    st.graphviz_chart(vae_diagram)

    # References section
    st.markdown("---")
    st.subheader("Références")
    st.markdown(
        """
    - [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) - Diederik P. Kingma, Max Welling (2013)
    - [β-VAE: LEARNING BASIC VISUAL CONCEPTS WITH A CONSTRAINED VARIATIONAL FRAMEWORK](https://openreview.net/forum?id=Sy2fzU9gl) - Irina Higgins et al. (2017)
    - [Simple and Effective VAE Training with Calibrated Decoders](https://orybkin.github.io/sigma-vae/) - Oleh Rybkin et al. (2020)
    """
    )

    # Footer
    st.markdown("---")
    st.markdown(
        "Développée par Lylian Challier et Mohamed-Amine Grini avec Streamlit et Pytorch."
    )
    st.markdown("[Repository GitHub du projet](https://github.com/LylianChallier/VAEs)")
