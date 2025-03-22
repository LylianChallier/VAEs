"""The main module of the app.

Contains only the corpus of the app.
All the functionalities are imported.
"""

import streamlit as st
import os

# Import modules
from viz import create_vae_diagram
from train import train_model

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List
import streamlit as st
import time

import streamlit as st
import matplotlib.pyplot as plt
import torch
import os
import torch.nn as nn

# Import des modules personnalisés
from vae import VAE
from utils import get_dataset, generate_model_id
from viz import plot_loss_curves, visualize_reconstructions, generate_samples


# Configuration de la page
st.set_page_config(
    page_title="Variational Autoencoder Explorer", page_icon="🖼", layout="wide"
)

# Assurez-vous que le dossier 'models' existe
if not os.path.exists("models"):
    os.makedirs("models")

# Structure principale de l'application avec deux colonnes
col1, col2 = st.columns([3, 1])


# Fonction pour créer et entrainer le modèle ou charger un modèle existant
def modelisation(
    dataset, latent_dim, hidden_layers, reconstruction_error, batch_size, epochs
):
    # Création du dataloader
    train_loader, test_loader, dim = get_dataset(dataset, batch_size)

    # Paramètres du modèle pour identifier les modèles déjà entraînés
    model_params = {
        "dataset": dataset,
        "latent_dim": latent_dim,
        "hidden_layers": hidden_layers,
        "reconstruction_error": reconstruction_error,
        "epochs": epochs,
    }

    model_id = generate_model_id(model_params)
    model_path = f"models/vae_{model_id}.pth"
    losses_path = f"models/vae_{model_id}_losses.pth"

    # Création du modèle VAE
    c, w, h = dim
    model = VAE(
        c,
        w,
        h,
        latent_dim=latent_dim,
        hidden_dims=hidden_layers,
        recon_loss_type=reconstruction_error,
    )

    # Container pour les visualisations après l'entraînement
    results_container = st.container()

    # Vérifier si le modèle existe déjà
    if os.path.exists(model_path) and os.path.exists(losses_path):
        st.info(f"Chargement d'un modèle existant avec les mêmes paramètres...")

        # Charger le modèle et les historiques de perte
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        losses = torch.load(losses_path, map_location=torch.device("cpu"))

        # Afficher le récapitulatif des pertes
        with results_container:
            st.subheader("Historique d'entraînement")
            fig_loss, ax_loss = plt.subplots(figsize=(10, 4))
            # Ajouter les points pour chaque epoch
            ax_loss.plot(losses["train_losses"], "b-", label="Train Loss")
            ax_loss.scatter(
                range(len(losses["train_losses"])), losses["train_losses"], color="blue"
            )
            ax_loss.plot(losses["test_losses"], "r-", label="Test Loss")
            ax_loss.scatter(
                range(len(losses["test_losses"])), losses["test_losses"], color="red"
            )
            ax_loss.set_xlabel("Epoch")
            ax_loss.set_ylabel("Loss")
            ax_loss.legend()
            ax_loss.grid(True)
            st.pyplot(fig_loss)
            plt.close(fig_loss)

            # Afficher message de confirmation
            st.success(f"Modèle chargé avec succès! (Nombre d'epochs: {epochs})")
    else:
        # Affichage barre de progression
        progress_container = st.container()
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()

        # Container pour les visualisations pendant l'entraînement
        training_vis_container = st.container()
        with training_vis_container:
            # Placeholders pour les graphiques mis à jour
            loss_placeholder = st.empty()
            images_placeholder = st.empty()

        # Entrainement du modèle
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

        # Sauvegarder le modèle entraîné et les historiques de pertes
        torch.save(model.state_dict(), model_path)
        torch.save(losses, losses_path)

        # Afficher un message de fin d'entrainement
        st.success(
            f"Entrainement terminé après {epochs} epochs! Modèle sauvegardé dans {model_path}"
        )

    # Afficher des visualisations dans le conteneur de résultats
    with results_container:
        # Obtenir un batch de test différent pour la reconstruction
        test_batch_for_recon = next(iter(test_loader))[0]

        # Afficher les images générées
        st.subheader("Images générées")
        fig_gen = plt.figure(figsize=(10, 10))
        generate_samples(model, num_samples=16, fig=fig_gen)
        st.pyplot(fig_gen)
        plt.close(fig_gen)

        # Afficher des reconstructions
        st.subheader("Reconstructions (sur un autre batch)")
        fig_recon = plt.figure(figsize=(10, 5))
        visualize_reconstructions(
            model, test_batch_for_recon, num_images=4, fig=fig_recon
        )
        st.pyplot(fig_recon)
        plt.close(fig_recon)

    return model


# Sidebar pour les hyperparamètres
st.sidebar.title("Paramètres du VAE")

# Sélection du dataset
dataset = st.sidebar.selectbox("Dataset", ["MNIST", "CIFAR10"])

# Dimension de l'espace latent
latent_dim = st.sidebar.slider(
    "Dimension de l'espace latent", min_value=2, max_value=200, value=40, step=1
)

# Couches convolutionnelles cachées
hidden_layers = st.sidebar.multiselect(
    "Couches convolutionnelles cachées",
    options=[32, 64, 128, 256, 512],
    default=[32, 64],
)

# Type d'erreur de reconstruction
reconstruction_error = st.sidebar.selectbox(
    "Erreur de reconstruction", ["gaussian", "laplace"]
)

# Taille du batch
batch_size = st.sidebar.selectbox("Taille du batch", options=[64, 128, 256], index=0)

# Nombre d'epochs
epochs = st.sidebar.slider(
    "Nombre d'epochs", min_value=1, max_value=50, value=5, step=1
)

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

# Colonne 2: Schéma du VAE (toujours visible)
with col2:
    st.markdown(4 * "<br>", unsafe_allow_html=True)
    st.subheader("Architecture")

    # Déterminer le nombre de canaux basé sur le dataset sélectionné
    dim = (1, 28, 28) if dataset.lower() == "mnist" else (3, 32, 32)

    # Créer et afficher le schéma du VAE qui est interactif avec les paramètres
    vae_diagram = create_vae_diagram(
        input_dim=dim, latent_dim=latent_dim, hidden_dims=hidden_layers
    )
    st.graphviz_chart(vae_diagram)

# Colonne 1: Contenu principal
with col1:
    st.title("Explorateur de Variational Autoencoder (VAE)")
    st.write(
        """
    Cette application permet d'explorer les Variational Autoencoders (VAEs), un type de modèle génératif
    qui apprend à reconstruire et générer des images à partir d'un espace latent continu.
    Ajustez les paramètres dans la barre latérale et lancez l'entrainement pour observer les résultats.
    """
    )

    # Section "En savoir plus sur les VAE"
    with st.expander("En savoir plus sur les VAE"):
        st.write(
            """
        ### Qu'est-ce qu'un Variational Autoencoder (VAE)?
        
        Un Variational Autoencoder est un type de réseau de neurones génératif qui apprend à représenter des données
        dans un espace latent continu. Contrairement aux autoencoders classiques, les VAEs imposent une structure sur
        l'espace latent en utilisant une approche probabiliste.
        
        ### Architecture
        
        Un VAE se compose de deux parties principales:
        
        1. **Encodeur**: Transforme les données d'entrée en distributions dans l'espace latent (moyenne μ et variance σ²)
        2. **Décodeur**: Reconstruit les données à partir d'échantillons de l'espace latent
        
        ### Fonction de perte
        
        La fonction de perte d'un VAE comprend deux termes:
        
        - **Erreur de reconstruction**: Mesure la différence entre les données d'entrée et leur reconstruction
        - **Divergence KL**: Force la distribution latente à se rapprocher d'une distribution normale standard
        
        La fonction de perte totale est: L = Reconstruction_Loss + β * KL_Divergence
        
        ### Génération de nouvelles données
        
        Après l'entrainement, on peut générer de nouvelles données en:
        1. Échantillonnant des points de l'espace latent suivant N(0, I)
        2. Décodant ces points en utilisant le décodeur
        
        ### Applications
        
        Les VAEs sont utilisés pour:
        - La génération d'images
        - La compression de données
        - L'apprentissage de représentations
        - L'interpolation entre différentes données
        """
        )

    # Bouton pour lancer l'entrainement
    if st.button("Entrainer le modèle"):
        with st.spinner("Entrainement/chargement du modèle en cours..."):
            trained_model = modelisation(
                dataset=dataset,
                latent_dim=latent_dim,
                hidden_layers=hidden_layers,
                reconstruction_error=reconstruction_error,
                batch_size=batch_size,
                epochs=epochs,
            )

    # Section Références
    st.markdown("---")
    st.subheader("Références")
    st.markdown(
        """
    - [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) - Diederik P. Kingma, Max Welling (2013)
    - [Tutorial on Variational Autoencoders](https://arxiv.org/abs/1606.05908) - Carl Doersch (2016)
    """
    )

    # Footer
    st.markdown("---")
    st.markdown(
        "Développée par [Lylian Challier](https://github.com/LylianChallier?tab=repositories) et Mohammed-Amine Grini avec Streamlit et Pytorch."
    )
