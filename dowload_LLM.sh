#!/bin/bash

# Installation d'Ollama sous Linux / WSL
echo "Installation d'Ollama..."

curl -fsSL https://ollama.com/install.sh | sh

# Vérifions que Ollama est bien installé
echo "Vérifions l'installation..."
ollama --version

# Lancement du service Ollama
echo "Démarrage du serveur Ollama..."
ollama serve &

# Attente quelques secondes pour laisser le serveur démarrer
sleep 5

# Téléchargement du modèle léger phi3:mini
echo "Téléchargement du modèle léger phi3:mini..."
ollama pull phi3:mini

echo "Téléchargement terminé !"

# Test rapide de génération
echo "Test rapide du modèle..."
ollama run phi3:mini
