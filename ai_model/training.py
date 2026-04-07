import sys
import os
# On récupère le chemin absolu du dossier parent (..) et on l'ajoute au sys.path
dossier_parent = os.path.abspath('..')
if dossier_parent not in sys.path:
    sys.path.append(dossier_parent)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import musdb
from ai_model.u_net import UnetAudioStemmer
from ai_model.data_loader import MUSDBDataset


# --- 1. PRÉPARATION DU MATÉRIEL ET DES OUTILS ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Matériel utilisé : {device}")

# On initialise les données 
musdb_root = "../data/dataset/"
musdb_train_data = MUSDBDataset(data_root=musdb_root,subset="train",chunk_duration=3.0)
musdb_train_loader = DataLoader(musdb_train_data,batch_size=4,shuffle=True,drop_last=True)
musdb_val_data = MUSDBDataset(data_root=musdb_root,subset="test",chunk_duration=3.0)
musdb_val_loader = DataLoader(musdb_val_data,batch_size=4,shuffle=True,drop_last=True)

# On instancie le modèle et on l'envoie sur la carte graphique
model = UnetAudioStemmer().to(device)

# Notre fonction d'erreur (L1 Loss, comme vu précédemment pour éviter le flou)
criterion = nn.L1Loss()

# L'optimiseur (Adam, avec un taux d'apprentissage standard)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Variables pour suivre l'évolution et sauvegarder le meilleur modèle
epochs = 2
best_val_loss = float('inf')

print("\n🔥 Début de l'entraînement...")

for epoch in range(epochs):
    # ==========================================
    #             PHASE D'ENTRAÎNEMENT
    # ==========================================

    model.train()
    epoch_train_loss = 0.0

    # On itère sur notre dataloader (qui nous donne des batchs de chunk_duration secondes)
    for batch_idx, (X_batch_db, y_batch_db) in enumerate(musdb_train_loader):
        X_batch_db = X_batch_db.to(device)
        y_batch_db = y_batch_db.to(device)

        # Remise à zéro des gradients
        optimizer.zero_grad()

        # Acquisition des prédictions
        y_predi_batch_db = model(X_batch_db)

        # Calcul de la loss
        loss = criterion(y_predi_batch_db, y_batch_db)
        epoch_train_loss += loss.item()

        # Backpropagation 
        loss.backward()

        # Mise à jour des gradients
        optimizer.step()

        epoch_train_loss += loss.item()

    # Calcul de la Loss moyenne sur toute l'époque
    avg_train_loss = epoch_train_loss / len(musdb_train_loader)

    # ==========================================
    #             PHASE DE VALIDATION
    # ==========================================

    model.eval() # Mode évaluation activé (gèle le BatchNorm pour qu'il soit stable)
    epoch_val_loss = 0.0

    # On itère sur notre dataloader (qui nous donne des batchs de chunk_duration secondes)
    with torch.no_grad():
        for batch_idx, (X_batch_db, X_batch_phase, y_batch_db, y_batch_phase) in enumerate(musdb_val_loader):
            X_batch_db = X_batch_db.to(device)
            y_batch_db = y_batch_db.to(device)

            # Acquisition des prédictions
            y_predi_batch_db = model(X_batch_db)

            # Calcul de la loss
            loss = criterion(y_predi_batch_db, y_batch_db)
            epoch_val_loss += loss.item()

    avg_val_loss = epoch_val_loss / len(musdb_val_loader)

    # ==========================================
    #             BILAN DE L'ÉPOQUE
    # ==========================================
    print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # 🏆 Sauvegarde du meilleur modèle ("Checkpointing")
    # Si la perte de validation est la plus basse jamais vue, on sauvegarde les poids.
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "./unet_models/meilleur_unet_vocal.pth")
        print(f"   💾 Nouveau record ! Modèle sauvegardé avec Val Loss: {best_val_loss:.4f}")

print("\n✅ Entraînement terminé ! Le meilleur modèle a été sauvegardé.")





