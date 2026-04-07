import sys
import os
# On récupère le chemin absolu du dossier parent (..) et on l'ajoute au sys.path
dossier_parent = os.path.abspath('..')
if dossier_parent not in sys.path:
    sys.path.append(dossier_parent)

import torch
from torch.utils.data import Dataset
import random
import musdb
from tool_box.converter import audio_to_spectrogram_db


class MUSDBDataset(Dataset):
    """
    Classe représentant les procédés de chargement des données de musique.
    """

    def __init__(self, data_root, subset, chunk_duration):
        super(MUSDBDataset, self).__init__()

        # Chargement de la base de donnée 
        self.mus = musdb.DB(root=data_root, subsets=subset, is_wav=False)
        self.tracks = self.mus.tracks

        # Constantes
        self.chunk_duration = chunk_duration
        self.n_fft = 2048
        self.hop_length = 512
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return len(self.tracks)
    
    def __getitem__(self, index):
        track = self.tracks[index]
        max_start = track.duration - self.chunk_duration
        track.chunk_start = random.uniform(0,max_start)
        X_audio = torch.from_numpy(track.audio.T).to(self.device)
        y_audio = torch.from_numpy(track.targets["vocals"].audio.T).to(self.device)
        X_spectro_mag, X_phase, X_spectro_db = audio_to_spectrogram_db(X_audio, 
                                         n_fft=self.n_fft, 
                                         hop_length=self.hop_length)
        y_spectro_mag, y_phase, y_spectro_db = audio_to_spectrogram_db(y_audio, 
                                         n_fft=self.n_fft, 
                                         hop_length=self.hop_length)
        
        return X_spectro_db, y_spectro_db

        
