#imports 
import torch 
import torchaudio
import matplotlib.pyplot as plt 

def audio_to_spectrogram_db(track, n_fft, hop_length):
    """
    Convert a file into a spectrogram usable for machine learning

    Args:
        track (torch): data structure of the track


    Returns:
        magnitude: 
        phase:
        magnitude_db:

    Raises:
        ValueError: (Optionnel) Documenter les erreurs que la fonction 
                    pourrait déclencher si on lui donne de mauvaises entrées.
    """

    #Reducing data sample size by two 
    if track.shape[0] > 1: 
        track = torch.mean(track,dim=0,keepdim=True)

    # creating the stft image 
    # return_complex=True est la norme sur les versions récentes de PyTorch
    stft_result = torch.stft(
        track,
        n_fft=n_fft,
        hop_length=hop_length,
        return_complex=True
    )
    
    #Récupération de la magnitude et de la phase 
    magnitude = torch.abs(stft_result)  # L'intensité (notre future "image")
    phase = torch.angle(stft_result)    # Le timing (à garder de côté pour la fin)
    magnitude_db = 20 * torch.log10(magnitude + 1e-5)
    
    return magnitude, phase, magnitude_db

