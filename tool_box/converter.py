#imports 
import torch 
import torchaudio
import matplotlib.pyplot as plt 

def audio_to_spectrogram(file_path):
    """
    Convert a file into a spectrogram usable for machine learning

    Args:
        file_path (str): link to a .mp3 or .wav file


    Returns:
        magnitude: 
        phase:
        magnitude_db:

    Raises:
        ValueError: (Optionnel) Documenter les erreurs que la fonction 
                    pourrait déclencher si on lui donne de mauvaises entrées.
    """
    waveform, sample_rate = torchaudio.load(file_path)
    
    #Reducing data sample size by two 
    if waveform.shape[0] > 1: 
        waveform = torch.mean(waveform,dim=0,keepdim=True)
    
    #stft parameters
    # n_fft : La "résolution" des fréquences (2048 est le standard en musique)
    # hop_length : Le nombre d'échantillons entre chaque "colonne" de pixels (le temps)
    n_fft = 2048
    hop_length = 512

    # creating the stft image 
    # return_complex=True est la norme sur les versions récentes de PyTorch
    stft_result = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        return_complex=True
    )

    #Récupération de la magnitude et de la phase 
    magnitude = torch.abs(stft_result)  # L'intensité (notre future "image")
    phase = torch.angle(stft_result)    # Le timing (à garder de côté pour la fin)
    magnitude_db = 20 * torch.log10(magnitude + 1e-5)
    
    return magnitude, phase, magnitude_db

