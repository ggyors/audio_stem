def pad_or_crop_to_multiple(tensor, multiple=8):
    """
    Ajuste les dimensions d'un tenseur 4D (Batch, Channels, Freq, Time) 
    pour qu'elles soient des multiples stricts.
    """
    # 1. Ajuster les fréquences (ex: passer de 1025 à 1024)
    freq_dim = tensor.size(2)
    new_freq = (freq_dim // multiple) * multiple
    tensor = tensor[:, :, :new_freq, :]
    
    # 2. Ajuster le temps
    time_dim = tensor.size(3)
    new_time = (time_dim // multiple) * multiple
    tensor = tensor[:, :, :, :new_time]
    
    return tensor