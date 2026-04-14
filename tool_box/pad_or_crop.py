def pad_or_crop_to_multiple(tensor, multiple=8):
    """
    Ajuste les dimensions d'un tenseur 4D (Batch, Channels, Freq, Time) 
    pour qu'elles soient des multiples stricts.
    """
    
    # 2. Ajuster le temps
    time_dim = tensor.shape[2]
    print(time_dim)
    new_time = (time_dim // multiple) * multiple
    tensor = tensor[ :, :, :new_time]
    print(tensor.shape)
    return tensor