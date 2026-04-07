import torch
import torch.nn as nn

class DownConvBlock(nn.Module):
    """
    Bloc de convolution pour l'Encodeur (La descente dans le U-Net).
    Objectif : Diviser la taille spatiale de l'image (spectrogramme) par 2 
    tout en extrayant les caractéristiques (augmentation des canaux).
    Donc on diminue la taille par 2, on se déplace de 2 pixel par 2 pixel et on a une largeur de padding de 2 autour pour se concentrer dans l'image 
    Ensuite on batch norm pour pas avoir de shift de divergence entre les blocks de convolution 
    Enfin activation une leaky relu pour éviter les neurones morts en négatif. 
    """
    def __init__ (self, in_channels, out_channels ):
        super(DownConvBlock, self).__init__()

        # 1. Convolution 2D : kernel 5x5, stride 2, padding 2 (Le standard Spleeter/Jansson)
        # Le stride de 2 est le secret : il divise la largeur et la hauteur de l'image par 2.
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=5,
            stride=2,
            padding=2)
        
        # 2. Batch Normalization : Crucial pour le son. 
        # Ça centre et réduit les valeurs pour éviter que le modèle ne diverge.
        self.batch_norm = nn.BatchNorm2d(
            num_features = out_channels
        )

        self.activation = nn.LeakyReLU(0.2)
    
    def forward(self,x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        return x 
    
class UpConvBlock(nn.Module):
    """
    Bloc de déconvolution pour le Décodeur (La remontée dans le U-Net).
    Objectif : Multiplier la taille spatiale de l'image par 2 (décompression) 
    et fusionner avec les détails de l'encodeur (Skip Connection).
    """

    def __init__(self, in_channels, out_channels):
        super(UpConvBlock,self).__init__()

        # 1. La Décompression (Transposed Convolution)
        # C'est l'opération inverse du stride=2. Elle double la taille de l'image.
        # Le output_padding=1 est crucial ici pour s'assurer que les dimensions 
        # retombent exactement au pixel près sur la taille d'origine.
        self.up_conv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size= 5,
            padding=2,
            stride=2,
            output_padding=1
        )

        # 2. La Fusion (Convolution classique après la concaténation)
        # Après avoir collé les canaux de l'encodeur avec ceux du décodeur, 
        # on aura 2 fois plus de canaux (out_channels * 2). Cette couche permet 
        # de mélanger ces informations et de retomber sur 'out_channels'.
        self.conv = nn.Conv2d(
            in_channels= 2 * out_channels,
            out_channels= out_channels,
            kernel_size= 3,
            padding= 1
        )

        self.batch_norm = nn.BatchNorm2d(num_features=out_channels)
        self.activation = nn.ReLU()

    def forward(self, x, skip_connection):
        # 1. On décompresse l'image qui vient du bas du réseau
        x = self.up_conv(x)

        # 2. LA MAGIE DU U-NET : La Skip Connection
        # On colle (concatène) l'image décompressée avec l'image haute définition 
        # que l'on avait gardée de côté pendant la descente. dim=1 correspond à l'axe des canaux.
        x = torch.cat([x,skip_connection],dim=1)

        # 3. On lisse et on active
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)

        return x

    
class UnetAudioStemmer(nn.Module):
    """
    Le modèle U-Net principal pour la séparation de voix (Stemming).
    """

    def __init__(self):
        super(UnetAudioStemmer,self).__init__()
        # --- L'ENCODEUR (La descente) ---
        # Notre entrée est un spectrogramme mono (1 canal).
        # On augmente progressivement le nombre de filtres pour comprendre le son.
        self.down1 = DownConvBlock(in_channels=1, out_channels=32)
        self.down2 = DownConvBlock(in_channels=32, out_channels=64)
        self.down3 = DownConvBlock(in_channels=64, out_channels=128)

        # --- LE BOTTLENECK (Le fond du "U") ---
        # L'endroit où l'image est la plus compressée. Le modèle a ici une vision 
        # globale de la musique (le rythme, la présence vocale globale).
        self.bottleneck = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=5,
                stride=2,
                padding=2
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        # --- LE DÉCODEUR (La remontée) ---
        # les blocs de déconvolution (ConvTranspose2d) !
        self.up1 = UpConvBlock(in_channels=256, out_channels=128)
        self.up2 = UpConvBlock(in_channels=128, out_channels=64)
        self.up3 = UpConvBlock(in_channels=64, out_channels=32)

        # Pour juste repasser à la bonne size comme dans l'entrée
        self.up4 = nn.ConvTranspose2d(
            in_channels=32, 
            out_channels=32, # On garde 16 canaux, on veut juste doubler la taille
            kernel_size=5, 
            stride=2, 
            padding=2, 
            output_padding=1
        )

        # --- COUCHE FINALE ---
        # On ramène nos 16 canaux à 1 seul canal (notre masque en noir et blanc)
        # On utilise kernel_size=1, ce qui revient à scanner chaque pixel individuellement 
        # sans regarder ses voisins, juste pour fusionner les canaux finaux.
        self.final_conv = nn.Conv2d(in_channels=32,
                                    out_channels=1,
                                    kernel_size=1,
                                    )
        
        # La Sigmoïde force la sortie finale entre 0 et 1 (C'est notre "Masque")
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1. ENCODAGE (Descente)
        d1 = self.down1(x) 
        d2 = self.down2(d1) 
        d3 = self.down3(d2) 

        # 2. BOTTLENECK
        bottom = self.bottleneck(d3) # 64

        #3. DECODAGE (Remontée)
        u1 = self.up1(bottom, d3) 
        u2 = self.up2(u1, d2)
        u3 = self.up3(u2,d1)

        #4. Upsample pour avoir la même shape que l'input
        u4 = self.up4(u3)

        # 4. MASQUE FINAL
        out = self.final_conv(u4)
        mask = self.sigmoid(out)
        # On multiplie le spectrogramme d'entrée original par le masque généré
        # pour isoler les fréquences de la voix.
        separated_audio_spectrogram = x * mask

        return separated_audio_spectrogram


        







