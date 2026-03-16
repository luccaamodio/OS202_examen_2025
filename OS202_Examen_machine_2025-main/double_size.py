# Ce programme double la taille d'une image en assayant de ne pas trop pixeliser l'image.

from PIL import Image
import os
import numpy as np
from scipy import signal

# Fonction pour doubler la taille d'une image sans trop la pixeliser
def double_size(image):
    # On charge l'image
    img = Image.open(image)
    print(f"Taille originale {img.size}")
    # Convertir la représentation RGB en HSV :
    img = img.convert('HSV')
    # On convertit l'image en tableau numpy
    img = np.array(img, dtype=np.double)
    # On double sa taille et on la normalise
    img = np.repeat(np.repeat(img, 2, axis=0), 2, axis=1)/255.
    print(f"Nouvelle taille : {img.shape}")
    # On crée un masque de flou gaussien
    mask = np.array([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]]) / 16.
    # On applique le filtre de flou
    blur_image = np.zeros_like(img, dtype=np.double)
    for i in range(3):
        blur_image[:,:,i] = signal.convolve2d(img[:,:,i], mask, mode='same')
    # On crée un masque de netteté
    mask = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    # On applique le filtre de netteté uniquement sur la luminance :
    sharpen_image = np.zeros_like(img, dtype=np.double)
    sharpen_image[:,:,:2] = blur_image[:,:,:2]
    sharpen_image[:,:,2] = np.clip(signal.convolve2d(blur_image[:,:,2], mask, mode='same'), 0., 1.)
    # On retourne l'image modifiée
    sharpen_image = (255.*sharpen_image).astype(np.uint8)
    return Image.fromarray(sharpen_image, 'HSV').convert('RGB')

path = "datas/"
image = path+"paysage.jpg"
doubled_image = double_size(image)
# On sauvegarde l'image modifiée
doubled_image.save("sorties/paysage_double.jpg")
print("Image sauvegardée")
