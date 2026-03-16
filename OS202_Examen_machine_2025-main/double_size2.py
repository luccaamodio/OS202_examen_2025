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
    img = img.convert('HSV')
    # On convertit l'image en tableau numpy
    img = np.array(img, dtype=np.double)
    # On double sa taille et on normalise
    img = np.repeat(np.repeat(img, 2, axis=0), 2, axis=1)/255.
    print(f"Nouvelle taille : {img.shape}")
    # On crée un masque de flou gaussien pour la teinte et la saturation (H et S)
    mask = np.array([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]]) / 16.
    # On applique le filtre de flou
    blur_image = np.zeros_like(img, dtype=np.double)
    for i in range(2):
        blur_image[:,:,i] = signal.convolve2d(img[:,:,i], mask, mode='same')
    blur_image[:,:,2] = img[:,:,2]
    # On crée un masque de flou 5x5 :
    mask = -np.array([[1., 4., 6., 4., 1.], [4., 16., 24., 16., 4.], [6., 24., -476., 24., 6.], [4., 16., 24., 16., 4.], [1., 4., 6., 4., 1.]]) / 256
    # On applique le filtre sur la luminance:
    blur_image[:,:,2] = np.clip(signal.convolve2d(blur_image[:,:,2], mask, mode='same'), 0., 1.)
    blur_image = (255.*blur_image).astype(np.uint8)
    # On retourne l'image modifiée
    return Image.fromarray(blur_image, 'HSV').convert('RGB')

path = "datas/"
image = path+"paysage.jpg"
doubled_image = double_size(image)
# On sauvegarde l'image modifiée
doubled_image.save("sorties/paysage_double_2.jpg")
print("Image sauvegardée")
