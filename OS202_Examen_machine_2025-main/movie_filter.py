# Ce programme va charger n images et y appliquer un filtre de netteté
# puis les sauvegarder dans un dossier de sortie

from PIL import Image
import os
import numpy as np
from scipy import signal
from mpi4py import MPI


# Fonction pour appliquer un filtre de netteté à une image
def apply_filter(image):
    # On charge l'image
    img = Image.open(image)
    print(f"Taille originale {img.size}")
    # Conversion en HSV :
    img = img.convert('HSV')
    # On convertit l'image en tableau numpy et on normalise
    img = np.repeat(np.repeat(np.array(img), 2, axis=0), 2, axis=1)
    img = np.array(img, dtype=np.double)/255.
    print(f"Nouvelle taille : {img.shape}")
    # Tout d'abord, on crée un masque de flou gaussien
    mask = np.array([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]]) / 16.
    # On applique le filtre de flou
    blur_image = np.zeros_like(img, dtype=np.double)
    for i in range(3):
        blur_image[:,:,i] = signal.convolve2d(img[:,:,i], mask, mode='same')
    # On crée un masque de netteté
    mask = np.array([[0., -1., 0.], [-1., 5., -1.], [0., -1., 0.]])
    # On applique le filtre de netteté
    sharpen_image = np.zeros_like(img)
    sharpen_image[:,:,:2] = blur_image[:,:,:2]
    sharpen_image[:,:,2] = np.clip(signal.convolve2d(blur_image[:,:,2], mask, mode='same'), 0., 1.)

    sharpen_image *= 255.
    sharpen_image = sharpen_image.astype(np.uint8)
    # On retourne l'image modifiée
    return Image.fromarray(sharpen_image, 'HSV').convert('RGB')


def main():
    # Initialisation de MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    path = "datas/perroquets/"
    out_dir = "sorties/perroquets"

    # Le processus 0 crée le répertoire de sortie si besoin
    if rank == 0 and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # On s'assure que le répertoire est créé avant d'écrire
    comm.Barrier()

    out_path = out_dir + "/"

    n_images = 37

    # Répartition simple : chaque processus traite un sous-ensemble d'images
    for i in range(rank, n_images, size):
        image_id = i + 1
        image = path + "Perroquet{:04d}.jpg".format(image_id)
        sharpen_image = apply_filter(image)
        sharpen_image.save(out_path + "Perroquet{:04d}.jpg".format(image_id))
        print(f"[rang {rank}] Image {image_id} traitée")

    comm.Barrier()
    if rank == 0:
        print("Traitement terminé - images sauvegardées")


if __name__ == "__main__":
    main()
