# Ce programme double la taille d'une image en assayant de ne pas trop pixeliser l'image.

from PIL import Image
import os
import numpy as np
from scipy import signal
from mpi4py import MPI

# Fonction pour doubler la taille d'une image sans trop la pixeliser
# en parallélisant le calcul des convolutions par bandes horizontales.
def double_size_parallel(image):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        # On charge l'image
        img = Image.open(image)
        print(f"Taille originale {img.size}")
        # Convertir la représentation RGB en HSV :
        img = img.convert('HSV')
        # On convertit l'image en tableau numpy
        img = np.array(img, dtype=np.double)
        # On double sa taille et on la normalise
        img = np.repeat(np.repeat(img, 2, axis=0), 2, axis=1) / 255.
        print(f"Nouvelle taille : {img.shape}")
        height = img.shape[0]
    else:
        img = None
        height = None

    # Tous les processus connaissent la hauteur totale pour définir leur bande
    height = comm.bcast(height, root=0)

    def bounds(h, p, r):
        start = r * h // p
        end = (r + 1) * h // p
        return start, end

    # Le rang 0 découpe et envoie les bandes avec une ligne fantôme
    if rank == 0:
        local_start, local_end = bounds(height, size, 0)
        halo_start = max(0, local_start - 1)
        halo_end = min(height, local_end + 1)
        local_img = img[halo_start:halo_end, :, :]

        for r in range(1, size):
            s, e = bounds(height, size, r)
            h_start = max(0, s - 1)
            h_end = min(height, e + 1)
            sub = img[h_start:h_end, :, :]
            meta = (sub.shape, s, e, h_start, h_end)
            comm.send(meta, dest=r, tag=0)
            if sub.size > 0:
                comm.Send([sub, MPI.DOUBLE], dest=r, tag=1)
    else:
        meta = comm.recv(source=0, tag=0)
        shape, local_start, local_end, halo_start, halo_end = meta
        if np.prod(shape) > 0:
            local_img = np.empty(shape, dtype=np.double)
            comm.Recv([local_img, MPI.DOUBLE], source=0, tag=1)
        else:
            local_img = np.empty(shape, dtype=np.double)

    # Filtre gaussien 3x3
    gauss_mask = np.array([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]]) / 16.
    blur_image = np.zeros_like(local_img, dtype=np.double)
    for i in range(3):
        if local_img.shape[0] > 0:
            blur_image[:, :, i] = signal.convolve2d(local_img[:, :, i], gauss_mask, mode='same')

    # Filtre de netteté 3x3 sur la composante V
    sharp_mask = np.array([[0., -1., 0.], [-1., 5., -1.], [0., -1., 0.]])
    sharpen_local = np.zeros_like(local_img, dtype=np.double)
    sharpen_local[:, :, :2] = blur_image[:, :, :2]
    if local_img.shape[0] > 0:
        sharpen_local[:, :, 2] = np.clip(
            signal.convolve2d(blur_image[:, :, 2], sharp_mask, mode='same'), 0., 1.
        )

    # On enlève les lignes fantômes (une éventuelle en haut et/ou en bas)
    local_height = local_img.shape[0]
    top_ghost = 1 if halo_start < local_start else 0
    bottom_ghost = 1 if halo_end > local_end else 0
    if local_height > 0:
        interior = sharpen_local[top_ghost:local_height - bottom_ghost, :, :]
    else:
        interior = sharpen_local

    # Rassemblement des bandes sur le rang 0
    if rank == 0:
        full_img = np.empty((height, local_img.shape[1], 3), dtype=np.double)
        full_img[local_start:local_end, :, :] = interior
        for r in range(1, size):
            meta = comm.recv(source=r, tag=2)
            s, e, shape = meta
            if np.prod(shape) > 0:
                buf = np.empty(shape, dtype=np.double)
                comm.Recv([buf, MPI.DOUBLE], source=r, tag=3)
                full_img[s:e, :, :] = buf
        sharpen_image = (255. * full_img).astype(np.uint8)
        return Image.fromarray(sharpen_image, 'HSV').convert('RGB')
    else:
        meta = (local_start, local_end, interior.shape)
        comm.send(meta, dest=0, tag=2)
        if interior.size > 0:
            comm.Send([interior, MPI.DOUBLE], dest=0, tag=3)
        return None


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    path = "datas/"
    image = path + "paysage.jpg"

    doubled_image = double_size_parallel(image)

    if rank == 0:
        if not os.path.exists("sorties"):
            os.makedirs("sorties")
        # On sauvegarde l'image modifiée
        doubled_image.save("sorties/paysage_double.jpg")
        print("Image sauvegardée")


if __name__ == "__main__":
    main()
