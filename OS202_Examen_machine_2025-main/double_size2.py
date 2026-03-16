# Ce programme double la taille d'une image en assayant de ne pas trop pixeliser l'image.

from PIL import Image
import os
import numpy as np
from scipy import signal
from mpi4py import MPI

# Fonction pour doubler la taille d'une image sans trop la pixeliser
# en appliquant un filtre gaussien sur H,S et un filtre 5x5 sur V
# de manière parallélisée par bandes horizontales.
def double_size_parallel(image):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        # On charge l'image
        img = Image.open(image)
        print(f"Taille originale {img.size}")
        img = img.convert('HSV')
        # On convertit l'image en tableau numpy
        img = np.array(img, dtype=np.double)
        # On double sa taille et on normalise
        img = np.repeat(np.repeat(img, 2, axis=0), 2, axis=1) / 255.
        print(f"Nouvelle taille : {img.shape}")
        height = img.shape[0]
    else:
        img = None
        height = None

    height = comm.bcast(height, root=0)

    def bounds(h, p, r):
        start = r * h // p
        end = (r + 1) * h // p
        return start, end

    # Découpage en bandes avec lignes fantômes
    if rank == 0:
        local_start, local_end = bounds(height, size, 0)
        # Rayon 2 pour le filtre 5x5 sur V
        halo_start = max(0, local_start - 2)
        halo_end = min(height, local_end + 2)
        local_img = img[halo_start:halo_end, :, :]

        for r in range(1, size):
            s, e = bounds(height, size, r)
            h_start = max(0, s - 2)
            h_end = min(height, e + 2)
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

    # Masque gaussien pour H,S
    gauss_mask = np.array([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]]) / 16.
    blur_image = np.zeros_like(local_img, dtype=np.double)
    for i in range(2):
        if local_img.shape[0] > 0:
            blur_image[:, :, i] = signal.convolve2d(local_img[:, :, i], gauss_mask, mode='same')
    # V est initialement simplement recopiée
    if local_img.shape[0] > 0:
        blur_image[:, :, 2] = local_img[:, :, 2]

    # Masque 5x5 combinant lissage et netteté sur V
    sharp_mask = -np.array([
        [1., 4., 6., 4., 1.],
        [4., 16., 24., 16., 4.],
        [6., 24., -476., 24., 6.],
        [4., 16., 24., 16., 4.],
        [1., 4., 6., 4., 1.]
    ]) / 256.
    if local_img.shape[0] > 0:
        blur_image[:, :, 2] = np.clip(
            signal.convolve2d(blur_image[:, :, 2], sharp_mask, mode='same'), 0., 1.
        )

    local_height = local_img.shape[0]
    # Nombre de lignes fantômes réellement ajoutées en haut et en bas
    top_ghost = local_start - halo_start
    bottom_ghost = halo_end - local_end
    if local_height > 0:
        interior = blur_image[top_ghost:local_height - bottom_ghost, :, :]
    else:
        interior = blur_image

    # Rassemblement sur le rang 0
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
        blur_image = (255. * full_img).astype(np.uint8)
        return Image.fromarray(blur_image, 'HSV').convert('RGB')
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
        doubled_image.save("sorties/paysage_double_2.jpg")
        print("Image sauvegardée")


if __name__ == "__main__":
    main()
