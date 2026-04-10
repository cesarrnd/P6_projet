import numpy as np
import h5py

nom_fichier = r"C:\Users\guilh\Desktop\UV-PROJET\gen4 reduced raw\mini_dataset\train\moorea_2019-02-19_004_td_610500000_670500000_td.dat"


def dat_to_h5_streaming(dat_path, h5_path, height=720, width=1280, slice_us=50000, chunk_size=1_000_000):
    """
    Conversion .dat -> .h5 en streaming
    dat_path: chemin vers le fichier .dat
    h5_path: chemin vers le fichier .h5 de sortie (Si le fichier n'existe pas, il sera créé)
    height: hauteur de l'image
    width: largeur de l'image
    slice_us: durée d'une tranche en microsecondes

    Output: dataset (T, 2, H, W)
    """

    def skip_header(f):
        while True:
            pos = f.tell()
            line = f.readline()
            if not line.startswith(b"%"):
                f.seek(pos)
                break

    with open(dat_path, "rb") as f, h5py.File(h5_path, "w") as h5:

        skip_header(f)

        # Dataset extensible
        dset = h5.create_dataset(
            "events",
            shape=(0, 2, height, width),
            maxshape=(None, 2, height, width),
            dtype=np.uint16,
            compression="gzip",
            chunks=(1, 2, height, width)
        )

        current_slice = np.zeros((2, height, width), dtype=np.uint16)

        current_slice_idx = 0
        first_timestamp = None

        total_slices_written = 0

        while True:
            # Lire chunk
            data = np.fromfile(f, dtype=np.uint32, count=chunk_size * 2)

            if data.size == 0:
                break

            data = data.reshape(-1, 2)

            t = data[:, 0]
            raw = data[:, 1]

            if first_timestamp is None:
                first_timestamp = t[0]

            t = t - first_timestamp

            x = raw & 0x3FFF
            y = (raw >> 14) & 0x3FFF
            p = (raw >> 28) & 1

            slice_idx = (t // slice_us).astype(np.int64)

            for i in range(len(t)):
                s = slice_idx[i]

                # Si on change de tranche → on écrit
                while s > current_slice_idx:
                    dset.resize(total_slices_written + 1, axis=0)
                    dset[total_slices_written] = current_slice

                    total_slices_written += 1
                    current_slice = np.zeros((2, height, width), dtype=np.uint16)
                    current_slice_idx += 1

                xi, yi, pi = x[i], y[i], p[i]

                if xi < width and yi < height:
                    current_slice[pi, yi, xi] += 1

        # Écriture dernière tranche
        dset.resize(total_slices_written + 1, axis=0)
        dset[total_slices_written] = current_slice

    print(f"Conversion terminée ✅")
    print(f"Nombre de tranches écrites: {total_slices_written + 1}")