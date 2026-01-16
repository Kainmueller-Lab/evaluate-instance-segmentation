# Support functions to unify visualization I/O in HDF and PNG
import numpy as np
import h5py
from skimage import io

def _mid_slice(arr):
    arr = np.asarray(arr)
    if arr.ndim == 3:              # (Z,Y,X)
        return arr[arr.shape[0] // 2]
    if arr.ndim == 4 and arr.shape[-1] in (3, 4):  # (Z,Y,X,C)
        return arr[arr.shape[0] // 2]
    return arr

def save_vis_hdf(outFn, data_dict, compression="gzip"):
    """
    data_dict: mapping name -> ndarray, will be written under volumes/<name>
    """
    with h5py.File(outFn + "_vis.hdf", "w") as f:
        for name, arr in data_dict.items():
            f.create_dataset(f"volumes/{name}", data=arr, compression=compression)

def export_vis_pngs(outFn, data_dict):
    """
    Writes PNG quicklooks next to the HDF. Scalar layers assumed in [0,1].
    RGB layers assumed uint8 or float in [0,255] / [0,1].
    """
    for name, arr in data_dict.items():
        a = _mid_slice(arr)

        if a.ndim == 3 and a.shape[-1] in (3, 4):
            # RGB(A)
            if a.dtype != np.uint8:
                a = np.clip(a, 0, 1)
                a = (a * 255).astype(np.uint8)
            io.imsave(outFn + f"_{name}.png", a)
        else:
            # scalar
            a = np.clip(a, 0, 1)
            io.imsave(outFn + f"_{name}.png", (a * 255).astype(np.uint8))
