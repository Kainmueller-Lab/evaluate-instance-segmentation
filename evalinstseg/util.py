import logging
import os

import h5py
import numpy as np
from scipy.optimize import linear_sum_assignment
from skimage.morphology import skeletonize_3d
from skimage.segmentation import relabel_sequential
import tifffile
import zarr

logger = logging.getLogger(__name__)


def crop(arr, shape):
    """center-crop arr to shape

    Args
    ----
    arr: ndarray
        data
    shape: list of int
        crop data to this shape, only last len(shape) dims

    Returns
    -------
    cropped array
    """

    target_shape = arr.shape()[:-len(shape)] + shape

    offset = tuple(
        (a - b)//2
        for a, b in zip(arr.shape(), target_shape))

    slices = tuple(
        slice(o, o + s)
        for o, s in zip(offset, target_shape))

    return arr[slices]


def check_and_fix_sizes(gt_labels, pred_labels, ndim):
    """check if prediction and gt have same size, otherwise crop the bigger one
    add channel dimension if missing (channel_first)

    E.g., if valid padding is used output might be smaller than input.

    Note
    ----
    only crops spatial dimensions, assumes channel_first
    """
    # add channel dim if not there already
    if gt_labels.ndim == ndim:
        logger.debug("adding channel dim to gt")
        gt_labels = np.expand_dims(gt_labels, axis=0)
    if pred_labels.ndim == ndim:
        logger.debug("adding channel dim to pred")
        pred_labels = np.expand_dims(pred_labels, axis=0)

    if gt_labels.shape[-ndim:] == pred_labels.shape[-ndim:]:
        logger.debug("no cropping necessary")
        return gt_labels, pred_labels

    if np.all([gt_s <= p_s for gt_s, p_s in zip(
            gt_labels.shape[-ndim:], pred_labels.shape[-ndim:])]):
        pred_labels = crop(pred_labels, gt_labels.shape[-ndim:])
    elif np.all([p_s <= gt_s for gt_s, p_s in zip(
            gt_labels.shape[-ndim:], pred_labels.shape[-ndim:])]):
        gt_labels = crop(gt_labels, pred_labels.shape[-ndim:])
    else:
        raise RuntimeError(
            "gt is bigger in some spatial dims, pred in others, unable to "
            "crop to same shape")

    logger.debug("gt shape cropped %s", gt_labels.shape)
    logger.debug("pred shape cropped %s", pred_labels.shape)

    return gt_labels, pred_labels


def binary_masks_to_uniq_ids(labels):
    """if label array has binary masks per channel, relabel to unique ids"""
    assert np.max(labels) <= 1, "labels is not a binary mask"
    labels = labels.astype(np.min_scalar_type(labels.shape[0]))
    for i in range(labels.shape[0]):
        labels[i] = labels[i] * (i + 1)

    return labels


def remove_empty_channels(labels):
    if labels.shape[0] == 1:
        return labels

    tmp_labels = []
    for i in range(labels.shape[0]):
        if np.max(labels[i]) > 0:
            tmp_labels.append(labels[i])

    return np.array(tmp_labels)


def check_fix_and_unify_ids(
        gt_labels, pred_labels, remove_small_components, foreground_only,
        dim_insts=[]):
    """unify prediction and gt labelling styles

    Note
    ----
    labelling should be
     - sequential, starting at 1 (background = 0)
     - if stored in multiple channels
       - masks in each channel should still have unique id
    """
    assert np.min(gt_labels) >= 0, "found negative ids in gt label array"
    assert np.min(pred_labels) >= 0, "found negative ids in pred label array"
    if np.max(gt_labels) == 0:
        logger.warning("gt label array is empty")
    if np.max(pred_labels) == 0:
        logger.warning("pred label array is empty")

    # optional: remove small components
    if remove_small_components is not None and remove_small_components > 0:
        logger.info(
            "remove small components with size < %i", remove_small_components)
        pred_labels = filter_components(pred_labels, remove_small_components)

    # optional:if foreground_only, remove all predictions within gt background
    # (rarely useful)
    if foreground_only:
        if (pred_labels.shape[0] == 1 and
            np.all(
                [ps == gs
                 for ps, gs in zip(pred_labels.shape, gt_labels.shape)])):
            pred_labels[gt_labels==0] = 0
        else:
            pred_labels[:, np.all(gt_labels, axis=0).astype(int)==0] = 0

    # after filtering, some channels might be empty
    pred_labels = remove_empty_channels(pred_labels)
    gt_labels = remove_empty_channels(gt_labels)

    # if each channel is a binary mask, still assign a unique id to each
    if np.max(pred_labels) == 1:
        pred_labels = binary_masks_to_uniq_ids(pred_labels)
    if np.max(gt_labels) == 1:
        gt_labels = binary_masks_to_uniq_ids(gt_labels)

    # relabel labels sequentially
    offset = 1
    for i in range(pred_labels.shape[0]):
        pred_labels[i], _, _ = relabel_sequential(
            pred_labels[i].astype(int), offset)
        offset = np.max(pred_labels[i]) + 1
    offset = 1
    fw_map = None
    for i in range(gt_labels.shape[0]):
        gt_labels[i], c_fw_map, _ = relabel_sequential(
            gt_labels[i].astype(int), offset)
        if fw_map is None:
            fw_map = c_fw_map
        else:
            ids = np.isin(c_fw_map.in_values, fw_map.in_values, invert=True)
            if np.any(ids):
                fw_map.in_values = np.concatenate(
                        [fw_map.in_values, c_fw_map.in_values[ids]])
                fw_map.out_values = np.concatenate(
                        [fw_map.out_values, c_fw_map.out_values[ids]])
        offset = np.max(gt_labels[i]) + 1

    if len(dim_insts) > 0:
        dim_insts_rel = list(fw_map[np.array(dim_insts)])
        return gt_labels, pred_labels, dim_insts_rel
    else:
        return gt_labels, pred_labels


def read_file(infn, key, read_dim=False):
    """read image/volume in hdf, tif and zarr format"""
    if infn.endswith(".hdf"):
        with h5py.File(infn, 'r') as f:
            volume = np.array(f[key])
    elif infn.endswith(".tif") or infn.endswith(".tiff") or \
         infn.endswith(".TIF") or infn.endswith(".TIFF"):
        volume = tifffile.imread(infn)
    elif infn.endswith(".zarr"):
        try:
            f = zarr.open(infn, 'r')
        except zarr.errors.PathNotFoundError as e:
            logger.info("File %s not found!", infn)
            raise e
        volume = np.array(f[key])
        if read_dim:
            if "dim_neurons" in f[key].attrs.keys():
                dim_insts = f[key].attrs["dim_neurons"]
    else:
        raise NotImplementedError("invalid file format %s", infn)
    if read_dim:
        return volume, dim_insts
    else:
        return volume


def get_output_name(
        out_dir, res_file, res_key, suffix, localization_criterion,
        assignment_strategy, remove_small_components):
    """constructs output file name based on used eval strategy"""
    outFnBase = os.path.join(
        out_dir,
        os.path.splitext(os.path.basename(res_file))[0] +
        "_" + res_key.replace("/","_") + suffix)
    # add localization criterion
    outFnBase += "_" + localization_criterion
    # add assignment strategy
    outFnBase += "_" + assignment_strategy
    # add remove small components
    if remove_small_components is not None and remove_small_components > 0:
        outFnBase += "_rm" + str(remove_small_components)
    outFn = outFnBase
    os.makedirs(os.path.dirname(outFnBase), exist_ok=True)

    return outFn

def replace(array, old_values, new_values):
    """fast function to replace set of values in array with new values"""
    values_map = np.arange(int(array.max() + 1), dtype=new_values.dtype)
    values_map[old_values] = new_values

    return values_map[array]


def filter_components(volume, thresh):
    """remove instances smaller than `thresh` pixels"""
    labels, counts = np.unique(volume, return_counts=True)
    small_labels = labels[counts <= thresh]

    volume = replace(
        volume,
        np.array(small_labels),
        np.array([0] * len(small_labels))
    )
    return volume


def get_gt_coverage(gt_labels, pred_labels, precMat, recallMat):
    # only take max gt label for each pred label to not count
    # pred labels twice for overlapping gt instances
    num_pred_labels = int(np.max(pred_labels))
    num_gt_labels = int(np.max(gt_labels))
    max_gt_ind = np.argmax(precMat, axis=0)

    gt_cov = []
    # if both gt and pred have overlapping instances
    if (np.any(np.sum(gt_labels, axis=0) != np.max(gt_labels, axis=0)) and
        np.any(np.sum(pred_labels, axis=0) != np.max(pred_labels, axis=0))):
        # recalculate clRecall for each gt and union of assigned
        # predictions, as predicted instances can potentially overlap
        max_gt_ind_unique = np.unique(max_gt_ind[max_gt_ind > 0])
        for gt_i in np.arange(1, num_gt_labels + 1):
            if gt_i in max_gt_ind_unique:
                pred_union = np.zeros(
                    pred_labels.shape[1:],
                    dtype=pred_labels.dtype)
                for pred_i in np.arange(num_pred_labels + 1)[max_gt_ind == gt_i]:
                    mask = np.max(pred_labels == pred_i, axis=0)
                    pred_union[mask] = 1
                gt_cov.append(get_centerline_overlap_single(
                    gt_labels, pred_union, gt_i, 1))
            else:
                gt_cov.append(0.0)
    else:
        # otherwise use previously computed values
        for i in range(1, recallMat.shape[0]):
            gt_cov.append(np.sum(recallMat[i, max_gt_ind==i]))
    return gt_cov


def get_centerline_overlap_single(
        to_skeletonize, compare_with, skeletonize_label, compare_label):
    """skeletonizes `to_skeletonize` and checks how much overlap
    `compare_with` has with the skeletons, for a single pair of labels)
    """
    to_skeletonize = to_skeletonize == skeletonize_label
    if to_skeletonize.ndim == 4:
        to_skeletonize = np.max(to_skeletonize, axis=0)
    # note: skeletonize_3d also works for 2d images
    skeleton = skeletonize_3d(to_skeletonize) > 0
    compare_with = compare_with == compare_label
    if compare_with.ndim == 4:
        compare_with == np.max(compare_with, axis=0)

    return (np.sum(compare_with[skeleton], dtype=float)
            / np.sum(skeleton, dtype=float))



