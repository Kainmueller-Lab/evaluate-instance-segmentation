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
        gt_labels, pred_labels, remove_small_components, foreground_only):
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
    for i in range(gt_labels.shape[0]):
        gt_labels[i], _, _ = relabel_sequential(
            gt_labels[i].astype(int), offset)
        offset = np.max(gt_labels[i]) + 1

    return gt_labels, pred_labels


def read_file(infn, key):
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
    else:
        raise NotImplementedError("invalid file format %s", infn)
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


def get_centerline_overlap(to_skeletonize, compare_with, match):
    """skeletonizes `to_skeletonize` and checks how much overlap
    `compare_with` has with the skeletons, for all pairs of labels
    (incl. the background label in `compare_with`)
    """
    # make sure that arrays are of type int
    to_skeletonize = to_skeletonize.astype(int)
    compare_with = compare_with.astype(int)
    # assumes channel_dim
    fg = np.max(compare_with > 0, axis=0).astype(np.uint8)

    # assumes uniquely labeled instances (i.e., no binary masks per channel)
    skeleton_ids = np.unique(to_skeletonize[to_skeletonize > 0])
    for skeleton_id in skeleton_ids:
        logger.debug("compute centerline overlap for %i", skeleton_id)
        # binarize
        mask = to_skeletonize == skeleton_id
        # remove channel dim via max projection
        mask = np.max(mask, axis=0)
        # skeletonize
        skeleton = skeletonize_3d(mask.astype(np.uint8)) > 0
        skeleton_size = np.sum(skeleton)

        compare_labels, compare_labels_cnt = np.unique(
            compare_with[:, skeleton], return_counts=True)

        # remember to correct for bg label
        if 0 in compare_labels:
            compare_fg, compare_fg_cnt = np.unique(
                fg[skeleton], return_counts=True)
            if 0 in compare_fg:
                assert compare_labels[0] == 0 and compare_fg[0] == 0
                compare_labels_cnt[0] = compare_fg_cnt[0]
            else:
                compare_labels = compare_labels[1:]
                compare_labels_cnt = compare_labels_cnt[1:]

        ratio = compare_labels_cnt / float(skeleton_size)
        match[skeleton_id, compare_labels] = ratio

    return match


# todo: define own functions per localization criterion
# todo: not use num_*_labels as parameters here?
def compute_localization_criterion(
        pred_labels, gt_labels, num_pred_labels, num_gt_labels,
        localization_criterion):
    """computes the localization part of the metric
    For each pair of labels in the prediction and gt,
    how much are they co-localized, based on the chosen criterion?
    """
    logger.debug("evaluate localization criterion for all gt and pred label pairs")

    # create matrices for pixelwise overlap measures
    locMat = np.zeros((num_gt_labels+1, num_pred_labels+1), dtype=np.float32)
    recallMat = np.zeros((num_gt_labels+1, num_pred_labels+1), dtype=np.float32)
    precMat = np.zeros((num_gt_labels+1, num_pred_labels+1), dtype=np.float32)
    recallMat_wo_overlap = None

    # intersection over union
    if localization_criterion == "iou":
        logger.debug("compute iou")
        pred_tile = [1, ] * pred_labels.ndim
        pred_tile[0] = gt_labels.shape[0]
        gt_tile = [1, ] * gt_labels.ndim
        gt_tile[1] = pred_labels.shape[0]
        pred_tiled = np.tile(pred_labels, pred_tile).flatten()
        gt_tiled = np.tile(gt_labels, gt_tile).flatten()
        mask = np.logical_or(pred_tiled > 0, gt_tiled > 0)
        overlay = np.array([pred_tiled[mask], gt_tiled[mask]])
        overlay_labels, overlay_labels_counts = np.unique(
            overlay, return_counts=True, axis=1)
        overlay_labels = np.transpose(overlay_labels)

        # get gt cell ids and the size of the corresponding cell
        gt_labels_list, gt_counts = np.unique(gt_labels, return_counts=True)
        gt_labels_count_dict = {}
        logger.debug("%s %s", gt_labels_list, gt_counts)
        for (l,c) in zip(gt_labels_list, gt_counts):
            gt_labels_count_dict[l] = c

        # get pred cell ids
        pred_labels_list, pred_counts = np.unique(
            pred_labels, return_counts=True)
        logger.debug("%s %s", pred_labels_list, pred_counts)
        pred_labels_count_dict = {}
        for (l,c) in zip(pred_labels_list, pred_counts):
            pred_labels_count_dict[l] = c

        for (u,v), c in zip(overlay_labels, overlay_labels_counts):
            iou = c / (gt_labels_count_dict[v] + pred_labels_count_dict[u] - c)

            locMat[v, u] = iou
            recallMat[v, u] = c / gt_labels_count_dict[v]
            precMat[v, u] = c / pred_labels_count_dict[u]

    # centerline dice
    elif localization_criterion == "cldice":
        logger.debug("compute cldice")
        # todo: transpose precMat
        precMat = get_centerline_overlap(
            pred_labels, gt_labels,
            np.transpose(precMat))
        precMat = np.transpose(precMat)
        recallMat = get_centerline_overlap(
            gt_labels, pred_labels,
            recallMat)

        if (np.any(np.sum(gt_labels, axis=0) != np.max(gt_labels, axis=0)) or
            np.any(np.sum(pred_labels, axis=0) != np.max(pred_labels, axis=0))):
            # get recallMat without overlapping gt labels for false merge
            # calculation later on
            gt_wo_overlap = gt_labels.copy()
            mask = np.sum(gt_labels > 0, axis=0) > 1
            gt_wo_overlap[:, mask] = 0
            pred_wo_overlap = pred_labels.copy()
            # todo: check how this should be done with overlapping inst in prediction
            pred_wo_overlap[:, mask] = 0

            recallMat_wo_overlap = get_centerline_overlap(
                gt_wo_overlap, pred_wo_overlap,
                np.zeros_like(recallMat))
        err = np.geterr()
        np.seterr(invalid='ignore')
        locMat = np.nan_to_num(2 * precMat * recallMat / (precMat + recallMat))
        np.seterr(invalid=err['invalid'])
    else:
        raise NotImplementedError
    return locMat, recallMat, precMat, recallMat_wo_overlap


def assign_labels(locMat, assignment_strategy, thresh, num_matches):
    """match (assign) prediction and gt labels

    Returns
    -------
    tp (int): number of true positive matches
    pred_ind (list of ints): pred labels that are matched as true positives (tp)
    gt_ind (list of ints): gt labels that are matched as tp

    Note
    ----
    lists have same length and x-th element in pred_ind list is matched to
    x-th element in gt_ind list
    """
    tp_pred_ind = []
    tp_gt_ind = []
    locFgMat = locMat[1:, 1:]

    # optimal hungarian matching
    if assignment_strategy == "hungarian":
        costs = -(locFgMat >= thresh).astype(float) - locFgMat / (2 * num_matches)
        logger.info("start computing lin sum assign for thresh %s",
                    thresh)
        gt_ind, pred_ind = linear_sum_assignment(costs)
        assert num_matches == len(gt_ind) == len(pred_ind)
        match_ok = locFgMat[gt_ind, pred_ind] >= thresh
        tp = np.count_nonzero(match_ok)

        # get true positive indices
        for idx, match in enumerate(match_ok):
            if match:
                tp_pred_ind.append(pred_ind[idx])
                tp_gt_ind.append(gt_ind[idx])

    # greedy matching by localization criterion
    elif assignment_strategy == "greedy":
        logger.info("start computing greedy assignment for thresh %s", thresh)
        gt_ind, pred_ind = np.nonzero(locFgMat > thresh) # > 0) if it should be
        # used before iterating through thresholds
        locs = locFgMat[gt_ind, pred_ind]
        # sort loc values in descending order
        sort = np.flip(np.argsort(locs))
        gt_ind = gt_ind[sort]
        pred_ind = pred_ind[sort]
        locs = locs[sort]

        # assign greedy by loc score
        for gt_idx, pred_idx, loc in zip(gt_ind, pred_ind, locs):
            if gt_idx not in tp_gt_ind and pred_idx not in tp_pred_ind:
                tp_gt_ind.append(gt_idx)
                tp_pred_ind.append(pred_idx)
        tp = len(tp_pred_ind)

    # todo: merge overlap_0_5 here
    #elif assignment_strategy == "overlap_0_5":
    else:
        raise NotImplementedError(
            "assignment strategy %s is not implemented (yet)",
            assignment_strategy)

    # correct indices to include background
    tp_pred_ind = np.array(tp_pred_ind) + 1
    tp_gt_ind = np.array(tp_gt_ind) + 1

    return tp, tp_pred_ind, tp_gt_ind


def get_false_labels(
        tp_pred_ind, tp_gt_ind, num_pred_labels, num_gt_labels, locMat,
        precMat, recallMat, thresh, unique_false_labels,
        recallMat_wo_overlap):

    # get false positive indices
    pred_ind_all = np.arange(1, num_pred_labels + 1)
    pred_ind_unassigned = pred_ind_all[np.isin(
        pred_ind_all, tp_pred_ind, invert=True)]
    fp_ind_only_bg = pred_ind_unassigned[np.argmax(
        precMat[:, pred_ind_unassigned], axis=0) == 0]
    if not unique_false_labels:
        # all unassigned pred labels
        fp_ind = pred_ind_unassigned
    else:
        # unassigned pred labels with maximal overlap for background
        fp_ind = fp_ind_only_bg
        #fp_ind = pred_ind_unassigned[np.argmax(
        #    precMat[:, pred_ind_unassigned], axis=0) == 0]
    logger.debug("false positive indices: %s", fp_ind)

    # get false split indices
    fs_ind = pred_ind_unassigned[
        np.argmax(precMat[:, pred_ind_unassigned], axis=0) > 0]
    logger.debug("false split indices: %s", fs_ind)

    # get false negative indices
    gt_ind_all = np.arange(1, num_gt_labels + 1)
    gt_ind_unassigned = gt_ind_all[np.isin(gt_ind_all, tp_gt_ind, invert=True)]
    fn_ind = gt_ind_unassigned
    logger.debug("false negative indices: %s", fn_ind)

    # get false merger indices
    # check if pred label covers more than one gt label with clDice > thresh
    # check if merger also exists when ignoring gt overlapping regions
    if recallMat_wo_overlap is not None:
        loc_mask = np.logical_and(
            recallMat[1:, 1:] > thresh, recallMat_wo_overlap[1:, 1:] > thresh)
    else:
        loc_mask = recallMat[1:, 1:] > thresh
    fm_pred_count = np.maximum(0, np.sum(loc_mask, axis=0) - 1)
    fm_count = np.sum(fm_pred_count)
    # we need fm_pred_ind and fm_gt_ind for visualization later on
    # correct indices to include background
    fm_pred_ind = np.nonzero(fm_pred_count)[0]
    fm_gt_ind = []
    for i in fm_pred_ind:
        fm_gt_ind.append(np.nonzero(loc_mask[:, i])[0] + 1)
    fm_pred_ind = np.array(fm_pred_ind) + 1
    logger.debug(
        "false merge indices (pred/gt/cnt): %s, %s, %i",
        fm_pred_ind, fm_gt_ind, fm_count)

    return (
        fp_ind, fn_ind, fs_ind, fm_pred_ind, fm_gt_ind, fm_count,
        fp_ind_only_bg)
