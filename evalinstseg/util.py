import logging
import os

import h5py
import numpy as np
from scipy.optimize import linear_sum_assignment
from skimage.morphology import skeletonize_3d
import tifffile
import zarr

logger = logging.getLogger(__name__)


def maybe_crop(pred_labels, gt_labels, overlapping_inst=False):
    """check if prediction and gt have same size, otherwise crop the bigger one

    E.g., if valid padding is used output might be smaller than input.

    Note
    ----
    only crops spatial dimensions, assumes channel_first
    """
    if overlapping_inst:
        if gt_labels.shape[1:] == pred_labels.shape[1:]:
            return pred_labels, gt_labels
        else:
            if gt_labels.shape == pred_labels.shape:
                return pred_labels, gt_labels
            if gt_labels.shape[-1] > pred_labels.shape[-1]:
                bigger_arr = gt_labels
                smaller_arr = pred_labels
                swapped = False
            else:
                bigger_arr = pred_labels
                smaller_arr = gt_labels
                swapped = True

            begin = (np.array(bigger_arr.shape[-2:]) -
                     np.array(smaller_arr.shape[-2:])) // 2
            end = np.array(bigger_arr.shape[-2:]) - begin
            if (np.array(bigger_arr.shape[-2:]) -
                np.array(smaller_arr.shape[-2:]))[-1] % 2 == 1:
                end[-1] -= 1
            if (np.array(bigger_arr.shape[-2:]) -
                np.array(smaller_arr.shape[-2:]))[-2] % 2 == 1:
                end[-2] -= 1
            bigger_arr = bigger_arr[...,
                                    begin[0]:end[0],
                                    begin[1]:end[1]]
            if not swapped:
                gt_labels = bigger_arr
                pred_labels = smaller_arr
            else:
                pred_labels = bigger_arr
                gt_labels = smaller_arr
            logger.debug("gt shape cropped %s", gt_labels.shape)
            logger.debug("pred shape cropped %s", pred_labels.shape)

            return pred_labels, gt_labels
    else:
        if gt_labels.shape == pred_labels.shape:
            return pred_labels, gt_labels
        if gt_labels.shape[0] > pred_labels.shape[0]:
            bigger_arr = gt_labels
            smaller_arr = pred_labels
            swapped = False
        else:
            bigger_arr = pred_labels
            smaller_arr = gt_labels
            swapped = True
        begin = (np.array(bigger_arr.shape) -
                 np.array(smaller_arr.shape)) // 2
        end = np.array(bigger_arr.shape) - begin
        if len(bigger_arr.shape) == 2:
            bigger_arr = bigger_arr[begin[0]:end[0],
                                    begin[1]:end[1]]
        else:
            if (np.array(bigger_arr.shape) -
                np.array(smaller_arr.shape))[2] % 2 == 1:
                end[2] -= 1
            bigger_arr = bigger_arr[begin[0]:end[0],
                                    begin[1]:end[1],
                                    begin[2]:end[2]]
        if not swapped:
            gt_labels = bigger_arr
            pred_labels = smaller_arr
        else:
            pred_labels = bigger_arr
            gt_labels = smaller_arr
        logger.debug("gt shape cropped %s", gt_labels.shape)
        logger.debug("pred shape cropped %s", pred_labels.shape)

        return pred_labels, gt_labels


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


def check_sizes(gt_labels, pred_labels, overlapping_inst, keep_gt_shape):
    # check: what if there are only one gt instance in overlapping_inst scenario?
    if gt_labels.shape[0] == 1 and not keep_gt_shape:
        gt_labels.shape = gt_labels.shape[1:]
    gt_labels = np.squeeze(gt_labels)
    if gt_labels.ndim > pred_labels.ndim and not keep_gt_shape:
        gt_labels = np.max(gt_labels, axis=0)
    logger.debug("gt shape %s", gt_labels.shape)

    # heads up: should not crop channel dimensions, assuming channels first
    if not keep_gt_shape:
        pred_labels, gt_labels = maybe_crop(
            pred_labels, gt_labels, overlapping_inst)
    else:
        if pred_labels.ndim < gt_labels.ndim:
            pred_labels = np.expand_dims(pred_labels, axis=0)
            pred_labels, gt_labels = maybe_crop(pred_labels, gt_labels, True)
            pred_labels = np.squeeze(pred_labels)
    logger.debug(
        "prediction %s, shape %s", np.unique(pred_labels), pred_labels.shape)
    logger.debug("gt %s, shape %s", np.unique(gt_labels), gt_labels.shape)
    return gt_labels, pred_labels


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
        skeletonize, compare, skeletonize_label, compare_label):
    """skeletonizes `to_skeletonize` and checks how much overlap
    `compare_with` has with the skeletons, for a single pair of labels)
    """
    mask = skeletonize == skeletonize_label
    if mask.ndim == 4:
        mask = np.max(mask, axis=0)
    skeleton = skeletonize_3d(mask) > 0
    skeleton_size = np.sum(skeleton)
    mask = compare == compare_label
    if mask.ndim == 4:
        mask == np.max(mask, axis=0)

    return np.sum(mask[skeleton]) / float(skeleton_size)


def get_centerline_overlap(to_skeletonize, compare_with, match):
    """skeletonizes `to_skeletonize` and checks how much overlap
    `compare_with` has with the skeletons, for all pairs of labels
    (incl. the background label in `compare_with`)
    """
    # heads up: only implemented for 3d data
    skeleton_one_inst_per_channel = True if to_skeletonize.ndim == 4 else False
    compare_one_inst_per_channel = True if compare_with.ndim == 4 else False

    if compare_one_inst_per_channel:
        fg = np.max(compare_with > 0, axis=0).astype(np.uint8)

    labels = np.unique(to_skeletonize[to_skeletonize > 0])

    for label in labels:
        logger.debug("compute centerline overlap for %i", label)
        if skeleton_one_inst_per_channel:
            #idx = np.unravel_index(np.argmax(mask), mask.shape)[0]
            mask = to_skeletonize[label - 1] #heads up: assuming increasing labels
        else:
            mask = to_skeletonize == label
        skeleton = skeletonize_3d(mask.astype(np.uint8)) > 0
        skeleton_size = np.sum(skeleton)

        # if one instance per channel for compare, we need to correct bg label
        if compare_one_inst_per_channel:
            compare_fg, compare_fg_cnt = np.unique(
                fg[skeleton], return_counts=True)
            if np.any(compare_fg > 0):
                compare_label, compare_label_cnt = np.unique(
                    compare_with[:, skeleton], return_counts=True)
                if np.any(compare_label == 0):
                    compare_label[0] = compare_fg[0]
                    compare_label_cnt[0] = compare_fg_cnt[0]
            else:
                compare_label = compare_fg
                compare_label_cnt = compare_fg_cnt
        else:
            compare_label, compare_label_cnt = np.unique(
                compare_with[skeleton], return_counts=True)
        ratio = compare_label_cnt / float(skeleton_size)
        match[label, compare_label] = ratio

    return match


# todo: define own functions per localization criterion
# todo: not use num_*_labels as parameters here?
def compute_localization_criterion(
        pred_labels_rel, gt_labels_rel, num_pred_labels, num_gt_labels,
        localization_criterion, overlapping_inst):
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
        # todo: implement iou for keep_gt_shape
        if overlapping_inst:
            pred_tile = [1, ] * pred_labels_rel.ndim
            pred_tile[0] = gt_labels_rel.shape[0]
            gt_tile = [1, ] * gt_labels_rel.ndim
            gt_tile[1] = pred_labels_rel.shape[0]
            pred_tiled = np.tile(pred_labels_rel, pred_tile).flatten()
            gt_tiled = np.tile(gt_labels_rel, gt_tile).flatten()
            mask = np.logical_or(pred_tiled > 0, gt_tiled > 0)
            overlay = np.array([pred_tiled[mask], gt_tiled[mask]])
            overlay_labels, overlay_labels_counts = np.unique(
                overlay, return_counts=True, axis=1)
            overlay_labels = np.transpose(overlay_labels)
        else:
            overlay = np.array(
                [pred_labels_rel.flatten(), gt_labels_rel.flatten()])
            logger.debug("overlay shape relabeled %s", overlay.shape)
            # get overlaying cells and the size of the overlap
            overlay_labels, overlay_labels_counts = np.unique(
                overlay, return_counts=True, axis=1)
            overlay_labels = np.transpose(overlay_labels)

        # get gt cell ids and the size of the corresponding cell
        gt_labels_list, gt_counts = np.unique(gt_labels_rel, return_counts=True)
        gt_labels_count_dict = {}
        logger.debug("%s %s", gt_labels_list, gt_counts)
        for (l,c) in zip(gt_labels_list, gt_counts):
            gt_labels_count_dict[l] = c

        # get pred cell ids
        pred_labels_list, pred_counts = np.unique(
            pred_labels_rel, return_counts=True)
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
            pred_labels_rel, gt_labels_rel,
            np.transpose(precMat))
        precMat = np.transpose(precMat)
        recallMat = get_centerline_overlap(
            gt_labels_rel, pred_labels_rel,
            recallMat)

        # get recallMat without overlapping gt labels for false merge calculation later on
        if overlapping_inst or \
           len(gt_labels_rel.shape) > len(pred_labels_rel.shape):
            gt_wo_overlap = gt_labels_rel.copy()
            mask = np.sum(gt_labels_rel > 0, axis=0) > 1
            gt_wo_overlap[:, mask] = 0
            pred_wo_overlap = pred_labels_rel.copy()
            if overlapping_inst:
                # todo: check how this should be done with overlapping inst in prediction
                pred_wo_overlap[:, mask] = 0
            else:
                pred_wo_overlap[mask] = 0

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
        logger.info("start computing greedy assignment for thresh %s, thresh")
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
        precMat, recallMat, thresh, overlapping_inst, unique_false_labels,
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
    if overlapping_inst:
        loc_mask = np.logical_and(
            recallMat[1:, 1:] > thresh, recallMat_wo_overlap[1:, 1:] > thresh)
    else:
        # todo: correct? not recallMat?
        loc_mask = locMat[1:, 1:] > thresh
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
