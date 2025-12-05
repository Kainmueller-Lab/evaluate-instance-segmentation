import logging

import numpy as np
from scipy.optimize import linear_sum_assignment
from skimage.morphology import skeletonize_3d
from skimage.segmentation import relabel_sequential

logger = logging.getLogger(__name__)


# TODO: define own functions per localization criterion
# TODO: not use num_*_labels as parameters here?
def compute_localization_criterion(
    pred_labels, gt_labels, num_pred_labels, num_gt_labels, localization_criterion
):
    """computes the localization part of the metric
    For each pair of labels in the prediction and gt,
    how much are they co-localized, based on the chosen criterion?
    """
    logger.debug("evaluate localization criterion for all gt and pred label pairs")

    # create matrices for pixelwise overlap measures
    locMat = np.zeros((num_gt_labels + 1, num_pred_labels + 1), dtype=np.float32)
    recallMat = np.zeros((num_gt_labels + 1, num_pred_labels + 1), dtype=np.float32)
    precMat = np.zeros((num_gt_labels + 1, num_pred_labels + 1), dtype=np.float32)
    recallMat_wo_overlap = None

    # intersection over union
    if localization_criterion == "iou":
        logger.debug("compute iou")
        pred_tile = [
            1,
        ] * pred_labels.ndim
        pred_tile[0] = gt_labels.shape[0]
        gt_tile = [
            1,
        ] * gt_labels.ndim
        gt_tile[1] = pred_labels.shape[0]
        pred_tiled = np.tile(pred_labels, pred_tile).flatten()
        gt_tiled = np.tile(gt_labels, gt_tile).flatten()
        mask = np.logical_or(pred_tiled > 0, gt_tiled > 0)
        overlay = np.array([pred_tiled[mask], gt_tiled[mask]])
        overlay_labels, overlay_labels_counts = np.unique(
            overlay, return_counts=True, axis=1
        )
        overlay_labels = np.transpose(overlay_labels)

        # get gt cell ids and the size of the corresponding cell
        gt_labels_list, gt_counts = np.unique(gt_labels, return_counts=True)
        gt_labels_count_dict = {}
        logger.debug("%s %s", gt_labels_list, gt_counts)
        for l, c in zip(gt_labels_list, gt_counts):
            gt_labels_count_dict[l] = c

        # get pred cell ids
        pred_labels_list, pred_counts = np.unique(pred_labels, return_counts=True)
        logger.debug("%s %s", pred_labels_list, pred_counts)
        pred_labels_count_dict = {}
        for l, c in zip(pred_labels_list, pred_counts):
            pred_labels_count_dict[l] = c

        for (u, v), c in zip(overlay_labels, overlay_labels_counts):
            iou = c / (gt_labels_count_dict[v] + pred_labels_count_dict[u] - c)

            locMat[v, u] = iou
            recallMat[v, u] = c / gt_labels_count_dict[v]
            precMat[v, u] = c / pred_labels_count_dict[u]

    # centerline dice
    elif localization_criterion == "cldice":
        logger.debug("compute cldice")
        # todo: transpose precMat
        precMat = get_centerline_overlap(pred_labels, gt_labels, np.transpose(precMat))
        precMat = np.transpose(precMat)
        recallMat = get_centerline_overlap(gt_labels, pred_labels, recallMat)

        if np.any(np.sum(gt_labels, axis=0) != np.max(gt_labels, axis=0)) or np.any(
            np.sum(pred_labels, axis=0) != np.max(pred_labels, axis=0)
        ):
            # get recallMat without overlapping gt labels for false merge
            # calculation later on
            gt_wo_overlap = gt_labels.copy()
            mask = np.sum(gt_labels > 0, axis=0) > 1
            gt_wo_overlap[:, mask] = 0
            pred_wo_overlap = pred_labels.copy()
            # todo: check how this should be done with overlapping inst in prediction
            pred_wo_overlap[:, mask] = 0

            recallMat_wo_overlap = get_centerline_overlap(
                gt_wo_overlap, pred_wo_overlap, np.zeros_like(recallMat)
            )
        err = np.geterr()
        np.seterr(invalid="ignore")
        locMat = np.nan_to_num(2 * precMat * recallMat / (precMat + recallMat))
        np.seterr(invalid=err["invalid"])
    else:
        raise NotImplementedError
    return locMat, recallMat, precMat, recallMat_wo_overlap


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

    # assumes uniquely and sequentially labeled instances
    # (i.e., no binary masks per channel)
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
            compare_with[:, skeleton], return_counts=True
        )

        # remember to correct for bg label
        if 0 in compare_labels:
            compare_fg, compare_fg_cnt = np.unique(fg[skeleton], return_counts=True)
            if 0 in compare_fg:
                assert compare_labels[0] == 0 and compare_fg[0] == 0
                compare_labels_cnt[0] = compare_fg_cnt[0]
            else:
                compare_labels = compare_labels[1:]
                compare_labels_cnt = compare_labels_cnt[1:]

        ratio = compare_labels_cnt / float(skeleton_size)
        match[skeleton_id, compare_labels] = ratio

    return match


def get_centerline_overlap_single(
    to_skeletonize, compare_with, skeletonize_label, compare_label
):
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
        compare_with = np.max(compare_with, axis=0)

    return np.sum(compare_with[skeleton], dtype=float) / np.sum(skeleton, dtype=float)
