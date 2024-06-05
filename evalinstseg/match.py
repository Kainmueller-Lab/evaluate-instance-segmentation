import logging
import os

import numpy as np
from scipy.optimize import linear_sum_assignment
from skimage.morphology import skeletonize_3d
from skimage.segmentation import relabel_sequential
from queue import PriorityQueue

logger = logging.getLogger(__name__)


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


def greedy_many_to_many_matching(gt_labels, pred_labels, locMat, thresh,
        only_one_gt=False, only_one_pred=False):

    matches = {}   # list of assigned pred instances for each gt
    locFgMat = locMat[1:, 1:]

    q = PriorityQueue()
    gt_skel = {}
    gt_avail = {}
    pred_avail = {}
    if not np.any(locFgMat > thresh):
        return None

    gt_ids, pred_ids = np.nonzero(locFgMat > thresh)
    for gt_id, pred_id in zip(gt_ids, pred_ids):
        # initialize clRecall priority queue
        q.put(((-1) * locFgMat[gt_id, pred_id], gt_id, pred_id))

    # initialize running instance masks with free/available pixel
    for gt_id in np.unique(gt_ids):
        # save skeletonized gt mask
        gt_skel[gt_id] = skeletonize_3d(gt_labels[gt_id]) > 0
        gt_avail[gt_id] = gt_skel[gt_id].copy()

    for pred_id in np.unique(pred_ids):
        pred_avail[pred_id] = pred_labels == (pred_id + 1) #todo: also for one inst per channel

    # iterate through clRecall values in descending order
    while len(q.queue) > 0:
        clr, gt_id, pred_id = q.get()
        # save as match
        if gt_id not in matches:
            matches[gt_id] = [pred_id]
        else:
            matches[gt_id] += [pred_id]

        # update running instance masks
        gt_avail[gt_id] = np.logical_and(gt_avail[gt_id],
                np.logical_not(pred_labels == (pred_id + 1)))
        pred_avail[pred_id] = np.logical_and(pred_avail[pred_id],
                np.logical_not(gt_labels[gt_id]))

        # check for other occurences of pred and gt labels in queue
        for o_clr, o_gt_id, o_pred_id in q.queue:
            if o_gt_id == gt_id:
                # remove old clRecall entry
                q.queue.remove((o_clr, o_gt_id, o_pred_id))
                if not only_one_gt:
                    # add updated clRecall entry
                    o_clr_new = np.sum(np.logical_and(gt_avail[o_gt_id],
                            pred_labels == (pred_id + 1))) / float(np.sum(gt_skel[o_gt_id]))
                    q.put(((-1) * o_clr_new, o_gt_id, o_pred_id))

            if o_pred_id == pred_id:
                # remove old clRecall entry
                q.queue.remove((o_clr, o_gt_id, o_pred_id))
                if not only_one_pred:
                    # add updated clRecall entry
                    o_clr_new = np.sum(np.logical_and(gt_skel[o_gt_id],
                        pred_avail[o_pred_id])) / np.sum(gt_skel[o_gt_id])
                    q.put(((-1) * o_clr_new, o_gt_id, o_pred_id))

    return matches


def get_false_labels(
        tp_pred_ind, tp_gt_ind, num_pred_labels, num_gt_labels, locMat,
        precMat, recallMat, thresh, recallMat_wo_overlap):

    # get false positive indices
    pred_ind_all = np.arange(1, num_pred_labels + 1)
    pred_ind_unassigned = pred_ind_all[np.isin(
        pred_ind_all, tp_pred_ind, invert=True)]
    fp_ind_only_bg = pred_ind_unassigned[np.argmax(
        precMat[:, pred_ind_unassigned], axis=0) == 0] #TODO: naming?
    # all unassigned pred labels
    fp_ind = pred_ind_unassigned
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

