import logging
import os

import numpy as np
from scipy.optimize import linear_sum_assignment
from skimage.morphology import skeletonize
from skimage.segmentation import relabel_sequential
from queue import PriorityQueue
from evalinstseg.util import LazyHeap

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
        costs = -(locFgMat > thresh).astype(float) - locFgMat / (2 * num_matches)
        logger.info("start computing lin sum assign for thresh %s",
                    thresh)
        gt_ind, pred_ind = linear_sum_assignment(costs)
        assert num_matches == len(gt_ind) == len(pred_ind)
        match_ok = locFgMat[gt_ind, pred_ind] > thresh
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
        order = np.lexsort((pred_ind, gt_ind, -locs))
        gt_ind  = gt_ind[order]
        pred_ind = pred_ind[order]
        locs    = locs[order]

        # assign greedy by loc score
        for gt_idx, pred_idx, loc in zip(gt_ind, pred_ind, locs):
            if gt_idx not in tp_gt_ind and pred_idx not in tp_pred_ind:
                tp_gt_ind.append(gt_idx)
                tp_pred_ind.append(pred_idx)
        tp = len(tp_pred_ind)

    # TODO: merge overlap_0_5 here
    #elif assignment_strategy == "overlap_0_5":
    else:
        raise NotImplementedError(
            "assignment strategy %s is not implemented (yet)",
            assignment_strategy)

    # correct indices to include background
    tp_pred_ind = np.array(tp_pred_ind) + 1
    tp_gt_ind = np.array(tp_gt_ind) + 1

    return tp, tp_pred_ind, tp_gt_ind

def instance_mask(labels, idx):
    """Returns a instance mask for a given instance index."""
    
    inst_id = idx + 1
    return np.any(labels == inst_id, axis=0)

def greedy_many_to_many_matching(gt_labels, pred_labels, locMat, thresh,
        only_one_gt=False, only_one_pred=False):

    matches = {}   # list of assigned pred instances for each gt
    locFgMat = locMat[1:, 1:]
    pq = LazyHeap()

    pred_avail = {}
    
    if not np.any(locFgMat > thresh):
        return None

    gt_ids, pred_ids = np.nonzero(locFgMat > thresh)
    for gt_id, pred_id in zip(gt_ids, pred_ids):
        # initialize clRecall priority queue
        priority = (-1) * locFgMat[gt_id, pred_id]
        pq.push((gt_id, pred_id), priority)

    gt_skel = {}
    gt_avail = {}
    # initialize running instance masks with free/available pixel
    for gt_id in np.unique(gt_ids):
        # save skeletonized gt mask
        gt_inst_mask = instance_mask(gt_labels, gt_id)
        gt_skel[gt_id] = skeletonize(gt_inst_mask) > 0
        gt_avail[gt_id] = gt_skel[gt_id].copy()

    for pred_id in np.unique(pred_ids): 
        pred_avail[pred_id] = instance_mask(pred_labels, pred_id) #todo: also for one inst per channel

    # iterate through clRecall values in descending order
    while not pq.empty():
        try:
            _, (gt_id, pred_id) = pq.pop()   
        except KeyError:
            break
        # save as match
        if gt_id not in matches:
            matches[gt_id] = [pred_id]
        else:
            matches[gt_id] += [pred_id]

        # update running instance masks
        gt_avail[gt_id] = np.logical_and(gt_avail[gt_id],
                np.logical_not(instance_mask(pred_labels, pred_id)))
        pred_avail[pred_id] = np.logical_and(pred_avail[pred_id],
                np.logical_not(instance_mask(gt_labels, gt_id)))

        # check for other occurences of pred and gt labels in queue
        # Update their clDice values and reinsert or remove from queue
        cand_pred_ids = np.nonzero(locFgMat[gt_id, :] > thresh)[0]
        denominator_gt = float(np.sum(gt_skel[gt_id])) or 1.0
        for o_pred_id in cand_pred_ids:
            if not only_one_gt:
                inter = np.logical_and(gt_avail[gt_id], instance_mask(pred_labels, o_pred_id))
                new_clr = np.sum(inter) / denominator_gt
                if new_clr > thresh:
                    pq.push((gt_id, o_pred_id), -new_clr)
                else:
                    pq.faulty_element((gt_id, o_pred_id))
            else:
                # If ONE-TO-MANY: Invalidate all other pairs for this gt_id
                pq.faulty_element((gt_id, o_pred_id))

        cand_gt_ids = np.nonzero(locFgMat[:, pred_id] > thresh)[0]
        for o_gt_id in cand_gt_ids:
            if not only_one_pred:
                denom = float(np.sum(gt_skel[o_gt_id])) or 1.0
                inter = np.logical_and(gt_skel[o_gt_id], pred_avail[pred_id])
                new_clr = np.sum(inter) / denom
                if new_clr > thresh:
                    pq.push((o_gt_id, pred_id), -new_clr)
                else:
                    pq.faulty_element((o_gt_id, pred_id))
            else:
                # If ONE-TO-MANY: Invalidate all other pairs for this pred_id
                pq.faulty_element((o_gt_id, pred_id))
    return matches


def get_false_labels(
        tp_pred_ind, tp_gt_ind, num_pred_labels, num_gt_labels,
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


def get_m2m_matches(locMat, thresh, gt_labels=None, pred_labels=None, overlaps=True):
    """Get many-to-many matches between gt and predicted labels. 
    If we have no overlaps, we can do easy matching based on thresholding the locMat."""

    # If we have overlapping instances, we need to do expensive greedy many-to-many matching
    if overlaps:
        if gt_labels is None or pred_labels is None:
            raise ValueError("gt_labels and pred_labels required when overlaps=True")
        matches = greedy_many_to_many_matching(gt_labels, pred_labels, locMat, thresh)
        if matches is not None:
            # key and values are 0-based, convert to 1-based
            matches = {k + 1: [v + 1 for v in val] for k, val in matches.items()}
        return matches
    else:
        # Simple matching based on threshold
        matches = {}
        locFgMat = locMat[1:, 1:] # excluding background
        rows, cols = np.nonzero(locFgMat > thresh)
        for gt_idx, pred_idx in zip(rows, cols):
            gt_id = gt_idx + 1 # 1-based IDs
            pred_id = pred_idx + 1
            if gt_id not in matches:
                matches[gt_id] = [pred_id]
            else:
                matches[gt_id].append(pred_id)
        return matches

