import logging

import numpy as np
from skimage.segmentation import relabel_sequential

from .localize import (
    get_centerline_overlap_single,
    get_centerline_overlap,
)
from .match import assign_labels, greedy_many_to_many_matching, get_m2m_matches

logger = logging.getLogger(__name__)


def get_gt_coverage(gt_labels, pred_labels, precMat, recallMat):
    # only take max gt label for each pred label to not count
    # pred labels twice for overlapping gt instances
    num_pred_labels = int(np.max(pred_labels))
    num_gt_labels = int(np.max(gt_labels))
    max_gt_ind = np.argmax(precMat, axis=0)

    gt_cov = []
    # if both gt and pred have overlapping instances
    if np.any(np.sum(gt_labels, axis=0) != np.max(gt_labels, axis=0)) and np.any(
        np.sum(pred_labels, axis=0) != np.max(pred_labels, axis=0)
    ):
        # recalculate clRecall for each gt and union of assigned
        # predictions, as predicted instances can potentially overlap
        max_gt_ind_unique = np.unique(max_gt_ind[max_gt_ind > 0])
        for gt_i in np.arange(1, num_gt_labels + 1):
            if gt_i in max_gt_ind_unique:
                pred_union = np.zeros(pred_labels.shape[1:], dtype=pred_labels.dtype)
                for pred_i in np.arange(num_pred_labels + 1)[max_gt_ind == gt_i]:
                    mask = np.max(pred_labels == pred_i, axis=0)
                    pred_union[mask] = 1
                gt_cov.append(
                    get_centerline_overlap_single(gt_labels, pred_union, gt_i, 1)
                )
            else:
                gt_cov.append(0.0)
    else:
        # otherwise use previously computed values
        for i in range(1, recallMat.shape[0]):
            gt_cov.append(np.sum(recallMat[i, max_gt_ind == i]))
    return gt_cov


# TODO: check consistent use gt_labels_rel and gt_labels!!!
def get_gt_coverage_dim(
    dim_insts,
    gt_labels_rel,
    pred_labels_rel,
    num_pred_labels,
    locMat,
    recallMat,
    assignment_strategy="greedy",
):
    tp_05_dim = 0
    tp_05_rel_dim = 0.0
    gt_covs_dim = []
    avg_cov_dim = 0
    gt_dim = len(dim_insts)
    if gt_dim > 0 and num_pred_labels > 0:
        # prepare arrays for subset
        subset_ids = np.array(dim_insts) - 1
        gt_labels_subset = gt_labels_rel[subset_ids]
        # relabel sequential
        offset = 1
        for i in range(gt_labels_subset.shape[0]):
            gt_labels_subset[i], _, _ = relabel_sequential(
                gt_labels_subset[i].astype(int), offset
            )
            offset = np.max(gt_labels_subset[i]) + 1

        precMat_subset = np.zeros((gt_dim + 1, num_pred_labels + 1), dtype=np.float32)
        precMat_subset = get_centerline_overlap(
            pred_labels_rel, gt_labels_subset, np.transpose(precMat_subset)
        )
        precMat_subset = np.transpose(precMat_subset)
        recallMat_subset = recallMat[
            [
                0,
            ]
            + dim_insts
        ]
        locMat_subset = locMat[
            [
                0,
            ]
            + dim_insts
        ]
        # compute coverage
        gt_covs_dim = get_gt_coverage(
            gt_labels_subset, pred_labels_rel, precMat_subset, recallMat_subset
        )
        avg_cov_dim = np.mean(gt_covs_dim)
        # compute tp
        if np.max(locMat_subset[1:, 1:]) > 0.5:
            tp_05_dim, _, _ = assign_labels(locMat_subset, assignment_strategy, 0.5, 1)
        tp_05_rel_dim = tp_05_dim / float(gt_dim)
    return gt_dim, tp_05_dim, tp_05_rel_dim, gt_covs_dim, avg_cov_dim


# TODO: merge with get_gt_coverage_dim to get_gt_coverage_subset
def get_gt_coverage_overlap(
    ovlp_inst_ids,
    gt_labels_rel,
    pred_labels_rel,
    num_pred_labels,
    locMat,
    recallMat,
    assignment_strategy="greedy",
):
    tp_05_ovlp = 0
    tp_05_rel_ovlp = 0.0
    gt_covs_ovlp = []
    avg_cov_ovlp = 0
    if 0 in ovlp_inst_ids:
        ovlp_inst_ids = np.delete(ovlp_inst_ids, 0)
    gt_ovlp = len(ovlp_inst_ids)
    if gt_ovlp > 0:
        # prepare arrays for subset
        subset_ids = np.array(ovlp_inst_ids) - 1
        gt_labels_subset = gt_labels_rel[subset_ids]
        # relabel sequential
        offset = 1
        for i in range(gt_labels_subset.shape[0]):
            gt_labels_subset[i], _, _ = relabel_sequential(
                gt_labels_subset[i].astype(int), offset
            )
            offset = np.max(gt_labels_subset[i]) + 1

        precMat_subset = np.zeros((gt_ovlp + 1, num_pred_labels + 1), dtype=np.float32)
        precMat_subset = get_centerline_overlap(
            pred_labels_rel, gt_labels_subset, np.transpose(precMat_subset)
        )
        precMat_subset = np.transpose(precMat_subset)
        recallMat_subset = recallMat[
            [
                0,
            ]
            + list(ovlp_inst_ids)
        ]
        locMat_subset = locMat[
            [
                0,
            ]
            + list(ovlp_inst_ids)
        ]
        # compute coverage
        gt_covs_ovlp = get_gt_coverage(
            gt_labels_subset, pred_labels_rel, precMat_subset, recallMat_subset
        )
        avg_cov_ovlp = np.mean(gt_covs_ovlp)

        # compute tp
        if np.max(locMat_subset[1:, 1:]) > 0.5:
            tp_05_ovlp, _, _ = assign_labels(locMat_subset, assignment_strategy, 0.5, 1)
        tp_05_rel_ovlp = tp_05_ovlp / float(gt_ovlp)
    return gt_ovlp, tp_05_ovlp, tp_05_rel_ovlp, gt_covs_ovlp, avg_cov_ovlp


def compute_m2m_stats(matches, num_pred_labels):
    '''Helper to compute false merges and false splits from many-to-many matches.'''
    
    fm = 0
    fs = 0
    if matches is not None:
        # FS calculation
        for k, v in matches.items():
            fs += max(0, len(v) - 1)

        # FM calculation
        if num_pred_labels > 0:
            fms = np.zeros(num_pred_labels)  # without 0 background
            for k, v in matches.items():
                for cv in v:
                    fms[cv - 1] += 1
            fms = np.maximum(fms - 1, np.zeros(num_pred_labels))
            fm = int(np.sum(fms))
    
    return fm, fs


def get_m2m_metrics(gt_labels, pred_labels, num_pred_labels, matchMat, thresh, overlaps=True):
    """
    Compute false merge and false split metrics for any localization criterion using many-to-many matching.
    
    Args:
        gt_labels: Ground truth labels
        pred_labels: Predicted labels
        num_pred_labels: Number of predicted labels
        matchMat: A matrix depending on the localization criterion (Recall matrix for clDice, IoU matrix for IoU)
        thresh: Threshold for matching
        overlaps: Whether to allow overlapping instances
        
    Returns:
        Tuple of (false_merge, false_split, matches)
    """
    matches = get_m2m_matches(
        matchMat, thresh, gt_labels, pred_labels, overlaps
    )
    fm, fs = compute_m2m_stats(matches, num_pred_labels)
    return fm, fs, matches
