import glob
import logging
import sys
import os

import argparse
import numpy as np
import toml
from natsort import natsorted
from skimage.segmentation import relabel_sequential
import pdb

from .metrics import Metrics
from .util import (
    check_and_fix_sizes,
    check_fix_and_unify_ids,
    get_centerline_overlap_single,
    get_output_name,
    read_file,
    get_gt_coverage,
)
from .localize import (
    compute_localization_criterion,
    get_centerline_overlap
)
from .match import (
    assign_labels,
    get_false_labels,
    greedy_many_to_many_matching,
)
from .visualize import (
    visualize_neurons,
    visualize_nuclei
)
from .summarize import summarize_metric_dict

logger = logging.getLogger(__name__)


def evaluate_file(
        res_file,
        gt_file,
        ndim,
        out_dir,
        res_key=None,
        gt_key=None,
        suffix="",
        localization_criterion="iou", # "iou", "cldice"
        assignment_strategy="greedy", # "hungarian", "greedy", "gt_0_5"
        add_general_metrics=[],
        visualize=False,
        visualize_type="nuclei", # "nuclei" or "neuron"
        overlapping_inst=False,
        partly=False,
        foreground_only=False,
        remove_small_components=None,
        evaluate_false_labels=False,
        check_for_metric=None,
        from_scratch=False,
        fm_thresh=0.1,
        fs_thresh=0.05,
        eval_dim=False,
        **kwargs
):
    """computes segmentation quality of file wrt. its ground truth

    Args
    -----
    res_file: str/Path
        path to computed segmentation results
    gt_file: str/Path
        path to ground truth segmentation
    ndim: int
        number of spatial dimensions of data (typically 2 or 3)
    out_dir: str/Path
        directory, where to store results
        (basename will be constructed based on sample name and args)
    res_key: str
        key of array in res_file (optional, not used for tif)
    gt_key: str
        key of array in gt_file (optional, not used for tif)
    other args: TODO

    Returns
    -------
    dict: contains computed metrics
    """
    # check for deprecated args
    if kwargs.get("use_linear_sum_assignment", False) == True:
        assignment_strategy = "hungarian"
    filterSz = kwargs.get("filterSz", None)
    if filterSz is not None and filterSz > 0:
        remove_small_components = filterSz

    # put together output filename
    outFn = get_output_name(
        out_dir, res_file, res_key, suffix, localization_criterion,
        assignment_strategy, remove_small_components)

    # if from_scratch is set, overwrite existing evaluation files
    # otherwise try to load precomputed metrics
    #   if check_for_metric is None, just check if matching file exists
    #   otherwise check if check_for_metric is contained within file
    if not from_scratch and \
       len(glob.glob(outFn + ".toml")) > 0:
        with open(outFn+".toml", 'r') as tomlFl:
            metricsDict = toml.load(tomlFl)
        if check_for_metric is None:
            return metricsDict
        try:
            metric = metricsDict
            for k in check_for_metric.split('.'):
                metric = metric[k]
            logger.info('Skipping evaluation, already exists! %s', outFn)
            return metricsDict
        except KeyError:
            logger.info(
                "Error (key %s missing) in existing evaluation "
                "for %s. Recomputing!", check_for_metric, res_file)

    # read preprocessed file
    pred_labels = read_file(res_file, res_key)
    logger.debug(
        "prediction min %f, max %f, shape %s", np.min(pred_labels),
        np.max(pred_labels), pred_labels.shape)
    pred_labels = np.squeeze(pred_labels)
    assert pred_labels.ndim == ndim or pred_labels.ndim == ndim+1

    # read ground truth data
    if eval_dim:
        gt_labels, dim_insts = read_file(gt_file, gt_key, read_dim=True)
    else:
        gt_labels = read_file(gt_file, gt_key)
        dim_insts = []
    logger.debug(
        "gt min %f, max %f, shape %s", np.min(gt_labels),
        np.max(gt_labels), gt_labels.shape)
    gt_labels = np.squeeze(gt_labels)
    assert gt_labels.ndim == ndim or gt_labels.ndim == ndim+1

    metrics = evaluate_volume(
        gt_labels,
        pred_labels,
        ndim,
        outFn,
        localization_criterion=localization_criterion,
        assignment_strategy=assignment_strategy,
        evaluate_false_labels=evaluate_false_labels,
        add_general_metrics=add_general_metrics,
        visualize=visualize,
        visualize_type=visualize_type,
        overlapping_inst=overlapping_inst,
        remove_small_components=remove_small_components,
        foreground_only=foreground_only,
        partly=partly,
        fm_thresh=fm_thresh,
        fs_thresh=fs_thresh,
        dim_insts=dim_insts
    )
    metrics.save()

    return metrics.metricsDict


def evaluate_volume(
        gt_labels,
        pred_labels,
        ndim,
        outFn,
        localization_criterion="iou",
        assignment_strategy="hungarian",
        evaluate_false_labels=False,
        add_general_metrics=[],
        visualize=False,
        visualize_type="nuclei",
        overlapping_inst=False,
        remove_small_components=None,
        foreground_only=False,
        partly=False,
        fm_thresh=0.1,
        fs_thresh=0.05,
        dim_insts=[]
):
    """computes segmentation quality of file wrt. its ground truth

    Args
    ----
    gt_labels: ndarray
        nd-array containing ground truth segmentation
    pred_labels: ndarray
        nd-array containing computed segmentation
    other args: TODO

    Returns
    -------
    dict: contains computed metrics
    """
    # if partly is set, then also set evaluate_false_labels to get false splits
    if partly:
        evaluate_false_labels = True

    # check sizes and crop if necessary
    gt_labels, pred_labels = check_and_fix_sizes(gt_labels, pred_labels, ndim)
    if len(dim_insts) > 0:
        gt_labels_rel, pred_labels_rel, dim_insts_rel = check_fix_and_unify_ids(
            gt_labels, pred_labels, remove_small_components, foreground_only,
            dim_insts=dim_insts)
    else:
        gt_labels_rel, pred_labels_rel = check_fix_and_unify_ids(
            gt_labels, pred_labels, remove_small_components, foreground_only)

    logger.debug(
        "are there pixels with multiple instances?: "
        f"{np.sum(np.sum(gt_labels_rel > 0, axis=0) > 1)}")

    # get number of labels
    num_pred_labels = int(np.max(pred_labels_rel))
    num_gt_labels = int(np.max(gt_labels_rel))
    num_matches = min(num_gt_labels, num_pred_labels)

    # get localization criterion -> TODO: check: do we still need recallMat_wo_overlap?
    locMat, recallMat, precMat, recallMat_wo_overlap = \
        compute_localization_criterion(
            pred_labels_rel, gt_labels_rel,
            num_pred_labels, num_gt_labels,
            localization_criterion)

    metrics = Metrics(outFn)
    tblNameGen = "general"
    metrics.addTable(tblNameGen)
    metrics.addMetric(tblNameGen, "Num GT", num_gt_labels)
    metrics.addMetric(tblNameGen, "Num Pred", num_pred_labels)

    # iterate through thresholds to compute multi-threshold metrics
    ths = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.55, 0.65, 0.75, 0.85, 0.95]
    aps = []
    fscores = []
    metrics.addTable("confusion_matrix")
    for th in ths:
        tblname = "confusion_matrix.th_" + str(th).replace(".", "_")
        metrics.addTable(tblname)

        # assign prediction to ground truth labels
        if num_matches > 0 and np.max(locMat) > th:
            tp, pred_ind, gt_ind = assign_labels(
                locMat, assignment_strategy, th, num_matches)
        else:
            tp = 0
            pred_ind = []
            gt_ind = []

        # get labels for each segmentation error
        if evaluate_false_labels == True or visualize == True:
            fp_ind, fn_ind, fs_ind, fm_pred_ind, fm_gt_ind, \
                fm_count, fp_ind_only_bg = get_false_labels(
                    pred_ind, gt_ind, num_pred_labels, num_gt_labels,
                    locMat, precMat, recallMat, th,
                    recallMat_wo_overlap)

        # get false positive and false negative counters
        if partly:
            fp = len(fs_ind)
            fp_ind = fs_ind
            fn = num_gt_labels - tp
        else:
            fp = num_pred_labels - tp
            fn = num_gt_labels - tp
        # add counter to metric dict
        metrics.addMetric(tblname, "AP_TP", tp)
        metrics.addMetric(tblname, "AP_FP", fp)
        metrics.addMetric(tblname, "AP_FN", fn)

        # calculate instance-level precision, recall, AP and fscore
        precision = 1. * (tp) / max(1, tp +  fp)
        recall = 1. * (tp) / max(1, tp +  fn)
        ap = precision * recall
        aps.append(ap)
        if (precision + recall) > 0:
            fscore = (2. * precision * recall) / (precision + recall)
        else:
            fscore = 0.0
        fscores.append(fscore)
        # add to metric dict
        metrics.addMetric(tblname, "precision", precision)
        metrics.addMetric(tblname, "recall", recall)
        metrics.addMetric(tblname, "AP", ap)
        metrics.addMetric(tblname, 'fscore', fscore)

        # add false labels to metric dict
        if evaluate_false_labels:
            metrics.addMetric(tblname, "false_split", len(fs_ind))
            metrics.addMetric(tblname, "false_merge", int(fm_count))

        # add one-to-one matched true positives to general dict
        if th == 0.5:
            tp_05 = tp
            tp_05_cldice = list(locMat[gt_ind, pred_ind])

        # visualize tp and false labels
        if visualize and th == 0.5:
            if visualize_type == "nuclei" and tp > 0:
                visualize_nuclei(
                    gt_labels_rel, pred_labels_rel, locMat, gt_ind, pred_ind,
                    th, outFn)
            elif visualize_type == "neuron" and \
                 localization_criterion == "cldice":
                visualize_neurons(
                    gt_labels_rel, pred_labels_rel, gt_ind, pred_ind,
                    outFn, fp_ind, fs_ind, fn_ind, fm_gt_ind, fm_pred_ind,
                    fp_ind_only_bg)
            else:
                raise NotImplementedError

    # save multi-threshold metrics to dict
    avAP19 = np.mean(aps[:-5])
    avAP59 = np.mean(aps[4:])
    metrics.addMetric("confusion_matrix", "avAP", avAP59)
    metrics.addMetric("confusion_matrix", "avAP59", avAP59)
    metrics.addMetric("confusion_matrix", "avAP19", avAP19)
    avFscore19 = np.mean(fscores[:-5])
    avFscore59 = np.mean(fscores[4:])
    metrics.addMetric("confusion_matrix", "avFscore", avFscore19)
    metrics.addMetric("confusion_matrix", "avFscore59", avFscore59)
    metrics.addMetric("confusion_matrix", "avFscore19", avFscore19)

    metrics.addMetric("general", "TP_05", tp_05)
    metrics.addMetric("general", "TP_05_rel", tp_05 / float(num_gt_labels))
    metrics.addMetric("general", "TP_05_cldice", tp_05_cldice)
    metrics.addMetric("general", "avg_TP_05_cldice", np.mean(tp_05_cldice))

    # additional metrics
    if len(add_general_metrics) > 0:
        # get coverage for ground truth instances
        if "avg_gt_skel_coverage" in add_general_metrics or \
           "avg_tp_skel_coverage" in add_general_metrics or \
           "avg_f1_cov_score" in add_general_metrics:
               gt_cov = get_gt_coverage(gt_labels_rel, pred_labels_rel, precMat, recallMat)
               gt_skel_coverage = np.mean(gt_cov)

        if "avg_gt_skel_coverage" in add_general_metrics:
            metrics.addMetric(tblNameGen, "gt_skel_coverage", gt_cov)
            metrics.addMetric(tblNameGen, "avg_gt_skel_coverage", gt_skel_coverage)

        # get coverage for true positive ground truth instances (> 0.5)
        if "avg_tp_skel_coverage" in add_general_metrics:
            gt_cov = np.array(gt_cov)
            tp_cov = gt_cov[gt_cov > 0.5]
            if len(tp_cov) > 0:
                tp_skel_coverage = np.mean(tp_cov)
            else:
                tp_skel_coverage = 0
            metrics.addMetric(
                "confusion_matrix.th_0_5", "tp_skel_coverage", tp_cov)
            metrics.addMetric(
                tblNameGen, "avg_tp_skel_coverage", tp_skel_coverage)

        if "avg_f1_cov_score" in add_general_metrics:
            avg_f1_cov_score = 0.5 * avFscore19 + 0.5 * gt_skel_coverage
            metrics.addMetric(tblNameGen, "avg_f1_cov_score", avg_f1_cov_score)

        if "false_merge" in add_general_metrics or \
                "false_split" in add_general_metrics:
            # call many-to-many matching based on clRecall
            # get false merges
            mmm = greedy_many_to_many_matching(gt_labels, pred_labels, recallMat, fm_thresh)
            if mmm is None:
                metrics.addMetric("general", "FM", 0)
            else:
                fms = np.zeros(num_pred_labels) # without 0 background
                for k, v in mmm.items():
                    for cv in v:
                        fms[cv-1] += 1
                fms = np.maximum(fms - 1, np.zeros(num_pred_labels))
                metrics.addMetric("general", "FM", int(np.sum(fms)))
            # get false splits
            if fm_thresh != fs_thresh:
                mmm = greedy_many_to_many_matching(gt_labels, pred_labels, recallMat, fs_thresh)
            if mmm is None:
                metrics.addMetric("general", "FS", 0)
            else:
                fs = 0
                for k, v in mmm.items():
                    fs += max(0, len(v) - 1)
                metrics.addMetric("general", "FS", fs)

        if "avg_gt_cov_dim" in add_general_metrics:
            tp_05_dim = 0
            tp_05_rel_dim = 0.0
            gt_covs_dim = []
            avg_cov_dim = 0
            gt_dim = len(dim_insts)
            if gt_dim > 0 and num_pred_labels > 0:
                # prepare arrays for subset
                subset_ids = np.array(dim_insts_rel) - 1
                gt_labels_subset = gt_labels_rel[subset_ids]
                # relabel sequential
                offset = 1
                for i in range(gt_labels_subset.shape[0]):
                    gt_labels_subset[i], _, _ = relabel_sequential(
                        gt_labels_subset[i].astype(int), offset)
                    offset = np.max(gt_labels_subset[i]) + 1

                precMat_subset = np.zeros((gt_dim+1, num_pred_labels+1), dtype=np.float32)
                precMat_subset = get_centerline_overlap(
                    pred_labels_rel, gt_labels_subset,
                    np.transpose(precMat_subset))
                precMat_subset = np.transpose(precMat_subset)
                recallMat_subset = recallMat[[0,] + dim_insts_rel]
                locMat_subset = locMat[[0,] + dim_insts_rel]
                # compute coverage
                gt_covs_dim = get_gt_coverage(gt_labels_subset, pred_labels_rel,
                        precMat_subset, recallMat_subset)
                avg_cov_dim = np.mean(gt_covs_dim)
                # compute tp
                if np.max(locMat_subset[1:, 1:]) > 0.5:
                    tp_05_dim, _, _ = assign_labels(
                        locMat_subset, assignment_strategy, 0.5, 1)
                tp_05_rel_dim = tp_05_dim / float(gt_dim)
            # add to metrics
            metrics.addMetric("general", "GT_dim", gt_dim)
            metrics.addMetric("general", "TP_05_dim", tp_05_dim)
            metrics.addMetric("general", "TP_05_rel_dim", tp_05_rel_dim)
            metrics.addMetric("general", "gt_covs_dim", gt_covs_dim)
            metrics.addMetric("general", "avg_gt_cov_dim", avg_cov_dim)

        if "avg_gt_cov_overlap" in add_general_metrics:
            tp_05_ovlp = 0
            tp_05_rel_ovlp = 0.0
            gt_covs_ovlp = []
            avg_cov_ovlp = 0
            overlap_mask = np.sum(gt_labels_rel > 0, axis=0) > 1

            ovlp_inst_ids = np.unique(gt_labels_rel[:, overlap_mask])
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
                        gt_labels_subset[i].astype(int), offset)
                    offset = np.max(gt_labels_subset[i]) + 1

                precMat_subset = np.zeros((gt_ovlp+1, num_pred_labels+1), dtype=np.float32)
                precMat_subset = get_centerline_overlap(
                    pred_labels_rel, gt_labels_subset,
                    np.transpose(precMat_subset))
                precMat_subset = np.transpose(precMat_subset)
                recallMat_subset = recallMat[[0,] + list(ovlp_inst_ids)]
                locMat_subset = locMat[[0,] + list(ovlp_inst_ids)]
                # compute coverage
                gt_covs_ovlp = get_gt_coverage(gt_labels_subset, pred_labels_rel,
                        precMat_subset, recallMat_subset)
                avg_cov_ovlp = np.mean(gt_covs_ovlp)

                # compute tp
                if np.max(locMat_subset[1:, 1:]) > 0.5:
                    tp_05_ovlp, _, _ = assign_labels(
                        locMat_subset, assignment_strategy, 0.5, 1)
                tp_05_rel_ovlp = tp_05_ovlp / float(gt_ovlp)
            # add to metrics
            metrics.addMetric("general", "GT_overlap", gt_ovlp)
            metrics.addMetric("general", "TP_05_overlap", tp_05_ovlp)
            metrics.addMetric("general", "TP_05_rel_overlap", tp_05_rel_ovlp)
            metrics.addMetric("general", "gt_covs_overlap", gt_covs_ovlp)
            metrics.addMetric("general", "avg_gt_cov_overlap", avg_cov_ovlp)

    return metrics


def average_flylight_score_over_instances(samples_foldn, result):
    # heads up: hard coded for 0.5 average F1 + 0.5 average gt coverage
    threshs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    fscores = []
    gt_covs = []
    tp = {}
    fp = {}
    fn = {}
    false_split = []    # to remove
    false_merge = []    # to remove
    fm = []
    fs = []
    tp_covs = []
    num_gt = []
    num_pred = []
    tp_05 = []
    tp_05_cldice = []
    gt_dim = []
    gt_covs_dim = []
    tp_05_dim = []
    gt_ovlp = []
    gt_covs_ovlp = []
    tp_05_ovlp = []

    for thresh in threshs:
        tp[thresh] = []
        fp[thresh] = []
        fn[thresh] = []
    for s in samples_foldn:
        # todo: move type conversion to evaluate_file
        c_gen = result[s]["general"]
        gt_covs += list(np.array(
            c_gen["gt_skel_coverage"], dtype=np.float32))
        num_gt.append(c_gen["Num GT"])
        num_pred.append(c_gen["Num Pred"])
        if "FM" in c_gen.keys():
            fm.append(c_gen["FM"])
            fs.append(c_gen["FS"])
            tp_05.append(c_gen["TP_05"])
            tp_05_cldice += list(np.array(
                c_gen["TP_05_cldice"], dtype=np.float32))
        if "GT_dim" in c_gen.keys():
            gt_dim.append(c_gen["GT_dim"])
            tp_05_dim.append(c_gen["TP_05_dim"])
            gt_covs_dim += list(np.array(c_gen["gt_covs_dim"], dtype=np.float32))
        if "GT_overlap" in c_gen.keys():
            gt_ovlp.append(c_gen["GT_overlap"])
            tp_05_ovlp.append(c_gen["TP_05_overlap"])
            gt_covs_ovlp += list(np.array(c_gen["gt_covs_overlap"], dtype=np.float32))

        for thresh in threshs:
            tp[thresh].append(result[s][
                                  "confusion_matrix"][
                                  "th_" + str(thresh).replace(".", "_")][
                                  "AP_TP"])
            fp[thresh].append(result[s][
                                  "confusion_matrix"][
                                  "th_" + str(thresh).replace(".", "_")][
                                  "AP_FP"])
            fn[thresh].append(result[s][
                                  "confusion_matrix"][
                                  "th_" + str(thresh).replace(".", "_")][
                                  "AP_FN"])
            if thresh == 0.5:
                false_split.append(result[s][
                                       "confusion_matrix"]["th_0_5"][
                                       "false_split"])
                false_merge.append(result[s][
                                       "confusion_matrix"]["th_0_5"][
                                       "false_merge"])
                tp_covs += list(np.array(
                    result[s]["confusion_matrix"]["th_0_5"][
                        "tp_skel_coverage"], dtype=np.float32))
    for thresh in threshs:
        fscores.append(2 * np.sum(tp[thresh]) / (
            2 * np.sum(tp[thresh]) + np.sum(fp[thresh]) + np.sum(fn[thresh])))
    avS = 0.5 * np.mean(fscores) + 0.5 * np.mean(gt_covs)

    per_instance_counts = {}
    per_instance_counts["general"] = {
        "Num GT": np.sum(num_gt),
        "Num Pred": np.sum(num_pred),
        "avg_gt_skel_coverage": np.mean(gt_covs),
        "avg_f1_cov_score": avS,
        "avFscore": np.mean(fscores),
        "FM": np.sum(fm),
        "FS": np.sum(fs),
        "TP_05": np.sum(tp_05),
        "TP_05_rel": np.sum(tp_05) / float(np.sum(num_gt)),
        "TP_05_cldice": tp_05_cldice,
        "avg_TP_05_cldice": np.mean(tp_05_cldice) if np.sum(tp_05) > 0 else 0.0,
        "GT_dim": np.sum(gt_dim),
        "TP_05_dim": np.sum(tp_05_dim),
        "TP_05_rel_dim": np.sum(tp_05_dim) / float(np.sum(gt_dim)),
        "avg_gt_cov_dim": np.mean(gt_covs_dim),
        "GT_overlap": np.sum(gt_ovlp),
        "TP_05_overlap": np.sum(tp_05_ovlp),
        "TP_05_rel_overlap": np.sum(tp_05_ovlp) / float(np.sum(gt_ovlp)),
        "avg_gt_cov_overlap": np.mean(gt_covs_ovlp)
    }
    per_instance_counts["confusion_matrix"] = {"avFscore": np.mean(fscores)}
    per_instance_counts["gt_covs"] = gt_covs
    per_instance_counts["false_split"] = np.sum(false_split)
    per_instance_counts["false_merge"] = np.sum(false_merge)
    per_instance_counts["tp"] = []
    per_instance_counts["fp"] = []
    per_instance_counts["fn"] = []
    for i, thresh in enumerate(threshs):
        per_instance_counts["tp"].append(np.sum(tp[thresh]))
        per_instance_counts["fp"].append(np.sum(fp[thresh]))
        per_instance_counts["fn"].append(np.sum(fn[thresh]))
        per_instance_counts["confusion_matrix"][
            "th_" + str(thresh).replace(".", "_")] = {
            "fscore": fscores[i]}
        if thresh == 0.5:
            per_instance_counts["confusion_matrix"]["th_0_5"]["AP_TP"] = np.sum(
                tp[thresh])
            per_instance_counts["confusion_matrix"]["th_0_5"]["AP_FP"] = np.sum(
                fp[0.5])
            per_instance_counts["confusion_matrix"]["th_0_5"]["AP_FN"] = np.sum(
                fn[0.5])
            per_instance_counts["confusion_matrix"]["th_0_5"][
                "false_split"] = np.sum(false_split)
            per_instance_counts["confusion_matrix"]["th_0_5"][
                "false_merge"] = np.sum(false_merge)
            per_instance_counts["confusion_matrix"]["th_0_5"][
                "avg_tp_skel_coverage"] = np.mean(tp_covs)

    return avS, per_instance_counts


# TODO: copy code from ppp
def average_sets(acc_a, dict_a, acc_b, dict_b):
    threshs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    acc = np.mean([acc_a, acc_b])
    fscore = np.mean([dict_a["general"]["avFscore"],
        dict_b["general"]["avFscore"]])
    gt_covs = list(dict_a["gt_covs"]) + list(dict_b["gt_covs"])
    num_gt = dict_a["general"]["Num GT"] + dict_b["general"]["Num GT"]
    tp_05 = dict_a["general"]["TP_05"] + dict_b["general"]["TP_05"]
    tp_05_cldice = list(dict_a["general"]["tp_05_cldice"]) + \
            list(dict_b["general"]["tp_05_cldice"])

    per_instance_counts = {}
    per_instance_counts["general"] = {
        "Num GT": num_gt,
        "Num Pred": dict_a["general"]["Num Pred"] + dict_b["general"]["Num Pred"],
        "avg_gt_skel_coverage": np.mean([
            dict_a["general"]["avg_gt_skel_coverage"],
            dict_b["general"]["avg_gt_skel_coverage"]]),
        "avg_f1_cov_score": acc,
        "avFscore": fscore,
        "FM": dict_a["general"]["FM"] + dict_b["general"]["FM"],
        "FS": dict_a["general"]["FS"] + dict_b["general"]["FS"],
        "TP_05": tp_05,
        "TP_05_rel": tp_05 / float(num_gt),
        "avg_TP_05_cldice": np.mean(tp_05_cldice)
    }
    per_instance_counts["confusion_matrix"] = {"avFscore": fscore}
    per_instance_counts["gt_covs"] = gt_covs
    per_instance_counts["false_split"] = dict_a["false_split"] + dict_b["false_split"]
    per_instance_counts["false_merge"] = dict_a["false_merge"] + dict_b["false_merge"]
    for i, thresh in enumerate(threshs):
        cm_a = dict_a["confusion_matrix"]["th_" + str(thresh).replace(".", "_")]
        cm_b = dict_b["confusion_matrix"]["th_" + str(thresh).replace(".", "_")]
        per_instance_counts["confusion_matrix"][
            "th_" + str(thresh).replace(".", "_")] = {
            "fscore": np.mean([cm_a["fscore"],cm_b["fscore"]])}
        if thresh == 0.5:
            per_instance_counts["confusion_matrix"]["th_0_5"]["AP_TP"] = \
                    cm_a["AP_TP"] + cm_b["AP_TP"]
            per_instance_counts["confusion_matrix"]["th_0_5"]["AP_FP"] = \
                    cm_a["AP_FP"] + cm_b["AP_FP"]
            per_instance_counts["confusion_matrix"]["th_0_5"]["AP_FN"] = \
                    cm_a["AP_FN"] + cm_b["AP_FN"]
            per_instance_counts["confusion_matrix"]["th_0_5"][
                "false_split"] = cm_a["false_split"] + cm_b["false_split"]
            per_instance_counts["confusion_matrix"]["th_0_5"][
                "false_merge"] = cm_a["false_merge"] + cm_b["false_merge"]
            per_instance_counts["confusion_matrix"]["th_0_5"][
                "avg_tp_skel_coverage"] = np.mean(
                        [cm_a["avg_tp_skel_coverage"],
                            cm_b["avg_tp_skel_coverage"]])
    return acc, per_instance_counts


def main():
    """main entry point if called from command line

    compare segmentation of args.res_file and args.gt_file.
    flags to select localization criterion, assignment strategy and
    which metrics to compute.

    TODO: option to just pass toml file instead of flags
    """
    parser = argparse.ArgumentParser()
    # input output
    parser.add_argument('--res_file', nargs="+", type=str,
            help='path to result file', required=True)
    parser.add_argument('--gt_file', nargs="+", type=str,
            help='path to ground truth file', required=True)
    parser.add_argument('--res_key', type=str,
            help='name result hdf/zarr key')
    parser.add_argument('--gt_key', type=str,
            help='name ground truth hdf/zarr key')
    parser.add_argument('--out_dir', nargs="+", type=str,
            help='output directory', required=True)
    parser.add_argument('--suffix', type=str,
            help='suffix (deprecated)', default='')
    parser.add_argument('--ndim', type=int,
            default=3,
            help='number of spatial dimensions')
    parser.add_argument('--summary_out_dir', type=str, default=None,
            help='output directory for summary')
    # metrics definitions
    parser.add_argument('--localization_criterion', type=str,
            help='localization_criterion', default='iou',
            choices=['iou', 'cldice'])
    parser.add_argument('--assignment_strategy', type=str,
            help='assignment strategy', default='greedy',
            choices=['hungarian', 'greedy'])
    parser.add_argument('--add_general_metrics', type=str,
            nargs='+', help='add general metrics', default=[])
    parser.add_argument('--metric', type=str,
            default='confusion_matrix.th_0_5.AP',
            help='check if this metric already has been computed in '\
                    'possibly existing result files')
    parser.add_argument('--summary', type=str, nargs='+',
            default=None, help='list of metrics to include in the summary')
    # visualize
    parser.add_argument('--visualize', help='visualize segmentation errors',
            action='store_true', default=False)
    parser.add_argument('--visualize_type', type=str,
            default='nuclei',
            help='which type of data should be visualized, '\
                    'e.g. nuclei, neurons.',)
    # other
    parser.add_argument('--overlapping_inst', action='store_true',
            default=False,
            help='if there can be multiple instances per pixel '\
                    'in ground truth and prediction')
    parser.add_argument('--partly', action='store_true',
            default=False,
            help='if ground truth is only labeled partly')
    parser.add_argument('--partly_list', nargs="+", type=int, default=None,
            help='use partly_list if you have a list of completely and partly inputs')
    parser.add_argument('--foreground_only', action='store_true',
            default=False,
            help='if background should be excluded')
    parser.add_argument('--remove_small_components', type=int,
            default=None,
            help='remove instances with less pixel than this threshold')
    parser.add_argument('--evaluate_false_labels', action='store_true',
            default=False,
            help='if false split and false merge should be computed '\
                    'besides false positives and false negatives')
    parser.add_argument('--app', type=str, default=None,
            help='set parameters for specific applications',
            choices=['flylight'])
    parser.add_argument('--fm_thresh', type=int, default=0.1,
            help='min overlap with gt to count as false merger')
    parser.add_argument('--fs_thresh', type=int, default=0.05,
            help='min overlap with gt to count as false split')
    parser.add_argument('--from_scratch', action='store_true',
            default=False,
            help='recompute everything (instead of checking '\
                    'if results are already there)')
    parser.add_argument("--debug", help="",
                        action="store_true")

    logger.debug("arguments %s",tuple(sys.argv))
    args = parser.parse_args()

    # shortcut if res_file and gt_file contain folders
    if len(args.res_file) == 1 and len(args.gt_file) == 1:
        res_file = args.res_file[0]
        gt_file = args.gt_file[0]
        if (os.path.isdir(res_file) and not res_file.endswith(".zarr")) \
                and (os.path.isdir(gt_file) and not gt_file.endswith(".zarr")):
            args.res_file = natsorted(glob.glob(res_file + "/*.hdf"))

            def get_gt_file(in_fn, gt_folder):
                out_fn = os.path.join(gt_folder,
                        os.path.basename(in_fn).split(".")[0] + ".zarr")
                assert os.path.basename(in_fn).split(".")[0] == os.path.basename(
                    out_fn).split(".")[0]
                return out_fn

            args.gt_file = [get_gt_file(fn, gt_file) for fn in args.res_file]

    # check same length for result and gt files
    assert len(args.res_file) == len(args.gt_file), \
            "Please check, not the same number of result and gt files"
    # set partly parameter for all samples if not done already
    if len(args.res_file) > 1:
        if args.partly_list is not None:
            assert len(args.partly_list) == len(args.res_file), \
                    "Please check, not the same number of result files "\
                    "and partly_list values"
            partly_list = np.array(args.partly_list, dtype=bool)
        else:
            partly_list = [args.partly] * len(args.res_file)
    else:
        partly_list = [args.partly]

    # check out_dir
    if len(args.res_file) > 1:
        if len(args.out_dir) > 1:
            assert len(args.res_file) == len(args.out_dir), \
                    "Please check, number of input files and output folders should correspond"
            outdir_list = args.out_dir
        else:
            outdir_list = args.out_dir * len(args.res_file)
    else:
        assert len(args.out_dir) == 1, "Please check number of output directories"
        outdir_list = args.out_dir
    # check output dir for summary
    if args.summary_out_dir is None:
        args.summary_out_dir = args.out_dir[0]

    if args.app is not None:
        if args.app == "flylight":
            print("Warning: parameter app is set and will overwrite parameters. "\
                    "This might not be what you want.")
            args.ndim = 3
            args.localization_criterion = "cldice"
            args.assignment_strategy = "greedy"
            args.overlapping_inst = True
            args.remove_small_components = 800
            args.evaluate_false_labels = True
            args.metric = "general.avg_f1_cov_score"
            args.add_general_metrics = ["avg_gt_skel_coverage",
                    "avg_tp_skel_coverage",
                    "avg_f1_cov_score",
                    "false_merge",
                    "false_split"
                    ]
            args.summary = ["general.Num GT",
                    "general.Num Pred",
                    "general.avg_f1_cov_score",
                    "confusion_matrix.avFscore",
                    "general.avg_gt_skel_coverage",
                    "confusion_matrix.th_0_1.fscore",
                    "confusion_matrix.th_0_2.fscore",
                    "confusion_matrix.th_0_3.fscore",
                    "confusion_matrix.th_0_4.fscore",
                    "confusion_matrix.th_0_5.fscore",
                    "confusion_matrix.th_0_6.fscore",
                    "confusion_matrix.th_0_7.fscore",
                    "confusion_matrix.th_0_8.fscore",
                    "confusion_matrix.th_0_9.fscore",
                    "confusion_matrix.th_0_5.AP_TP", "confusion_matrix.th_0_5.AP_FP",
                    "confusion_matrix.th_0_5.AP_FN",
                    "confusion_matrix.th_0_5.false_split",
                    "confusion_matrix.th_0_5.false_merge",
                    "confusion_matrix.th_0_5.avg_tp_skel_coverage",
                    "general.FM",
                    "general.FS",
                    "general.TP_05",
                    "general.TP_05_rel",
                    "general.avg_TP_05_cldice"
                    ]
            args.visualize_type = "neuron"
            args.fm_thresh = 0.1
            args.fs_thresh = 0.05

    samples = []
    metric_dicts = []
    for res_file, gt_file, partly, out_dir in zip(args.res_file, args.gt_file, partly_list, outdir_list):
        sample_name = os.path.basename(res_file).split(".")[0]
        print("sample_name: ", sample_name)
        print("res_file: ", res_file)
        print("gt_file: ", gt_file)
        print("partly: ", partly)
        print("localization: ", args.localization_criterion)
        print("assignment: ", args.assignment_strategy)
        print("from scratch: ", args.from_scratch)
        print("add general metrics: ", args.add_general_metrics)

        samples.append(os.path.basename(res_file).split(".")[0])
        metric_dict = evaluate_file(
                res_file,
                gt_file,
                args.ndim,
                out_dir,
                res_key=args.res_key,
                gt_key=args.gt_key,
                suffix=args.suffix,
                localization_criterion=args.localization_criterion,
                assignment_strategy=args.assignment_strategy,
                add_general_metrics=args.add_general_metrics,
                visualize=args.visualize,
                visualize_type=args.visualize_type,
                partly=partly,
                overlapping_inst=args.overlapping_inst,
                foreground_only=args.foreground_only,
                remove_small_components=args.remove_small_components,
                evaluate_false_labels=args.evaluate_false_labels,
                fm_thresh=args.fm_thresh,
                fs_thresh=args.fs_thresh,
                from_scratch=args.from_scratch,
                debug=args.debug
                )
        metric_dicts.append(metric_dict)
        print(metric_dict)

    # aggregate over instances
    metrics = {}
    metrics_full = {}
    acc_all_instances = None
    for metric_dict, sample in zip(metric_dicts, samples):
        if metric_dict is None:
            continue
        metrics_full[sample] = metric_dict
    if len(np.unique(partly_list)) > 1:
        print("averaging for combined")
        # get average over instances for completely
        samples = np.array(samples)
        acc_cpt, acc_inst_cpt = average_flylight_score_over_instances(
                samples[partly_list==False], metrics_full)
        acc_prt, acc_inst_prt = average_flylight_score_over_instances(
                samples[partly_list==True], metrics_full)
        acc, acc_all_instances = average_sets(acc_cpt, acc_inst_cpt,
                acc_prt, acc_inst_prt)

    else:
        acc, acc_all_instances = average_flylight_score_over_instances(
            samples, metrics_full)
    summarize_metric_dict(metric_dicts, samples, args.summary,
                          os.path.join(args.summary_out_dir, "summary.csv"),
                          agg_inst_dict=acc_all_instances
                          )


if __name__ == "__main__":
    main()

