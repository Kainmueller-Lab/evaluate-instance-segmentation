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
    get_output_name,
    read_file,
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
from .compute import (
    get_gt_coverage,
    get_gt_coverage_dim,
    get_gt_coverage_overlap,
    get_m2m_fm,
    get_m2m_fs
)
from .visualize import (
    visualize_neurons,
    visualize_nuclei
)
from .summarize import (
    summarize_metric_dict,
    average_flylight_score_over_instances,
    average_sets
)

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

    # check sizes and crop if necessary, unify input into per instance arrays
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
               gt_cov = get_gt_coverage(gt_labels_rel, pred_labels_rel,
                       precMat, recallMat)
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

        # TODO: rename "false_merge" and "false_splits" to sth with many-to-many?
        m2m_matches = None
        if "false_merge" in add_general_metrics:
            fm, m2m_matches = get_m2m_fm(gt_labels_rel, pred_labels_rel, num_pred_labels,
                    recallMat, fm_thresh)
            metrics.addMetric("general", "FM", fm)
        if "false_split" in add_general_metrics:
            # if fm and fs thresh are different, reset matches, reuse otherwise
            if fm_thresh != fs_thresh:
                m2m_matches = None
            fs, _ = get_m2m_fs(gt_labels_rel, pred_labels_rel, recallMat,
                    fs_thresh, m2m_matches)
            metrics.addMetric("general", "FS", fs)

        if "avg_gt_cov_dim" in add_general_metrics:
            gt_dim, tp_05_dim, tp_05_rel_dim, gt_covs_dim, avg_cov_dim = \
                    get_gt_coverage_dim(dim_insts_rel, gt_labels_rel, pred_labels_rel,
                            num_pred_labels, locMat, recallMat, assignment_strategy)
            # add to metrics
            metrics.addMetric("general", "GT_dim", gt_dim)
            metrics.addMetric("general", "TP_05_dim", tp_05_dim)
            metrics.addMetric("general", "TP_05_rel_dim", tp_05_rel_dim)
            metrics.addMetric("general", "gt_covs_dim", gt_covs_dim)
            metrics.addMetric("general", "avg_gt_cov_dim", avg_cov_dim)

        if "avg_gt_cov_overlap" in add_general_metrics:
            overlap_mask = np.sum(gt_labels_rel > 0, axis=0) > 1
            ovlp_inst_ids = np.unique(gt_labels_rel[:, overlap_mask])

            gt_ovlp, tp_05_ovlp, tp_05_rel_ovlp, gt_covs_ovlp, avg_cov_ovlp = \
                    get_gt_coverage_overlap(ovlp_inst_ids, gt_labels_rel, pred_labels_rel,
                            num_pred_labels, locMat, recallMat, assignment_strategy)
            # add to metricss
            metrics.addMetric("general", "GT_overlap", gt_ovlp)
            metrics.addMetric("general", "TP_05_overlap", tp_05_ovlp)
            metrics.addMetric("general", "TP_05_rel_overlap", tp_05_rel_ovlp)
            metrics.addMetric("general", "gt_covs_overlap", gt_covs_ovlp)
            metrics.addMetric("general", "avg_gt_cov_overlap", avg_cov_ovlp)

    return metrics


#TODO: option to just pass config (toml) file instead of flags
def main():
    """main entry point if called from command line

    compare segmentation of args.res_file and args.gt_file.
    flags to select localization criterion, assignment strategy and
    which metrics to compute.

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
    parser.add_argument('--fm_thresh', type=float, default=0.1,
            help='min overlap with gt to count as false merger')
    parser.add_argument('--fs_thresh', type=float, default=0.05,
            help='min overlap with gt to count as false split')
    parser.add_argument('--from_scratch', action='store_true',
            default=False,
            help='recompute everything (instead of checking '\
                    'if results are already there)')
    parser.add_argument('--eval_dim', action='store_true',
            default=False,
            help='if flag is set challenging cases like dim'\
                    ' and overlapping neurons are evaluated')
    parser.add_argument("--debug", help="",
                        action="store_true")

    logger.debug("arguments %s", tuple(sys.argv))
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
                    "avg_f1_cov_score",
                    "false_merge",
                    "false_split",
                    "avg_gt_cov_dim",
                    "avg_gt_cov_overlap"
                    ]
            args.summary = ["general.Num GT",
                    "general.Num Pred",
                    "general.avg_f1_cov_score",
                    "confusion_matrix.avFscore",
                    "general.avg_gt_skel_coverage",
                    "confusion_matrix.th_0_5.AP_TP",
                    "confusion_matrix.th_0_5.AP_FP",
                    "confusion_matrix.th_0_5.AP_FN",
                    "general.FM",
                    "general.FS",
                    "general.TP_05",
                    "general.TP_05_rel",
                    "general.avg_TP_05_cldice",
                    "general.GT_dim",
                    "general.TP_05_dim",
                    "general.TP_05_rel_dim",
                    "general.avg_gt_cov_dim",
                    "general.GT_overlap",
                    "general.TP_05_overlap",
                    "general.TP_05_rel_overlap",
                    "general.avg_gt_cov_overlap",
                    "confusion_matrix.th_0_1.fscore",
                    "confusion_matrix.th_0_2.fscore",
                    "confusion_matrix.th_0_3.fscore",
                    "confusion_matrix.th_0_4.fscore",
                    "confusion_matrix.th_0_5.fscore",
                    "confusion_matrix.th_0_6.fscore",
                    "confusion_matrix.th_0_7.fscore",
                    "confusion_matrix.th_0_8.fscore",
                    "confusion_matrix.th_0_9.fscore",
                    ]
            args.visualize_type = "neuron"
            args.fm_thresh = 0.1
            args.fs_thresh = 0.05
            args.eval_dim = True

    samples = []
    metric_dicts = []
    for res_file, gt_file, partly, out_dir in zip(
            args.res_file, args.gt_file, partly_list, outdir_list):
        sample_name = os.path.basename(res_file).split(".")[0]
        logger.info("sample_name: %s", sample_name)
        logger.info("res_file: %s", res_file)
        logger.info("gt_file: %s", gt_file)
        logger.info("partly: %s", partly)
        logger.info("localization: %s", args.localization_criterion)
        logger.info("assignment: %s", args.assignment_strategy)
        logger.info("from scratch: %s", args.from_scratch)
        logger.info("add general metrics: %s", args.add_general_metrics)

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
                eval_dim=args.eval_dim,
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
    if args.summary:
        summarize_metric_dict(metric_dicts, samples, args.summary,
                            os.path.join(args.summary_out_dir, "summary.csv"),
                            agg_inst_dict=acc_all_instances
                            )


if __name__ == "__main__":
    main()

