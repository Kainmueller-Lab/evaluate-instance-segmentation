import glob
import logging
import sys
import os

import argparse
import numpy as np
import toml
from natsort import natsorted

from .util import (
    assign_labels,
    check_and_fix_sizes,
    check_fix_and_unify_ids,
    compute_localization_criterion,
    get_centerline_overlap_single,
    get_false_labels,
    get_output_name,
    read_file,
)
from .visualize import (
    visualize_neurons,
    visualize_nuclei
)
from .summarize import summarize_metric_dict

logger = logging.getLogger(__name__)


class Metrics:
    """class that stores variety of computed metrics

    Attributes
    ----------
    metricsDict: dict
        python dict containing results.
        filled by calling add/getTable and addMetric.
    fn: str/Path
        filename without toml extension. results will be written to this file.
    """
    def __init__(self, fn):
        self.metricsDict = {}
        self.fn = fn

    def save(self):
        """dump results to toml file."""
        logger.info("saving %s", self.fn)
        with open(self.fn+".toml", 'w') as tomlFl:
            toml.dump(self.metricsDict, tomlFl)

    def addTable(self, name, dct=None):
        """add new sub-table to result dict

        pass name containing '.' for nested tables,
        e.g., passing "confusion_matrix.th_0_5" results in:
        `dict = {"confusion_matrix": {"th_0_5": result}}`
        """
        levels = name.split(".")
        if dct is None:
            dct = self.metricsDict
        if levels[0] not in dct:
            dct[levels[0]] = {}
        if len(levels) > 1:
            name = ".".join(levels[1:])
            self.addTable(name, dct[levels[0]])

    def getTable(self, name, dct=None):
        """access existing sub-table in result dict

        pass name containing '.' to access nested tables.
        """
        levels = name.split(".")
        if dct is None:
            dct = self.metricsDict
        if len(levels) == 1:
            return dct[levels[0]]
        else:
            name = ".".join(levels[1:])
            return self.getTable(name, dct=dct[levels[0]])

    def addMetric(self, table_name, name, value):
        """add result for metric `name` to sub-table `table_name` """
        tbl = self.getTable(table_name)
        tbl[name] = value


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
        unique_false_labels=False,
        check_for_metric=None,
        from_scratch=False,
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
    gt_labels = read_file(gt_file, gt_key)
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
        unique_false_labels=unique_false_labels,
        add_general_metrics=add_general_metrics,
        visualize=visualize,
        visualize_type=visualize_type,
        overlapping_inst=overlapping_inst,
        remove_small_components=remove_small_components,
        foreground_only=foreground_only,
        partly=partly
    )
    metrics.save()

    return metrics.metricsDict


# todo: should pixelwise neuron evaluation also be possible?
def evaluate_volume(
        gt_labels,
        pred_labels,
        ndim,
        outFn,
        localization_criterion="iou",
        assignment_strategy="hungarian",
        evaluate_false_labels=False,
        unique_false_labels=False,
        add_general_metrics=[],
        visualize=False,
        visualize_type="nuclei",
        overlapping_inst=False,
        remove_small_components=None,
        foreground_only=False,
        partly=False
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
    gt_labels_rel, pred_labels_rel = check_fix_and_unify_ids(
        gt_labels, pred_labels, remove_small_components, foreground_only)

    logger.debug(
        "are there pixels with multiple instances?: "
        f"{np.sum(np.sum(gt_labels_rel > 0, axis=0) > 1)}")

    # get number of labels
    num_pred_labels = int(np.max(pred_labels_rel))
    num_gt_labels = int(np.max(gt_labels_rel))
    num_matches = min(num_gt_labels, num_pred_labels)

    # get localization criterion
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
                    locMat, precMat, recallMat, th, unique_false_labels,
                    recallMat_wo_overlap)

        # get false positive and false negative counters
        if unique_false_labels:
            fp = len(fp_ind)
            fn = len(fn_ind)
        elif partly:
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

    # additional metrics
    if len(add_general_metrics) > 0:
        # get coverage for ground truth instances
        if "avg_gt_skel_coverage" in add_general_metrics or \
           "avg_tp_skel_coverage" in add_general_metrics or \
           "avg_f1_cov_score" in add_general_metrics:
            # only take max gt label for each pred label to not count
            # pred labels twice for overlapping gt instances
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
                            pred_labels_rel.shape[1:],
                            dtype=pred_labels_rel.dtype)
                        for pred_i in np.arange(num_pred_labels + 1)[max_gt_ind == gt_i]:
                            mask = np.max(pred_labels_rel == pred_i, axis=0)
                            pred_union[mask] = 1
                        gt_cov.append(get_centerline_overlap_single(
                            gt_labels_rel, pred_union, gt_i, 1))
                    else:
                        gt_cov.append(0.0)
            else:
                # otherwise use previously computed values
                for i in range(1, recallMat.shape[0]):
                    gt_cov.append(np.sum(recallMat[i, max_gt_ind==i]))
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

    return metrics


def average_flylight_score_over_instances(samples_foldn, result):
    # heads up: hard coded for 0.5 average F1 + 0.5 average gt coverage
    threshs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    fscores = []
    gt_covs = []
    tp = {}
    fp = {}
    fn = {}
    false_split = []
    false_merge = []
    tp_covs = []
    num_gt = []
    num_pred = []

    for thresh in threshs:
        tp[thresh] = []
        fp[thresh] = []
        fn[thresh] = []
    for s in samples_foldn:
        # todo: move type conversion to evaluate_file
        gt_covs += list(np.array(
            result[s]["general"]["gt_skel_coverage"], dtype=np.float32))
        num_gt.append(result[s]["general"]["Num GT"])
        num_pred.append(result[s]["general"]["Num Pred"])
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
        print(tp[thresh], fp[thresh], fn[thresh])
        fscores.append(2 * np.sum(tp[thresh]) / (
            2 * np.sum(tp[thresh]) + np.sum(fp[thresh]) + np.sum(fn[thresh])))
    print(fscores)
    avS = 0.5 * np.mean(fscores) + 0.5 * np.mean(gt_covs)

    per_instance_counts = {}
    per_instance_counts["general"] = {
        "Num GT": np.sum(num_gt),
        "Num Pred": np.sum(num_pred),
        "avg_gt_skel_coverage": np.mean(gt_covs),
        "avg_f1_cov_score": avS,
        "avFscore": np.mean(fscores)
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
    parser.add_argument('--out_dir', type=str,
            help='output directory', required=True)
    parser.add_argument('--suffix', type=str,
            help='suffix (deprecated)', default='')
    parser.add_argument('--ndim', type=int,
            default=3,
            help='number of spatial dimensions')
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
    parser.add_argument('--partly_list', nargs="+", type=bool, default=None,
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
            partly_list = args.partly_list
        else:
            partly_list = [args.partly] * len(args.res_file)
    else:
        partly_list = [args.partly]

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
                    "avg_f1_cov_score"
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
                    "confusion_matrix.th_0_5.avg_tp_skel_coverage"
                    ]

    samples = []
    metric_dicts = []
    for res_file, gt_file, partly in zip(args.res_file, args.gt_file, partly_list):
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
                args.out_dir,
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

    acc, acc_all_instances = average_flylight_score_over_instances(
        samples, metrics_full)
    summarize_metric_dict(metric_dicts, samples, args.summary,
                          os.path.join(args.out_dir, "summary.csv"),
                          agg_inst_dict=acc_all_instances
                          )


if __name__ == "__main__":
    main()

