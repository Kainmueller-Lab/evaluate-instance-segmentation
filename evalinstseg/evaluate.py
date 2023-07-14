import glob
import logging
import sys

import argparse
import numpy as np

from skimage.segmentation import relabel_sequential
import toml

from .util import (
    read_file,
    check_sizes,
    get_output_name,
    get_centerline_overlap_single,
    compute_localization_criterion,
    assign_labels,
)
from .visualize import (
    visualize_neurons,
    visualize_nuclei
)

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
        e.g., passing "confusion_matrix.th_0.5" results in:
        `dict = {"confusion_matrix": {"th_0.5": result}}`
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
        res_file, gt_file, res_key=None, gt_key=None,
        out_dir=None, suffix="",
        localization_criterion="iou", # "iou", "cldice"
        assignment_strategy="hungarian", # "hungarian", "greedy", "gt_0_5"
        add_general_metrics=[],
        visualize=False,
        visualize_type="nuclei", # "nuclei" or "neuron"
        overlapping_inst=False,
        partly=False,
        foreground_only=False,
        remove_small_components=None,
        evaluate_false_labels=False,
        unique_false_labels=False,
        **kwargs
):
    """computes segmentation quality of file wrt. its ground truth

    Args
    -----
    res_file: str/Path
        path to computed segmentation results
    gt_file: str/Path
        path to ground truth segmentation
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

    # put together output filename with suffix
    outFn = get_output_name(out_dir, res_file, res_key, suffix,
            localization_criterion, assignment_strategy,
            remove_small_components)

    # if from_scratch is set, overwrite existing evaluation files
    # otherwise try to load precomputed metric
    if not kwargs.get("from_scratch") and \
            len(glob.glob(outFn + ".toml")) > 0: # heads up: changed here from outFnBase *.toml
        with open(outFn+".toml", 'r') as tomlFl:
            metrics = toml.load(tomlFl)
        if kwargs.get('metric', None) is None:
            return metrics
        try:
            metric = metrics
            for k in kwargs['metric'].split('.'):
                metric = metric[k]
            logger.info('Skipping evaluation, already exists! %s',
                        outFn)
            return metrics
        except KeyError:
            logger.info("Error (key %s missing) in existing evaluation "
                        "for %s. Recomputing!",
                        kwargs['metric'], res_file)

    # read preprocessed file
    pred_labels = read_file(res_file, res_key)
    logger.debug("prediction min %f, max %f, shape %s", np.min(pred_labels),
                 np.max(pred_labels), pred_labels.shape)
    pred_labels = np.squeeze(pred_labels)
    logger.debug("prediction shape %s", pred_labels.shape)

    # read ground truth data
    gt_labels = read_file(gt_file, gt_key)
    logger.debug("gt min %f, max %f, shape %s", np.min(gt_labels),
                 np.max(gt_labels), gt_labels.shape)

    # check sizes and crop if necessary
    gt_labels, pred_labels = check_sizes(
            gt_labels, pred_labels, overlapping_inst, **kwargs)

    # remove small components
    if remove_small_components is not None and remove_small_components > 0:
        logger.info("call remove small components with filter size %i",
                remove_small_components)
        logger.debug("prediction %s, shape %s", np.unique(pred_labels),
                     pred_labels.shape)
        pred_labels = filter_components(pred_labels, remove_small_components)
        logger.debug("prediction %s, shape %s", np.unique(pred_labels),
                     pred_labels.shape)

    # if foreground_only is selected, remove all predictions within gt background
    if foreground_only:
        try:
            pred_labels[gt_labels==0] = 0
        except IndexError:
            pred_labels[:, np.any(gt_labels, axis=0).astype(int)==0] = 0
    logger.info("processing %s %s", res_file, gt_file)

    # relabel gt labels in case of binary mask per channel
    if overlapping_inst and np.max(gt_labels) == 1:
        for i in range(gt_labels.shape[0]):
            gt_labels[i] = gt_labels[i] * (i + 1)

    return evaluate_volume(
            gt_labels, pred_labels, outFn,
            localization_criterion,
            assignment_strategy,
            evaluate_false_labels,
            unique_false_labels,
            add_general_metrics,
            visualize,
            visualize_type,
            overlapping_inst,
            partly
            )


# todo: should pixelwise neuron evaluation also be possible?
# keep_gt_shape not in pixelwise overlap so far
def evaluate_volume(
        gt_labels, pred_labels, outFn,
        localization_criterion="iou",
        assignment_strategy="hungarian",
        evaluate_false_labels=False,
        unique_false_labels=False,
        add_general_metrics=[],
        visualize=False,
        visualize_type="nuclei",
        overlapping_inst=False,
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

    # relabel labels sequentially
    pred_labels_rel, _, _ = relabel_sequential(pred_labels.astype(int))
    gt_labels_rel, _, _ = relabel_sequential(gt_labels)

    print("check for overlap: ", np.sum(np.sum(gt_labels_rel > 0, axis=0) > 1))

    # get number of labels
    num_pred_labels = int(np.max(pred_labels_rel))
    num_gt_labels = int(np.max(gt_labels_rel))
    num_matches = min(num_gt_labels, num_pred_labels)

    # get localization criterion
    locMat, recallMat, precMat, recallMat_wo_overlap = \
            compute_localization_criterion(
                    pred_labels_rel, gt_labels_rel,
                    num_pred_labels, num_gt_labels,
                    localization_criterion, overlapping_inst)

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
            check_wo_overlap = overlapping_inst or \
                    len(gt_labels_rel.shape) > len(pred_labels_rel.shape)
            fp_ind, fn_ind, fs_ind, fm_pred_ind, fm_gt_ind, \
                    fm_count, fp_ind_only_bg = get_false_labels(
                            pred_ind, gt_ind, num_pred_labels, num_gt_labels,
                            locMat, precMat, recallMat, th,
                            check_wo_overlap, unique_false_labels,
                            recallMat_wo_overlap
                            )

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
                    gt_labels_rel, locMat, gt_ind, pred_ind, th, outFn)
            elif visualize_type == "neuron" and localization_criterion == "cldice":
                visualize_neurons(
                    gt_labels_rel, pred_labels_rel, gt_ind, pred_ind,
                    outFn, fp_ind, fn_ind, fm_gt_ind)
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
        if "avg_gt_skel_coverage" in add_general_metrics:
            # only take max gt label for each pred label to not count
            # pred labels twice for overlapping gt instances
            max_gt_ind = np.argmax(precMat, axis=0)
            gt_cov = []
            # if gt and pred have overlapping instances
            if overlapping_inst:
                # recalculate clRecall for each gt and union of assigned
                # predictions, as predicted instances can potentially overlap
                max_gt_ind_unique = np.unique(max_gt_ind[max_gt_ind > 0])
                for gt_i in np.arange(1, num_gt_labels + 1):
                    if gt_i in max_gt_ind_unique:
                        pred_union = np.zeros(
                                pred_labels_rel.shape[1:],
                                dtype=pred_labels_rel.dtype)
                        for pred_i in np.arange(num_pred_labels + 1)[max_gt_ind == gt_i]:
                            mask = pred_labels_rel[pred_i - 1] > 0
                            pred_union[mask] = 1
                        gt_cov.append(get_centerline_overlap_single(gt_labels_rel,
                                pred_union, gt_i, 1))
                    else:
                        gt_cov.append(0.0)
            else:
                # if gt has overlapping instances, but not prediction
                if len(gt_labels_rel.shape) > len(pred_labels_rel.shape):
                    for i in range(1, recallMat.shape[0]):
                        gt_cov.append(np.sum(recallMat[i, max_gt_ind==i]))
                # if none has overlapping instances
                else:
                    gt_cov = np.sum(recallMat[1:, 1:], axis=1)
            gt_skel_coverage = np.mean(gt_cov)
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
                metrics.addMetric(tblNameGen, "tp_skel_coverage", tp_cov)
                metrics.addMetric(tblNameGen, "avg_tp_skel_coverage", tp_skel_coverage)

        if "avg_f1_cov_score" in add_general_metrics:
            avg_f1_cov_score = 0.5 * avFscore19 + 0.5 * gt_skel_coverage
            metrics.addMetric(tblNameGen, "avg_f1_cov_score", avg_f1_cov_score)

    metrics.save()
    return metrics.metricsDict


def main():
    """main entry point if called from command line

    compare segmentation of args.res_file and args.gt_file.
    flags to select localization criterion, assignment strategy and
    which metrics to compute.

    TODO: option to just pass toml file instead of flags
    """
    parser = argparse.ArgumentParser()
    # input output
    parser.add_argument('--res_file', type=str,
            help='path to result file', required=True)
    parser.add_argument('--gt_file', type=str,
            help='path to ground truth file', required=True)
    parser.add_argument('--res_key', type=str,
            help='name result hdf/zarr key')
    parser.add_argument('--gt_key', type=str,
            help='name ground truth hdf/zarr key')
    parser.add_argument('--out_dir', type=str,
            help='output directory', required=True)
    parser.add_argument('--suffix', type=str,
            help='suffix (deprecated)', default='')
    # metrics definitions
    parser.add_argument('--localization_criterion', type=str,
            help='localization_criterion', default='iou',
            choices=['iou', 'cldice'])
    parser.add_argument('--assignment_strategy', type=str,
            help='assignment strategy', default='hungarian',
            choices=['hungarian', 'greedy'])
    parser.add_argument('--add_general_metrics', type=str,
            nargs='+', help='add general metrics', default=[])
    parser.add_argument('--metric', type=str,
            default='confusion_matrix.th_0_5.AP',
            help='check if this metric already has been computed in '\
                    'possibly existing result files')
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
    parser.add_argument('--keep_gt_shape', action='store_true',
            default=False,
            help='if there can be multiple instances per pixel '\
                    'in ground truth but not prediction')
    parser.add_argument('--partly', action='store_true',
            default=False,
            help='if ground truth is only labeled partly')
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
    parser.add_argument('--unique_false_labels', action='store_true',
            default=False,
            help='if false positives should not include false splits '\
                    'and false negatives not false merges') # take out?
    parser.add_argument('--from_scratch', action='store_true',
            default=False,
            help='recompute everything (instead of checking '\
                    'if results are already there)')
    parser.add_argument("--debug", help="",
                        action="store_true")

    logger.debug("arguments %s",tuple(sys.argv))
    args = parser.parse_args()

    evaluate_file(
        args.res_file, args.gt_file,
        res_key=args.res_key,
        gt_key=args.gt_key,
        out_dir=args.out_dir,
        suffix=args.suffix,
        localization_criterion=args.localization_criterion,
        assignment_strategy=args.assignment_strategy,
        add_general_metrics=args.add_general_metrics,
        visualize=args.visualize,
        visualize_type=args.visualize_type,
        overlapping_inst=args.overlapping_inst,
        keep_gt_shape=args.keep_gt_shape,
        partly=args.partly,
        foreground_only=args.foreground_only,
        remove_small_components=args.remove_small_components,
        evaluate_false_labels=args.evaluate_false_labels,
        unique_false_labels=args.unique_false_labels,
        from_scratch=args.from_scratch,
        debug=args.debug
    )


if __name__ == "__main__":
    main()
