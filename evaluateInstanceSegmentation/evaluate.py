import glob
import logging
import os
import sys

import h5py
import argparse
import numpy as np
import scipy.ndimage
from scipy.optimize import linear_sum_assignment
from skimage.segmentation import relabel_sequential
import tifffile
import toml
import zarr
from skimage.morphology import skeletonize, skeletonize_3d
from skimage import io
from matplotlib.colors import to_rgb
#from pylab import cm

logger = logging.getLogger(__name__)


vis_cmap = [
        [ 49, 130, 189],
        [230,  85,  13],
        [ 49, 163,  84],
        [117, 107, 177],
        #[99, 99, 99],
        [107, 174, 214],
        [253, 141,  60],
        [116, 196, 118],
        [158, 154, 200],
        #[150, 150, 150],
        [158, 202, 225],
        [253, 174, 107],
        [161, 217, 155],
        [188, 189, 220],
        #[189, 189, 189],
        [198, 219, 239],
        [253, 208, 162],
        [199, 233, 192],
        [218, 218, 235],
        #[217, 217, 217]
        ]

gt_cmap = [
        "#88B04B", "#9F00A7", "#EFC050", "#34568B", "#E47A2E", "#BC70A4",
        "#92A8D1", "#A3B18A", "#45B8AC", "#6B5B95",
        "#F7CAC9", "#E8A798", "#9C9A40", "#9C4722", "#6B5876",
        "#CE3175", "#00A591", "#EDD59E", "#1E7145", "#E9FF70",
        ]
pred_cmap = [
        "#FDAC53", "#9BB7D4", "#B55A30", "#F5DF4D", "#0072B5",
        "#A0DAA9", "#E9897E", "#00A170", "#926AA6", "#EFE1CE",
        "#9A8B4F", "#FFA500", "#56C6A9", "#4B5335", "#798EA4",
        "#E0B589", "#00758F", "#FA7A35", "#578CA9", "#95DEE3"
        ]
class Metrics:
    def __init__(self, fn):
        self.metricsDict = {}
        self.metricsArray = []
        self.fn = fn
        self.outFl = open(self.fn+".txt", 'w')

    def save(self):
        self.outFl.close()
        logger.info("saving %s", self.fn)
        tomlFl = open(self.fn+".toml", 'w')
        toml.dump(self.metricsDict, tomlFl)

    def addTable(self, name, dct=None):
        levels = name.split(".")
        if dct is None:
            dct = self.metricsDict
        if levels[0] not in dct:
            dct[levels[0]] = {}
        if len(levels) > 1:
            name = ".".join(levels[1:])
            self.addTable(name, dct[levels[0]])

    def getTable(self, name, dct=None):
        levels = name.split(".")
        if dct is None:
            dct = self.metricsDict
        if len(levels) == 1:
            return dct[levels[0]]
        else:
            name = ".".join(levels[1:])
            return self.getTable(name, dct=dct[levels[0]])

    def addMetric(self, table, name, value):
        as_str = "{}: {}".format(name, value)
        self.outFl.write(as_str+"\n")
        self.metricsArray.append(value)
        tbl = self.getTable(table)
        tbl[name] = value


def maybe_crop(pred_labels, gt_labels, overlapping_inst=False):
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
    # check for different formats
    if infn.endswith(".hdf"):
        with h5py.File(infn, 'r') as f:
            volume = np.array(f[key])
    elif infn.endswith(".tif") or infn.endswith(".tiff") or \
         infn.endswith(".TIF") or infn.endswith(".TIFF"):
        volume = tifffile.imread(infn)
    elif infn.endswith(".zarr"):
        print(infn)
        try:
            f = zarr.open(infn, 'r')
        except zarr.errors.PathNotFoundError as e:
            logger.info("File %s not found!", infn)
            raise e
        volume = np.array(f[key])
    else:
        raise NotImplementedError("invalid file format %s", infn)
    return volume


def check_sizes(gt_labels, pred_labels, overlapping_inst, **kwargs):
    # check: what if there are only one gt instance in overlapping_inst scenario?
    keep_gt_shape = kwargs.get("keep_gt_shape", False)
    if gt_labels.shape[0] == 1 and not keep_gt_shape:
        gt_labels.shape = gt_labels.shape[1:]
    gt_labels = np.squeeze(gt_labels)
    if gt_labels.ndim > pred_labels.ndim and not keep_gt_shape:
        gt_labels = np.max(gt_labels, axis=0)
    logger.debug("gt shape %s", gt_labels.shape)

    # heads up: should not crop channel dimensions, assuming channels first
    if not keep_gt_shape:
        pred_labels, gt_labels = maybe_crop(pred_labels, gt_labels,
                                            overlapping_inst)
    else:
        if pred_labels.ndim < gt_labels.ndim:
            pred_labels = np.expand_dims(pred_labels, axis=0)
            pred_labels, gt_labels = maybe_crop(pred_labels, gt_labels, True)
            pred_labels = np.squeeze(pred_labels)
    logger.debug("prediction %s, shape %s", np.unique(pred_labels),
                 pred_labels.shape)
    logger.debug("gt %s, shape %s", np.unique(gt_labels),
                 gt_labels.shape)
    return gt_labels, pred_labels


def evaluate_file(
        res_file, gt_file, res_key=None, gt_key=None,
        out_dir=None, suffix="",
        localization_criterion="iou", # "iou", "cldice"
        assignment_strategy="hungarian", # "hungarian", "greedy", "gt_0_5"
        add_general_metrics=[],
        add_multi_thresh_metrics=[],
        visualize=False,
        visualize_type="nuclei", # "nuclei" or "neuron"
        overlapping_inst=False,
        partly=False,
        background=0, 
        foreground_only=False, 
        remove_small_components=None,
        evaluate_false_labels=False,
        unique_false_labels=False,
        **kwargs
        ):

    # check for deprecated args
    if kwargs.get("use_linear_sum_assignment", False) == True:
        assignment_strategy = "hungarian"
    filterSz = kwargs.get("filterSz", None)
    if filterSz is not None and filterSz > 0:
        remove_small_components = filterSz
    
    # put together output filename with suffix
    # todo: define own function
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
            pred_labels[:, np.any(gt_labels, axis=0).astype(np.int)==0] = 0
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
            add_multi_thresh_metrics,
            visualize,
            visualize_type,
            overlapping_inst,
            partly
            )


# todo: define own functions per localization criterion
# todo: rename iouMat, not use num_*_labels as parameters here?
def compute_localization_criterion(pred_labels_rel, gt_labels_rel, 
        num_pred_labels, num_gt_labels,
        localization_criterion, overlapping_inst):
    logger.debug("evaluate localization criterion for all gt and pred label pairs")
    
    # create matrices for pixelwise overlap measures
    iouMat = np.zeros((num_gt_labels+1, num_pred_labels+1),
                      dtype=np.float32)
    recallMat = np.zeros((num_gt_labels+1, num_pred_labels+1),
                         dtype=np.float32)
    precMat = np.zeros((num_gt_labels+1, num_pred_labels+1),
                       dtype=np.float32)
    fscoreMat = np.zeros((num_gt_labels+1, num_pred_labels+1),
                         dtype=np.float32)
    iouMat_wo_overlap = None
    recallMat_wo_overlap = None

    # intersection over union
    if localization_criterion == "iou":
        logger.debug("compute iou")
        print(overlapping_inst, pred_labels_rel.shape, gt_labels_rel.shape)
        # todo: implement iou for keep_gt_shape
        if overlapping_inst:
            pred_tile = [1, ] * pred_labels_rel.ndim
            pred_tile[0] = gt_labels_rel.shape[0]
            gt_tile = [1, ] * gt_labels_rel.ndim
            gt_tile[1] = pred_labels_rel.shape[0]
            pred_tiled = np.tile(pred_labels_rel, pred_tile).flatten()
            gt_tiled = np.tile(gt_labels_rel, gt_tile).flatten()
            mask = np.logical_or(pred_tiled > 0, gt_tiled > 0)
            overlay = np.array([
                pred_tiled[mask],
                gt_tiled[mask]
            ])
            overlay_labels, overlay_labels_counts = np.unique(
                overlay, return_counts=True, axis=1)
            overlay_labels = np.transpose(overlay_labels)
        else:
            overlay = np.array([pred_labels_rel.flatten(),
                                gt_labels_rel.flatten()])
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
        pred_labels_list, pred_counts = np.unique(pred_labels_rel,
                                                  return_counts=True)
        logger.debug("%s %s", pred_labels_list, pred_counts)
        pred_labels_count_dict = {}
        for (l,c) in zip(pred_labels_list, pred_counts):
            pred_labels_count_dict[l] = c

        for (u,v), c in zip(overlay_labels, overlay_labels_counts):
            iou = c / (gt_labels_count_dict[v] + pred_labels_count_dict[u] - c)

            iouMat[v, u] = iou
            recallMat[v, u] = c / gt_labels_count_dict[v]
            precMat[v, u] = c / pred_labels_count_dict[u]
            fscoreMat[v, u] = 2 * (precMat[v, u] * recallMat[v, u]) / \
                                  (precMat[v, u] + recallMat[v, u])
    
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
        if overlapping_inst or len(gt_labels_rel.shape) > len(pred_labels_rel.shape):
            gt_wo_overlap = gt_labels_rel.copy()
            mask = np.sum(gt_labels_rel > 0, axis=0) > 1
            gt_wo_overlap[:, mask] = 0
            pred_wo_overlap = pred_labels_rel.copy()
            if overlapping_inst:
                # todo: check how this should be done with overlapping inst in prediction
                pred_wo_overlap[:, mask] = 0
            else:
                pred_wo_overlap[mask] = 0
            
            #precMat_wo_overlap = get_centerline_overlap(
            #        pred_wo_overlap, gt_wo_overlap,
            #        np.transpose(np.zeros_like(precMat)))
            #precMat_wo_overlap = np.transpose(precMat_wo_overlap)
            recallMat_wo_overlap = get_centerline_overlap(
                    gt_wo_overlap, pred_wo_overlap,
                    np.zeros_like(recallMat))
            #iouMat_wo_overlap = np.nan_to_num(
            #        2 * precMat_wo_overlap * recallMat_wo_overlap / (
            #            precMat_wo_overlap + recallMat_wo_overlap)
            #        ) 
        iouMat = np.nan_to_num(2 * precMat * recallMat / (precMat + recallMat))
    else:
        raise NotImplementedError
    print(iouMat.shape, recallMat.shape, precMat.shape, fscoreMat.shape)
    return iouMat, recallMat, precMat, fscoreMat, recallMat_wo_overlap


def assign_labels(iouMat, assignment_strategy, thresh, num_matches):
    """
    Assigns prediction and gt labels

            Returns:
                    tp (int): number of true positive matches
                    pred_ind (list of ints): pred labels that are matched as true positives (tp)
                    gt_ind (list of ints): gt labels that are matched as tp
    """
    tp_pred_ind = []
    tp_gt_ind = []
    iouFgMat = iouMat[1:, 1:]
    
    # optimal hungarian matching
    if assignment_strategy == "hungarian":
        costs = -(iouFgMat >= thresh).astype(float) - iouFgMat / (2 * num_matches)
        logger.info("start computing lin sum assign for thresh %s",
                    thresh)
        gt_ind, pred_ind = linear_sum_assignment(costs)
        assert num_matches == len(gt_ind) == len(pred_ind)
        match_ok = iouFgMat[gt_ind, pred_ind] >= thresh
        tp = np.count_nonzero(match_ok)

        # get true positive indices
        for idx, match in enumerate(match_ok):
            if match:
                tp_pred_ind.append(pred_ind[idx])
                tp_gt_ind.append(gt_ind[idx])

    # greedy matching by localization criterion
    elif assignment_strategy == "greedy":
        logger.info("start computing greedy assignment for thresh %s, thresh")
        gt_ind, pred_ind = np.nonzero(iouFgMat > thresh) # > 0) if it should be 
        # used before iterating through thresholds
        ious = iouFgMat[gt_ind, pred_ind]
        # sort iou values in descending order
        sort = np.flip(np.argsort(ious))
        gt_ind = gt_ind[sort]
        pred_ind = pred_ind[sort]
        ious = ious[sort]
        
        # assign greedy by iou score
        for gt_idx, pred_idx, iou in zip(gt_ind, pred_ind, ious):
            print(gt_idx, pred_idx, iou)
            if gt_idx not in tp_gt_ind and pred_idx not in tp_pred_ind:
                tp_gt_ind.append(gt_idx)
                tp_pred_ind.append(pred_idx)
        tp = len(tp_pred_ind)
    
    # todo: merge overlap_0_5 here
    #elif assignment_strategy == "overlap_0_5":
    else:
        raise NotImplementedError(
                "assignment strategy %s is not implemented yet", 
                assignment_strategy)
    
    # correct indices to include background
    tp_pred_ind = np.array(tp_pred_ind) + 1
    tp_gt_ind = np.array(tp_gt_ind) + 1
    
    return tp, tp_pred_ind, tp_gt_ind


def get_false_labels(tp_pred_ind, tp_gt_ind, num_pred_labels, num_gt_labels,
        iouMat, precMat, recallMat, thresh, 
        overlapping_inst, unique_false_labels, recallMat_wo_overlap):
    
    # get false positive indices 
    pred_ind_all = np.arange(1, num_pred_labels + 1)
    pred_ind_unassigned = pred_ind_all[np.isin(
        pred_ind_all, tp_pred_ind, invert=True)]
    fp_ind_only_bg = pred_ind_unassigned[np.argmax(
        precMat[:, pred_ind_unassigned], axis=0) == 0]
    if unique_false_labels == False:
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
        iou_mask = np.logical_and(
                recallMat[1:, 1:] > thresh, recallMat_wo_overlap[1:, 1:] > thresh)
    else:
        iou_mask = iouMat[1:, 1:] > thresh
    fm_pred_count = np.maximum(0, np.sum(iou_mask, axis=0) - 1)
    fm_count = np.sum(fm_pred_count)
    # we need fm_pred_ind and fm_gt_ind for visualization later on
    # correct indices to include background
    fm_pred_ind = np.nonzero(fm_pred_count)[0]
    fm_gt_ind = []
    for i in fm_pred_ind:
        fm_gt_ind.append(np.nonzero(iou_mask[:, i])[0] + 1)
    fm_pred_ind = np.array(fm_pred_ind) + 1
    logger.debug("false merge indices (pred/gt/cnt): %s, %s, %i", 
            fm_pred_ind, fm_gt_ind, fm_count)

    return fp_ind, fn_ind, fs_ind, fm_pred_ind, fm_gt_ind, fm_count, fp_ind_only_bg


# todo: should pixelwise neuron evaluation also be possible?
# keep_gt_shape not in pixelwise overlap so far
def evaluate_volume(gt_labels, pred_labels, outFn,
        localization_criterion="iou", 
        assignment_strategy="hungarian",
        evaluate_false_labels=False,
        unique_false_labels=False,
        add_general_metrics=[],
        add_multi_thresh_metrics=[],
        visualize=False,
        visualize_type="nuclei",
        overlapping_inst=False,  
        partly=False
        ):
    
    # if partly is set, then also set evaluate_false_labels to get false splits
    if partly:
        evaluate_false_labels = True
    
    # relabel labels sequentially
    pred_labels_rel, _, _ = relabel_sequential(pred_labels.astype(np.int))
    gt_labels_rel, _, _ = relabel_sequential(gt_labels)

    print("check for overlap: ", np.sum(np.sum(gt_labels_rel > 0, axis=0) > 1))

    # get number of labels
    num_pred_labels = int(np.max(pred_labels_rel))
    num_gt_labels = int(np.max(gt_labels_rel))
    num_matches = min(num_gt_labels, num_pred_labels)
    
    # get localization criterion
    iouMat, recallMat, precMat, fscoreMat, recallMat_wo_overlap = \
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
        if num_matches > 0 and np.max(iouMat) > th:
            tp, pred_ind, gt_ind = assign_labels(
                    iouMat, assignment_strategy, th, num_matches)
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
                            iouMat, precMat, recallMat, th, 
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
            fscore = (2. * precision * recall) / max(1, precision + recall)
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

        # add additional multi-threshold metrics
        # todo: move to separate def
        if len(add_multi_thresh_metrics) > 0:
            if "avg_tp_skel_coverage" in add_multi_thresh_metrics:
                tp_cov = []
                if th == 0.5:
                    if tp > 0:
                        max_gt_ind = np.argmax(precMat, axis=0)
                        tp_cov = []
                        for i in gt_ind:
                            tp_cov.append(np.sum(recallMat[i, max_gt_ind==i]))
                        tp_skel_coverage = np.mean(tp_cov)
                        #tp_skel_coverage = np.mean(np.sum(recallMat[gt_ind, 1:], axis=1))
                    else:
                        tp_skel_coverage = 0
                    metrics.addMetric(tblname, "tp_skel_coverage", tp_cov)
                    metrics.addMetric(tblname, "avg_tp_skel_coverage", tp_skel_coverage)

        # visualize tp and false labels
        if visualize and th == 0.5:
            if visualize_type == "nuclei" and tp > 0:
                visualize_nuclei(gt_labels_rel, iouMat, gt_ind, pred_ind)
            elif visualize_type == "neuron" and localization_criterion == "cldice":
                visualize_neuron(
                        gt_labels_rel, pred_labels_rel, gt_ind, pred_ind,
                        outFn, fs_ind, fp_ind, fn_ind, fm_pred_ind, fm_gt_ind, fp_ind_only_bg)
            else:
                raise NotImplementedError
    
    # save multi-threshold metrics to dict
    avAP19 = np.mean(aps)   # todo: check with Peter, should also include 0.05s?
    avAP59 = np.mean(aps[4:])
    metrics.addMetric("confusion_matrix", "avAP", avAP59)
    metrics.addMetric("confusion_matrix", "avAP59", avAP59)
    metrics.addMetric("confusion_matrix", "avAP19", avAP19)
    avFscore19 = np.mean(fscores[:-5])
    avFscore59 = np.mean(fscores[4:])
    metrics.addMetric("confusion_matrix", "avFscore", avFscore19)
    metrics.addMetric("confusion_matrix", "avFscore59", avFscore59)

    # additional metrics
    if len(add_general_metrics) > 0:
        if "avg_gt_skel_coverage" in add_general_metrics:
            # get gt coverage 
            # only take max gt label for each pred label to not count pred labels 
            # twice for overlapping gt instances
            # todo: recalculate clRecall for each gt and union of assigned predictions, 
            # because predictions could potentially overlap (not in our predictions so far)
            max_gt_ind = np.argmax(precMat, axis=0)
            print("max_gt_ind: ", max_gt_ind)
            gt_cov = []
            for i in range(1, recallMat.shape[0]):
                gt_cov.append(np.sum(recallMat[i, max_gt_ind==i]))
            print("gt cov: ", gt_cov)
            #gt_skel_coverage = np.sum(recallMat[1:, 1:], axis=1)
            gt_skel_coverage = np.mean(gt_cov)
            metrics.addMetric(tblNameGen, "gt_skel_coverage", gt_cov)
            metrics.addMetric(tblNameGen, "avg_gt_skel_coverage", gt_skel_coverage)
        if "avg_f1_cov_score" in add_general_metrics:
            avg_f1_cov_score = 0.5 * avFscore19 + 0.5 * gt_skel_coverage
            metrics.addMetric(tblNameGen, "avg_f1_cov_score", avg_f1_cov_score)

    metrics.save()
    return metrics.metricsDict


def set_boundary(labels_rel, label, target):
    coords_z, coords_y, coords_x = np.nonzero(labels_rel == label)
    coords = {}
    for z,y,x in zip(coords_z, coords_y, coords_x):
        coords.setdefault(z, []).append((z, y, x))
    max_z = -1
    max_z_len = -1
    for z, v in coords.items():
        if len(v) > max_z_len:
            max_z_len = len(v)
            max_z = z
    tmp = np.zeros_like(labels_rel[max_z], dtype=np.float32)
    tmp = labels_rel[max_z]==label
    struct = scipy.ndimage.generate_binary_structure(2, 2)
    eroded_tmp = scipy.ndimage.binary_erosion(
        tmp,
        iterations=1,
        structure=struct,
        border_value=1)
    bnd = np.logical_xor(tmp, eroded_tmp)
    target[max_z][bnd] = 1


def replace(array, old_values, new_values):
    values_map = np.arange(int(array.max() + 1), dtype=new_values.dtype)
    values_map[old_values] = new_values

    return values_map[array]


def filter_components(volume, thresh):

    labels, counts = np.unique(volume, return_counts=True)
    small_labels = labels[counts <= thresh]

    volume = replace(
        volume,
        np.array(small_labels),
        np.array([0] * len(small_labels))
    )
    return volume


def get_centerline_overlap(skeletonize, compare, match):
    skeleton_one_inst_per_channel = True if skeletonize.ndim == 4 else False
    compare_one_inst_per_channel = True if compare.ndim == 4 else False
    
    if compare_one_inst_per_channel:
        fg = np.max(compare > 0, axis=0).astype(np.uint8)

    labels, labels_cnt = np.unique(
        skeletonize[skeletonize > 0],
        return_counts=True
    )
    
    for label, label_count in zip(labels, labels_cnt):
        logger.debug("compute centerline overlap for %i", label)
        if skeleton_one_inst_per_channel:
            #idx = np.unravel_index(np.argmax(mask), mask.shape)[0]
            mask = skeletonize[label - 1] #heads up: assuming increasing labels
        else:
            mask = skeletonize == label
        skeleton = skeletonize_3d(mask) > 0
        skeleton_size = np.sum(skeleton)

        # if one instance per channel for compare, we need to correct bg label
        if compare_one_inst_per_channel:
            compare_fg, compare_fg_cnt = np.unique(
                    fg[skeleton], return_counts=True)
            if np.any(compare_fg > 0):
                compare_label, compare_label_cnt = np.unique(
                    compare[:, skeleton], return_counts=True)
                if np.any(compare_label == 0):
                    compare_label[0] = compare_fg[0]
                    compare_label_cnt[0] = compare_fg_cnt[0]
            else:
                compare_label = compare_fg
                compare_label_cnt = compare_fg_cnt
        else:
            compare_label, compare_label_cnt = np.unique(
                compare[skeleton], return_counts=True)
        ratio = compare_label_cnt / float(skeleton_size)
        match[label, compare_label] = ratio

    return match


# todo: this is probably not working anymore, add parameters and test
def visualize_nuclei(gt_labels_rel, iouMat, gt_ind, pred_ind):
    vis_tp = np.zeros_like(gt_labels_rel, dtype=np.float32)
    vis_fp = np.zeros_like(gt_labels_rel, dtype=np.float32)
    vis_fn = np.zeros_like(gt_labels_rel, dtype=np.float32)
    vis_tp_seg = np.zeros_like(gt_labels_rel, dtype=np.float32)
    vis_tp_seg2 = np.zeros_like(gt_labels_rel, dtype=np.float32)
    vis_fp_seg = np.zeros_like(gt_labels_rel, dtype=np.float32)
    vis_fn_seg = np.zeros_like(gt_labels_rel, dtype=np.float32)
    if len(gt_labels_rel.shape) == 3:
        vis_fp_seg_bnd = np.zeros_like(gt_labels_rel, dtype=np.float32)
        vis_fn_seg_bnd = np.zeros_like(gt_labels_rel, dtype=np.float32)

    cntrs_gt = scipy.ndimage.measurements.center_of_mass(
        gt_labels_rel > 0,
        gt_labels_rel, sorted(list(np.unique(gt_labels_rel)))[1:])
    cntrs_pred = scipy.ndimage.measurements.center_of_mass(
        pred_labels_rel > 0,
        pred_labels_rel, sorted(list(np.unique(pred_labels_rel)))[1:])
    sz = 1
    for gti, pi, in zip(gt_ind, pred_ind):
        if iouMat[gti, pi] < th:
            vis_fn_seg[gt_labels_rel == gti+1] = 1
            if len(gt_labels_rel.shape) == 3:
                set_boundary(gt_labels_rel, gti+1,
                             vis_fn_seg_bnd)
            vis_fp_seg[pred_labels_rel == pi+1] = 1
            if len(gt_labels_rel.shape) == 3:
                set_boundary(pred_labels_rel, pi+1,
                             vis_fp_seg_bnd)
            cntr = cntrs_gt[gti]
            if len(gt_labels_rel.shape) == 3:
                vis_fn[int(cntr[0]), int(cntr[1]), int(cntr[2])] = 1
            else:
                vis_fn[int(cntr[0]), int(cntr[1])] = 1
            cntr = cntrs_pred[pi]
            if len(gt_labels_rel.shape) == 3:
                vis_fp[int(cntr[0]), int(cntr[1]), int(cntr[2])] = 1
            else:
                vis_fp[int(cntr[0]), int(cntr[1])] = 1
        else:
            vis_tp_seg[gt_labels_rel == gti+1] = 1
            cntr = cntrs_gt[gti]
            if len(gt_labels_rel.shape) == 3:
                vis_tp[int(cntr[0]), int(cntr[1]), int(cntr[2])] = 1
            else:
                vis_tp[int(cntr[0]), int(cntr[1])] = 1
            vis_tp_seg2[pred_labels_rel == pi+1] = 1
    vis_tp = scipy.ndimage.gaussian_filter(vis_tp, sz, truncate=sz)
    for gti in range(num_gt_labels):
        if gti in gt_ind:
            continue
        vis_fn_seg[gt_labels_rel == gti+1] = 1
        if len(gt_labels_rel.shape) == 3:
            set_boundary(gt_labels_rel, gti+1,
                         vis_fn_seg_bnd)
        cntr = cntrs_gt[gti]
        if len(gt_labels_rel.shape) == 3:
            vis_fn[int(cntr[0]), int(cntr[1]), int(cntr[2])] = 1
        else:
            vis_fn[int(cntr[0]), int(cntr[1])] = 1
    vis_fn = scipy.ndimage.gaussian_filter(vis_fn, sz, truncate=sz)
    for pi in range(num_pred_labels):
        if pi in pred_ind:
            continue
        vis_fp_seg[pred_labels_rel == pi+1] = 1
        if len(gt_labels_rel.shape) == 3:
            set_boundary(pred_labels_rel, pi+1,
                         vis_fp_seg_bnd)
        cntr = cntrs_pred[pi]
        if len(gt_labels_rel.shape) == 3:
            vis_fp[int(cntr[0]), int(cntr[1]), int(cntr[2])] = 1
        else:
            vis_fp[int(cntr[0]), int(cntr[1])] = 1
    vis_fp = scipy.ndimage.gaussian_filter(vis_fp, sz, truncate=sz)
    vis_tp = vis_tp/np.max(vis_tp)
    vis_fp = vis_fp/np.max(vis_fp)
    vis_fn = vis_fn/np.max(vis_fn)
    with h5py.File(outFn + "_vis.hdf", 'w') as fi:
        fi.create_dataset(
            'volumes/vis_tp',
            data=vis_tp,
            compression='gzip')
        fi.create_dataset(
            'volumes/vis_fp',
            data=vis_fp,
            compression='gzip')
        fi.create_dataset(
            'volumes/vis_fn',
            data=vis_fn,
            compression='gzip')
        fi.create_dataset(
            'volumes/vis_tp_seg',
            data=vis_tp_seg,
            compression='gzip')
        fi.create_dataset(
            'volumes/vis_tp_seg2',
            data=vis_tp_seg2,
            compression='gzip')
        fi.create_dataset(
            'volumes/vis_fp_seg',
            data=vis_fp_seg,
            compression='gzip')
        fi.create_dataset(
            'volumes/vis_fn_seg',
            data=vis_fn_seg,
            compression='gzip')
        if len(gt_labels_rel.shape) == 3:
            fi.create_dataset(
                'volumes/vis_fp_seg_bnd',
                data=vis_fp_seg_bnd,
                compression='gzip')
            fi.create_dataset(
                'volumes/vis_fn_seg_bnd',
                data=vis_fn_seg_bnd,
                compression='gzip')


def rgb(idx, cmap):
    return (np.array(to_rgb(cmap[idx % len(cmap)])) * 255).astype(np.uint8)


def proj_label(lbl):
    dst = np.zeros(lbl.shape[1:], dtype=np.uint8)
    for i in range(lbl.shape[0]-1, -1, -1):
        dst[np.max(dst, axis=-1)==0,:] = lbl[i][np.max(dst, axis=-1)==0,:]

    return dst

def visualize_neuron(gt_labels_rel, pred_labels_rel, gt_ind, pred_ind, outFn, 
        fs_ind, fp_ind, fn_ind, fm_pred_ind, fm_gt_ind, fp_ind_only_bg):
    if len(gt_labels_rel.shape) == 4:
        gt = np.max(gt_labels_rel, axis=0)
    else:
        gt = gt_labels_rel
    if len(pred_labels_rel.shape) == 4:
        pred = np.max(pred_labels_rel, axis=0)
    else:
        pred = pred_labels_rel
    print(pred_ind, gt_ind)
    num_gt = np.max(gt_labels_rel)
    num_pred = np.max(pred_labels_rel)
    #gray_gt_cmap = (np.arange(num_gt + 1) / float(num_gt) * 255).astype(np.uint8)
    #gray_pred_cmap = (np.arange(num_pred + 1) / float(num_pred) * 255).astype(np.uint8)
    dst = np.zeros_like(gt, dtype=np.uint8)
    dst = np.stack([dst, dst, dst], axis=-1)

    # visualize gt
    vis = np.zeros_like(dst)
    for i in range(1, num_gt + 1):
        vis[gt == i] = rgb(i-1, gt_cmap) 
    mip = proj_label(vis)
    mip_gt_mask = np.max(mip>0, axis=-1)
    io.imsave(
        outFn + '_gt.png',
        mip.astype(np.uint8)
    )
    
    # visualize pred
    vis = np.zeros_like(dst)
    for i in range(1, num_pred + 1):
        vis[pred == i] = rgb(i-1, pred_cmap)
    mip = proj_label(vis)
    mip_pred_mask = np.max(mip>0, axis=-1)
    io.imsave(
        outFn + '_pred.png',
        mip.astype(np.uint8)
    )
    
    # visualize tp pred + fp + fs
    vis = np.zeros_like(dst)
    for i in pred_ind:
        vis[pred == i] = rgb(i-1, pred_cmap)
    for i in fp_ind:
        vis[pred == i] = [255, 0, 0]
    mip = proj_label(vis)
    mask = np.logical_and(mip_gt_mask, np.logical_not(np.max(mip > 0, axis=-1)))
    mip[mask] = [200, 200, 200]
    io.imsave(
        outFn + '_tp_pred_fp_fs.png',
        mip.astype(np.uint8)
    )

    # visualize false negative in color, false merger in red
    vis = np.zeros_like(dst, dtype=np.uint8)
    fm_merged = np.unique(np.array(
        [ind for inds in fm_gt_ind for ind in inds]).flatten())
    #for i in range(1, num_pred + 1):
    #    vis[pred == i] = [gray_pred_cmap[i],] * 3
    for i in fn_ind:
        if i not in fm_merged:
            vis[gt == i] = rgb(i-1, gt_cmap)
    for i in fm_merged:
        if i not in gt_ind:
            vis[gt == i] = [255, 0, 0]
    mip = proj_label(vis)
    mask = np.logical_and(mip_pred_mask, np.logical_not(np.max(mip > 0, axis=-1)))
    mip[mask] = [200, 200, 200]
    io.imsave(
        outFn + '_fn_fm.png',
        mip.astype(np.uint8)
    )

    # version 2 of gt errors
    # visualize false negative in color, false merger in red
    vis = np.zeros_like(dst, dtype=np.uint8)
    #fm_merged = np.unique(np.array(fm_gt_ind).flatten())
    #for i in range(1, num_pred + 1):
    #    vis[pred == i] = [gray_pred_cmap[i],] * 3
    for i in gt_ind:
        vis[gt == i] = rgb(i-1, gt_cmap)
    for i in fn_ind:
        if i not in fm_merged:
            vis[gt == i] = [255, 64, 64]
    for i in fm_merged:
       if i not in gt_ind:
           vis[gt == i] = [192, 0, 0]
    mip = proj_label(vis)
    mask = np.logical_and(mip_pred_mask, np.logical_not(np.max(mip > 0, axis=-1)))
    mip[mask] = [200, 200, 200]
    io.imsave(
        outFn + '_fn_fm_v2.png',
        mip.astype(np.uint8)
    )
