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
#from pylab import cm
import pdb

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

def evaluate_file(res_file, gt_file, background=0,
                  foreground_only=False, **kwargs):
    # TODO: maybe remove, should be superfluous with proper arg parsing
    if 'res_file_suffix' in kwargs:
        res_file = res_file.replace(".hdf", kwargs['res_file_suffix'] + ".hdf")
    logger.info("loading %s %s", res_file, gt_file)

    # read preprocessed hdf file
    if res_file.endswith(".hdf"):
        with h5py.File(res_file, 'r') as f:
            pred_labels= np.array(f[kwargs['res_key']])
    # or read preprocessed tif files
    elif res_file.endswith(".tif") or res_file.endswith(".tiff") or \
         res_file.endswith(".TIF") or res_file.endswith(".TIFF"):
        pred_labels = tifffile.imread(res_file)
    else:
        raise NotImplementedError("invalid file format %s", res_file)
    logger.debug("prediction min %f, max %f, shape %s", np.min(pred_labels),
                 np.max(pred_labels), pred_labels.shape)
    pred_labels = np.squeeze(pred_labels)
    logger.debug("prediction shape %s", pred_labels.shape)

    # read ground truth data
    if gt_file.endswith(".hdf"):
        with h5py.File(gt_file, 'r') as f:
            gt_labels = np.array(f[kwargs['gt_key']])
    elif gt_file.endswith(".tif") or gt_file.endswith(".tiff") or \
         gt_file.endswith(".TIF") or gt_file.endswith(".TIFF"):
        gt_labels = tifffile.imread(gt_file)
    elif gt_file.endswith(".zarr"):
        gt = zarr.open(gt_file, 'r')
        gt_labels = np.array(gt[kwargs['gt_key']])
    else:
        raise NotImplementedError("invalid file format %s", gt_file)
    logger.debug("gt min %f, max %f, shape %s", np.min(gt_labels),
                 np.max(gt_labels), gt_labels.shape)
    
    # check: what if there are only one gt instance in overlapping_inst scenario?
    keep_gt_shape = kwargs.get("keep_gt_shape", False)
    if gt_labels.shape[0] == 1 and not keep_gt_shape:
        gt_labels.shape = gt_labels.shape[1:]
    gt_labels = np.squeeze(gt_labels)
    if gt_labels.ndim > pred_labels.ndim and not keep_gt_shape:
        gt_labels = np.max(gt_labels, axis=0)
    # check if pred.dim < gt.dim
    # if pred_labels.ndim < gt_labels.ndim:
    #     print("WARNING: changing prediction to one instance per channel")
    # lbls = np.unique(pred_labels)
    # pred_1instpch = np.zeros((np.sum(lbls > 0),
    #                                           ) + pred_labels.shape,
    #                                          dtype=pred_labels.dtype)
    # i = 0
    # for lbl in lbls:
    #     if lbl == 0:
    #         continue
    #     pred_1instpch[i][pred_labels == lbl] = i + 1
    #     print("check: ", i, lbl, np.sum(pred_1instpch == i + 1),
    #           np.sum(pred_labels == lbl))
    #     i += 1

    # pred_labels = pred_1instpch
    logger.debug("gt shape %s", gt_labels.shape)

    # heads up: should not crop channel dimensions, assuming channels first
    overlapping_inst = kwargs.get('overlapping_inst', False)
    if not keep_gt_shape:
        pred_labels, gt_labels = maybe_crop(pred_labels, gt_labels,
                                            overlapping_inst)
    else:
        # todo: take out duplicate last frame, should be done beforehand
        print("debug: ", pred_labels.ndim, gt_labels.ndim, pred_labels.shape, gt_labels.shape)
        if pred_labels.shape[0] < gt_labels.shape[1]:
            print("duplicate last frame")
            print(pred_labels.shape)
            pred_labels = np.concatenate(
                    [pred_labels,
                        np.reshape(pred_labels[-1], 
                            (1, pred_labels.shape[1], pred_labels.shape[2]))])
            print(pred_labels.shape)

        if pred_labels.ndim < gt_labels.ndim:
            pred_labels = np.expand_dims(pred_labels, axis=0)
            pred_labels, gt_labels = maybe_crop(pred_labels, gt_labels, True)
            pred_labels = np.squeeze(pred_labels)
    logger.debug("prediction %s, shape %s", np.unique(pred_labels),
                 pred_labels.shape)
    logger.debug("gt %s, shape %s", np.unique(gt_labels),
                 gt_labels.shape)

    rm_sm_comp = kwargs.get("remove_small_components", None) 
    if rm_sm_comp is not None and rm_sm_comp > 0:
        pred_labels = remove_small_components(pred_labels, rm_sm_comp)
        logger.debug("prediction %s, shape %s", np.unique(pred_labels),
                     pred_labels.shape)

    # if pred_labels.shape[0] == 536:
    #     print(pred_labels.shape, gt_labels.shape)
    #     pred_labels = pred_labels[12:-12, 12:-12]
    #     gt_labels = gt_labels[12:-12, 12:-12]
    #     print(pred_labels.shape, gt_labels.shape)

    if foreground_only:
        try:
            pred_labels[gt_labels==0] = 0
        except IndexError:
            pred_labels[:, np.any(gt_labels, axis=0).astype(np.int)==0] = 0

    logger.info("processing %s %s", res_file, gt_file)
    outFnBase = os.path.join(
        kwargs['out_dir'],
        '' if kwargs.get('filterSz', None) is None else str(kwargs['filterSz']),
        os.path.splitext(os.path.basename(res_file))[0] +
        kwargs['res_key'].replace("/","_") + kwargs['suffix'])
    if kwargs.get('use_linear_sum_assignment'):
        outFnBase += "_linear"
    else:
        outFnBase += "_seg"
    if res_file.endswith(".hdf"):
        outFn = outFnBase + "_hdf_scores"
    else:
        outFn = outFnBase + "_tif_scores"
    os.makedirs(os.path.dirname(outFnBase), exist_ok=True)

    if not kwargs.get("from_scratch") and \
       len(glob.glob(outFnBase + "*.toml")) > 0:
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

    # relabel gt labels in case of binary mask per channel
    if overlapping_inst and np.max(gt_labels) == 1:
        for i in range(gt_labels.shape[0]):
            gt_labels[i] = gt_labels[i] * (i + 1)

    if kwargs.get('use_linear_sum_assignment'):
        return evaluate_linear_sum_assignment(
                gt_labels, pred_labels, outFn,
                overlapping_inst,
                kwargs.get('filterSz', None),
                visualize=kwargs.get("visualize", False),
                localization_criterion=kwargs.get("localization_criterion", "iou"),
                partly=kwargs.get("partly", False),
                visualize_type=kwargs.get("visualize_type", "nuclei"),
                greedy_by_score=kwargs.get("greedy_by_score", False)
                )

    # get gt cell ids and the size of the corresponding cell
    gt_labels_list, gt_counts = np.unique(gt_labels, return_counts=True)
    gt_labels_count_dict = {}
    logger.debug("%s %s", gt_labels_list, gt_counts)
    for (l, c) in zip(gt_labels_list, gt_counts):
        gt_labels_count_dict[l] = c

    # get pred cell ids
    pred_labels_list, pred_counts = np.unique(pred_labels,
                                              return_counts=True)
    logger.debug("%s %s", pred_labels_list, pred_counts)
    pred_labels_count_dict = {}
    for (l, c) in zip(pred_labels_list, pred_counts):
        pred_labels_count_dict[l] = c

    # get overlapping labels
    if overlapping_inst:
        pred_tile = [1,] * pred_labels.ndim
        pred_tile[0] = gt_labels.shape[0]
        gt_tile = [1,] * gt_labels.ndim
        gt_tile[1] = pred_labels.shape[0]
        pred_tiled = np.tile(pred_labels, pred_tile).flatten()
        gt_tiled = np.tile(gt_labels, gt_tile).flatten()
        mask = np.logical_or(pred_tiled > 0, gt_tiled > 0)
        overlay = np.array([
            pred_tiled[mask],
            gt_tiled[mask]
        ])
        overlay_labels, overlay_labels_counts = np.unique(
            overlay, return_counts=True, axis=1)
        overlay_labels = np.transpose(overlay_labels)
    else:
        overlay = np.array([pred_labels.flatten(),
                            gt_labels.flatten()])
        logger.debug("overlay shape %s", overlay.shape)
        # get overlaying cells and the size of the overlap
        overlay_labels, overlay_labels_counts = np.unique(overlay,
                                             return_counts=True, axis=1)
        overlay_labels = np.transpose(overlay_labels)

    # identify overlaying cells where more than 50% of gt cell is covered
    matchesSEG = np.asarray([c > 0.5 * float(gt_counts[gt_labels_list == v])
        for (u,v), c in zip(overlay_labels, overlay_labels_counts)],
                            dtype=np.bool)

    # get their ids
    matches_labels = overlay_labels[matchesSEG]

    # remove background
    if background is not None:
        pred_labels_list = pred_labels_list[pred_labels_list != background]
        gt_labels_list = gt_labels_list[gt_labels_list != background]

    matches_mat = np.zeros((len(pred_labels_list), len(gt_labels_list)))
    for (u, v) in matches_labels:
        if u > 0 and v > 0:
            matches_mat[np.where(pred_labels_list == u),
                        np.where(gt_labels_list == v)] = 1

    diceGT = {}
    iouGT = {}
    segGT = {}
    diceP = {}
    iouP = {}
    segP = {}
    segPrev = {}
    for (u,v), c in zip(overlay_labels, overlay_labels_counts):
        dice = 2.0 * c / (gt_labels_count_dict[v] + pred_labels_count_dict[u])
        iou = c / (gt_labels_count_dict[v] + pred_labels_count_dict[u] - c)

        if c > 0.5 * gt_labels_count_dict[v]:
            seg = iou
        else:
            seg = 0
        if c > 0.5 * pred_labels_count_dict[u]:
            seg2 = iou
        else:
            seg2 = 0

        if v not in diceGT:
            diceGT[v] = []
            iouGT[v] = []
            segGT[v] = []
        if u not in diceP:
            diceP[u] = []
            iouP[u] = []
            segP[u] = []
            segPrev[u] = []
        diceGT[v].append(dice)
        iouGT[v].append(iou)
        segGT[v].append(seg)
        diceP[u].append(dice)
        iouP[u].append(iou)
        segP[u].append(seg)
        segPrev[u].append(seg2)

    if background is not None:
        iouP.pop(background)
        iouGT.pop(background)
        diceP.pop(background)
        diceGT.pop(background)
        segP.pop(background)
        segPrev.pop(background)
        segGT.pop(background)

    if not diceGT:
        logger.error("%s: No labels found in gt image", gt_file)
        return
    if not diceP:
        logger.error("%s: No labels found in pred image", res_file)
        return

    dice = 0
    cnt = 0
    for (k, vs) in diceGT.items():
        vs = sorted(vs, reverse=True)
        dice += vs[0]
        cnt += 1
    diceGT = dice/max(1, cnt)

    dice = 0
    cnt = 0
    for (k, vs) in diceP.items():
        vs = sorted(vs, reverse=True)
        dice += vs[0]
        cnt += 1
    diceP = dice/max(1, cnt)

    iou = []
    instances = gt_labels.copy().astype(np.float32)
    for (k, vs) in iouGT.items():
        vs = sorted(vs, reverse=True)
        iou.append(vs[0])
        instances[instances==k] = vs[0]
    iouGT = np.array(iou)
    iouGTMn = np.mean(iouGT)
    if kwargs['debug']:
        with h5py.File(outFnBase + "GT.hdf", 'w') as fi:
            fi.create_dataset(
                'images/instances',
                data = instances,
                compression='gzip')
            for dataset in ['images/instances']:
                fi[dataset].attrs['offset'] = (0, 0)
                fi[dataset].attrs['resolution'] = (1, 1)

    iou = []
    iouIDs = []
    instances = pred_labels.copy().astype(np.float32)
    for (k, vs) in iouP.items():
        vs = sorted(vs, reverse=True)
        iou.append(vs[0])
        iouP[k] = vs
        iouIDs.append(k)
        instances[instances==k] = vs[0]
    if kwargs['debug']:
        with h5py.File(outFnBase + "P.hdf", 'w') as fi:
            fi.create_dataset(
                'images/instances',
                data = instances,
                compression='gzip')
            for dataset in ['images/instances']:
                fi[dataset].attrs['offset'] = (0, 0)
                fi[dataset].attrs['resolution'] = (1, 1)

    iouP_2 = np.array(iou)
    iouIDs = np.array(iouIDs)
    iouPMn = np.mean(iouP_2)

    seg = 0
    cnt = 0
    for (k, vs) in segGT.items():
        vs = sorted(vs, reverse=True)
        seg += vs[0]
        cnt += 1
    segGT = seg/max(1, cnt)

    seg = 0
    cnt = 0
    for (k, vs) in segP.items():
        vs = sorted(vs, reverse=True)
        seg += vs[0]
        cnt += 1
    segP = seg/max(1, cnt)

    seg = 0
    cnt = 0
    for (k, vs) in segPrev.items():
        vs = sorted(vs, reverse=True)
        seg += vs[0]
        cnt += 1
    segPrev = seg/max(1, cnt)

    # non-split vertices num non-empty cols - num non-empty rows
    # (more than one entry in col: predicted cell with more than one
    # ground truth cell assigned)
    # (other way around not possible due to 50% rule)
    ns = np.sum(np.count_nonzero(matches_mat, axis=0)) \
            - np.sum(np.count_nonzero(matches_mat, axis=1) > 0)
    ns = int(ns)

    # false negative: empty cols
    # (no predicted cell for ground truth cell)
    fn = np.sum(np.sum(matches_mat, axis=0) == 0)
    # tmp = np.sum(matches_mat, axis=0)==0
    # for i in range(len(tmp)):
    #     print(i, tmp[i], gt_labels_list[i])
    fn = int(fn)

    # false positive: empty rows
    # (predicted cell for non existing ground truth cell)
    fp = np.sum(np.sum(matches_mat, axis=1) == 0)
    # tmp = np.sum(matches_mat, axis=1)==0
    # for i in range(len(tmp)):
    #     print(i, tmp[i], pred_labels_list[i])
    # print(np.sum(matches_mat, axis=1)==0)
    fp = int(fp)

    # true positive: row with single entry (can be 0, 1, or more)
    tpP = np.sum(np.sum(matches_mat, axis=1) == 1)
    tpP = int(tpP)

    # true positive: non-empty col (can only be 0 or 1)
    tpGT = np.sum(np.sum(matches_mat, axis=0) > 0)
    tpGT = int(tpGT)


    metrics = Metrics(outFn)
    tblNameGen = "general"
    metrics.addTable(tblNameGen)
    metrics.addMetric(tblNameGen, "Num GT", len(gt_labels_list))
    metrics.addMetric(tblNameGen, "Num Pred", len(pred_labels_list))
    metrics.addMetric(tblNameGen, "GT/Ref -> Pred mean dice", diceGT)
    metrics.addMetric(tblNameGen, "Pred -> GT/Ref mean dice", diceP)
    metrics.addMetric(tblNameGen, "GT/Ref -> Pred mean iou", iouGTMn)
    metrics.addMetric(tblNameGen, "Pred -> GT/Ref mean iou", iouPMn)
    metrics.addMetric(tblNameGen, "GT/Ref -> Pred mean seg", segGT)
    metrics.addMetric(tblNameGen, "Pred -> GT/Ref mean seg", segP)
    metrics.addMetric(tblNameGen, "Pred -> GT/Ref mean seg rev", segPrev)
    metrics.addMetric(tblNameGen, "Pred -> GT/Ref NS", ns)
    metrics.addMetric(tblNameGen, "Pred -> GT/Ref FP", fp)
    metrics.addMetric(tblNameGen, "Pred -> GT/Ref TP", tpP)
    metrics.addMetric(tblNameGen, "GT/Ref -> Pred FN", fn)
    metrics.addMetric(tblNameGen, "GT/Ref -> Pred TP", tpGT)

    ths = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    aps = []
    metrics.addTable("confusion_matrix")
    for th in ths:
        tblname = "confusion_matrix.th_"+str(th).replace(".","_")
        metrics.addTable(tblname)
        apTP = 0
        for pID in np.nonzero(iouP_2 > th)[0]:
            if len(iouP[iouIDs[pID]]) == 0:
                pass
            elif len(iouP[iouIDs[pID]]) == 1:
                apTP += 1
            elif len(iouP[iouIDs[pID]]) > 1 and iouP[iouIDs[pID]][1] < th:
                apTP += 1
        metrics.addMetric(tblname, "AP_TP", apTP)
        apTP = np.count_nonzero(iouP_2[iouP_2>th])
        apFP = np.count_nonzero(iouP_2[iouP_2<=th])
        apFN = np.count_nonzero(iouGT[iouGT<=th])
        metrics.addMetric(tblname, "AP_TP", apTP)
        metrics.addMetric(tblname, "AP_FP", apFP)
        metrics.addMetric(tblname, "AP_FN", apFN)
        p = 1.*(apTP) / max(1, apTP +  apFP)
        rec = 1.*(apTP) / max(1, apTP +  apFN)
        aps.append(p*rec)
        metrics.addMetric(tblname, "AP", p*rec)

        precision = 1.*(apTP) / max(1, len(pred_labels_list))
        metrics.addMetric(tblname, "precision", precision)
        recall = 1.*(apTP) / max(1, len(gt_labels_list))
        metrics.addMetric(tblname, "recall", recall)
        if (precision + recall) > 0:
            fscore = (2. * precision * recall) / max(1, precision + recall)
        else:
            fscore = 0.0
        metrics.addMetric(tblname, 'fscore', fscore)

    avAP = np.mean(aps)
    metrics.addMetric("confusion_matrix", "avAP", avAP)

    metrics.save()
    return metrics.metricsDict


# todo: rename def, evaluate_volume( ?
# todo: should pixelwise neuron evaluation also be possible? restructure code accordingly
def evaluate_linear_sum_assignment(gt_labels, pred_labels, outFn,
                                   overlapping_inst=False, filterSz=None,
                                   visualize=False, localization_criterion="iou",
                                   partly=False, visualize_type="nuclei",
                                   greedy_by_score=False):
    if filterSz is not None:
        ls, cs = np.unique(pred_labels, return_counts=True)
        pred_labels2 = np.copy(pred_labels)
        print(sorted(zip(cs, ls)))
        for l, c in zip(ls, cs):
            if c < filterSz:
                pred_labels[pred_labels==l] = 0
            # else:
            #     pred_labels2[pred_labels==l] = 0
        print(outFn)
        # with h5py.File(outFn + ".hdf", 'w') as f:
        #     f.create_dataset(
        #         'volumes/small_inst',
        #         data=pred_labels2,
        #         compression='gzip')
    
    # relabel labels sequentially
    pred_labels_rel, _, _ = relabel_sequential(pred_labels.astype(np.int))
    gt_labels_rel, _, _ = relabel_sequential(gt_labels)

    # get number of labels
    num_pred_labels = int(np.max(pred_labels_rel))
    num_gt_labels = int(np.max(gt_labels_rel))
    num_matches = min(num_gt_labels, num_pred_labels)
    iouMat = np.zeros((num_gt_labels+1, num_pred_labels+1),
                      dtype=np.float32)
    recallMat = np.zeros((num_gt_labels+1, num_pred_labels+1),
                         dtype=np.float32)
    precMat = np.zeros((num_gt_labels+1, num_pred_labels+1),
                       dtype=np.float32)
    fscoreMat = np.zeros((num_gt_labels+1, num_pred_labels+1),
                         dtype=np.float32)
    
    # get localization criterion
    if localization_criterion == "iou":
        logger.debug("evaluate iou localization_criterion")
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

    elif localization_criterion == "cldice":
        logger.debug("evaluate cldice localization_criterion")

        cl_prec = get_centerline_overlap(
                pred_labels_rel, gt_labels_rel,
                np.zeros((num_pred_labels + 1, num_gt_labels + 1), dtype=np.float32))
        cl_recall = get_centerline_overlap(
                gt_labels_rel, pred_labels_rel,
                np.zeros((num_gt_labels + 1, num_pred_labels + 1), dtype=np.float32))
        
        # get cl_recall without overlapping gt labels for false merge calculation later on
        gt_labels_rel_wo_overlap = None
        cl_recall_wo_overlap = None
        if gt_labels_rel.ndim == 4:
            mask = np.sum(gt_labels_rel > 0, axis=0) > 1
            if np.sum(mask) > 0:
                pdb.set_trace()
                gt_labels_rel_wo_overlap = gt_labels_rel
                gt_labels_rel_wo_overlap[mask] = 0
                cl_recall_wo_overlap = get_centerline_overlap(
                        gt_labels_rel_wo_overlap, pred_labels_rel,
                        np.zeros((num_gt_labels + 1, num_pred_labels + 1), dtype=np.float32))
        
        cl_prec = np.transpose(cl_prec)
        iouMat = np.nan_to_num(2 * cl_prec * cl_recall / (cl_prec + cl_recall))
        cl_prec = np.transpose(cl_prec)
        
    else:
        raise NotImplementedError
    
    iouMat = iouMat[1:, 1:]
    recallMat = recallMat[1:, 1:]
    precMat = precMat[1:, 1:]
    fscoreMat = fscoreMat[1:, 1:]

    metrics = Metrics(outFn)
    tblNameGen = "general"
    metrics.addTable(tblNameGen)
    metrics.addMetric(tblNameGen, "Num GT", num_gt_labels)
    metrics.addMetric(tblNameGen, "Num Pred", num_pred_labels)

    if localization_criterion == "cldice":
        # get gt skeleton coverage, 
        # todo: only take max gt label for each pred label to not count pred labels 
        # twice for overlapping gt instances
        #np.argmax(cl_prec[pred_ind_unassigned], axis=1)
        gt_skel_coverage = np.sum(cl_recall[1:, 1:], axis=1)
        gt_skel_coverage = np.mean(gt_skel_coverage)
        metrics.addMetric(tblNameGen, "avg_gt_skel_coverage", gt_skel_coverage)

    ths = [0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    #ths = [0.3]
    aps = []
    metrics.addTable("confusion_matrix")
    for th in ths:
        tblname = "confusion_matrix.th_"+str(th).replace(".", "_")
        metrics.addTable(tblname)
        fscore = 0
        pred_ind_ok = []
        gt_ind_ok = []
        fp_ind = None
        fn_ind = None
        false_split_ind = []
        if num_matches > 0 and np.max(iouMat) > th:
            fscore_cnt = 0
            # todo: use other like assignment_strategy = = [linear, greedy, overlap_gt_0_5]
            if greedy_by_score == False:
                # assignment_strategy
                # do optimal hungarian matching
                costs = -(iouMat >= th).astype(float) - iouMat / (2*num_matches)
                logger.info("start computing lin sum assign for th %s (%s)",
                            th, outFn)
                gt_ind, pred_ind = linear_sum_assignment(costs)
                assert num_matches == len(gt_ind) == len(pred_ind)
                match_ok = iouMat[gt_ind, pred_ind] >= th
                tp = np.count_nonzero(match_ok)
                for idx, match in enumerate(match_ok):
                    if match:
                        pred_ind_ok.append(pred_ind[idx])
                        gt_ind_ok.append(gt_ind[idx])
                        fscore = fscoreMat[gt_ind[idx], pred_ind[idx]]
                        if fscore >= 0.8:
                            fscore_cnt += 1
            else:
                # greedy matching by localization criterion
                # this could be done before iterating through thresholds and then 
                # only threshold
                gt_ind, pred_ind = np.nonzero(iouMat > 0) # > th)
                ious = iouMat[gt_ind, pred_ind]
                # sort iou values in descending order
                sort = np.flip(np.argsort(ious))
                gt_ind = gt_ind[sort]
                pred_ind = pred_ind[sort]
                ious = ious[sort]
                
                # assign greedy by iou score
                for gt_idx, pred_idx, iou in zip(gt_ind, pred_ind, ious):
                    if gt_idx not in gt_ind_ok and pred_idx not in pred_ind_ok and iou > th:
                        gt_ind_ok.append(gt_idx)
                        pred_ind_ok.append(pred_idx)
                    # if gt label is already assinged, count pred label as false split
                    elif gt_idx in gt_ind_ok and pred_idx not in pred_ind_ok \
                            and pred_idx not in false_split_ind:
                        false_split_ind.append(pred_idx)
                # get fn (unassigned gt_ind)
                gt_ind_all = np.arange(1, num_gt_labels + 1)
                fn_ind = gt_ind_all[np.isin(gt_ind_all, gt_ind_ok, invert=True)]

                # get fp (unassigned pred_ind)
                pred_ind_all = np.arange(1, num_pred_labels + 1)
                fp_ind = pred_ind_all[np.isin(pred_ind_all, 
                    np.concatenate([pred_ind_ok, false_split_ind]), invert=True)]

                print(pred_ind_ok, gt_ind_ok, fp_ind, fn_ind, false_split_ind)
                tp = len(pred_ind_ok)
            
            # correct indices to include background
            pred_ind_ok = np.array(pred_ind_ok) + 1
            gt_ind_ok = np.array(gt_ind_ok) + 1    
        else:
            tp = 0
            fscore_cnt = 0
        
        metrics.addMetric(tblname, "Fscore_cnt", fscore_cnt)
        # todo: take fp count from fp_ind excluding false splits as well
        fp = num_pred_labels - tp
        fn = num_gt_labels - tp
        metrics.addMetric(tblname, "AP_TP", tp)
        metrics.addMetric(tblname, "AP_FP", fp)
        metrics.addMetric(tblname, "AP_FN", fn)

        p = 1.*(tp) / max(1, tp +  fp)
        rec = 1.*(tp) / max(1, tp +  fn)
        aps.append(p*rec)
        metrics.addMetric(tblname, "AP", p*rec)

        precision = tp / max(1, tp + fp)
        metrics.addMetric(tblname, "precision", precision)
        recall = tp / max(1, tp + fn)
        metrics.addMetric(tblname, "recall", recall)
        if (precision + recall) > 0:
            fscore = (2. * precision * recall) / max(1, precision + recall)
        else:
            fscore = 0.0
        metrics.addMetric(tblname, 'fscore', fscore)
        
        # report false split and false merge and tp skeleton coverage
        #if kwargs.get("false_split_thresh") ?
        if localization_criterion == "cldice":
            if fp_ind is None:
                # necessary if other than greedy matching
                # move somewhere else?
                pred_ind_all = np.arange(1, num_pred_labels + 1)
                pred_ind_unassigned = pred_ind_all[np.isin(
                    pred_ind_all, pred_ind_ok, invert=True)]
                # count all unassigned pred labels which maximal overlap label 
                # is not background as false split
                fp_ind = pred_ind_unassigned[np.argmax(
                    cl_prec[pred_ind_unassigned], axis=1) == 0]
                false_split_ind = pred_ind_unassigned[
                        np.argmax(cl_prec[pred_ind_unassigned], axis=1) > 0]
            false_split = len(false_split_ind)
            metrics.addMetric(tblname, "false_split", int(false_split))

            # get false negative indices
            if fn_ind is None:
                # necessary if other than greedy matching
                # move somewhere else?
                gt_ind_all = np.arange(1, num_gt_labels + 1)
                gt_ind_unassigned = gt_ind_all[np.isin(gt_ind_all, gt_ind_ok, invert=True)]
                fn_ind = gt_ind_unassigned
            
            # get false merges
            #gt_ind_all = np.arange(1, num_gt_labels + 1)
            #gt_ind_unassigned = gt_ind_all[np.isin(gt_ind_all, gt_ind_ok, invert=True)]
            #false_merge = np.sum(np.any(cl_recall[gt_ind_unassigned, 1:] > 0.1, axis=1))
            # todo: define false merges with clDice in greedy matching?
            if gt_labels_rel.ndim == 4 and cl_recall_wo_overlap is not None:
                fm = np.sum(cl_recall[1:, 1:] > th, axis=0)
            else:
                fm = np.sum(cl_recall[1:, 1:] > th, axis=0) 
            fm_bg = (cl_prec[1:, 0] > 0.1).astype(np.uint8)
            fm_bg[fm <= 0] = 0
            fm_pred_ind = np.array(np.nonzero(
                np.maximum([0] * num_pred_labels, fm + fm_bg - 1) > 0)[0])
            fm_pred_ind += 1

            if gt_labels_rel.ndim == 4 and cl_recall_wo_overlap is not None:
                fm_gt_ind = np.nonzero(cl_recall_wo_overlap[:, fm_pred_ind] > th)[0]
            else:
                fm_gt_ind = np.nonzero(cl_recall[:, fm_pred_ind] > th)[0]
            false_merge = np.sum(np.maximum(
                    [0] * num_pred_labels,
                    fm + fm_bg - 1
                    ))
            metrics.addMetric(tblname, "false_merge", int(false_merge))
            
            if tp > 0:
                tp_skel_coverage = np.mean(np.sum(cl_recall[gt_ind_ok, 1:], axis=1))
            else:
                tp_skel_coverage = 0
            metrics.addMetric(tblname, "avg_tp_skel_coverage", tp_skel_coverage)

        # visualize tp and errors
        if visualize and tp > 0 and th == 0.5:
            if visualize_type == "nuclei":
                visualize_nuclei(gt_labels_rel, iouMat, gt_ind, pred_ind)
            elif visualize_type == "neuron" and localization_criterion == "cldice":
                visualize_neuron(
                        gt_labels_rel, pred_labels_rel, gt_ind_ok, pred_ind_ok, 
                        outFn, false_split_ind, fp_ind, fn_ind, fm_pred_ind, fm_gt_ind)
            else:
                raise NotImplementedError
            
    avAP19 = np.mean(aps)
    avAP59 = np.mean(aps[4:])
    metrics.addMetric("confusion_matrix", "avAP", avAP59)
    metrics.addMetric("confusion_matrix", "avAP59", avAP59)
    metrics.addMetric("confusion_matrix", "avAP19", avAP19)

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


def remove_small_components(array, size=0):

    labels, counts = np.unique(array, return_counts=True)
    small_labels = labels[counts <= size]

    array = replace(
        array,
        np.array(small_labels),
        np.array([0] * len(small_labels))
    )
    return array


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


def visualize_neuron(gt_labels_rel, pred_labels_rel, gt_ind, pred_ind, outFn, 
        false_split_ind, fp_ind, fn_ind, fm_pred_ind, fm_gt_ind):
    # todo: find colormap
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
    gray_gt_cmap = (np.arange(num_gt + 1) / float(num_gt) * 255).astype(np.uint8)
    gray_pred_cmap = (np.arange(num_pred + 1) / float(num_pred) * 255).astype(np.uint8)
    dst = np.zeros_like(gt, dtype=np.uint8)
    dst = np.stack([dst, dst, dst], axis=-1)

    # visualize gt with same colormap
    vis = np.zeros_like(dst)
    for i in range(1, num_gt + 1):
        vis[gt == i] = vis_cmap[(i-1) % len(vis_cmap)]
    mip = np.max(vis, axis=0)
    io.imsave(
        outFn + '_gt.png',
        mip.astype(np.uint8)
    )
    
    # visualize tp, fp and false split with gt in gray
    # tp green, fp blue, false split red
    gt_gray = np.zeros_like(dst, dtype=np.uint8)
    for i in range(1, num_gt + 1):
        print(gray_gt_cmap[i])
        gt_gray[gt == i] = [gray_gt_cmap[i],] * 3
    vis = np.zeros_like(dst, dtype=np.uint8)
    for i in pred_ind:
        vis[pred == i] = [0, 255, 0]
    for i in false_split_ind:
        vis[pred == i] = [255, 0, 0]
    for i in fp_ind:
        vis[pred == i] = [0, 0, 255]

    #with h5py.File(outFn + "_vis.hdf", 'w') as fi:
    #    fi.create_dataset(
    #        'volumes/vis_tp',
    #        data=vis_tp,
    #        compression='gzip')
    mip = np.max(vis, axis=0)
    gt_mip = np.max(gt_gray, axis=0)
    mask = np.logical_not(np.any(mip > 0, axis=-1))
    mip[mask] = gt_mip[mask]
    io.imsave(
        outFn + '_tp.png',
        mip.astype(np.uint8)
    )
    
    # visualize false negative in color, pred in grey
    vis = np.zeros_like(dst, dtype=np.uint8)
    for i in range(1, num_pred + 1):
        vis[pred == i] = [gray_pred_cmap[i],] * 3
    for i in fn_ind:
        vis[gt == i] = vis_cmap[(i-1) % len(vis_cmap)]
    mip = np.max(vis, axis=0)
    io.imsave(
        outFn + '_fn.png',
        mip.astype(np.uint8)
    )
    
    # show false merge errors
    vis = np.zeros_like(dst)
    print(fm_gt_ind, fm_pred_ind)
    for i in fm_gt_ind:
        print(i)
        vis[np.max(gt_labels_rel == i, axis=0)] = vis_cmap[(i-1) % len(vis_cmap)]
        print(np.sum(vis>0))
    for i in fm_pred_ind:
        print(i)
        vis[pred == i] = [gray_pred_cmap[i],] * 3
        print(np.sum(vis>0))

    mip = np.max(vis, axis=0)
    io.imsave(
        outFn + '_false_merge.png',
        mip.astype(np.uint8)
    )

    # overlay gt and pred
    pred = ((pred_labels_rel > 0) * 255).astype(np.uint8)
    gt = ((gt > 0) * 255).astype(np.uint8)
    vis = np.stack([pred, gt, np.zeros_like(pred)], axis=-1).astype(np.uint8)
    mip = np.max(vis, axis=0)
    io.imsave(
        outFn + '_overlayed.png',
        mip.astype(np.uint8)
    )
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--res_file', type=str,
                        help='path to res_file', required=True)
    parser.add_argument('--res_file_suffix', type=str,
                        help='res_file suffix (deprecated)')
    parser.add_argument('--res_key', type=str,
                        help='name labeling hdf key')
    parser.add_argument('--gt_file', type=str,
                        help='path to gt_file', required=True)
    parser.add_argument('--gt_key', type=str,
                        help='name gt hdf key')
    parser.add_argument('--out_dir', type=str,
                        help='output directory', required=True)
    parser.add_argument('--metric', type=str,
                        default="confusion_matrix.th_0_5.AP",
                        help='check if this metric already has been computed in possibly existing result files')
    parser.add_argument('--suffix', type=str,
                        help='suffix (deprecated)', default="")
    parser.add_argument('--background', type=int,
                        help='label for background (use -1 for None)',
                        default="0")
    parser.add_argument("--use_gt_fg", help="usually not used (deprecated)",
                        action="store_true")
    parser.add_argument("--overlapping_inst", help="if there can be multiple instances per pixel",
                        action="store_true")
    parser.add_argument("--from_scratch",
                        help="recompute everything (instead of checking if results are already there)",
                        action="store_true")
    parser.add_argument("--no_use_linear_sum_assignment",
                        help="don't use Hungarian matching",
                        dest='use_linear_sum_assignment',
                        action="store_false")
    parser.add_argument("--visualize", help="",
                        action="store_true")
    parser.add_argument("--debug", help="",
                        action="store_true")
    parser.add_argument('--visualize_type', type=str,
                        help='which type of data should be visualized, e.g. nuclei, neurons.', 
                        default="nuclei")

    logger.debug("arguments %s",tuple(sys.argv))
    args = parser.parse_args()
    if args.use_gt_fg:
        logger.info("using gt foreground")

    evaluate_file(args.res_file, args.gt_file,
                  foreground_only=args.use_gt_fg,
                  background=args.background, res_key=args.res_key,
                  gt_key=args.gt_key, out_dir=args.out_dir, suffix=args.suffix,
                  debug=args.debug, visualize=args.visualize, 
                  visualize_type=args.visualize_type)
