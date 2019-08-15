import glob
import logging
import os
import sys

import h5py
import argparse
import numpy as np
import tifffile
import toml
import zarr

logger = logging.getLogger(__name__)


class Metrics:
    def __init__(self, fn):
        self.metricsDict = {}
        self.metricsArray = []
        self.fn = fn
        self.outFl = open(self.fn+".txt", 'w')

    def __del__(self):
        self.outFl.close()
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


def maybe_crop(pred_labels, gt_labels):
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
        # TODO: check if necessary/correct
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
    # TODO: check if reshaping necessary
    if pred_labels.shape[0] == 1:
        pred_labels.shape = pred_labels.shape[1:]
    logger.debug("prediction shape %s", pred_labels.shape)

    # read ground truth data
    if gt_file.endswith(".hdf"):
        with h5py.File(gt_file, 'r') as f:
            # gt_labels = np.array(f['volumes/gt_labels'])
            # gt_labels = np.array(f['images/gt_instances'])
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
    # TODO: check if reshaping necessary
    if gt_labels.shape[0] == 1:
        gt_labels.shape = gt_labels.shape[1:]
    gt_labels = np.squeeze(gt_labels)
    if gt_labels.ndim > pred_labels.ndim:
        gt_labels = np.max(gt_labels, axis=0)
    logger.debug("gt shape %s", gt_labels.shape)

    # TODO: check if necessary
    pred_labels, gt_labels = maybe_crop(pred_labels, gt_labels)

    if foreground_only:
        pred_labels[gt_labels==0] = 0

    logger.info("processing %s %s", res_file, gt_file)
    outFnBase = os.path.join(
        kwargs['out_dir'],
        os.path.splitext(os.path.basename(res_file))[0] +
        kwargs['res_key'].replace("/","_") + kwargs['suffix'])
    if res_file.endswith(".hdf"):
        outFn = outFnBase + "_hdf_scores.txt"
    else:
        outFn = outFnBase + "_tif_scores.txt"

    if len(glob.glob(outFnBase + "*")) > 0:
        logger.info('Skipping evaluation for %s. Already exists!', res_file)
        tomlFl = open(outFn+".toml", 'r')
        return toml.load(tomlFl)

    overlay = np.array([pred_labels.flatten(),
                        gt_labels.flatten()])
    logger.debug("overlay shape %s", overlay.shape)
    # get overlaying cells and the size of the overlap
    overlay_labels, overlay_labels_counts = np.unique(overlay,
                                         return_counts=True, axis=1)
    overlay_labels = np.transpose(overlay_labels)

    # get gt cell ids and the size of the corresponding cell
    gt_labels_list, gt_counts = np.unique(gt_labels, return_counts=True)
    gt_labels_count_dict = {}
    for (l,c) in zip(gt_labels_list, gt_counts):
        gt_labels_count_dict[l] = c

    # get pred cell ids
    pred_labels_list, pred_counts = np.unique(pred_labels, return_counts=True)
    pred_labels_count_dict = {}
    for (l,c) in zip(pred_labels_list, pred_counts):
        pred_labels_count_dict[l] = c

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

        # print("c %s gt_label %s num %s pred_label %s num %s dice %s iou %s"
        # %(c,v,gt_labels_count_dict[v],u,pred_labels_count_dict[u],dice,iou))
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
    diceGT = dice/cnt

    dice = 0
    cnt = 0
    for (k, vs) in diceP.items():
        vs = sorted(vs, reverse=True)
        dice += vs[0]
        cnt += 1
    diceP = dice/cnt

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
    segGT = seg/cnt

    seg = 0
    cnt = 0
    for (k, vs) in segP.items():
        vs = sorted(vs, reverse=True)
        seg += vs[0]
        cnt += 1
    segP = seg/cnt

    seg = 0
    cnt = 0
    for (k, vs) in segPrev.items():
        vs = sorted(vs, reverse=True)
        seg += vs[0]
        cnt += 1
    segPrev = seg/cnt

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

    ths = [0.5, 0.6, 0.7, 0.8, 0.9]
    ths.extend([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
    ths.extend([0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
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
        # TODO: check, are both versions equal? which is correct
        metrics.addMetric(tblname, "AP_TP", apTP)
        apTP = np.count_nonzero(iouP_2[iouP_2>th])
        apFP = np.count_nonzero(iouP_2[iouP_2<=th])
        apFN = np.count_nonzero(iouGT[iouGT<=th])
        metrics.addMetric(tblname, "AP_TP", apTP)
        metrics.addMetric(tblname, "AP_FP", apFP)
        metrics.addMetric(tblname, "AP_FN", apFN)
        ap = 1.*(apTP) / (apTP + apFN + apFP)
        aps.append(ap)
        metrics.addMetric(tblname, "AP", ap)
        precision = 1.*(apTP) / len(pred_labels_list)
        metrics.addMetric(tblname, "precision", precision)
        recall = 1.*(apTP) / len(gt_labels_list)
        metrics.addMetric(tblname, "recall", recall)
        if (precision + recall) > 0:
            fscore = (2. * precision * recall) / (precision + recall)
        else:
            fscore = 0.0
        metrics.addMetric(tblname, 'fscore', fscore)

    avAP = np.mean(aps)
    metrics.addMetric("confusion_matrix", "avAP", avAP)

    return metrics.metricsDict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--res_file', type=str,
                        help='path to res_file', required=True)
    parser.add_argument('--res_file_suffix', type=str,
                        help='res_file suffix')
    parser.add_argument('--res_key', type=str,
                        help='name labeling hdf key')
    parser.add_argument('--gt_file', type=str,
                        help='path to gt_file', required=True)
    parser.add_argument('--gt_key', type=str,
                        help='name gt hdf key')
    parser.add_argument('--out_dir', type=str,
                        help='output directory', required=True)
    parser.add_argument('--suffix', type=str,
                        help='suffix', default="")
    parser.add_argument('--background', type=int,
                        help='label for background (use -1 for None)',
                        default="0")
    parser.add_argument("--use_gt_fg", help="",
                    action="store_true")
    parser.add_argument("--debug", help="",
                    action="store_true")

    logger.debug("arguments %s",tuple(sys.argv))
    args = parser.parse_args()
    if args.use_gt_fg:
        logger.info("using gt foreground")

    evaluate_file(args.res_file, args.gt_file,
                  foreground_only=args.use_gt_fg,
                  background=args.background, res_key=args.res_key,
                  gt_key=args.gt_key, out_dir=args.out_dir, suffix=args.suffix,
                  debug=args.debug)
