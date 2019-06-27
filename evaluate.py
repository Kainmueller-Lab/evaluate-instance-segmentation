import sys
import os

import h5py
import argparse
import numpy as np
import tifffile
import toml


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

    def addMetric(self, name, value):
        as_str = "{}: {}".format(name, value)
        print(as_str)
        self.outFl.write(as_str+"\n")
        self.metricsArray.append(value)
        self.metricsDict[name] = value


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
        pred_labels = bigger_arr_arr
        gt_labels = smaller_arr
    print("gt shape cropped", gt_labels.shape)
    print("pred shape cropped", pred_labels.shape)

    return pred_labels, gt_labels

def evaluate_files(args, res_file, gt_file, background=0,
                   foreground_only=False):
    # TODO: maybe remove, should be superfluous with proper arg parsing
    if args.resFileSuffix is not None:
        res_file = res_file.replace(".hdf", args.resFileSuffix + ".hdf")
    print("loading", res_file, gt_file)

    # read preprocessed hdf file
    if res_file.endswith(".hdf"):
        with h5py.File(res_file, 'r') as f:
            pred_labels= np.array(f[args.dataset])
    # or read preprocessed tif files
    elif res_file.endswith(".tif") or res_file.endswith(".tiff") or \
         res_file.endswith(".TIF") or res_file.endswith(".TIFF"):
        pred_labels = tifffile.imread(res_file)
    print("prediction max/min/shape", np.max(pred_labels),
          np.min(pred_labels), pred_labels.shape)
    # TODO: check if reshaping necessary
    if pred_labels.shape[0] == 1:
        pred_labels.shape = pred_labels.shape[1:]
    print("prediction shape", pred_labels.shape)

    # read ground truth data
    if gt_file.endswith(".hdf"):
        with h5py.File(gt_file, 'r') as f:
            # gt_labels = np.array(f['volumes/gt_labels'])
            # gt_labels = np.array(f['images/gt_instances'])
            try:
                if args.gt_dataset is not None:
                    gt_labels = np.array(f[args.gt_dataset])
                else:
                    gt_labels = np.array(f['images/gt_labels'])
            except:
                gt_labels = np.array(f['images/gt_instances'])
    elif gt_file.endswith(".tif") or gt_file.endswith(".tiff") or \
         gt_file.endswith(".TIF") or gt_file.endswith(".TIFF"):
        gt_labels = tifffile.imread(gt_file)
    print("gt max/min/shape", np.max(gt_labels), np.min(gt_labels),
          gt_labels.shape)
    # TODO: check if reshaping necessary
    if gt_labels.shape[0] == 1:
        gt_labels.shape = gt_labels.shape[1:]
    gt_labels = np.squeeze(gt_labels)
    print("gt shape", gt_labels.shape)

    # TODO: check if necessary
    pred_labels, gt_labels = maybe_crop(pred_labels, gt_labels)

    if foreground_only:
        pred_labels[gt_labels==0] = 0

    print("processing", res_file, gt_file)
    outFnBase = os.path.join(
        args.outDir,
        os.path.splitext(os.path.basename(args.resFile))[0] +
        args.dataset.replace("/","_") + args.suffix)

    overlay = np.array([pred_labels.flatten(),
                        gt_labels.flatten()])
    print("overlay shape", overlay.shape)
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
    if args.debug:
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
    if args.debug:
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

    metrics = {}
    if res_file.endswith(".hdf"):
        outFn = outFnBase + "_hdf_scores.txt"
    else:
        outFn = outFnBase + "_tif_scores.txt"

    metrics = Metrics(outFn)
    metrics.addMetric("Num GT", len(gt_labels_list))
    metrics.addMetric("Num Pred", len(pred_labels_list))
    metrics.addMetric("GT/Ref -> Pred mean dice", diceGT)
    metrics.addMetric("Pred -> GT/Ref mean dice", diceP)
    metrics.addMetric("GT/Ref -> Pred mean iou", iouGTMn)
    metrics.addMetric("Pred -> GT/Ref mean iou", iouPMn)
    metrics.addMetric("GT/Ref -> Pred mean seg", segGT)
    metrics.addMetric("Pred -> GT/Ref mean seg", segP)
    metrics.addMetric("Pred -> GT/Ref mean seg rev", segPrev)
    metrics.addMetric("Pred -> GT/Ref NS", ns)
    metrics.addMetric("Pred -> GT/Ref FP", fp)
    metrics.addMetric("Pred -> GT/Ref TP", tpP)
    metrics.addMetric("GT/Ref -> Pred FN", fn)
    metrics.addMetric("GT/Ref -> Pred TP", tpGT)

    ths = [0.5, 0.6, 0.7, 0.8, 0.9]
    ths.extend([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
    ths.extend([0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
    aps = []
    for th in ths:
        apTP = 0
        for pID in np.nonzero(iouP_2 > th)[0]:
            if len(iouP[iouIDs[pID]]) == 0:
                pass
            elif len(iouP[iouIDs[pID]]) == 1:
                apTP += 1
            elif len(iouP[iouIDs[pID]]) > 1 and iouP[iouIDs[pID]][1] < th:
                apTP += 1
        metrics.addMetric("apTP (iou {})".format(th), apTP)
        apTP = np.count_nonzero(iouP_2[iouP_2>th])
        apFP = np.count_nonzero(iouP_2[iouP_2<=th])
        apFN = np.count_nonzero(iouGT[iouGT<=th])
        metrics.addMetric("apTP (iou {})".format(th), apTP)
        metrics.addMetric("apFP (iou {})".format(th), apFP)
        metrics.addMetric("apFN (iou {})".format(th), apFN)
        ap = 1.*(apTP) / (apTP + apFN + apFP)
        aps.append(ap)
        metrics.addMetric("AP (iou {})".format(th), ap)
        precision = 1.*(apTP) / len(pred_labels_list)
        metrics.addMetric("precision (iou {})".format(th), precision)
        recall = 1.*(apTP) / len(gt_labels_list)
        metrics.addMetric("recall (iou {})".format(th), recall)

    avAP = np.mean(aps)
    metrics.addMetric("avAP", avAP)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--resFile', type=str,
                        help='path to resFile', required=True)
    parser.add_argument('--resFileSuffix', type=str,
                        help='resFile suffix')
    parser.add_argument('--gtFile', type=str,
                        help='path to gtFile', required=True)
    parser.add_argument('--dataset', type=str,
                        help='name labeling dataset', required=True)
    parser.add_argument('--gt_dataset', type=str,
                        help='name gt dataset', required=True)
    parser.add_argument('--outDir', type=str,
                        help='outdir', required=True)
    parser.add_argument('--suffix', type=str,
                        help='suffix', default="")
    parser.add_argument("--use_gt_fg", help="",
                    action="store_true")
    parser.add_argument("--debug", help="",
                    action="store_true")

    args = parser.parse_args()
    res_file = args.resFile
    gt_file = args.gtFile
    print(sys.argv)
    if args.use_gt_fg:
        print("using gt foreground")
        evaluate_files(args, res_file, gt_file, foreground_only=True)
    else:
        evaluate_files(args, res_file, gt_file, foreground_only=False)
