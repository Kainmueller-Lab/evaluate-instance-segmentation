from joblib import Parallel, delayed
import sys
import h5py
import numpy as np
import json
import scipy.ndimage
import tifffile
import os
import operator

def evaluate_files(res_file, gt_file):
    if len(sys.argv) > 4:
        res_file = res_file.replace(".hdf", sys.argv[4] + ".hdf")
    print("loading", res_file, gt_file)
    # read preprocessed hdf file
    if "hdf" in res_file:
        with h5py.File(res_file, 'r') as f:
            pred_labels= np.array(f['volumes/'+sys.argv[3]])
    # or read preprocessed tif files
    else:
        pred_labels = tifffile.imread(res_file)
    print(np.max(pred_labels), np.min(pred_labels), pred_labels.shape)
    if pred_labels.shape[0] == 1:
        pred_labels.shape = pred_labels.shape[1:]

    # read ground truth data
    if "hdf" in gt_file:
        with h5py.File(gt_file, 'r') as f:
            gt_labels = np.array(f['volumes/gt_labels'])
    else:
        gt_labels = tifffile.imread(gt_file)
    print(np.max(gt_labels), np.min(gt_labels), gt_labels.shape)
    if gt_labels.shape[0] == 1:
        gt_labels.shape = gt_labels.shape[1:]
    begin = (np.asarray(gt_labels.shape)-np.asarray(pred_labels.shape))//2
    end = np.asarray(gt_labels.shape)-begin
    print(begin, end)
    if gt_labels.shape[0] != pred_labels.shape[0] and \
       pred_labels.shape[2] % 2 == 1:
        end[2] -= 1
    gt_labels = gt_labels[begin[0]:end[0],
                          begin[1]:end[1],
                          begin[2]:end[2]]
    print(gt_labels.shape)

    print("processing", res_file, gt_file)
    overlay = np.array([pred_labels.flatten(),
                        gt_labels.flatten()])
    print(overlay.shape)
    # get overlaying cells and the size of the overlap
    conn_labels, conn_counts = np.unique(overlay,
                                         return_counts=True, axis=1)
    conn_labels = np.transpose(conn_labels)

    # get gt cell ids and the size of the corresponding cell
    gt_labelsT, gt_counts = np.unique(gt_labels, return_counts=True)
    # get pred cell ids
    gt_labelsD = {}
    for (l,c) in zip(gt_labelsT, gt_counts):
        gt_labelsD[l] = c
    pred_labelsT, pred_counts = np.unique(pred_labels, return_counts=True)
    pred_labelsD = {}
    for (l,c) in zip(pred_labelsT, pred_counts):
        pred_labelsD[l] = c
    # identify overlaying cells where more than 50% of gt cell is covered
    conn = np.asarray([c > 0.5 * float(gt_counts[gt_labelsT == v])
        for (u,v), c in zip(conn_labels, conn_counts)], dtype=np.bool)

    # get their ids
    conn_labelsT = conn_labels[conn]
    # remove background
    pred_labelsT = pred_labelsT[pred_labelsT > 0]
    gt_labelsT = gt_labelsT[gt_labelsT > 0]
    con_mat = np.zeros((len(pred_labelsT), len(gt_labelsT)))
    for (u,v) in conn_labelsT:
        if u > 0 and v > 0:
            con_mat[np.where(pred_labelsT == u),
                    np.where(gt_labelsT == v)] = 1

    diceGT = {}
    iouGT = {}
    segGT = {}
    diceP = {}
    iouP = {}
    segP = {}
    segP2 = {}
    for (u,v), c in zip(conn_labels, conn_counts):
        dice = 2*c/(gt_labelsD[v]+pred_labelsD[u])
        iou = c/(gt_labelsD[v]+pred_labelsD[u]-c)
        if c > 0.5 * gt_labelsD[v]:
            seg = iou
        else:
            seg = 0
        if c > 0.5 * pred_labelsD[u]:
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
            segP2[u] = []
        diceGT[v].append(dice)
        iouGT[v].append(iou)
        segGT[v].append(seg)
        diceP[u].append(dice)
        iouP[u].append(iou)
        segP[u].append(seg)
        segP2[u].append(seg2)
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
    for (k, vs) in iouGT.items():
        vs = sorted(vs, reverse=True)
        iou.append(vs[0])
    iouGT = np.array(iou)
    iouGTMn = np.mean(iouGT)
    iou = []
    for (k, vs) in iouP.items():
        vs = sorted(vs, reverse=True)
        iou.append(vs[0])
    iouP = np.array(iou)
    iouPMn = np.mean(iouP)

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
    for (k, vs) in segP2.items():
        vs = sorted(vs, reverse=True)
        seg += vs[0]
        cnt += 1
    segP2 = seg/cnt

    # non-split vertices num non-empty cols - num non-empty rows
    # (more than one entry in col: this ground truth cell has been
    # assigned to more than one predicted cell)
    # (other way around not possible due to 50% rule)
    ns = np.sum(np.count_nonzero(con_mat, axis=0)) \
            - np.sum(np.count_nonzero(con_mat, axis=1)>0)
    ns = int(ns)

    # false negative: empty cols
    # (no predicted cell for ground truth cell)
    fn = np.sum(np.sum(con_mat, axis=0)==0)
    fn = int(fn)

    # false positive: empty rows
    # (predicted cell for non existing ground truth cell)
    fp = np.sum(np.sum(con_mat, axis=1)==0)
    fp = int(fp)

    # true positive: row with single entry (can be 0, 1, or more)
    tpP = np.sum(np.sum(con_mat, axis=1)==1)
    tpP = int(tpP)

    # true positive: non-empty col (can only be 0 or 1)
    tpGT = np.sum(np.sum(con_mat, axis=0)>0)
    tpGT = int(tpGT)

    arrTmp = []
    if "hdf" in res_file:
        outFn = sys.argv[1][:-4] + sys.argv[3] + sys.argv[4] + "_hdf_scores.txt"
    else:
        outFn = sys.argv[1][:-4] + "_tif_scores.txt"
    outFl = open(outFn, 'w')
    print("Num GT: ", len(gt_labelsT))
    arrTmp.append(len(gt_labelsT))
    outFl.write("Num GT: {}\n".format(len(gt_labelsT)))
    print("Num Pred:", len(pred_labelsT))
    arrTmp.append(len(pred_labelsT))
    outFl.write("Num Pred: {}\n".format(len(pred_labelsT)))
    print("GT/Ref -> Pred mean dice", diceGT)
    arrTmp.append(diceGT)
    outFl.write("GT/Ref -> Pred mean dice: {:.3f}\n".format(diceGT))
    print("Pred -> GT/Ref mean dice", diceP)
    arrTmp.append(diceP)
    outFl.write("Pred -> GT/Ref mean dice: {:.3f}\n".format(diceP))
    print("GT/Ref -> Pred mean iou", iouGTMn)
    arrTmp.append(iouGTMn)
    outFl.write("GT/Ref -> Pred mean iou: {:.3f}\n".format(iouGTMn))
    print("Pred -> GT/Ref mean iou", iouPMn)
    arrTmp.append(iouPMn)
    outFl.write("Pred -> GT/Ref mean iou: {:.3f}\n".format(iouPMn))
    print("GT/Ref -> Pred mean seg", segGT)
    arrTmp.append(segGT)
    outFl.write("GT/Ref -> Pred mean seg: {:.3f}\n".format(segGT))
    print("Pred -> GT/Ref mean seg", segP)
    arrTmp.append(segP)
    outFl.write("Pred -> GT/Ref mean seg: {:.3f}\n".format(segP))
    print("Pred -> GT/Ref mean rev seg", segP2)
    arrTmp.append(segP2)
    outFl.write("Pred -> GT/Ref mean rev seg: {:.3f}\n".format(segP2))

    print("Pred -> GT/Ref NS", ns)
    arrTmp.append(ns)
    outFl.write("Pred -> GT/Ref NS: {}\n".format(ns))
    print("Pred -> GT/Ref FP", fp)
    arrTmp.append(fp)
    outFl.write("Pred -> GT/Ref FP: {}\n".format(fp))
    print("Pred -> GT/Ref TP", tpP)
    arrTmp.append(tpP)
    outFl.write("Pred -> GT/REf TP: {}\n".format(tpP))
    print("GT/Ref -> Pred FN", fn)
    arrTmp.append(fn)
    outFl.write("GT/Ref -> Pred FN: {}\n".format(fn))
    print("GT/Ref -> Pred TP", tpGT)
    arrTmp.append(tpGT)
    outFl.write("GT/Ref -> Pred TP: {}\n".format(tpGT))

    ths = [0.5, 0.6, 0.7, 0.8]
    aps = []
    for th in ths:
        apTP = np.count_nonzero(iouP[iouP>th])
        apFP = np.count_nonzero(iouP[iouP<=th])
        apFN = np.count_nonzero(iouGT[iouGT<=th])
        ap = (apTP) / (apTP + apFN + apFP)
        aps.append(ap)
        arrTmp.append(ap)
        print("Average Precision: ", th, ap)
        outFl.write("Average Precision, th={}: {:.3f}\n".format(th, ap))

    outFl.close()

    if "hdf" in res_file:
        outFn = sys.argv[1][:-4] + sys.argv[3] + sys.argv[4] + "_hdf_scores.csv"
    else:
        outFn = sys.argv[1][:-4] + "_tif_scores.csv"
    with open(outFn, 'w') as outFl:
        for arr in arrTmp:
            if isinstance(arr, int):
                outFl.write("{}, ".format(arr))
            else:
                outFl.write("{:.3f}, ".format(arr))

if __name__ == "__main__":
    res_file = sys.argv[1]
    gt_file = sys.argv[2]
    evaluate_files(res_file, gt_file)
