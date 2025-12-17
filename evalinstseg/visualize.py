# evalinstseg/visualize.py
import logging

from matplotlib.colors import to_rgb
import numpy as np
import scipy.ndimage
from skimage import io

from .vis_io import save_vis_hdf, export_vis_pngs

logger = logging.getLogger(__name__)

gt_cmap_ = [
    "#88B04B", "#9F00A7", "#EFC050", "#34568B", "#E47A2E",
    "#BC70A4", "#92A8D1", "#A3B18A", "#45B8AC", "#6B5B95",
    "#F7CAC9", "#E8A798", "#9C9A40", "#9C4722", "#6B5876",
    "#CE3175", "#00A591", "#EDD59E", "#1E7145", "#E9FF70",
]

pred_cmap_ = [
    "#FDAC53", "#9BB7D4", "#B55A30", "#F5DF4D", "#0072B5",
    "#A0DAA9", "#E9897E", "#00A170", "#926AA6", "#EFE1CE",
    "#9A8B4F", "#FFA500", "#56C6A9", "#4B5335", "#798EA4",
    "#E0B589", "#00758F", "#FA7A35", "#578CA9", "#95DEE3"
]


def rgb(idx, cmap):
    """convert color hex code to uint8 rgb tuple"""
    return (np.array(to_rgb(cmap[idx % len(cmap)])) * 255).astype(np.uint8)


def proj_label(lbl):
    """project label map along the first axis (z), rgb-aware"""
    dst = np.zeros(lbl.shape[1:], dtype=np.uint8)
    for i in range(lbl.shape[0] - 1, -1, -1):
        dst[np.max(dst, axis=-1) == 0, :] = lbl[i][np.max(dst, axis=-1) == 0, :]
    return dst


def paint_boundary(labels_rel, label_id, target):
    """
    converts dense label map to boundary label map
    NOTE: border_value=0 to avoid edge artifacts.
    """
    if len(labels_rel.shape) == 3:
        coords_ = np.nonzero(labels_rel == label_id)
        coords = {}
        for z, y, x in zip(*coords_):
            coords.setdefault(z, []).append((z, y, x))
        max_z = max(coords.keys(), key=lambda z: len(coords[z]))
        tmp = (labels_rel[max_z] == label_id)

        struct = scipy.ndimage.generate_binary_structure(2, 2)
        eroded_tmp = scipy.ndimage.binary_erosion(
            tmp,
            iterations=1,
            structure=struct,
            border_value=1,
        )
        bnd = np.logical_xor(tmp, eroded_tmp)
        target[max_z][bnd] = 1
    else:
        tmp = (labels_rel == label_id)
        struct = scipy.ndimage.generate_binary_structure(2, 2)
        eroded_tmp = scipy.ndimage.binary_erosion(
            tmp,
            iterations=1,
            structure=struct,
            border_value=1)
        bnd = np.logical_xor(tmp, eroded_tmp)
        target[bnd] = 1


def visualize_nuclei(gt_labels_rel, pred_labels_rel, locMat, gt_ind, pred_ind, th, outFn, export_png=True):
    """visualize nuclei (blob-like) segmentation results"""
    # GT-basiert
    vis_tp = np.zeros_like(gt_labels_rel, dtype=np.float32)
    vis_fn = np.zeros_like(gt_labels_rel, dtype=np.float32)
    vis_tp_seg = np.zeros_like(gt_labels_rel, dtype=np.float32)
    vis_fn_seg = np.zeros_like(gt_labels_rel, dtype=np.float32)
    vis_fn_seg_bnd = np.zeros_like(gt_labels_rel, dtype=np.float32)

    # Pred-basiert
    vis_fp = np.zeros_like(pred_labels_rel, dtype=np.float32)
    vis_tp_seg2 = np.zeros_like(pred_labels_rel, dtype=np.float32)
    vis_fp_seg = np.zeros_like(pred_labels_rel, dtype=np.float32)
    vis_fp_seg_bnd = np.zeros_like(pred_labels_rel, dtype=np.float32)

    num_gt_labels = int(np.max(gt_labels_rel))
    num_pred_labels = int(np.max(pred_labels_rel))

    labels_gt = list(range(1, num_gt_labels + 1))
    labels_pred = list(range(1, num_pred_labels + 1))

    cntrs_gt = scipy.ndimage.center_of_mass(gt_labels_rel > 0, gt_labels_rel, labels_gt)
    cntrs_pred = scipy.ndimage.center_of_mass(pred_labels_rel > 0, pred_labels_rel, labels_pred)

    sz = 1
    for gti, pi in zip(gt_ind, pred_ind):
        if locMat[gti, pi] < th:
            vis_fn_seg[gt_labels_rel == gti] = 1
            paint_boundary(gt_labels_rel, gti, vis_fn_seg_bnd)
            vis_fp_seg[pred_labels_rel == pi] = 1
            paint_boundary(pred_labels_rel, pi, vis_fp_seg_bnd)
            idx = tuple(int(round(c)) for c in cntrs_gt[gti - 1])
            vis_fn[idx] = 1

            idx = tuple(int(round(c)) for c in cntrs_pred[pi - 1])
            vis_fp[idx] = 1
        else:
            vis_tp_seg[gt_labels_rel == gti] = 1
            idx = tuple(int(round(c)) for c in cntrs_gt[gti - 1])
            vis_tp[idx] = 1
            vis_tp_seg2[pred_labels_rel == pi] = 1

    # FN
    for label_id in range(1, num_gt_labels + 1):
        if label_id in gt_ind:
            continue
        vis_fn_seg[gt_labels_rel == label_id] = 1
        paint_boundary(gt_labels_rel, label_id, vis_fn_seg_bnd)
        idx = tuple(int(round(c)) for c in cntrs_gt[label_id - 1])
        vis_fn[idx] = 1

    # FP
    for label_id in range(1, num_pred_labels + 1):
        if label_id in pred_ind:
            continue
        vis_fp_seg[pred_labels_rel == label_id] = 1
        paint_boundary(pred_labels_rel, label_id, vis_fp_seg_bnd)
        idx = tuple(int(round(c)) for c in cntrs_pred[label_id - 1])
        vis_fp[idx] = 1

    # blur markers
    vis_tp = scipy.ndimage.gaussian_filter(vis_tp, sz, truncate=sz)
    vis_fn = scipy.ndimage.gaussian_filter(vis_fn, sz, truncate=sz)
    vis_fp = scipy.ndimage.gaussian_filter(vis_fp, sz, truncate=sz)

    # normalize markers to [0,1]
    for arr in (vis_tp, vis_fn, vis_fp):
        m = np.max(arr)
        if m > 0:
            arr /= m

    data = {
        "vis_tp": vis_tp,
        "vis_fp": vis_fp,
        "vis_fn": vis_fn,
        "vis_tp_seg": vis_tp_seg,
        "vis_tp_seg2": vis_tp_seg2,
        "vis_fp_seg": vis_fp_seg,
        "vis_fn_seg": vis_fn_seg,
        "vis_fp_bnd": vis_fp_seg_bnd,
        "vis_fn_bnd": vis_fn_seg_bnd,
    }

    save_vis_hdf(outFn, data)
    if export_png:
        export_vis_pngs(outFn, data)


def visualize_neurons(
        gt_labels_rel, pred_labels_rel, gt_ind, pred_ind, outFn,
        fp_ind, fs_ind, fn_ind, fm_gt_ind, fm_pred_ind, fp_ind_only_bg, export_hdf=True):
    """visualize neuron (tree-like) segmentation results
    Note
    ----
    currently unused: unused: fs_ind , fm_pred_ind, fp_ind_only_bg
    """
    if len(gt_labels_rel.shape) == 4:
        gt = np.max(gt_labels_rel, axis=0)
    else:
        gt = gt_labels_rel
    if len(pred_labels_rel.shape) == 4:
        pred = np.max(pred_labels_rel, axis=0)
    else:
        pred = pred_labels_rel
    # unify shapes for vis if only z differs (instance stack)
    if gt.ndim == 3 and pred.ndim == 3 and gt.shape[1:] == pred.shape[1:] and gt.shape[0] != pred.shape[0]:
        gt = np.max(gt, axis=0)
        pred = np.max(pred, axis=0)
    assert gt.shape == pred.shape, f"Spatial shape mismatch: gt {gt.shape}, pred {pred.shape}"

    num_gt = int(np.max(gt_labels_rel))
    num_pred = int(np.max(pred_labels_rel))

    # base RGB canvas
    dst = np.zeros_like(gt, dtype=np.uint8)
    dst = np.stack([dst, dst, dst], axis=-1)

    # visualize gt
    # one color per instance
    vis = np.zeros_like(dst)
    for i in range(1, num_gt + 1):
        vis[gt == i] = rgb(i - 1, gt_cmap_)
    if vis.ndim == 3:
        mip_gt = vis
    else:
        mip_gt = proj_label(vis)
    mip_gt_mask = np.max(mip_gt > 0, axis=-1)
    io.imsave(outFn + "_gt.png", mip_gt.astype(np.uint8))

    # visualize pred
    # one color per instance
    vis = np.zeros_like(dst)
    for i in range(1, num_pred + 1):
        vis[pred == i] = rgb(i - 1, pred_cmap_)
    if vis.ndim == 3:
        mip_pred = vis
    else:
        mip_pred = proj_label(vis)
    mip_pred_mask = np.max(mip_pred > 0, axis=-1)
    io.imsave(outFn + "_pred.png", mip_pred.astype(np.uint8))

    # visualize tp pred + fp + fs
    # tp pred in color
    # fp and fs (false splits) in red
    # fn pixels in gray
    vis = np.zeros_like(dst)
    for i in pred_ind:
        vis[pred == i] = rgb(i - 1, pred_cmap_)
    for i in fp_ind:
        vis[pred == i] = [255, 0, 0]
    if vis.ndim == 3:
        mip_tp_pred_fp_fs = vis
    else:
        mip_tp_pred_fp_fs = proj_label(vis)
    mask = np.logical_and(mip_gt_mask, np.logical_not(np.max(mip_tp_pred_fp_fs > 0, axis=-1)))
    mip_tp_pred_fp_fs[mask] = [200, 200, 200]
    io.imsave(outFn + "_tp_pred_fp_fs.png", mip_tp_pred_fp_fs.astype(np.uint8))

    # ---- FN / FM ----
    vis = np.zeros_like(dst, dtype=np.uint8)
    fm_merged = np.unique(np.array([ind for inds in fm_gt_ind for ind in inds]).flatten()) if len(fm_gt_ind) else np.array([])
    for i in fn_ind:
        if i not in fm_merged:
            vis[gt == i] = rgb(i - 1, gt_cmap_)
    for i in fm_merged:
        if i not in gt_ind:
            vis[gt == i] = [255, 0, 0]
    if vis.ndim == 3:
        mip_fn_fm = vis
    else:
        mip_fn_fm = proj_label(vis)
    mask = np.logical_and(mip_pred_mask, np.logical_not(np.max(mip_fn_fm > 0, axis=-1)))
    mip_fn_fm[mask] = [200, 200, 200]
    io.imsave(outFn + "_fn_fm.png", mip_fn_fm.astype(np.uint8))

    # visualize tp pred + fp + fs
    # tp pred in color
    # fp and fs (false splits) in red
    # fn pixels in gray
    vis = np.zeros_like(dst, dtype=np.uint8)
    for i in gt_ind:
        vis[gt == i] = rgb(i - 1, gt_cmap_)
    for i in fn_ind:
        if i not in fm_merged:
            vis[gt == i] = [255, 64, 64]
    for i in fm_merged:
        if i not in gt_ind:
            vis[gt == i] = [192, 0, 0]
    if vis.ndim == 3:
        mip_fn_fm_v2 = vis
    else:
        mip_fn_fm_v2 = proj_label(vis)
    mask = np.logical_and(mip_pred_mask, np.logical_not(np.max(mip_fn_fm_v2 > 0, axis=-1)))
    mip_fn_fm_v2[mask] = [200, 200, 200]
    io.imsave(outFn + "_fn_fm_v2.png", mip_fn_fm_v2.astype(np.uint8))

    if export_hdf:
        data = {
            "gt_rgb": mip_gt.astype(np.uint8),
            "pred_rgb": mip_pred.astype(np.uint8),
            "tp_pred_fp_fs_rgb": mip_tp_pred_fp_fs.astype(np.uint8),
            "fn_fm_rgb": mip_fn_fm.astype(np.uint8),
            "fn_fm_v2_rgb": mip_fn_fm_v2.astype(np.uint8),
        }
        save_vis_hdf(outFn, data)
