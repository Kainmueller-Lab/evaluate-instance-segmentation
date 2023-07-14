import logging

import h5py
from matplotlib.colors import to_rgb
import numpy as np
import scipy.ndimage
from skimage import io

logger = logging.getLogger(__name__)


# colors selected based on:
# - at least somewhat usable in case of color blindness
# - no two colors should be too similar
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
    for i in range(lbl.shape[0]-1, -1, -1):
        dst[np.max(dst, axis=-1)==0,:] = lbl[i][np.max(dst, axis=-1)==0,:]

    return dst


def paint_boundary(labels_rel, label, target):
    """converts dense label map to boundary label map
    (for 3d label maps only slice with largest xy-extend is filled)
    """
    if len(labels_rel.shape) == 3:
        coords_ = np.nonzero(labels_rel == label)
        coords = {}
        for z, y, x in zip(*coords_):
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
    else:
        tmp = np.zeros_like(labels_rel, dtype=np.float32)
        tmp = labels_rel==label
        struct = scipy.ndimage.generate_binary_structure(2, 2)
        eroded_tmp = scipy.ndimage.binary_erosion(
            tmp,
            iterations=1,
            structure=struct,
            border_value=1)
        bnd = np.logical_xor(tmp, eroded_tmp)
        target[bnd] = 1


def visualize_nuclei(
        gt_labels_rel, pred_labels_rel, locMat, gt_ind, pred_ind, th, outFn):
    """visualize nuclei (blob-like) segmentation results"""
    vis_tp = np.zeros_like(gt_labels_rel, dtype=np.float32)
    vis_fp = np.zeros_like(gt_labels_rel, dtype=np.float32)
    vis_fn = np.zeros_like(gt_labels_rel, dtype=np.float32)
    vis_tp_seg = np.zeros_like(gt_labels_rel, dtype=np.float32)
    vis_tp_seg2 = np.zeros_like(gt_labels_rel, dtype=np.float32)
    vis_fp_seg = np.zeros_like(gt_labels_rel, dtype=np.float32)
    vis_fn_seg = np.zeros_like(gt_labels_rel, dtype=np.float32)
    vis_fp_seg_bnd = np.zeros_like(gt_labels_rel, dtype=np.float32)
    vis_fn_seg_bnd = np.zeros_like(gt_labels_rel, dtype=np.float32)

    cntrs_gt = scipy.ndimage.measurements.center_of_mass(
        gt_labels_rel > 0,
        gt_labels_rel, sorted(list(np.unique(gt_labels_rel)))[1:])
    cntrs_pred = scipy.ndimage.measurements.center_of_mass(
        pred_labels_rel > 0,
        pred_labels_rel, sorted(list(np.unique(pred_labels_rel)))[1:])
    num_gt_labels = np.max(gt_labels_rel)
    num_pred_labels = np.max(pred_labels_rel)

    sz = 1
    for gti, pi, in zip(gt_ind, pred_ind):
        if locMat[gti, pi] < th:
            vis_fn_seg[gt_labels_rel == gti+1] = 1
            paint_boundary(gt_labels_rel, gti+1, vis_fn_seg_bnd)
            vis_fp_seg[pred_labels_rel == pi+1] = 1
            paint_boundary(pred_labels_rel, pi+1, vis_fp_seg_bnd)
            cntr = cntrs_gt[gti]
            vis_fn[(int(c) for c in cntr)] = 1
            cntr = cntrs_pred[pi]
            vis_fp[(int(c) for c in cntr)] = 1
        else:
            vis_tp_seg[gt_labels_rel == gti+1] = 1
            cntr = cntrs_gt[gti]
            vis_tp[(int(c) for c in cntr)] = 1
            vis_tp_seg2[pred_labels_rel == pi+1] = 1
    vis_tp = scipy.ndimage.gaussian_filter(vis_tp, sz, truncate=sz)
    for gti in range(num_gt_labels):
        if gti in gt_ind:
            continue
        vis_fn_seg[gt_labels_rel == gti+1] = 1
        paint_boundary(gt_labels_rel, gti+1, vis_fn_seg_bnd)
        cntr = cntrs_gt[gti]
        vis_fn[(int(c) for c in cntr)] = 1
    vis_fn = scipy.ndimage.gaussian_filter(vis_fn, sz, truncate=sz)
    for pi in range(num_pred_labels):
        if pi in pred_ind:
            continue
        vis_fp_seg[pred_labels_rel == pi+1] = 1
        paint_boundary(pred_labels_rel, pi+1, vis_fp_seg_bnd)
        cntr = cntrs_pred[pi]
        vis_fp[(int(c) for c in cntr)] = 1
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
        fi.create_dataset(
            'volumes/vis_fp_seg_bnd',
            data=vis_fp_seg_bnd,
            compression='gzip')
        fi.create_dataset(
            'volumes/vis_fn_seg_bnd',
            data=vis_fn_seg_bnd,
            compression='gzip')


def visualize_neurons(
        gt_labels_rel, pred_labels_rel, gt_ind, pred_ind, outFn,
        fp_ind, fn_ind, fm_gt_ind):
    """visualize neuron (tree-like) segmentation results"""
    # unused: fs_ind , fm_pred_ind, fp_ind_only_bg
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
    dst = np.zeros_like(gt, dtype=np.uint8)
    dst = np.stack([dst, dst, dst], axis=-1)

    # visualize gt
    # one color per instance
    vis = np.zeros_like(dst)
    for i in range(1, num_gt + 1):
        vis[gt == i] = rgb(i-1, gt_cmap_)
    mip = proj_label(vis)
    mip_gt_mask = np.max(mip>0, axis=-1)
    io.imsave(outFn + '_gt.png', mip.astype(np.uint8))

    # visualize pred
    # one color per instance
    vis = np.zeros_like(dst)
    for i in range(1, num_pred + 1):
        vis[pred == i] = rgb(i-1, pred_cmap_)
    mip = proj_label(vis)
    mip_pred_mask = np.max(mip>0, axis=-1)
    io.imsave(outFn + '_pred.png', mip.astype(np.uint8))

    # visualize tp pred + fp + fs
    # tp pred in color
    # fp and fs (false splits) in red
    # fn pixels in gray
    vis = np.zeros_like(dst)
    for i in pred_ind:
        vis[pred == i] = rgb(i-1, pred_cmap_)
    for i in fp_ind:
        vis[pred == i] = [255, 0, 0]
    mip = proj_label(vis)
    mask = np.logical_and(mip_gt_mask, np.logical_not(np.max(mip > 0, axis=-1)))
    mip[mask] = [200, 200, 200]
    io.imsave(outFn + '_tp_pred_fp_fs.png', mip.astype(np.uint8))

    # visualize fn/fm
    # fn in color
    # fm (false merges) in red
    # fp pixels in gray
    vis = np.zeros_like(dst, dtype=np.uint8)
    fm_merged = np.unique(np.array(
        [ind for inds in fm_gt_ind for ind in inds]).flatten())
    for i in fn_ind:
        if i not in fm_merged:
            vis[gt == i] = rgb(i-1, gt_cmap_)
    for i in fm_merged:
        if i not in gt_ind:
            vis[gt == i] = [255, 0, 0]
    mip = proj_label(vis)
    mask = np.logical_and(mip_pred_mask, np.logical_not(np.max(mip > 0, axis=-1)))
    mip[mask] = [200, 200, 200]
    io.imsave(outFn + '_fn_fm.png', mip.astype(np.uint8))

    # visualize fn/fm v2
    # tp gt in color
    # fn in bright red
    # fm in dark red
    # fp pixels in gray
    vis = np.zeros_like(dst, dtype=np.uint8)
    for i in gt_ind:
        vis[gt == i] = rgb(i-1, gt_cmap_)
    for i in fn_ind:
        if i not in fm_merged:
            vis[gt == i] = [255, 64, 64]
    for i in fm_merged:
       if i not in gt_ind:
           vis[gt == i] = [192, 0, 0]
    mip = proj_label(vis)
    mask = np.logical_and(mip_pred_mask, np.logical_not(np.max(mip > 0, axis=-1)))
    mip[mask] = [200, 200, 200]
    io.imsave(outFn + '_fn_fm_v2.png', mip.astype(np.uint8))
