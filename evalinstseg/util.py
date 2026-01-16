import logging
import os

import h5py
import numpy as np
from skimage.segmentation import relabel_sequential
import tifffile
import zarr

logger = logging.getLogger(__name__)


def crop(arr, target_shape):
    """center-crop arr to shape

    Args
    ----
    arr: ndarray
        data
    shape: list of int
        crop data to this shape, only last len(shape) dims

    Returns
    -------
    cropped array
    """
    target_shape = tuple(target_shape)
    n_spatial = len(target_shape)

    # Offsets computed over the last axes only
    spatial_in = arr.shape[-n_spatial:]
    offset = tuple((a - b) // 2 for a, b in zip(spatial_in, target_shape))

    # Build slices: preserve leading axes, crop trailing axes
    leading = (slice(None),) * (arr.ndim - n_spatial)
    trailing = tuple(slice(o, o + s) for o, s in zip(offset, target_shape))

    return arr[leading + trailing]


def check_and_fix_sizes(gt_labels, pred_labels, ndim):
    """check if prediction and gt have same size, otherwise crop the bigger one
    add channel dimension if missing (channel_first)

    E.g., if valid padding is used output might be smaller than input.

    Note
    ----
    only crops spatial dimensions, assumes channel_first
    """
    # add channel dim if not there already
    if gt_labels.ndim == ndim:
        logger.debug("adding channel dim to gt")
        gt_labels = np.expand_dims(gt_labels, axis=0)
    if pred_labels.ndim == ndim:
        logger.debug("adding channel dim to pred")
        pred_labels = np.expand_dims(pred_labels, axis=0)

    if gt_labels.shape[-ndim:] == pred_labels.shape[-ndim:]:
        logger.debug("no cropping necessary")
        return gt_labels, pred_labels

    if np.all([gt_s <= p_s for gt_s, p_s in zip(
            gt_labels.shape[-ndim:], pred_labels.shape[-ndim:])]):
        pred_labels = crop(pred_labels, gt_labels.shape[-ndim:])
    elif np.all([p_s <= gt_s for gt_s, p_s in zip(
            gt_labels.shape[-ndim:], pred_labels.shape[-ndim:])]):
        gt_labels = crop(gt_labels, pred_labels.shape[-ndim:])
    else:
        raise RuntimeError(
            "gt is bigger in some spatial dims, pred in others, unable to "
            "crop to same shape")

    logger.debug("gt shape cropped %s", gt_labels.shape)
    logger.debug("pred shape cropped %s", pred_labels.shape)

    return gt_labels, pred_labels


def binary_masks_to_uniq_ids(labels):
    """if label array has binary masks per channel, relabel to unique ids"""
    assert np.max(labels) <= 1, "labels is not a binary mask"
    labels = labels.astype(np.min_scalar_type(labels.shape[0]))
    for i in range(labels.shape[0]):
        labels[i] = labels[i] * (i + 1)

    return labels


def remove_empty_channels(labels):
    if labels.shape[0] == 1:
        return labels

    tmp_labels = []
    for i in range(labels.shape[0]):
        if np.max(labels[i]) > 0:
            tmp_labels.append(labels[i])

    return np.array(tmp_labels)


# todo: assert for integers (no floats)
def check_fix_and_unify_ids(
        gt_labels, pred_labels, remove_small_components, foreground_only,
        dim_insts=[]):
    """unify prediction and gt labelling styles 

    Note
    ----
    labelling should be
     - sequential, starting at 1 (background = 0)
     - if stored in multiple channels
       - masks in each channel should still have unique id
    """
    assert np.min(gt_labels) >= 0, "found negative ids in gt label array"
    assert np.min(pred_labels) >= 0, "found negative ids in pred label array"
    if np.max(gt_labels) == 0:
        logger.warning("gt label array is empty")
    if np.max(pred_labels) == 0:
        logger.warning("pred label array is empty")

    # optional: remove small components
    if remove_small_components is not None and remove_small_components > 0:
        logger.info(
            "remove small components with size < %i", remove_small_components)
        pred_labels = filter_components(pred_labels, remove_small_components)

    # optional:if foreground_only, remove all predictions within gt background
    # (rarely useful) 
    if foreground_only:
        if (pred_labels.shape[0] == 1 and
            np.all(
                [ps == gs
                 for ps, gs in zip(pred_labels.shape, gt_labels.shape)])):
            pred_labels[gt_labels==0] = 0
        else:
            gt_foreground_mask = np.any(gt_labels > 0, axis=0)
            pred_labels[:, ~gt_foreground_mask] = 0
    # after filtering, some channels might be empty
    pred_labels = remove_empty_channels(pred_labels)
    gt_labels = remove_empty_channels(gt_labels)

    # if each channel is a binary mask, still assign a unique id to each
    if np.max(pred_labels) == 1:
        pred_labels = binary_masks_to_uniq_ids(pred_labels)
    if np.max(gt_labels) == 1:
        gt_labels = binary_masks_to_uniq_ids(gt_labels)

    # relabel labels sequentially
    offset = 1
    for i in range(pred_labels.shape[0]):
        pred_labels[i], _, _ = relabel_sequential(
            pred_labels[i].astype(int), offset)
        offset = np.max(pred_labels[i]) + 1
    offset = 1
    fw_map = None
    for i in range(gt_labels.shape[0]):
        gt_labels[i], c_fw_map, _ = relabel_sequential(
            gt_labels[i].astype(int), offset)
        if fw_map is None:
            fw_map = c_fw_map
        else:
            ids = np.isin(c_fw_map.in_values, fw_map.in_values, invert=True)
            if np.any(ids):
                fw_map.in_values = np.concatenate(
                        [fw_map.in_values, c_fw_map.in_values[ids]])
                fw_map.out_values = np.concatenate(
                        [fw_map.out_values, c_fw_map.out_values[ids]])
        offset = np.max(gt_labels[i]) + 1

    if len(dim_insts) > 0:
        dim_insts_rel = list(fw_map[np.array(dim_insts)])
        return gt_labels, pred_labels, dim_insts_rel
    else:
        return gt_labels, pred_labels


def read_file(infn, key, read_dim=False):
    """read image/volume in hdf, tif and zarr format"""
    dim_insts = []
    if infn.endswith(".hdf"):
        with h5py.File(infn, 'r') as f:
            volume = np.array(f[key])
    elif infn.endswith(".tif") or infn.endswith(".tiff") or \
         infn.endswith(".TIF") or infn.endswith(".TIFF"):
        volume = tifffile.imread(infn)
    elif infn.endswith(".zarr"):
        try:
            f = zarr.open(infn, mode='r')
        except (zarr.errors.NodeNotFoundError, FileNotFoundError) as e:
            logger.info("File %s not found!", infn)
            raise e
        volume = np.array(f[key])
        if read_dim:
            if "dim_neurons" in f[key].attrs.keys():
                dim_insts = f[key].attrs["dim_neurons"]
    else:
        raise NotImplementedError("invalid file format %s", infn)
    if read_dim:
        return volume, dim_insts
    else:
        return volume


def get_output_name(
        out_dir, res_file, res_key, suffix, localization_criterion,
        assignment_strategy, remove_small_components):
    """constructs output file name based on used eval strategy"""
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

    return outFn

def replace(array, old_values, new_values):
    """fast function to replace set of values in array with new values"""
    values_map = np.arange(int(array.max() + 1), dtype=new_values.dtype)
    values_map[old_values] = new_values

    return values_map[array]


def filter_components(volume, thresh):
    """remove instances smaller than `thresh` pixels"""
    labels, counts = np.unique(volume, return_counts=True)
    small_labels = labels[counts <= thresh]

    volume = replace(
        volume,
        np.array(small_labels),
        np.array([0] * len(small_labels))
    )
    return volume

from itertools import count
import heapq

from itertools import count
import heapq

class LazyHeap:
    """A min-heap with lazy invalidation.

    - push(key, priority): inserts/upserts an item identified by 'key' with given priority.
    - pop(): returns (priority, key) for the next *live* item; skips stale/invalid entries.
    - faulty_element(key): marks the current version of 'key' as invalid so queued entries are ignored.
    - empty(): returns True iff there is no *live* item left (also lazily discards stale head items).
    """
    def __init__(self):
        self._heap = []                 # entries: (priority, tie_breaker, version, key)
        self._version = {}              # key: latest version int; -1 means invalidated
        self._tb = count()              # tie-breaker to keep heap stable

    def push(self, key, priority):
        v = self._version.get(key, 0) + 1
        self._version[key] = v
        heapq.heappush(self._heap, (priority, next(self._tb), v, key))

    def faulty_element(self, key):
        # Mark the current version as invalid; queued entries with that version will be skipped.
        if key in self._version:
            self._version[key] = -1

    def discard_stale_head(self):
        # Drop stale/invalid head entries so empty() can work correctly.
        while self._heap:
            prio, _, v, key = self._heap[0]
            if self._version.get(key) == v:
                # live head
                return
            # if stale/invalid, keep removing and going
            heapq.heappop(self._heap)

    def pop(self):
        while self._heap:
            prio, _, v, key = heapq.heappop(self._heap)
            if self._version.get(key) == v:
                # consume live entry
                self._version.pop(key, None)
                return prio, key
            # else stale/invalid, continue loop
        raise KeyError("pop from empty priority queue")

    def empty(self):
        self.discard_stale_head()
        return not self._heap
