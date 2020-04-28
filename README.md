Evaluation scripts for instance segmentation
=======================================================

Developed for nuclei instances segmentation in 2d and 3d.
Computes average precision (AP = TP/(TP+FP+FN))

usage:
-------

``` shell
    python evaluate.py --res_file <pred-file> --res_key <hdf-key> --gt_file <gt-file> --gt_key <hdf-key> --out_dir output --background 0
```
(`--res_key` and `--gt_key` only required for hdf or zarr input)

By default hungarian matching is used to compute the matches of prediction and ground truth (can be disabled with `--no_use_linear_sum_assignment`.<br>
AP is computed for multiple thresholds.

output:
--------
- evaluation metrics are written to toml-file and returned as dict
