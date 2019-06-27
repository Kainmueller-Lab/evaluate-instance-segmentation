Evaluation scripts for instance segmentation
=======================================================

usage:
-------
- with tif-files:

``` shell
    python evaluate.py --res_file <pred-file> --gt_file <gt-file> --out_dir output --background 0
```

- with hdf-files:

``` shell
    python evaluate.py --res_file <pred-file> --res_key images/pred_affs --gt_file <gt-file> --gt_key images/instances --out_dir output --background 0
```

output:
--------
- evaluation metrics are written to toml-file
  - dice score
  - iou (intersection over union)
  - seg (0 if pred covers less than 50% of gt instance, otherwise iou)
  - #ns: non-split, one prediction for multiple gt instances
  - #fp: predicted cell for non existing ground truth instances
  - #tpGT: number of gt instances with pred instance (unique due to seg test)
  - #tpP: number of pred instances with exactly one gt instance
  - #fn: no predicted cell for ground truth cell
  - average precision: tpP/(tpP+fn+fp)

(Pred -> GT: for each predicted instance, ..
GT/Ref -> Pred: for each ground truth instance, ..)
