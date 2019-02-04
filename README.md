Evaluation scripts for instance segmentation
=======================================================

usage:
-------
- with tif-files:
  python evaluate.py <pred-file> <gt-file>
- with hdf-files:
  python evaluate.py <pred-file> <gt-file> <hdf-volume-name> <option-fl-suffix>

output:
--------
- output is written to terminal/txt-file (and just the numbers to csv)
- dice
- iou
- seg (0 if pred covers less than 50% of gt instance, otherwise iou)
- #ns: non-split, one prediction for multiple gt instances
- #fp: predicted cell for non existing ground truth instances
- #tpGT: number of gt instances with pred instance (unique due to seg test)
- #tpP: number of pred instances with exactly one gt instance
- #fn: no predicted cell for ground truth cell
- average precision: tpP/(tpP+fn+fp)

(Pred -> GT: for each predicted instance, ..
GT/Ref -> Pred: for each ground truth instance, ..)
