Evaluation scripts for instance segmentation
=======================================================

Developed for nuclei instances segmentation in 2d and 3d.

usage:
-------

``` shell
    python evaluate.py --res_file <pred-file> --gt_file <gt-file> --out_dir output --background 0
```

add `--res_key images/pred_affs` for hdf input

output:
--------
- evaluation metrics are written to toml-file and returned as dict
