Evaluation scripts for instance segmentation
=======================================================

*currently under construction*

Developed for instances segmentation in 2d and 3d.
Computes a number of evaluation metrics (e.g., AP, F1, coverage, precision, recall).
Provides visualization of the errors.

installation:
-------------
The recommended way is to install it into your conda/python virtual environment.

``` shell
conda activate <<your-env-name>>
git clone https://github.com/Kainmueller-Lab/evaluate-instance-segmentation
cd evaluate-instance-segmentation
pip install -e .
```

usage:
-------
You can either import the module and call the `evaluate_file` function,
or you can use the command line, similar to here:

```
evalinstseg \
--res_file tests/pred/R14A02-20180905_65_A6.hdf \
--res_key volumes/gmm_label_cleaned \
--gt_file tests/gt/R14A02-20180905_65_A6.zarr \
--gt_key volumes/gt_instances \
--out_dir tests/results \
--app flylight
```

output:
--------
- evaluation metrics are written to toml-file and returned as dict
