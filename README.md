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
It is currently use by importing the module and calling the `evaluate_file` function.
It will be updated to be callable from the command line.

output:
--------
- evaluation metrics are written to toml-file and returned as dict
