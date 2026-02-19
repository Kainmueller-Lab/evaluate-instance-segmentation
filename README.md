<h1 align="center">FISBe: A real-world benchmark dataset for instance segmentation of long-range thin filamentous structures
</h1>

![Alt text](assets/Cover_Image.png)

## About

This is the official implementation of the **FISBe (FlyLight Instance Segmentation Benchmark)** evaluation pipeline. It is the first publicly available multi-neuron light microscopy dataset with pixel-wise annotations.

**Download the dataset:** [https://kainmueller-lab.github.io/fisbe/](https://kainmueller-lab.github.io/fisbe/)

The benchmark supports 2D and 3D segmentations and computes a wide range of commonly used evaluation metrics (e.g., AP, F1, coverage). Crucially, it provides specialized error attribution for topological errors (False Merges, False Splits) relevant to filamentous structures.

### Features
- **Standard Metrics:** AP, F1, Precision, Recall.
- **FISBe Metrics:** Greedy many-to-many matching for False Merges (FM) and False Splits (FS).
- **Flexibility:** Supports HDF5 (`.hdf`, `.h5`) and Zarr (`.zarr`) files.
- **Modes:** Run on single files, entire folders, or in stability analysis mode.
- **Partly Labeled Data:** Robust evaluation ignoring background conflicts for sparse Ground Truth.

---

## Installation

The recommended way to install is using `uv` (fastest) or `micromamba`.

### Option 1: Using `uv` (Fastest)

```bash
# 1. Install uv (if not installed)
pip install uv

# 2. Clone and install
git clone https://github.com/Kainmueller-Lab/evaluate-instance-segmentation.git
cd evaluate-instance-segmentation
uv venv
uv pip install -e .
```

### Option 2: Using `micromamba` or `conda`

```bash
micromamba create -n evalinstseg python=3.10
micromamba activate evalinstseg

git clone https://github.com/Kainmueller-Lab/evaluate-instance-segmentation.git
cd evaluate-instance-segmentation
pip install -e .
```

## Usage: Command Line (CLI)
The `evalinstseg` command is automatically available after installation.

### 1. Evaluate a Single File
```bash
evalinstseg \
  --res_file tests/pred/R14A02-20180905.hdf \
  --res_key volumes/labels \
  --gt_file tests/gt/R14A02-20180905.zarr \
  --gt_key volumes/gt_instances \
  --out_dir tests/results \
  --app flylight
```

### 2. Evaluate an Entire Folder
If you provide a directory path to `--res_file`, the tool will look for matching Ground Truth files in the `--gt_file` folder. Files are matched by name.

```bash
evalinstseg \
  --res_file /path/to/predictions_folder \
  --res_key volumes/labels \
  --gt_file /path/to/ground_truth_folder \
  --gt_key volumes/gt_instances \
  --out_dir /path/to/output_folder \
  --app flylight
```

### 3. Stability & Robustness Mode
Compute the **Mean ± Std** of metrics across exactly 3 different training runs (e.g., different random seeds).

```bash
evalinstseg \
  --stability_mode \
  --run_dirs experiments/seed1 experiments/seed2 experiments/seed3 \
  --gt_file data/ground_truth_folder \
  --out_dir results/stability_report \
  --app flylight
```

**Requirements:**
- `--run_dirs`: Provide exactly 3 folders.
- `--gt_file`: The folder containing Ground Truth files (filenames must match predictions).

### 4. Partly Labeled Data
If your ground truth is sparse (not fully dense), use the `--partly` flag. T

## Usage: Python Package
You can integrate the benchmark directly into your Python scripts or notebooks.

### Evaluate a File
```python
from evalinstseg import evaluate_file

# Run evaluation
metrics = evaluate_file(
    res_file="tests/pred/sample_01.hdf",
    gt_file="tests/gt/sample_01.zarr",
    res_key="volumes/labels",
    gt_key="volumes/gt_instances",
    out_dir="output_folder",
    ndim=3,
    app="flylight",  # Applies default FISBe config
    partly=False     # Set True for sparse GT
)

# Access metrics directly
print("AP:", metrics['confusion_matrix']['avAP'])
print("False Merges:", metrics['general']['FM'])
```

### Evaluate Raw Numpy Arrays
If you already have the arrays loaded in memory:

```python
import numpy as np
from evalinstseg import evaluate_volume

pred_array = np.load(...) # Shape: (Z, Y, X)
gt_array = np.load(...)

metrics = evaluate_volume(
    gt_labels=gt_array,
    pred_labels=pred_array,
    ndim=3,
    outFn="output_path_prefix",
    localization_criterion="cldice",  # or 'iou'
    assignment_strategy="greedy",
    add_general_metrics=["false_merge", "false_split"]
)
```
### 4. Partly Labeled Data (`--partly`)
Some samples contain sparse / incomplete GT annotations. In this setting, counting all unmatched predictions as false positives is not meaningful.

When `--partly` is enabled, we approximate FP by counting only **unmatched predictions whose best match is a foreground GT instance** (based on the localization matrix used for evaluation, e.g. clPrecision for `cldice`).  
Unmatched predictions whose best match is **background** are ignored.

Concretely, we compute for each unmatched prediction the index of the GT label with maximal overlap score; it is counted as FP only if that index is > 0 (foreground), not 0 (background).

---

## Metrics Explanation

### 1. Standard Instance Metrics (TP/FP/FN, F-score, AP proxy)
These metrics are computed from a **one-to-one matching** between GT and prediction instances (Hungarian or greedy), using a chosen localization criterion (default for FlyLight is `cldice`).

- **TP**: matched pairs above threshold  
- **FP**: unmatched predictions (or, in `--partly`, only those whose best match is foreground)  
- **FN**: unmatched GT instances  
- **precision** = TP / (TP + FP)  
- **recall** = TP / (TP + FN)  
- **fscore** = 2 * precision * recall / (precision + recall)  
- **AP**: we report a simple AP proxy `precision × recall` at each threshold and average it across thresholds (this is not COCO-style AP).

### 2. FISBe Error Attribution (False Splits / False Merges)
False splits (FS) and false merges (FM) aim to quantify **instance topology errors** for long-range thin filamentous structures.

We compute FS/FM using **greedy many-to-many matching with consumption**:
- Candidate GT–Pred pairs above threshold are processed in descending score order.
- After selecting a match, we update “available” pixels so that already explained structure is not matched again.
- FS counts when one GT is explained by multiple preds (excess preds per GT).
- FM counts when one pred explains multiple GTs (excess GTs per pred).

This produces an explicit attribution of split/merge errors rather than only TP/FP/FN.

### Metric Definitions

#### Instance-Level (per threshold)
| Metric | Description |
| :--- | :--- |
| **AP_TP** | True Positives (1-to-1 match) |
| **AP_FP** | False Positives (unmatched preds; in `--partly`: only unmatched preds whose best match is foreground) |
| **AP_FN** | False Negatives (unmatched GT) |
| **precision** | TP / (TP + FP) |
| **recall** | TP / (TP + FN) |
| **fscore** | Harmonic mean of precision and recall |

#### Global / FISBe
| Metric | Description |
| :--- | :--- |
| **avAP** | Mean AP proxy across thresholds ≥ 0.5 |
| **FM** | False Merges (many-to-many matching with consumption) |
| **FS** | False Splits (many-to-many matching with consumption) |
| **avg_gt_skel_coverage** | Mean skeleton coverage of GT instances by associated predictions (association via best-match mapping) |
