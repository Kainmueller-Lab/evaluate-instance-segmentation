<h1 align="center">FISBe: A real-world benchmark dataset for instance segmentation of long-range thin filamentous structures
</h1>

![Alt text](assets/Cover_Image.png)

## About

⚠️ *Currently under construction.*

This is the official implementation of the **FISBe (FlyLight Instance Segmentation Benchmark)** evaluation pipeline. It is the first publicly available multi-neuron light microscopy dataset with pixel-wise annotations.

**Download the dataset:** [https://kainmueller-lab.github.io/fisbe/](https://kainmueller-lab.github.io/fisbe/)

The benchmark supports 2D and 3D segmentations and computes a wide range of commonly used evaluation metrics (e.g., AP, F1, coverage). Crucially, it provides specialized error attribution for topological errors (False Merges, False Splits) relevant to filamentous structures.

### Key Features
- **Official Protocol:** Implements the exact ranking score ($S$) and matching logic defined in the FISBe paper.
- **Topology-Aware:** Uses skeleton-based localization (`clDice`) to handle thin structures robustly.
- **Error Attribution:** Explicitly quantifies False Merges (FM) and False Splits (FS) via many-to-many matching.
- **Flexibility:** Supports HDF5 (`.hdf`, `.h5`) and Zarr (`.zarr`) files.
- **Modes:** Single file, folder evaluation, or 3x stability analysis.
- **Partly Labeled Support:** Robust evaluation that ignores background conflicts for sparse Ground Truth.

---

## Installation

The recommended way to install is using `uv` (fastest) or `micromamba`.

### Option 1: Using `uv` (Fastest)

```bash
pip install uv
git clone https://github.com/Kainmueller-Lab/evaluate-instance-segmentation.git
cd evaluate-instance-segmentation
uv venv
uv pip install -e .
```

### Option 2: Using micromamba or conda

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
  --res_file tests/pred/sample_01.hdf \
  --res_key volumes/gmm_label_cleaned \
  --gt_file tests/gt/sample_01.zarr \
  --gt_key volumes/gt_instances \
  --split_file assets/sample_list_per_split.txt \
  --out_dir tests/results \
  --app flylight
```

### 2. Evaluate an Entire Folder
If you provide a directory path to `--res_file`, the tool will look for matching Ground Truth files in the `--gt_file` folder. Files are matched by name.

```bash
evalinstseg \
  --res_file /path/to/predictions_folder \
  --res_key volumes/gmm_label_cleaned \
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
If your ground truth is sparse (not fully dense), use the `--partly` flag. See the **Partly Labeled Data Mode** section for details on how False Positives are handled.

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

## FISBe Benchmark Protocol
For a complete reference of all calculated metrics, see [docs/METRICS.md](docs/METRICS.md).
> **Note:** Some output keys use internal names; see the documentation for the exact mapping to website/leaderboard columns.

### Official FlyLight Configuration (`--app flylight`)
The `flylight` preset implements the specific metrics described in the FISBe paper for evaluating long-range thin filamentous neuronal structures.

**Primary Ranking Score ($S$)**
The single scalar used to rank methods on the leaderboard:
$$S = 0.5 \cdot \text{avF1} + 0.5 \cdot C$$

**Key Metrics**
- **avF1**: Average F1 score across clDice thresholds.
- **C (Coverage)**: Average ground truth skeleton coverage (assigned via max clPrecision; score via clRecall on union of matches).  
- **clDiceTP**: Average clDice score of matched True Positives (at threshold 0.5).
- **tp**: Relative number of True Positives at threshold 0.5 ($TP_{0.5} / N_{GT}$).
- **FS (False Splits)**: $\sum_{gt} \max(0, N_{\text{assigned\_pred}} - 1)$
- **FM (False Merges)**: $\sum_{pred} \max(0, N_{\text{assigned\_gt}} - 1)$

### Partly Labeled Data Mode (`--partly`)
FISBe includes 71 partly labeled images where only a subset of neurons is annotated.
- **Logic**: Unmatched predictions are only counted as False Positives if they match a **Foreground GT instance**.
- **Background Exclusion**: Predictions matching background (unlabeled regions) are ignored.

## Output Structure
Metrics returned by the API or saved to disk are grouped into category-specific dictionaries:

```python
metrics["confusion_matrix"]
├── TP / FP / FN         # Counts across all images
├── precision / recall   # Standard detection metrics
└── avAP                 # Mean precision × recall proxy

metrics["general"]
├── aggregate_score      # S (Official Ranking Score)
├── avg_gt_skel_coverage # C (Coverage)
├── FM                   # Global False Merge count
└── FS                   # Global False Split count

metrics["curves"]
└── F1_0.1 … F1_0.9     # Per-threshold performance
```