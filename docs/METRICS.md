# FISBe Metrics Reference

This document details all evaluation metrics computed by the pipeline.

## 1. Official FlyLight Benchmark Metrics (`--app flylight`)
These metrics determine the **FISBe Leaderboard** ranking. They are designed for long-range, thin filamentous structures.

### Primary Ranking Score ($S$)
The single scalar used to rank methods.
$$S = 0.5 \cdot \text{avF1} + 0.5 \cdot C$$

### Website leaderboard column mapping
| Website column | Meaning | Pipeline key |
|---|---|---|
| S | $0.5 \cdot \text{avF1} + 0.5 \cdot C$ | `general.aggregate_score` |
| avF1 | mean F1 over th=0.1..0.9 | `confusion_matrix.avFscore19` |
| C | avg GT coverage (union-of-preds, clRecall) | `general.avg_gt_skel_coverage` |
| clDiceTP | mean clDice of TP matches at th=0.5 | `general.avg_TP_05_cldice` |
| tp | (#TP matches at th=0.5) / (#GT) | `general.TP_05_rel` |
| FS | false splits count | `general.FS` |
| FM | false merges count | `general.FM` |

---

### Official Ranking Metrics Definition
| Metric Key | Definition | Matching Strategy |
| :--- | :--- | :--- |
| **`avFscore19`** (avF1) | Mean F1 score averaged over clDice thresholds 0.1 to 0.9. | Greedy 1-to-1 |
| **`avg_gt_skel_coverage`** (C) | Average GT centerline coverage computed per GT using the union of matched predictions (see below). | One-to-Many |

#### Coverage (C) definition (official protocol)
**Localization:** compute clPrecision for all (pred, gt) pairs.  
**Matching (one-to-many):** assign each predicted instance to the GT instance with the highest clPrecision score (a GT can receive multiple predictions).  
**Computation:** for each GT, compute clRecall using the union of all predictions assigned to that GT (to avoid double-counting in overlaps), then average over all GT instances.

#### avF1 computation details (official protocol)
For each threshold th ∈ {0.1, 0.2, …, 0.9} after greedy 1-to-1 matching by clDice:
- **TP**: matched (pred, gt) with clDice > th
- **FP**: unmatched predicted instances
- **FN**: unmatched GT instances

Compute F1 = 2TP / (2TP + FP + FN) aggregated across all images, then average across thresholds.

### Topology Error Attribution
Explicit counts of topological errors specific to filamentous structures.

| Metric Key | Definition | Matching Strategy |
| :--- | :--- | :--- |
| **`FS`** (False Splits) | $\sum_{gt} \max(0, N_{\text{assigned\_pred}} - 1)$ | Many-to-Many (Consumption) |
| **`FM`** (False Merges) | $\sum_{pred} \max(0, N_{\text{assigned\_gt}} - 1)$ | Many-to-Many (Consumption) |

### Localization Criteria
**clDice (Benchmark Default)**
Centerline Dice evaluates the agreement between skeletonized structures. It is the robust choice for thin filamentous objects where standard pixel-level IoU is mathematically unstable due to small boundary variations.
- **clPrecision**: Fraction of the predicted skeleton lying within the ground truth mask.
- **clRecall**: Fraction of the ground truth skeleton covered by the predicted mask.
- **clDice**: Harmonic mean of centerline precision and recall.

### Matching Strategies
**1. Greedy One-to-One Matching**
Pairs are sorted by localization score and matched if both remain unassigned.
*Used for: TP/FP/FN counts, F1-scores, and Precision/Recall curves.*

**2. One-to-Many Matching**
Assigns each predicted instance to the single ground truth instance with which it shares the highest clPrecision.
*Used for: Average GT Coverage (C).*

**3. Greedy Many-to-Many Matching with Consumption**
Iteratively matches GT and predictions while "consuming" pixels—removing them from availability once matched. This prevents double-counting of overlapping structures.
*Used for: False Splits (FS) and False Merges (FM).*

---

## 2. Standard Instance Metrics (Reported per Threshold)
These metrics are computed for every threshold (e.g., `th_0_5`) using **Greedy 1-to-1 matching**.

| Metric Key | Description |
| :--- | :--- |
| **`AP_TP`** | True Positives count. |
| **`AP_FP`** | False Positives count (filtered in `--partly` mode). |
| **`AP_FN`** | False Negatives count. |
| **`precision`** | $TP / (TP + FP)$ |
| **`recall`** | $TP / (TP + FN)$ |
| **`fscore`** | Harmonic mean of precision and recall. |
| **`AP`** | Precision $\times$ Recall (at specific threshold). |

> **Note on Averages:** The code also aggregates these across thresholds:
> * **`avAP`** (or `avAP59`): Mean AP over thresholds **0.5 to 0.9**.
> * **`avFscore`** (or `avFscore19`): Mean F1 over thresholds **0.1 to 0.9**.

---

## 3. Diagnostic & Sub-group Metrics
The code calculates these additional metrics for deep-dive analysis. They do not affect the primary ranking score.

### General Stats
| Metric Key | Description |
| :--- | :--- |
| **`Num GT`** | Total number of ground truth instances. |
| **`Num Pred`** | Total number of predicted instances. |
| **`TP_05`** | Count of True Positives at threshold 0.5. |
| **`TP_05_rel`** | Fraction of GT instances detected at threshold 0.5 ($TP_{0.5} / N_{GT}$). |

### Quality of Matches
> **Note:** `avg_TP_05_cldice` corresponds to the leaderboard metric **clDiceTP**, and `TP_05_rel` corresponds to **tp**.

| Metric Key | Description |
| :--- | :--- |
| **`avg_TP_05_cldice`** | Average `clDice` score of matched True Positives (at $th=0.5$). Indicates segmentation quality of detected objects. |
| **`TP_05_cldice`** | List of individual `clDice` scores for all TP pairs. |

### Challenging Subsets (Dim & Overlapping)
Metrics evaluated only on specific subsets of Ground Truth neurons.

| Metric Key | Description |
| :--- | :--- |
| **`GT_dim`** | Total number of dim (low contrast) GT neurons. |
| **`TP_05_dim`** | Number of dim neurons correctly detected ($clDice > 0.5$). |
| **`TP_05_rel_dim`** | Fraction of dim neurons detected. |
| **`avg_gt_cov_dim`** | Coverage ($C$) score computed only on dim neurons. |
| **`GT_overlap`** | Total number of GT neurons involved in overlaps. |
| **`TP_05_overlap`** | Number of overlapping neurons correctly detected ($clDice > 0.5$). |
| **`TP_05_rel_overlap`** | Fraction of overlapping neurons detected. |
| **`avg_gt_cov_overlap`** | Coverage ($C$) score computed only on overlapping neurons. |

---

## 4. Evaluation Logic

### Localization Criterion
* **Default:** `clDice` (Centerline Dice). Robust to boundary variations in thin structures.
* **Options:** `iou` (Intersection over Union) is available via API but not used for the benchmark.

### Matching Strategies
* **Greedy 1-to-1:** Used for all standard detection metrics (TP, FP, F1).
* **One-to-Many:** Used for Coverage ($C$). Assigns a prediction to the GT it covers best.
* **Many-to-Many (Consumption):** Used for FS/FM. "Consumes" pixels to prevent double-counting in overlaps.

### Partly Labeled Data (`--partly`)
Handles sparse Ground Truth where not all neurons are annotated.
* **Logic:** Unmatched predictions are counted as False Positives (**FP**) *only* if their best match is a **Foreground** GT instance.
* **Background:** Predictions matching background (unlabeled regions) are ignored.
