# Explainable Multimodal Machine Learning for rTMS Treatment Risk Assessment in Alzheimer's Disease

A reproducible multimodal biomarker-discovery and risk-stratification pipeline for repetitive Transcranial Magnetic Stimulation (rTMS) in mild-to-moderate Alzheimer's Disease (AD). The pipeline combines stability-driven feature selection, a cost-sensitive XGBoost classifier, and multi-protocol SHAP agreement analysis to deliver safety-prioritized predictions and biologically interpretable biomarkers under small-sample constraints (N = 58).

This README accompanies the manuscript **"Explainable Multimodal Machine Learning for rTMS Treatment Risk Assessment in Alzheimer's Disease"** and documents every experiment, section, and artifact produced by `rtms_biomarker_analysis.ipynb`.

---

## Table of Contents

- [1. Project Summary](#1-project-summary)
- [2. Clinical & Statistical Context](#2-clinical--statistical-context)
- [3. Dataset](#3-dataset)
- [4. Environment & Installation](#4-environment--installation)
- [5. Repository Layout](#5-repository-layout)
- [6. Notebook Structure (Top-Down)](#6-notebook-structure-top-down)
  - [Section 1 — Environment Setup](#section-1--environment-setup)
  - [Section 2 — Data Loading](#section-2--data-loading)
  - [Section 3 — Data Preparation](#section-3--data-preparation)
  - [Section 4 — Exploratory Data Analysis](#section-4--exploratory-data-analysis)
  - [Section 5 — Statistical Feature Analysis](#section-5--statistical-feature-analysis)
  - [Section 6 — Machine Learning Modeling Framework](#section-6--machine-learning-modeling-framework)
  - [Section 7 — Results Visualization & Interpretation](#section-7--results-visualization--interpretation)
  - [Section 8 — Model Selection, Validation & Explainability](#section-8--model-selection-validation--explainability)
  - [Section 9 — Appendix: Data Scaling Pipeline](#section-9--appendix-data-scaling-pipeline)
- [7. Five Validation Protocols at a Glance](#7-five-validation-protocols-at-a-glance)
- [8. Headline Findings](#8-headline-findings)
- [9. Artifacts & Output Directories](#9-artifacts--output-directories)
- [10. Reproducing the Results](#10-reproducing-the-results)
- [11. File-by-File Reference](#11-file-by-file-reference)
- [12. Citation](#12-citation)
- [13. License & Acknowledgments](#13-license--acknowledgments)

---

## 1. Project Summary

This codebase implements a three-stage *triangulated design* for pre-intervention identification of rTMS treatment decliners in AD:

| Stage | Goal | Method | Primary Artifact |
|---|---|---|---|
| **1 — Robust Feature Selection** | Identify stable multivariate biomarkers | Multi-seed Sequential Forward Selection (SFS) with an RBF-SVM evaluator and stratified 3-fold inner CV | `new_analysis_sfs_evidence_synthesis/sfs/` |
| **2 — Safety-Prioritized Diagnosis** | Classifier that minimizes missed decliners (FN) | Cost-sensitive XGBoost (decision stumps, `scale_pos_weight = 12`, threshold `T = 0.30`) | `xgb_main_v11_diag_v5/` |
| **3 — Cross-Protocol Explainability** | Confirm that learned decision rules are biologically interpretable and protocol-agnostic | Multi-seed SHAP under 5 complementary training protocols + Jaccard/Spearman agreement + SFS↔SHAP triangulation | `shap_xgb_*/`, `cross_protocol_comparison/` |

**Headline biomarkers** converging across all three stages: **global white-matter-hyperintensity (WMH) burden** (decline-driving) and **preserved DLPFC gray-matter (GM) asymmetry** (protective).

---

## 2. Clinical & Statistical Context

- **Intervention**: 20 Hz rTMS at 100 % resting motor threshold, applied serially to left and right DLPFC.
- **Cohort source**: active arm of a multisite, double-blind, placebo-controlled trial in mild-to-moderate AD (`N = 105` enrolled; `N = 58` with complete Week-5 assessments and valid TIV measurements).
- **Outcome**: ΔADAS-Cog between baseline and immediate post-intervention assessment, binarized with a strict one-unit threshold:
  - **Improved (Class 0)**: ΔADAS-Cog ≤ −1
  - **Declined (Class 1)**: ΔADAS-Cog ≥ +1
  - *No-change* cases (−1 < Δ < +1) are excluded to maximize class separability.
- **Class balance**: `46 / 58` improved (79.3 %) vs `12 / 58` declined (20.7 %). Severely imbalanced; all stratified schemes preserve this ratio.
- **Clinical priority**: *false negatives are clinically worse than false positives* (administering rTMS to a future decliner is more harmful than withholding). All validation protocols are tuned to minimize FN (maximize Recall) under a fixed operating threshold `T = 0.30`.

---

## 3. Dataset

A 16-feature multimodal vector per participant, grouped into three categories:

| Category | Features |
|---|---|
| **Clinical & Demographic** | `Age`, `Gender`, `CDR`, `MoCA`, `CSDD`, `HIS_2_Threshold` (binary: HIS > 2 vs ≤ 2) |
| **Structural Integrity** (TIV-normalized) | `V_R_DLPFC_GM`, `V_L_DLPFC_GM`, `V_R_DLPFC_WM`, `V_L_DLPFC_WM` |
| **Structural Symmetry** | `GM_Asymmetry_Ind`, `WM_Asymmetry_Ind` *(|L − R| / (0.5·(L + R)))* |
| **Vascular Health (WMH)** (TIV-normalized) | `WMH` (global), `WMH_FL` (frontal lobe), `WMH_R_DLPFC_mm`, `WMH_L_DLPFC_mm` |

**Preprocessing**: T1-weighted MRI → CAT12 within SPM12 (bias-field correction, skull-stripping, MNI normalization). Neuroimaging volumes are normalized to Total Intracranial Volume (TIV).

**Data file expected**: `Combined_v6_Updated.xlsx` (proprietary trial data, not redistributed). On Kaggle, attach via *Add data → Your datasets*. On local runs, place the file at the path the auto-detection block in Section 2 discovers.

---

## 4. Environment & Installation

The notebook auto-detects Windows, Linux, Colab, and Kaggle environments and sets `PROJECT_DIR` + `DATASET_PATH` automatically. No manual path edits are needed.

```bash
conda create -n p python=3.11
conda activate p
pip install -r requirements.txt
```

GPU acceleration is optional; XGBoost and the LOOCV-SHAP protocol will use CUDA if `device="cuda"` is set and a compatible GPU is present. Fall back to `device="cpu"` otherwise.

**Python**: 3.11+ recommended.
**Core libraries**: `numpy`, `pandas`, `scipy`, `scikit-learn` (≥ 1.6.1), `xgboost`, `shap`, `matplotlib`, `seaborn`, `joblib`, `openpyxl`.

---

## 5. Repository Layout

```
.
├── rtms_biomarker_analysis.ipynb      # Main analysis notebook (all 9 sections)
├── manuscript.txt                     # LaTeX source of the EMBC manuscript
├── README.md                          # This document
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Excludes large/derived artifacts from git
│
├── new_analysis_sfs_evidence_synthesis/
│   ├── sfs/                           # SFS per-seed rankings, summaries, CSVs
│   ├── feature_directionality/        # Cohen's d + signed-AUC evidence
│   ├── evidence_synthesis/            # Multi-method evidence synthesis reports
│   └── XGB Diagnostic/                # Legacy diagnostic XGBoost plots
│
├── l1_main_v7_diag_v26/               # L1-Logistic Regression training runs
├── l1_main_v16_diag_v26/
├── rf_main_v2_diag_v6/                # Random Forest training runs
├── xgb_main_v11_diag_v5/              # XGBoost (headline model) training runs
├── combined_l1_rf_xgb_1st/            # Combined model-comparison outputs
│
├── shap_xgb_full_100seeds/            # Multi-seed SHAP: full-data re-fit, 100 seeds
├── shap_xgb_full_1000seeds/           #   …                                1000 seeds
├── shap_xgb_holdout_100seeds/         # Multi-seed SHAP: fixed 80/20 (seed=42)
├── shap_xgb_holdout_1000seeds/
├── shap_xgb_holdout_varysplit_100seeds/   # MCCV (varying splits)
├── shap_xgb_holdout_varysplit_1000seeds/
├── shap_xgb_loocv_100seeds/           # LOOCV-SHAP
├── shap_xgb_loocv_1000seeds/
├── shap_xgb_OVERFIT_CEILING_1000seeds/ # Overfit-ceiling (noise reference)
├── cross_protocol_comparison/        # Jaccard, Spearman, top-K agreement tables
│
├── confusion_matrix_xgb_safety_prioritized.{png,pdf,eps}   # Headline manuscript figures
└── confusion_matrix_xgb_diag_ieee.png
```

> **Note**: SHAP experiment folders and classifier training folders are produced by running the notebook; they are gitignored by default (see `.gitignore`).

---

## 6. Notebook Structure (Top-Down)

The notebook is organized into 9 top-level sections. Cell numbering below refers to the logical sections visible in the notebook outline.

### Section 1 — Environment Setup
- System configuration, disk-space probe, cross-platform detection (Windows / Linux / Colab / Kaggle).
- Defines `PROJECT_DIR` and `DATASET_PATH` for downstream artifact writes.

### Section 2 — Data Loading
- **2.1 Verify Data Path** — locates `Combined_v6_Updated.xlsx` in the detected environment.
- Generates six filtered DataFrames spanning `{binary target-0, binary target-1, multiclass} × {active, sham, active+sham}`. The pipeline downstream uses **`df_binary_1_a`** (active-arm binary target, *One-Threshold* outcome definition).

### Section 3 — Data Preparation
- **3.1** Load and filter dataset.
- **3.2** Display settings for wide DataFrames.
- **3.3** Preview loaded DataFrames.
- **3.4** Encode target (Declined = 1, Improved = 0) and binarize `Gender`, `HIS_2_Threshold`.
- **3.5** Verify encoded data (type/NaN checks).

### Section 4 — Exploratory Data Analysis
- **4.1** Descriptive statistics stratified by outcome class.
- **4.2** Stratified 80/20 train/test split (with `seed=42` for the fixed hold-out probe).
- **4.3** Gender distribution by outcome.

### Section 5 — Statistical Feature Analysis
Univariate and multivariate feature-importance analyses feeding into the Evidence Synthesis.

- **5.0 Univariate Statistics** — Pearson, Spearman, φ-coefficient, ROC-AUC (signed), mutual information.
- **5.1 ANOVA Feature Selection** — univariate F-test group-difference ranking.
- **5.1.0 Feature Directionality** — Cohen's *d* effect size (direction of the decline-vs-improve contrast).
- **5.1.1 ANCOVA** — covariate-adjusted analysis (adjusting for Age and Gender; then "Full" covariate set).
- **5.1.2 PCA-Based Feature Ranking** — variance-contribution summary across principal components.
- **5.1.3 Correlation-Based Redundancy Removal** — identify feature pairs with |ρ| above threshold; used to prune redundant columns.
- **5.2 Sequential Forward Selection (SFS) with SVM** — *headline feature-selection experiment*.
  - RBF-SVM evaluator (C = 0.01, γ = 1), `class_weight = "balanced"`.
  - Scoring metric: Average Precision (AP).
  - **10 random seeds**; each seed jointly shuffles the inner 3-fold stratified partition *and* the SVM `random_state` (so the 10 runs sample distinct fold structures, not just initializations).
  - A feature is designated a **robust biomarker** iff it appears in the SFS-optimal subset in **strictly > 50 %** of runs.
  - Saved to `new_analysis_sfs_evidence_synthesis/sfs/latest_version/`.
- **5.3 Comprehensive Feature Evidence Synthesis** — integrates all of 5.0–5.2 into a composite **Evidence Score (0–100)** with tiered tags (★★★ Strong / ★★ Moderate / ★ Suggestive). Saved to `new_analysis_sfs_evidence_synthesis/evidence_synthesis/latest_version/`.

### Section 6 — Machine Learning Modeling Framework
- **6.1 Framework Definition** — reusable training harness for L1-logistic regression, Random Forest, and XGBoost with per-model hyperparameter grids and cost-sensitive settings.
- **6.2 Model Training Execution** — runs the three classifiers; artifacts saved under `l1_main_v*`, `rf_main_v*`, `xgb_main_v11_diag_v5/`.
- **6.3 Leave-One-Out Cross-Validation (LOOCV) Evaluation** — per-patient hold-out evaluation. Baseline LOOCV metrics for each model class.
- **Overfitting diagnostic cell** — ceiling-check fit on full training data.

### Section 7 — Results Visualization & Interpretation
- **7.0 TabPFN Foundation-Model Analysis** — runs the TabPFN tabular foundation model as a non-gradient-boosted comparison baseline; includes **7.0.1 Analysis Summary**.
- **7.1 Plotting Functions** — confusion matrices, SHAP summary plots (beeswarm / bar), per-seed distribution plots.

### Section 8 — Model Selection, Validation & Explainability

This is the **core experimental section** and maps directly to the manuscript's Probe 1 and Probe 2.

- **8.1 Version Comparison Utilities** — compare multiple (main × diagnostic) model versions to select the deployed configuration (`xgb_main_v11_diag_v5`).
- **8.2 Multi-Seed SHAP Analysis — Robust Feature Importance**
  - Implements `run_multi_seed_shap()`, the workhorse function for protocol-level SHAP aggregation.
  - Parameters: `n_seeds`, `test_size`, `vary_split` (False = fixed-split, True = MCCV), `device` (cpu/cuda).
  - Artifacts per run (in `shap_experiments/...`): `meta.json`, `ranking_A.csv` (mean|SHAP|), `ranking_B.csv` (alt aggregation), `per_seed_metrics_summary_thre<T>.csv`, `stability_topk.csv`, `beeswarm.pdf`, `beeswarm_all_seeds.pdf`, `importance_bar.pdf`, `confusion_matrix_*_ieee_thre<T>.pdf`.
- **8.3 Multi-Seed Classification Metrics (Re-use Saved Models)** — recomputes performance metrics from saved runs at arbitrary thresholds without retraining.
- **8.4 MCCV Threshold Sweep** — scans `T ∈ [0.1, 0.9]` across `vary_split=True` draws to identify the tightest operating point that preserves the zero-miss requirement (the sweep that selected `T = 0.30`).
- **8.5 LOOCV-SHAP — Out-of-Sample Explanations for Every Patient**
  - Implements `run_loocv_shap()` (CUDA-default): for each of 58 patients, trains on the remaining 57 and records SHAP attributions plus predictions. Repeated over `n_seeds` model-initialization seeds.
  - Bagged predictions are produced by per-patient majority vote across seeds.
  - Artifacts match the holdout layout (confusion matrix, beeswarm, rankings, per-seed metrics).
- **8.6 Overfit-Ceiling SHAP — Noise Calibration via Intentional Overfitting**
  - Deliberately fits an over-capacity XGBoost to the full training set and extracts its SHAP profile. This serves as a *noise ceiling*: features that appear here but nowhere else are not generalizable biomarkers.
- **8.7 Cross-Protocol SHAP Agreement — Signal vs Noise**
  - Loads `ranking_B.csv` from all 5 SHAP protocols (LOOCV, MCCV, Fixed hold-out, Full-data safety re-fit, Overfit ceiling).
  - Computes pairwise **Jaccard** overlap of top-K feature sets and **Spearman rank correlation** across the full 16-feature ranking.
  - Classifies each feature into one of four verdicts:
    - **`bulletproof`** — top-K in *all five* protocols (generalizable + recovered under overfit).
    - **`oos_only`** — top-K in all four out-of-sample protocols but not overfit (strong out-of-sample signal, hidden by overfit noise).
    - **`oos_partial`** — top-K in some but not all out-of-sample protocols.
    - **`overfit_noise`** — top-K *only* under the overfit ceiling (non-generalizable).
  - Output: `cross_protocol_comparison/cross_protocol_feature_classification.csv`, plus heatmaps.

### Section 9 — Appendix: Data Scaling Pipeline
Optional, documentational cells showing alternative scaling strategies (StandardScaler, RobustScaler, TIV-only) evaluated against the raw feature space. Not part of the deployed pipeline.

---

## 7. Five Validation Protocols at a Glance

| Protocol | `vary_split` | Train set | Test set | Purpose | Primary metric |
|---|---|---|---|---|---|
| **LOOCV** (primary) | — | 57 patients | 1 patient × 58 folds × `n_seeds` | Strictest small-sample performance probe | Bagged Recall, FN count |
| **MCCV** (split robustness) | True | Stratified 80 % × `n_seeds` independent draws | Stratified 20 % (n = 12) | Partition sensitivity | Per-seed Recall ± SD |
| **Fixed 80/20 hold-out** (single draw) | False | Stratified 80 % with `split_seed=42` | Stratified 20 % (n = 12) | Reference single draw; quantify partition luck | Per-seed mean ± SD |
| **Full-data re-fit** | N/A | 58 patients | — (SHAP-only) | Full-signal SHAP reference | SHAP rankings |
| **Overfit ceiling** | N/A | 58 patients, overfit | — (SHAP-only) | Noise-ceiling reference | SHAP rankings |

At the deployment threshold `T = 0.30` on `N = 58` with 12 decliners, the model achieves:

| Metric (Table III in the manuscript) | LOOCV | MCCV | Fixed 80/20 |
|---|---|---|---|
| Recall | 0.99 ± 0.03 | 0.92 ± 0.20 | 0.89 ± 0.21 |
| Precision | 0.26 ± 0.01 | 0.21 ± 0.06 | 0.33 ± 0.08 |
| F1 | 0.42 ± 0.01 | 0.34 ± 0.08 | 0.47 ± 0.11 |
| Accuracy | 0.43 ± 0.02 | 0.40 ± 0.12 | 0.67 ± 0.08 |

Bagged LOOCV confusion matrix: **TP = 12, FN = 0, FP = 32, TN = 14** → Recall = 1.00 (95 % Wilson CI [0.76, 1.00]), zero missed decliners.

---

## 8. Headline Findings

1. **Safety objective met end-to-end** — under bagged LOOCV the model identifies every decliner (FN = 0); MCCV and fixed 80/20 sustain the safety priority on average.
2. **Triangulated biomarker convergence** — `WMH` (global) and `GM_Asymmetry_Ind` are
   - the **two highest-consensus SFS features** (100 % and 80 % of 10 seeds),
   - **`bulletproof`** under cross-protocol SHAP agreement (top-5 in all 5 protocols, including the overfit ceiling),
   - biologically plausible as a **vascular-burden vs structural-reserve antagonism**.
3. **Direction of effect** — high `WMH` pushes SHAP toward *Declined*; low `GM_Asymmetry_Ind` (loss of hemispheric asymmetry) pushes toward *Declined*. The decline-risk regime is **high WMH ∩ low DLPFC GM asymmetry**.
4. **`V_R_DLPFC_GM`** is `bulletproof` under SHAP but not promoted by SFS; explained by its partial redundancy with `GM_Asymmetry_Ind` (which is derived from bilateral DLPFC GM volumes).
5. **Noise features** — `V_R_DLPFC_WM` and `HIS_2_Threshold` appear only under the overfit ceiling (`overfit_noise` verdict) and are excluded from the clinical signature.

---

## 9. Artifacts & Output Directories

### SFS (Section 5.2)
- `new_analysis_sfs_evidence_synthesis/sfs/latest_version/`
  - `01_svm_tuning_<ts>.txt`, `02_svm_results_<ts>.txt` — SVM hyperparameter traces.
  - `03_sfs_seed_<N>_<ts>.txt` — per-seed forward-selection traces (one per random seed).
  - `04_sfs_summary_<ts>.txt` — consensus summary across seeds.
  - `05_ranking_comparison_<ts>.txt` — comparison across seeds.
  - `feature_rankings_<ts>.csv`, `sfs_all_results_<ts>.csv`, `svm_cv_results_<ts>.csv` — tabular ranking and CV outputs.

### Evidence Synthesis (Section 5.3)
- `new_analysis_sfs_evidence_synthesis/evidence_synthesis/latest_version/`
  - `evidence_report_<ts>.txt` — human-readable tiered report.
  - `evidence_table_<ts>.csv` — per-feature composite scores and tiers.
  - `justification.txt` — narrative justification of the Evidence Synthesis method.
- `new_analysis_sfs_evidence_synthesis/feature_directionality/feature_directionality.txt` — Cohen's d / signed-AUC per feature.

### Multi-Seed SHAP Experiments (Sections 8.2, 8.5, 8.6)
Each `shap_xgb_*` folder contains, at minimum:
- `meta.json` — run configuration (seeds, threshold, split policy, device, protocol label).
- `ranking_A.csv`, `ranking_B.csv` — SHAP feature rankings (two aggregation flavors).
- `stability_topk.csv`, `stability_plot.pdf` — top-K stability across seeds.
- `per_seed_metrics_summary_thre0.30.csv` — per-seed classification metrics at `T = 0.30`.
- `beeswarm.pdf`, `beeswarm_all_seeds.pdf` — SHAP beeswarm (single-seed / stacked-seeds).
- `importance_bar.pdf` — mean |SHAP| bar chart.
- `confusion_matrix_<protocol>_ieee_thre0.30.pdf` — single-seed / bagged confusion matrices.

### Cross-Protocol Comparison (Section 8.7)
- `cross_protocol_comparison/`
  - `cross_protocol_topk.csv` — top-5 per protocol (mirrors manuscript Table VI).
  - `cross_protocol_spearman.csv` — pairwise Spearman ρ heatmap data.
  - `cross_protocol_jaccard.csv` — pairwise Jaccard heatmap data.
  - `cross_protocol_feature_classification.csv` — per-feature verdict (bulletproof / oos_only / oos_partial / overfit_noise) — mirrors manuscript Table IV.
  - `cross_protocol_comparison.pdf` — aggregate comparison figure.

### Classifier Training (Sections 6.2, 6.3)
- `xgb_main_v11_diag_v5/` — deployed XGBoost (safety-prioritized: `n_estimators=30`, `max_depth=1`, `learning_rate=0.1`, `scale_pos_weight=12`).
- `l1_main_v7_diag_v26/`, `l1_main_v16_diag_v26/` — L1-logistic-regression variants.
- `rf_main_v2_diag_v6/` — Random Forest comparator.
- `combined_l1_rf_xgb_1st/` — combined model-comparison outputs.

### Manuscript Figures (repo root)
- `confusion_matrix_xgb_safety_prioritized.{png,pdf,eps}` — single-seed hold-out confusion matrix (Figure baseline).
- `confusion_matrix_xgb_diag_ieee.png` — diagnostic (non-safety-prioritized) reference.
- *(Manuscript figures `beeswarm_all_seeds.pdf` and `confusion_matrix_loocv_bagged_ieee_thre0.30.pdf` are produced by the LOOCV cell in Section 8.5 and should be copied from `shap_xgb_loocv_1000seeds/` to the Overleaf project.)*

---

## 10. Reproducing the Results

```bash
conda activate p
jupyter notebook rtms_biomarker_analysis.ipynb
```

Run in order:

1. **Sections 1–4** — once, top to bottom (deterministic).
2. **Section 5 (Statistical Feature Analysis)** — SFS (5.2) needs `n_seeds = 10`; Evidence Synthesis (5.3) consumes all prior outputs.
3. **Section 6 (Modeling Framework)** — generates baseline L1 / RF / XGBoost artifacts; the manuscript's deployed model is **`xgb_main_v11_diag_v5`**.
4. **Section 8 (SHAP Validation)** — the five SHAP protocols can run independently; each produces a self-contained `shap_xgb_*/` folder.
   - Full-data SHAP, Holdout-fixed SHAP, Holdout-MCCV SHAP: Section 8.2 driver cells with different `vary_split` settings.
   - LOOCV-SHAP: Section 8.5 driver cells (CUDA recommended; each 1000-seed run ≈ 2–6 h on a mid-range GPU).
   - Overfit ceiling: Section 8.6.
   - Cross-protocol agreement: Section 8.7 (consumes all five `ranking_B.csv` files).
5. **Regenerate manuscript figures** — copy the chosen confusion matrix and beeswarm PDFs into Overleaf.

### Determinism

All stochastic cells accept explicit seeds. The `1000-seed` runs sweep `random_state ∈ {0, 1, …, 999}`. With `vary_split = True`, both the split and the model initialization move together; with `vary_split = False`, only the model initialization varies (split fixed at `seed = 42`).

### Performance notes

- Full-data and fixed-hold-out SHAP at 1000 seeds: ≈ 10–30 min on CPU.
- MCCV SHAP at 1000 seeds: ≈ 30–60 min on CPU.
- LOOCV-SHAP at 1000 seeds: ≈ 2–6 h on CUDA (58 × 1000 = 58 000 model fits).
- Overfit-ceiling SHAP at 1000 seeds: ≈ 10–20 min on CPU.

---

## 11. File-by-File Reference

| File | Purpose |
|---|---|
| `rtms_biomarker_analysis.ipynb` | Main analysis notebook — all 9 sections. |
| `manuscript.txt` | LaTeX source of the EMBC submission (self-contained; depends on `refs.bib` on Overleaf). |
| `requirements.txt` | Python dependency list. |
| `README.md` | This document. |
| `.gitignore` | Keeps the repo light (tracks only `.ipynb`, `README.md`, `requirements.txt`). |
| `_cell94.py` | Scratch Python script extracted from the notebook during development. |
| `Comments.txt`, `models.txt`, `linear_output.txt` | Development notes / scratch logs. |

---

## 12. Citation

If you use this pipeline in your own work, please cite the EMBC manuscript:

```bibtex
@inproceedings{tehrani2026rtms,
  author    = {Arshia Esmail Tehrani and Milad Pourmohammadi Langroudi and Javad Ebrahimi and Alireza Bakhshai and Zahra Moussavi},
  title     = {Explainable Multimodal Machine Learning for {rTMS} Treatment Risk Assessment in {A}lzheimer's Disease},
  booktitle = {Proc. IEEE Engineering in Medicine and Biology Society (EMBC)},
  year      = {2026}
}
```

---

## 13. License & Acknowledgments

- **Code**: released for research and reproducibility purposes.
- **Data**: proprietary trial data; not redistributed via this repository. Access is governed by the parent clinical-trial data-use agreement.
- **Acknowledgments**: parent trial `Moussavi et al., 2024`; CAT12 / SPM12 for neuroimaging preprocessing; the SHAP, scikit-learn, and XGBoost open-source communities.
