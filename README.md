# Multimodal Biomarker Feature Importance for rTMS in Alzheimer's Disease

A machine learning pipeline to identify predictive biomarkers for treatment response in Alzheimer's Disease patients undergoing repetitive Transcranial Magnetic Stimulation (rTMS).

## Overview

This project analyzes clinical and neuroimaging data to predict patient outcomes (Improved vs. Declined) following rTMS treatment. Multiple statistical and ML methods are combined for robust feature selection and interpretability.

## Notebook Structure

1. **Environment Setup** — System configuration
2. **Data Loading** — Load and prepare the dataset
3. **Data Preparation** — Filtering, encoding, and train/test split
4. **Exploratory Data Analysis** — Descriptive statistics and distributions
5. **Statistical Feature Analysis**
   - **ANOVA** — Univariate group difference testing
   - **Feature Directionality** — Cohen's d effect size and direction of effect
   - **ANCOVA** — Covariate-adjusted analysis (Age+Gender and Full)
   - **PCA Ranking** — Variance contribution analysis
   - **Correlation Analysis** — Redundancy identification
   - **SFS with SVM** — Multi-seed Sequential Forward Selection
   - **Evidence Synthesis** — Comprehensive integration of all methods
6. **Machine Learning Modeling** — L1 Logistic Regression, Random Forest, XGBoost
7. **Results Visualization** — Performance metrics, SHAP plots, feature importance
8. **Model Selection** — Compare model versions and select best configuration

## Feature Selection Pipeline

### Multi-Method Evidence Synthesis

The notebook integrates multiple feature selection approaches into a unified evidence score:

| Method | Purpose |
|--------|---------|
| ANOVA | Univariate significance testing |
| ANCOVA | Controls for Age/Gender confounders |
| Cohen's d | Effect size and directionality |
| PCA | Variance contribution ranking |
| Correlation | Redundancy detection |
| SFS (Multi-Seed) | Multivariate selection stability |

**Output:** Features ranked by composite evidence score (0-100) with tiered significance (★★★ Strong, ★★ Moderate, ★ Suggestive).

### Outputs

- **`SFS/`** — SFS results, rankings, plots
- **`Evidence_Synthesis/`** — Comprehensive report, evidence tables, visualizations

## Installation

```bash
pip install -r requirements.txt
```

## Project Structure

```
├── rtms_biomarker_analysis.ipynb   # Main analysis notebook
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── SFS/                             # SFS outputs
└── Evidence_Synthesis/              # Evidence synthesis reports
```

## License

This project is for research purposes.
