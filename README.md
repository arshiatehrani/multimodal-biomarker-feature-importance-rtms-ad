# Multimodal Biomarker Feature Importance for rTMS in Alzheimer's Disease

A machine learning pipeline to identify predictive biomarkers for treatment response in Alzheimer's Disease patients undergoing repetitive Transcranial Magnetic Stimulation (rTMS).

## Overview

This project analyzes clinical and neuroimaging data to predict patient outcomes (Improved vs. Declined) following rTMS treatment. Multiple machine learning models are trained and compared, with emphasis on interpretability and feature importance analysis.

## Notebook Structure

1. **Environment Setup** — System configuration
2. **Data Loading** — Load and prepare the dataset
3. **Data Preparation** — Filtering, encoding, and train/test split
4. **Exploratory Data Analysis** — Descriptive statistics and distributions
5. **Statistical Feature Analysis**
   - Univariate correlation and association tests (Pearson, Spearman, Phi, MI)
   - ANOVA F-test for feature significance
   - Sequential Forward Selection (SFS) with SVM
6. **Machine Learning Modeling** — L1 Logistic Regression, Random Forest, XGBoost
7. **Results Visualization** — Performance metrics, SHAP plots, feature importance
8. **Model Selection** — Compare model versions and select best configuration

## Feature Selection Pipeline

### Sequential Forward Selection (SFS)

The notebook includes a robust feature selection pipeline using SVM:

- **SVM Hyperparameter Tuning** — GridSearchCV to find optimal kernel, C, and gamma
- **Multi-Seed Analysis** — Runs SFS with 5 different random seeds for stability
- **Consensus Features** — Identifies features selected by majority of seeds
- **Ranking Comparison** — Analyzes agreement between seeds using Spearman correlation

**Outputs saved to `SFS/` folder:**
- SVM tuning results (`.txt`, `.csv`)
- Per-seed SFS results (`.txt`)
- Combined results and rankings (`.csv`)
- Performance curves and ranking heatmaps (`.png`)

## Models

- **L1 Logistic Regression** — Sparse linear model with feature selection
- **Random Forest** — Ensemble of decision trees
- **XGBoost** — Gradient boosted trees (GPU-accelerated)

## Installation

```bash
pip install -r requirements.txt
```

### GPU Support

- **XGBoost**: Automatically uses CUDA if available (`device="cuda"`)
- **PyTorch** (optional): `pip install torch --index-url https://download.pytorch.org/whl/cu124`
- **cuML/RAPIDS**: Linux only (not supported on Windows)

## Project Structure

```
├── Phase_1.ipynb          # Main analysis notebook
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── SFS/                   # Sequential Feature Selection outputs
│   ├── *.txt              # Captured console outputs
│   ├── *.csv              # Results tables
│   └── *.png              # Plots
└── model_params_*/        # Saved model artifacts
```

## License

This project is for research purposes.
