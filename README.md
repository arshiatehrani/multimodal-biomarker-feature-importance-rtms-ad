# Multimodal Biomarker Feature Importance for rTMS in Alzheimer's Disease

A machine learning pipeline to identify predictive biomarkers for treatment response in Alzheimer's Disease patients undergoing repetitive Transcranial Magnetic Stimulation (rTMS).

## Overview

This project analyzes clinical and neuroimaging data to predict patient outcomes (Improved vs. Declined) following rTMS treatment. Multiple machine learning models are trained and compared, with emphasis on interpretability and feature importance analysis.

## Notebook Structure

1. **Environment Setup** — System configuration
2. **Data Loading** — Load and prepare the dataset
3. **Data Preparation** — Filtering, encoding, and train/test split
4. **Exploratory Data Analysis** — Descriptive statistics and distributions
5. **Statistical Feature Analysis** — Univariate correlation and association tests
6. **Machine Learning Modeling** — L1 Logistic Regression, Random Forest, XGBoost
7. **Results Visualization** — Performance metrics, SHAP plots, feature importance
8. **Model Selection** — Compare model versions and select best configuration

## Models

- **L1 Logistic Regression** — Sparse linear model with feature selection
- **Random Forest** — Ensemble of decision trees
- **XGBoost** — Gradient boosted trees

## Installation

```bash
pip install -r requirements.txt
```

## License

This project is for research purposes.
