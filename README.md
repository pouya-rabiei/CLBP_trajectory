# CLBP_trajectory

Machine learning classification of 1500 chronic low back pain patients into 4 clinical clusters using nested cross-validation and SHAP for model interpretation.

---

## Project Overview

This repository contains the full machine learning pipeline for classifying chronic low back pain (CLBP) patients into four data-driven clusters using baseline biopsychosocial features. The pipeline uses **nested cross-validation** for robust model evaluation and **SHAP (SHapley Additive exPlanations)** for model interpretability.

---

## Objective

- Predict **cluster membership** based on baseline features
- Identify key predictors driving classification decisions

---

## Methods

### Models Used:
- Logistic Regression
- Random Forest
- XGBoost
- Support Vector Classifier (SVC)

### Validation Strategy:
- **Nested 5-Fold Cross-Validation**
  - Outer loop: unbiased performance estimation
  - Inner loop: hyperparameter tuning and feature selection

### Interpretability:
- SHAP summary plots
- Cluster-wise feature impact using SHAP values

---

## Dataset

- **Sample size**: individuals with chronic low back pain
- **Features include**:
  - *Psychological*: depression, fear, catastrophizing, sleep quality
  - *Pain-related*: intensity, duration, interference, impact
  - *Demographic*: age, sex, BMI, smoking, education, employment status, comorbidities
- **Target**: 4 pre-defined clusters (from Latent Class Growth Analysis)

---

## Installation

Make sure Python ≥ 3.8 is installed. Then install dependencies:

```bash
pip install -r requirements.txt

If SHAP fails to install, try:

pip install shap
or
conda install -c conda-forge shap
