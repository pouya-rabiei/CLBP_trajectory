# CLBP_trajectory
Machine learning classification of 1500 chronic low back pain patients into 4 clinical clusters using nested cross-validation and SHAP for model interpretation.

# CLBP Cluster Classification

This repository contains the full machine learning pipeline for classifying chronic low back pain (CLBP) patients into four data-driven clusters using baseline features. The pipeline uses **nested cross-validation** for robust model selection and evaluation, and **SHAP (SHapley Additive exPlanations)** for interpretability.

---
## Objective

- Build predictive models for **cluster membership** based on baseline biopsychosocial variables.

---

## Methods

### Models Used:
- Logistic Regression
- Random Forest
- XGBoost
- Support Vector Classification (SVC)
- Multi-Layer Perceptron (MLP)

### Validation:
- **Nested 5-Fold Cross-Validation**
  - Outer loop: unbiased model evaluation
  - Inner loop: hyperparameter tuning & feature selection

### Interpretability:
- SHAP summary plots
- SHAP feature impact per cluster

---

## Dataset

- **Sample Size**: Near 1500 individuals with chronic low back pain
- **Features**:
  - Psychological: depression, fear, catastrophizing, sleep quality
  - Pain-related: intensity, duration, interference, impact
  - Demographics: age, sex, BMI, smoking, education, employment status, comorbidities
- **Target**: 4 pre-defined clusters (derived via Latent Class Growth analysis)

---

### Installation
you may need to install shap via:

```bash
pip install -r requirements.txt

License
This project is licensed under the MIT License – see the LICENSE file for details.
