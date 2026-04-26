# Predicting High‑Cost Hospitals Using AI and MSPB Data
AI‑based prediction of high‑cost hospitals using MSPB and CMS quality data, with a focus on interpretability and healthcare decision support.

# Overview
Medicare Spending Per Beneficiary (MSPB) is a key metric used by the Centers for Medicare and Medicaid Services (CMS) to evaluate how efficiently hospitals deliver care. MSPB captures all Medicare Part A and Part B spending during a patient’s full episode of care, including:

1. The three days before admission

2. The inpatient stay

3. The thirty days after discharge

4. High MSPB values indicate higher‑than‑expected spending after adjusting for patient risk and geographic payment differences.

This project builds machine learning models to predict which U.S. hospitals are likely to have above‑median MSPB spending, identifies the hospital characteristics most associated with high spending, and uses SHAP to produce interpretable outputs.

# Data Sources
Both datasets are publicly available from CMS:

# Dataset	CMS Label
Medicare Spending Per Beneficiary – Hospital	Medicare Hospital Spending Per Patient
Hospital General Information	Hospital General Information


The two datasets are merged on facility_id.

# Project Structure
Code files:

# Code
Predicting_High‑Cost_Hospitals_Using_AI_and_MSPB_Data_2.ipynb
README.md

# Section Descriptions
1. Dependencies	Library imports and environment setup
2. Load Data	Upload and read CMS datasets
3. Clean and Merge	Standardize columns, create binary target, merge datasets
4. Exploratory Data Analysis	Score distributions, regional analysis, correlation heatmap, choropleth map
5. Feature Engineering & Preprocessing	Feature selection, imputation, scaling
6. Modeling	Logistic Regression, Decision Tree, Random Forest, XGBoost
7. Evaluation	ROC curves and AUC comparison
8. Interpretability	SHAP summary and dependence plots for XGBoost


# Target Variable
The binary target high_cost is defined as:

1 — Hospital MSPB score is above the national median

0 — Hospital MSPB score is at or below the national median

# Features
The model uses four hospital quality‑measure counts as predictors:

count_of_facility_mort_measures — Mortality measures reported

count_of_facility_safety_measures — Safety measures reported

count_of_facility_readm_measures — Readmission measures reported

count_of_facility_te_measures — Timeliness and efficiency measures reported

The raw MSPB score is excluded to prevent data leakage, since the target variable is derived directly from it.

# Models
Logistic Regression	Scaled, imputed features
Decision Tree	Imputed features
Random Forest	300 estimators
XGBoost	300 estimators, learning rate 0.05, max depth 4


All models are evaluated using accuracy, classification report, confusion matrix, and AUC.

# Interpretability
SHAP (SHapley Additive exPlanations) is applied to the XGBoost model to identify which features drive predictions at both the global and individual level.

# Requirements
Code
pandas  
numpy  
matplotlib  
seaborn  
scikit-learn  
xgboost  
shap  
geopandas  
shapely
Install dependencies:

# Code
pip install shap xgboost geopandas shapely

Notes
Designed to run in Google Colab (files.upload() handles data ingestion).

Causal inference exploration is planned for a future version.

# Author
Dylan Reid — GitHub
