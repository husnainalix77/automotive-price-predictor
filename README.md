# Automotive Price Prediction & Vehicle Classification
## End-to-End Data Science Pipeline

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red)](https://streamlit.io/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4-orange)](https://scikit-learn.org/)

---

## Project Overview

This project implements a complete end-to-end data science pipeline on the 
**Car Features and MSRP Dataset** (Kaggle — CooperUnion). It covers every 
stage from raw data acquisition to an interactive Streamlit dashboard — 
including exploratory data analysis, data cleaning, feature engineering, 
regression modelling, classification modelling, clustering, and deployment.

| Detail | Info |
|---|---|
| **Department** | Mechatronics and Control Engineering |
| **University** | University of Engineering & Technology (UET), Lahore |
| **Dataset** | Car Features and MSRP — Kaggle (CooperUnion) |
| **Regression Target** | MSRP — Manufacturer's Suggested Retail Price ($) |
| **Classification Target** | Vehicle Size (Compact / Midsize / Large) |

---

## Dataset

| Property | Value |
|---|---|
| Source | [Kaggle — CooperUnion Car Dataset](https://www.kaggle.com/datasets/CooperUnion/cardataset) |
| Licence | CC0 — Public Domain |
| Raw Rows | 11,914 |
| Raw Columns | 16 |
| Cleaned Rows | 11,592 |
| Final Features | 18 (after encoding + feature engineering) |

---

## Project Structure
```
automotive-price-predictor/
├── notebook/
│   ├── data.csv ← raw dataset
│   └── automotiv-price-analysis.ipynb  ← fully executed notebook
├── dashboard/
│   ├── app.py                                ← Streamlit application
│   ├── requirements.txt                      ← Python dependencies
│   ├── data/
│   │   └── cleaned_data.csv                  ← preprocessed dataset
│   └── models/
│       ├── regression_model.pkl              ← Ridge Regression (R²=0.839)
│       ├── classification_model.pkl          ← Random Forest (Accuracy=96.2%)
│       ├── kmeans_model.pkl                  ← K-Means (k=3)
│       ├── pca_model.pkl                     ← PCA (2 components)
│       ├── scaler.pkl                        ← StandardScaler
│       └── feature_columns.json             ← feature column order
└── README.md                                 ← this file
```
---

## Pipeline Overview (9 Phases)

| Phase | Description | Key Output |
|---|---|---|
| 1 | Dataset Acquisition & Exploration | Data card, target definitions |
| 2 | Exploratory Data Analysis | 6 plot types, EDA narrative |
| 3 | Data Cleaning & Preprocessing | 11,592 clean rows, encoded features |
| 4 | Data Visualization & Storytelling | 6 publication-quality plots |
| 5 | Feature Engineering | 3 new features, Lasso selection, Pipeline R²=0.839 |
| 6 | Regression Modelling | Ridge R²=0.839, RMSE=$7,947 |
| 7 | Classification Modelling | Random Forest Accuracy=96.2% |
| 8 | Clustering | K-Means k=3, Silhouette=0.185 |
| 9 | Streamlit Dashboard | 4-tab interactive app |

---

## Model Performance Summary

### Regression — Predicting MSRP

| Model | R² | RMSE | MAE |
|---|---|---|---|
| Baseline (mean predictor) | 0.000 | $19,789 | $15,267 |
| Linear Regression | 0.839 | $7,947 | $6,052 |
| **Ridge Regression** ← selected | **0.839** | **$7,947** | **$6,051** |

### Classification — Predicting Vehicle Size

| Model | Accuracy | Macro F1 | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 67.0% | 0.653 | 0.851 |
| **Random Forest** ← selected | **96.2%** | **0.961** | **0.996** |

### Clustering — K-Means

| Parameter | Value |
|---|---|
| Optimal k | 3 |
| Silhouette Score | 0.185 |
| Alignment with Vehicle Size | Partial — boundary cases overlap |

---

## Dashboard — Install and Run

### Step 1 — Install Dependencies

Open PowerShell or terminal and run:

```bash
pip install -r requirements.txt
```

### Step 2 — Navigate to Dashboard Folder

```bash
cd path\to\automotive-price-predictor\dashboard
```

### Step 3 — Launch Dashboard

```bash
streamlit run app.py
```

Browser opens automatically at:
http://localhost:8501
---

## Dashboard Tabs

### 📊 Tab A — Dataset Overview
Browse the cleaned dataset, view column distributions and the full 
Pearson correlation heatmap. Use the column selector to visualise 
any feature's distribution dynamically.

### 🔍 Tab B — Exploratory Analysis
Interactively explore relationships between any two numerical features 
via scatter plots. Toggle trend lines, colour points by a third feature, 
and filter data to zoom into subpopulations using a range slider.

### 🎯 Tab C — Model Prediction
Enter car specifications using sliders and dropdowns. Click Predict 
to simultaneously get the MSRP price prediction (Ridge Regression) 
and Vehicle Size classification (Random Forest) with confidence 
probabilities. Colour-coded results indicate economy, mid-range or 
premium vehicle category.

### 🔵 Tab D — Cluster Explorer
Explore the three K-Means clusters discovered without using Vehicle 
Size labels. Select any cluster to highlight it in the PCA scatter 
plot. View a feature comparison table and delta bar chart showing 
what makes each cluster distinctive from the overall dataset.

---

## Key Findings

1. **Engine HP is the strongest predictor of MSRP** (r = 0.83) — 
   more powerful engines cost significantly more.

2. **Random Forest dramatically outperforms Logistic Regression** 
   for vehicle size classification (96.2% vs 67.0%) — confirming 
   non-linear boundaries between size classes.

3. **highway MPG and city mpg are nearly redundant** (r = 0.94) — 
   severe multicollinearity identified and addressed in feature selection.

4. **K-Means discovered k=3 natural clusters** that partially align 
   with Vehicle Size categories — confirming genuine structure in 
   the automotive feature space.

5. **Winsorisation of MSRP** at $74,078 reduced skewness from 
   γ1 = 11.77 to γ1 = 0.48 — significantly improving model stability.

---

## Technologies Used

| Technology | Version | Purpose |
|---|---|---|
| Python | 3.12 | Core language |
| Pandas | 2.2.0 | Data manipulation |
| NumPy | 1.26.4 | Numerical computing |
| Matplotlib | 3.8.3 | Visualisation |
| Seaborn | 0.13.2 | Statistical plots |
| Scikit-learn | 1.4.0 | ML models + preprocessing |
| SciPy | 1.12.0 | Statistical analysis |
| Streamlit | 1.32.0 | Interactive dashboard |
| Joblib | 1.3.2 | Model serialisation |

---

## Academic Integrity

All interpretations, explanations, markdown narratives and modelling 
decisions in this project were written and understood by both group 
members. AI tools were used only for syntax assistance, debugging 
and boilerplate code generation as permitted by course policy.

---

*University of Engineering & Technology (UET), Lahore — 2026*
