# Customer Churn Prediction — Telco Dataset

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)
![License](https://img.shields.io/badge/License-MIT-green)

End-to-end machine learning project predicting customer churn on the IBM Telco dataset (7,043 customers · 21 features). Covers the full pipeline: EDA → preprocessing → class balancing → model comparison → hyperparameter tuning → SHAP explainability → business recommendations.

---

## Results

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC |
|---|---|---|---|---|---|
| Logistic Regression | ~0.80 | ~0.65 | ~0.56 | ~0.60 | ~0.84 |
| Random Forest | ~0.82 | ~0.68 | ~0.59 | ~0.63 | ~0.84 |
| XGBoost | ~0.80 | ~0.62 | ~0.57 | ~0.59 | ~0.81 |
| LightGBM | ~0.80 | ~0.63 | ~0.56 | ~0.59 | ~0.82 |
| **XGBoost (Tuned + Optuna)** | **~0.82** | **~0.67** | **~0.62** | **~0.64** | **~0.85** |

> All models evaluated on a held-out test set (20%). SMOTE applied inside CV folds to prevent data leakage.

---

## Key Findings

- **Contract type** is the single strongest churn predictor — month-to-month customers churn ~42% vs ~3% for 2-year contracts
- **Tenure** has a strong inverse relationship with churn — risk halves after month 12, nearly disappears after month 24
- **Monthly charges** above $65 meaningfully increase churn probability
- **Fiber optic** customers churn 2× more than DSL customers despite paying the highest bills
- High-risk profile: new customer (< 12 months), month-to-month contract, fiber optic internet, no TechSupport

---

## Project Structure

```
customer-churn-prediction/
├── Customer_Churn_Prediction.ipynb   ← Main notebook (13 sections)
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── requirements.txt
└── README.md
```

---

## Notebook Sections

| # | Section |
|---|---------|
| 1 | Imports & Setup |
| 2 | Data Loading & Overview |
| 3 | Exploratory Data Analysis |
| 4 | Feature Engineering & Preprocessing |
| 5 | Class Imbalance — SMOTE |
| 6 | Model Training & Comparison |
| 7 | Model Evaluation (ROC, PR curves, Confusion Matrix) |
| 8 | Hyperparameter Tuning — Optuna (40 trials) |
| 9 | Native Feature Importance |
| 10 | Classification Threshold Optimization |
| 11 | Model Explainability — SHAP |
| 12 | Business Insights & Recommendations |
| 13 | Conclusion & Leaderboard |

---

## Setup

```bash
git clone https://github.com/BerkeKutay/customer-churn-prediction.git
cd customer-churn-prediction
pip install -r requirements.txt
jupyter notebook Customer_Churn_Prediction.ipynb
```

---

## Tech Stack

| Category | Libraries |
|---|---|
| Data | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| ML Models | Scikit-learn, XGBoost, LightGBM |
| Class Balancing | imbalanced-learn (SMOTE) |
| Tuning | Optuna (TPE sampler, 40 trials) |
| Explainability | SHAP (TreeExplainer) |

---

## Dataset

[IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — publicly available on Kaggle.
