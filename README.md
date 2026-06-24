# ✈️ Big Data Flight Delay Prediction
### A Two-Stage Machine Learning Approach with Explicit Data Leakage Analysis
**PySpark · Databricks**

---

## 📌 Project Overview

Flight delays affect roughly **1 in 5 U.S. domestic flights**, imposing substantial costs on passengers and airlines. This project develops a **two-stage machine learning system** using the [US 2023 Civil Flights Kaggle dataset](https://www.kaggle.com/datasets/bordanova/2023-us-civil-flights-delay-meteo-and-aircraft) (~6.7M flights joined with airport-level weather data).

The problem is framed as:
1. **Binary Classification** — will a flight be delayed more than 15 minutes? (FAA standard threshold)
2. **Regression** — if delayed, how long will the delay be (in minutes)?

### 🔑 Central Contribution
Every model is evaluated under **two configurations**:
- **With leakage features** — includes post-arrival columns (e.g., `Arr_Delay`) unavailable at departure time
- **Without leakage features** — strictly pre-flight information only (realistic deployment scenario)

This explicit leakage quantification exposes a gap commonly overlooked in published flight delay literature.

---

## 📊 Key Results

### Classification (No Leakage — Realistic Pre-Flight Scenario)
| Model | AUC-ROC | Accuracy | F1 |
|---|---|---|---|
| Random Forest | **0.665** | 0.802 | 0.714 |
| Logistic Regression | 0.665 | 0.802 | 0.720 |
| Linear SVM | 0.620 | 0.802 | 0.714 |
| Decision Tree | 0.533 | 0.805 | 0.733 |

> ⚠️ With leakage features included, all classifiers achieve AUC-ROC ≈ **0.999** — entirely due to data leakage, not genuine predictive ability.

### Regression (No Leakage — Realistic Pre-Flight Scenario)
| Model | RMSE (min) | MAE (min) | R² |
|---|---|---|---|
| Gradient Boosted Trees | 94.74 | 49.23 | 0.036 |
| Random Forest | 94.72 | 49.67 | 0.036 |
| Decision Tree | 95.50 | 49.81 | 0.024 |
| Linear Regression | 95.03 | 50.07 | 0.030 |

### Two-Stage Hybrid Architecture
| Stage | MAE (min) | R² |
|---|---|---|
| Single-stage baseline | ~49.00 | 0.030 |
| + Bayesian Tuning (LightGBM + Optuna, 30 trials) | 44.02 | 0.262 |
| **Two-Stage Hybrid (Normal Range)** | **20.75** | — |

> The two-stage architecture separates severe delays (>143 min, top 10%) from normal-range flights, achieving **MAE of 20.75 minutes — a 58% improvement** over single-stage baselines.

---

## 📦 Dataset

**Source:** [Kaggle — 2023 US Civil Flights: Delay, Meteo and Aircraft](https://www.kaggle.com/datasets/bordanova/2023-us-civil-flights-delay-meteo-and-aircraft)

- **~6.7M** domestic U.S. flight records for 2023
- Joined with daily airport-level weather data
- Cancelled and diverted records excluded

### Leakage Features Removed (No-Leakage Configuration)
| Feature | Reason |
|---|---|
| `Arr_Delay` | Only available post-landing |
| `Dep_Delay` | Only available post-departure |
| `Delay_Carrier` | Post-flight attribution |
| `Delay_Weather` | Post-flight attribution |
| `Delay_NAS` | Post-flight attribution |
| `Delay_Security` | Post-flight attribution |
| `Delay_LastAircraft` | Post-flight attribution |

---

## ⚙️ Tech Stack

| Layer | Tool |
|---|---|
| Distributed Computing | Apache PySpark |
| Environment | Databricks Community Edition |
| ML Framework | PySpark MLlib |
| Boosting | LightGBM |
| Hyperparameter Tuning | Optuna (30 Bayesian trials) |
| Visualization | Matplotlib, Seaborn |
| Version Control | Git + GitHub |

---

## 🔬 Methodology

### Stage 1 — Binary Classification
**Task:** Predict whether a flight will be delayed more than 15 minutes (FAA standard)

**Models benchmarked:**
- Decision Tree (maxDepth=10)
- Random Forest (30 trees, maxDepth=10)
- Logistic Regression (L2, λ=0.01, StandardScaler)
- Linear SVM (λ=0.1)

**Split:** 80/20 stratified train-test

**Primary metric:** AUC-ROC (robust to class imbalance; ~20% of flights delayed)

### Stage 2 — Regression
**Task:** Predict delay duration (minutes) for flights predicted as delayed

**Models benchmarked:**
- Decision Tree
- Random Forest (30 trees)
- Gradient Boosted Trees (50 iterations)
- Linear Regression (L2, λ=0.1)

**Primary metrics:** RMSE, MAE, R²

### Two-Stage Hybrid Architecture
- **Tier-1:** Separates severe delays (>143 min, top 10% of delayed flights) from normal range
- **Tier-2:** Dedicated regressor predicts exact duration for normal-range flights only
- Severe delays issued as anomaly alerts rather than overconfident point estimates

---

## 📈 Key Findings

1. **Leakage gap is massive:** AUC-ROC drops from 0.999 → 0.665 after removing post-arrival features — exposing inflated benchmarks common in published literature
2. **Pre-flight regression is genuinely hard:** R² ≈ 0.03 for all single-stage models without leakage features
3. **Two-stage architecture solves the long-tail problem:** Separating outliers reduces MAE from ~49 → 20.75 minutes (58% improvement)
4. **Random Forest wins classification:** Most robust classifier due to non-linear feature interaction capture
5. **Temporal patterns matter:** Afternoon/evening flights, Fridays/Sundays, and June–July show highest delay rates

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
```

### Data Setup
1. Download dataset from [Kaggle](https://www.kaggle.com/datasets/bordanova/2023-us-civil-flights-delay-meteo-and-aircraft)
2. Upload CSVs to Databricks FileStore or DBFS
3. Run notebooks in order: 01 → 05

### Running on Databricks
1. Upload notebooks folder to your Databricks workspace
2. Attach to a cluster with PySpark runtime
3. Run notebooks sequentially

---

## 👥 Team

| Member | Contribution |
|---|---|
| **Nayan Kadhre** | Data ingestion, multi-table joins, feature selection, leakage removal, classification pipeline |
| Young Soo Ko | EDA, data cleaning, trend analysis, iterative improvement (LightGBM + Optuna) |
| Jenna Childress | EDA, feature engineering, regression pipeline |

---

## 📄 Report

Full methodology and results documented in `docs/report.pdf`.
