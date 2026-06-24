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

## 🏗️ Project Architecture
