# UK Road Accident Severity Prediction and Forward Hotspot Forecasting

This repository contains a full end-to-end data science project on **UK road accident severity prediction** and **future hotspot forecasting** using structured accident records, temporal features, spatial features, class-imbalance handling, explainable machine learning, and forward-looking risk modeling.

The project was developed as part of an MSc Data Science final project and is designed to be useful to both researchers and practitioners who want to understand how to move from accident-level classification to operational hotspot prioritisation.

---

## 1. Project Overview

Road traffic accidents do not all have the same consequences. Some incidents result in slight injuries, others in serious harm, and a small minority in fatal outcomes. From a public safety perspective, the real challenge is not only to classify accident severity accurately, but also to convert those predictions into actionable intelligence for future risk monitoring.

This project addresses both tasks:

1. **Severity Classification**  
   Predict accident severity as one of three classes:
   - Slight
   - Serious
   - Fatal

2. **Forward Hotspot Forecasting**  
   Convert severity probabilities into a spatial risk score, aggregate that risk over time and space, and forecast next-month hotspot burden for geographic cells.

The result is a workflow that supports both **retrospective severity modelling** and **prospective risk prioritisation**.

---

## 2. Research Objective

The central aim of this project is:

> To evaluate whether UK road accident severity can be predicted using spatio-temporal, environmental, and vehicle-related features, and whether those predictions can be extended into future-oriented hotspot forecasting.

This is not just a model-comparison exercise. The project also explores:
- imbalance-aware evaluation,
- temporal generalisation,
- explainable tree ensembles,
- and spatial decision-support outputs.

---

## 3. Dataset

The project uses the **Road Accident Casualties Dataset** accessed from Kaggle:

- Dataset link: <https://www.kaggle.com/datasets/nezukokamaado/road-accident-casualties-dataset>

Based on the project materials, the raw dataset contained:
- **660,679 rows**
- **14 original variables**

After data cleaning, duplicate removal, imputation, capping, and feature engineering, the final modelling dataset contained:
- **660,579 rows**
- **17 engineered features** used in the main structured modelling pipeline

### Target variable
- `Accident_Severity`
  - Slight
  - Serious
  - Fatal

### Observed project characteristics
- Strong class imbalance
- Clear temporal patterns across months and weekdays
- Spatial clustering across geographic regions
- Right-skewed distributions in casualty and vehicle count features

---

## 4. Project Workflow

The project follows a practical data science lifecycle:

### A. Problem framing
- Define accident severity prediction as an imbalanced multiclass classification task
- Extend the task into future hotspot forecasting

### B. Data understanding
- Inspect distributions, missing values, duplicates, and feature types
- Explore severity imbalance, spatial concentration, and temporal trends

### C. Data preparation
- Remove duplicates
- Impute missing values
- Cap extreme values in count-based variables
- Create engineered features such as:
  - weekend flag
  - season
  - grouped vehicle category
  - simplified weather category
  - regional risk score
  - cyclical month/day-of-week encodings

### D. Modelling
- Train baseline and imbalance-aware classifiers
- Tune the strongest candidate with Optuna
- Explain the final classifier with SHAP
- Translate class probabilities into hotspot scores
- Build a leakage-safe forecasting pipeline for next-month hotspot burden

### E. Evaluation
- Evaluate classification using:
  - Accuracy
  - Macro F1
  - Macro Recall
  - ROC-AUC
- Evaluate forecasting using:
  - MAE
  - RMSE
  - Top-10 hotspot hit rate

### F. Interpretation and deployment thinking
- Identify influential features
- Map risk hotspots
- Produce a forward hotspot forecast for unseen future periods

---

## 5. Repository Contents

The project materials used in the report include the following core files:

```text
.
├── Problem_Framing,EDA,Data_Preprocessing,Feature_Engineering.ipynb
├── UK_Road_Accident_Severity_Models.ipynb
├── Hotspot Forecasting.ipynb
├── CatBoost_Optuna_Tuned.cbm
├── CatBoost_Optuna_Tuned.pkl
├── cleaned_data.csv
├── best_model_performance_record.csv
├── table (2).csv
├── table (3).csv
├── model_comparison_results.xlsx
├── catboost_info
├── shap_outputs
├── .gitignore
├── LICENSE
└── README.md

```

### What each file does

#### `Problem_Framing,EDA,Data_Preprocessing,Feature_Engineering.ipynb`
Handles:
- problem framing
- dataset inspection
- exploratory data analysis
- missing value handling
- duplicate removal
- outlier capping
- feature engineering

#### `UK_Road_Accident_Severity_Models.ipynb`
Handles:
- train/validation/test splitting
- model training
- baseline and SMOTE-based comparisons
- Optuna tuning
- SHAP explainability
- classification performance comparison

#### `Hotspot Forecasting.ipynb`
Handles:
- conversion of class probabilities into hotspot scores
- monthly spatial aggregation
- leakage-safe lag and rolling features
- CatBoost regression for next-month hotspot burden
- hotspot ranking evaluation

#### `CatBoost_Optuna_Tuned.cbm` and `CatBoost_Optuna_Tuned.pkl`
Saved versions of the final tuned CatBoost classifier.

#### `model_comparison_results.*` and `best_model_performance_record.csv`
Saved evaluation summaries for model benchmarking.

---

## 6. Data Preparation Summary

The project used a structured preprocessing pipeline grounded in the notebooks.

### Key steps

- Converted accident date into datetime format
- Extracted year, month, and day-of-week features
- Removed duplicate records in multiple stages
- Imputed missing values for both numerical and categorical fields
- Applied 99th-percentile style capping to reduce leverage from extreme count values
- Grouped complex categorical features into simpler analytical categories
- Built a `Regional_Risk_Score` from area-level accident concentration
- Applied cyclical encoding to month and weekday signals

### Why this matters

This preprocessing work is one of the strongest parts of the project because it improves:
- model stability,
- interpretability,
- temporal realism,
- and compatibility with tree-based algorithms.

---

## 7. Train / Validation / Test Strategy

A major strength of the project is that it does **not** rely on a random split.

Instead, it uses a **temporal split**:
- **Training:** data up to 2020
- **Validation:** 2021
- **Test:** 2022

This matters because it tests whether the model can generalise to **future periods**, which is more realistic for deployment than a shuffled split.

---

## 8. Models Trained

The project compares several structured-data classifiers.

### Baseline models
- CatBoost
- LightGBM
- XGBoost
- Random Forest
- Decision Tree
- Artificial Neural Network with class weighting

### Imbalance-aware variants
- CatBoost + SMOTE
- LightGBM + SMOTE
- XGBoost + SMOTE
- Random Forest + SMOTE
- Decision Tree + SMOTE
- ANN + SMOTE

### Final tuned model
- **CatBoost (Optuna tuned)**

The tuning process searched over parameters such as:
- iterations
- learning rate
- depth
- L2 regularisation
- random strength
- bagging temperature

---

## 9. Best Classification Results

The strongest final model in the project is the **Optuna-tuned CatBoost classifier**.

### Final test performance

| Model | Accuracy | Macro F1 | Macro Recall | ROC-AUC |
|---|---:|---:|---:|---:|
| CatBoost_Optuna_Tuned | 0.8592 | 0.3102 | 0.3343 | 0.6878 |

### Interpretation

This result is important because:
- the dataset is highly imbalanced,
- fatal cases are very rare,
- and raw accuracy alone would be misleading.

Some simpler models achieved a higher Macro F1, but they did so with a much weaker ROC-AUC and lower overall discrimination. For this reason, the tuned CatBoost model was chosen as the strongest deployment-oriented option.

---

## 10. Explainability with SHAP

To make the model more interpretable, the project applies **SHAP** analysis to the saved CatBoost classifier.

### High-impact features reported in the project
- Number of vehicles
- Urban or rural area
- Number of casualties
- Light conditions
- Road type
- Latitude
- Longitude
- Regional risk score

### Why this is useful

SHAP helps answer an important practical question:

> Which features are driving severity predictions, and do those drivers make sense in the real world?

This makes the model easier to justify in an academic report and more suitable for decision-support use.

---

## 11. Hotspot Scoring Logic

After classification, the project converts class probabilities into a severity-weighted hotspot score:

```text
Hotspot Score = 0.5 × P(Serious) + 1.0 × P(Fatal)
```

This design gives:
- moderate weight to serious collisions
- full weight to fatal-risk probability

That means the hotspot layer does not simply count accidents. It prioritises places with a higher burden of severe outcomes.

---

## 12. Forward Hotspot Forecasting

One of the most valuable extensions in the repository is the forecasting pipeline.

### Forecasting workflow
- Aggregate accident-level hotspot scores into monthly spatial cells
- Create a complete cell-month panel
- Add leakage-safe lagged and rolling features
- Train a CatBoost regressor to predict next-month hotspot burden

### Forecasting performance

| Split | MAE | RMSE |
|---|---:|---:|
| Validation | 0.0187 | 0.0383 |
| Test | 0.0176 | 0.0357 |

### Ranking usefulness
- Mean **Top-10 hotspot hit rate** on the 2022 test period: **0.3273**

This means the model was able to recover roughly three out of the ten highest-burden cells, on average, during the test months.

### First unseen forecast month
For **January 2023**, the forecasting notebook generated predictions for **85,976 spatial cells**. The top-ranked forecast cell was:

- `51.51_-0.13`

with a predicted hotspot total of approximately **0.9802**.

---

## 13. Why This Project Matters

This repository is useful because it shows how to move beyond a single classification model and build a richer, operationally meaningful workflow.

### Practical value
- Supports road safety monitoring
- Helps prioritise high-risk areas
- Adds interpretability through SHAP
- Uses future-period validation rather than random evaluation
- Demonstrates how probability outputs can be converted into planning signals

### Technical value
- Good example of handling imbalanced multiclass data
- Good example of structured feature engineering for tabular ML
- Strong demonstration of temporal validation
- Useful reference for connecting classification and forecasting in one project

---

## 14. How to Run the Project

### Step 1: Clone the repository

```bash
git clone <your-repository-url>
cd <your-repository-folder>
```

### Step 2: Create and activate a virtual environment

```bash
python -m venv .venv
```

#### Windows
```bash
.venv\Scripts\activate
```

#### macOS / Linux
```bash
source .venv/bin/activate
```

### Step 3: Install dependencies

If you have a `requirements.txt`, run:

```bash
pip install -r requirements.txt
```

If not, install the main libraries used in the notebooks manually:

```bash
pip install pandas numpy matplotlib scikit-learn catboost lightgbm xgboost imbalanced-learn shap optuna jupyter openpyxl
```

> Depending on your environment, you may also need additional plotting or geospatial libraries used in the notebooks.

### Step 4: Launch Jupyter

```bash
jupyter notebook
```

### Step 5: Run notebooks in order

Recommended sequence:

1. `Problem_Framing,EDA,Data_Preprocessing,Feature_Engineering.ipynb`
2. `UK_Road_Accident_Severity_Models.ipynb`
3. `Hotspot Forecasting(1).ipynb`

---

## 15. Suggested Output Artifacts

When fully reproduced, the project can generate outputs such as:
- cleaned modelling dataset
- model comparison tables
- SHAP summary plots
- hotspot maps
- monthly burden plots
- Top-10 hotspot hit-rate charts
- saved trained models
- forecast priority-band summaries

---

## 16. Limitations

This project has several realistic limitations that other programmers should understand before reuse:

1. **Class imbalance remains challenging**  
   Fatal accidents are rare, so minority-class performance is naturally hard.

2. **Dataset source verification should be checked carefully**  
   The project uses the Kaggle access point, but anyone reusing this work should verify the original source, licence, and downstream reuse conditions.

3. **Forecasting is stronger as a ranking tool than a perfect burden estimator**  
   The forecasting model tracks direction and concentration reasonably well, but predicted totals can be smoother than actual totals.

4. **Geospatial representation is cell-based**  
   Rounded coordinates are useful for operational analysis, but they simplify real road-network structure.

---

## 17. Ideas for Improvement

If you want to extend this project, good next steps include:

- adding a deployment app with Streamlit or Flask
- building a proper `requirements.txt`
- packaging preprocessing and inference into reusable functions
- creating a pipeline script for training from raw data
- improving calibration of severity probabilities
- testing alternative hotspot definitions
- experimenting with graph-based or spatial deep learning methods
- adding boundary-based aggregation with districts or local authorities
- comparing CatBoost forecasting against time-series baselines

---

## 18. Ethical and Academic Note

This repository is intended for learning, reproducibility, and portfolio use.

If you adapt it for academic submission:
- verify your dataset permissions,
- document all reused or adapted code clearly,
- and make sure your final work reflects your own understanding and implementation.

For public-safety applications, this project should be treated as a **decision-support prototype**, not a fully validated production system.

---

## 19. Citation

If you use this repository in your own work, cite:

```text
A B M Masum. UK Road Accident Severity Prediction and Forward Hotspot Forecasting.
MSc Data Science Project, University of Hertfordshire.
```

You can replace this with your preferred citation format.

---

## 20. Contact

- Name: A B M Masum
- Email: abmmasum.abm@gmail.com

---

## 21. Quick Summary

If you only read one paragraph, read this:

This project builds a full machine learning pipeline for classifying UK road accident severity and extends those predictions into forward hotspot forecasting. It combines careful preprocessing, temporal validation, imbalance-aware evaluation, CatBoost optimisation, SHAP explainability, and spatial risk forecasting. For anyone working on tabular machine learning, public-safety analytics, or interpretable predictive systems, this repository provides a practical reference for turning raw accident records into useful decision-support outputs.
