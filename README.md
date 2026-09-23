# Maternal Health Risk Prediction

A complete, literature-grounded machine learning project that classifies a
pregnant woman's health risk level (**low / mid / high**) from six routine
physiological vitals — built by reviewing and consolidating the best
practices across multiple published studies on this exact problem.

## Why this design

This project is not built from a single paper — it synthesizes the design
choices that recur across the strongest published results on the maternal
health risk problem:

| Design choice | Why (source) |
|---|---|
| **Dataset**: UCI "Maternal Health Risk" (Ahmed et al., 2020) — 1,014 IoT-sensor records from rural Bangladeshi clinics, 6 vitals, 3-class risk label | Used as the benchmark dataset across essentially every paper found (UCI ML Repo id=863; Nature *Sci Reports* 2024; multiple 2025–2026 comparative studies) |
| **De-duplication** of repeated sensor readings before modeling | A 2026 reproducibility study found 562 duplicate rows in the raw 1,014-row file and flagged this as a leakage risk if not removed first |
| **Engineered clinical features**: Mean Arterial Pressure, Shock Index, Pulse Pressure, BP Ratio, Composite Risk Score | Same 2026 study built these 5 features on top of the 6 raw vitals and evaluated both feature sets across 7 classifiers |
| **SMOTE oversampling applied strictly after the train/test split** | Standard fix across studies for the "mid risk" class being under-represented; applying it post-split (not pre-split) avoids data leakage into the test set |
| **Comparing 6–9 classifiers**: Logistic Regression, Decision Tree, Random Forest, SVM, KNN, Naive Bayes, XGBoost (+ Ranger/LogitBoost in some papers) | This is the common model shortlist across the comparative-analysis papers |
| **Random Forest as the strongest single model** | A 2025–2026 comparative study (5-fold CV) found Random Forest to be the top performer, ~89% accuracy, macro-AUC ~0.97 |
| **Ensembling (stacking/voting) for a further lift** | The Nature Scientific Reports 2024 paper built a "Quad-Ensemble" (QEML-MHRC) combining multiple ensemble techniques and found the **high-risk class** the easiest to separate (~90% correct), with the biggest confusion between low/mid risk — the same pattern this project's confusion matrix reproduces |
| **Reporting per-class precision/recall/F1 + macro-AUC, not just accuracy** | Necessary because risk-level classes are imbalanced and clinically the cost of misses differs by class |

## ⚠️ Important note on the data

This sandbox environment cannot reach `archive.ics.uci.edu`, so the pipeline
ships with `data/maternal_health_risk_SAMPLE.csv` — a **1,014-row synthetic
dataset** built to match the real dataset's schema and general clinical
logic (same 6 columns, same 3 classes, similar class balance). It lets you
run the whole pipeline immediately, but **the accuracy numbers you'll see
here are not the real published numbers** (real-data studies report
85–100% accuracy with Random Forest/XGBoost/SVM; this synthetic set has more
label noise by design so it currently scores ~65-67% accuracy).

**To get real results:** run `python src/get_data.py` on your own machine
(with internet access) — it will automatically download the real UCI
dataset via the `ucimlrepo` package and everything downstream (features,
models, evaluation) will use it in place of the sample data with zero code
changes.

## Project structure

```
maternal_health_risk_project/
├── data/
│   └── maternal_health_risk_SAMPLE.csv   # synthetic stand-in (see note above)
├── notebooks/
│   └── maternal_health_risk_prediction.ipynb   # full narrative walk-through
├── src/
│   ├── get_data.py          # downloads real UCI data OR generates sample data
│   ├── features.py          # clinical feature engineering
│   ├── train_pipeline.py    # full training/evaluation pipeline (script form)
│   └── predict.py           # CLI: predict risk for a new patient
├── models/
│   └── best_model.joblib    # trained pipeline (preprocessing + best model)
├── reports/
│   ├── figures/              # EDA + confusion matrix plots
│   └── results_summary.json  # CV + test-set metrics for every model
└── requirements.txt
```

## How to run

```bash
pip install -r requirements.txt

# 1. (optional but recommended) get the real dataset
python src/get_data.py

# 2. run the full pipeline: EDA -> feature engineering -> SMOTE -> model
#    comparison -> hyperparameter tuning -> ensemble -> evaluation -> save model
cd src && python train_pipeline.py

# 3. predict risk for a new patient
python predict.py --age 32 --systolic 140 --diastolic 95 --bs 12.0 --bodytemp 99.5 --heartrate 88
```

Or open `notebooks/maternal_health_risk_prediction.ipynb` for the same
pipeline with inline explanations and visualizations.

## Methodology summary

1. **Data cleaning** — drop duplicate rows, remove physiologically impossible
   sensor readings (e.g. heart rate < 40 bpm from sensor glitches).
2. **EDA** — class distribution, correlation heatmap, per-class boxplots for
   every vital (`reports/figures/`).
3. **Feature engineering** — adds Mean Arterial Pressure, Shock Index, Pulse
   Pressure, BP Ratio, and a Composite Risk Score on top of the 6 raw vitals.
4. **Split first, then balance** — stratified 80/20 train/test split, then
   SMOTE applied only to the training fold to fix the "mid risk" imbalance.
5. **Model comparison** — 5-fold stratified CV (macro-F1) across Logistic
   Regression, Decision Tree, Random Forest, SVM, KNN, Naive Bayes, XGBoost.
6. **Tuning** — GridSearchCV on the strongest candidate (Random Forest).
7. **Ensembling** — a Stacking classifier (RF + XGBoost + SVM + KNN base
   learners, Logistic Regression meta-learner) is compared against the tuned
   single model; whichever wins on macro-F1 is saved as the final model.
8. **Evaluation** — accuracy, macro-F1, per-class precision/recall/F1,
   macro-averaged one-vs-rest AUC, and a confusion matrix.

## Results (on the bundled synthetic sample — replace with real data for true numbers)

See `reports/results_summary.json` and `reports/figures/` for the full
breakdown. On the real UCI dataset, published studies consistently report
**Random Forest / XGBoost / SVM in the 85–100% accuracy range**, with the
**high-risk class** the most reliably detected and the **low/mid-risk
boundary** the main source of confusion — worth keeping in mind when
interpreting any model's output clinically.

## Key sources reviewed for this project

- Ahmed, M. (2020). *Maternal Health Risk* [Dataset]. UCI Machine Learning
  Repository. https://doi.org/10.24432/C5DP5D
- Ahmed, M., Kashem, M.A., Rahman, M., Khatun, S. (2020). *Review and
  Analysis of Risk Factor of Maternal Health in Remote Area Using IoT.*
  Lecture Notes in Electrical Engineering, vol 632.
- *Ensemble machine learning framework for predicting maternal health risk
  during pregnancy*, Scientific Reports (Nature), 2024.
- *Comparative Evaluation of Machine Learning Models for Maternal Health
  Risk Prediction Using IoT-Based Data*, International Journal of
  Statistical Sciences, 2025.
- *Maternal Health Risk Prediction in Bangladesh Using Machine Learning*
  (reproducibility/feature-engineering study), 2026.
- *Machine Learning-Based Maternal Health Risk Assessment: A Comparative
  Analysis of Classification Algorithms*, International Journal of
  Statistics in Medical Research, 2025.

## Limitations & next steps (also flagged consistently across the papers)

- The dataset has only 6 raw features and no demographic/socioeconomic
  variables — several authors note this limits real-world generalizability.
- All data comes from one region (rural Bangladesh) — external validation on
  other populations is an open gap.
- Next steps worth exploring: SHAP-based explainability for clinical trust,
  a simple web/Streamlit front-end around `predict.py`, and — if you gather
  more data — recalibrating thresholds per population.
