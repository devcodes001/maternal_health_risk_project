"""
train_pipeline.py
------------------
End-to-end training pipeline for Maternal Health Risk classification.

Steps (each grounded in what the published literature on this exact problem
converges on -- see README.md "Methodology" section for citations):
  1. Load data (real UCI set if present, else the synthetic sample)
  2. Clean: drop duplicate sensor readings, clip physiologically impossible values
  3. Feature engineering: MAP, Shock Index, Pulse Pressure, BP Ratio, Composite score
  4. Train/test split (stratified) BEFORE any resampling, to avoid leakage
  5. SMOTE oversampling on the TRAIN split only (fixes class imbalance toward "mid risk")
  6. Train + 5-fold CV compare: Logistic Regression, Decision Tree, Random Forest,
     SVM, KNN, Naive Bayes, XGBoost
  7. Hyperparameter-tune the best single model with GridSearchCV
  8. Build a Stacking ensemble (the approach most papers found gives the biggest
     accuracy lift over any single model) and compare it to the tuned single model
  9. Evaluate the final chosen model on the held-out test set: accuracy, precision,
     recall, F1 (macro + per-class), confusion matrix, ROC-AUC (OvR)
  10. Save the fitted pipeline (preprocessing + model) to models/best_model.joblib
"""
import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from features import BASE_COLS, ENGINEERED_COLS, add_clinical_features

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "reports" / "figures"
MODEL_DIR = ROOT / "models"
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# 1-3. Load, clean, engineer
# ---------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    real = DATA_DIR / "maternal_health_risk.csv"
    sample = DATA_DIR / "maternal_health_risk_SAMPLE.csv"
    path = real if real.exists() else sample
    print(f"Loading data from: {path.name}")
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    # Physiologically implausible heart-rate sensor readings reported in the
    # literature (values as low as 7 bpm) are treated as sensor noise and dropped.
    df = df[(df["HeartRate"] >= 40) & (df["HeartRate"] <= 140)].reset_index(drop=True)
    print(f"Cleaning: {before} -> {len(df)} rows after de-duplication / sensor-noise removal")
    return df


def prepare_features(df: pd.DataFrame):
    df = add_clinical_features(df)
    X = df[BASE_COLS + ENGINEERED_COLS]
    y = df["RiskLevel"]
    return X, y


# ---------------------------------------------------------------------------
# EDA artifacts
# ---------------------------------------------------------------------------
def run_eda(df: pd.DataFrame):
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(5, 4))
    order = df["RiskLevel"].value_counts().index
    sns.countplot(data=df, x="RiskLevel", order=order, palette="viridis")
    plt.title("Class distribution: Maternal Health Risk Level")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "class_distribution.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 6))
    corr = df[BASE_COLS].assign(
        RiskCode=LabelEncoder().fit_transform(df["RiskLevel"])
    ).corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Feature correlation matrix")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "correlation_heatmap.png", dpi=150)
    plt.close()

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, col in zip(axes.ravel(), BASE_COLS):
        sns.boxplot(data=df, x="RiskLevel", y=col, order=order, ax=ax, palette="viridis")
        ax.set_title(col)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "feature_boxplots_by_risk.png", dpi=150)
    plt.close()

    print(f"EDA figures saved to {FIG_DIR}")


# ---------------------------------------------------------------------------
# 4-5. Split + SMOTE
# ---------------------------------------------------------------------------
def split_and_balance(X, y):
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, stratify=y_enc, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Class balance BEFORE SMOTE (train):", np.bincount(y_train))
    sm = SMOTE(random_state=RANDOM_STATE)
    X_train_bal, y_train_bal = sm.fit_resample(X_train_scaled, y_train)
    print("Class balance AFTER SMOTE (train):", np.bincount(y_train_bal))

    return X_train_bal, X_test_scaled, y_train_bal, y_test, scaler, le


# ---------------------------------------------------------------------------
# 6. Compare baseline models with 5-fold CV
# ---------------------------------------------------------------------------
def compare_models(X_train, y_train):
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "DecisionTree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
        "SVM": SVC(probability=True, random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "NaiveBayes": GaussianNB(),
        "XGBoost": XGBClassifier(
            eval_metric="mlogloss", random_state=RANDOM_STATE
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_macro")
        results[name] = {"mean_f1_macro": scores.mean(), "std_f1_macro": scores.std()}
        print(f"{name:18s} macro-F1 = {scores.mean():.4f} (+/- {scores.std():.4f})")

    return results, models


# ---------------------------------------------------------------------------
# 7. Tune the strongest candidate (Random Forest, per literature consensus)
# ---------------------------------------------------------------------------
def tune_random_forest(X_train, y_train):
    param_grid = {
        "n_estimators": [200, 400],
        "max_depth": [None, 12, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }
    grid = GridSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE),
        param_grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        scoring="f1_macro",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    print("Best RF params:", grid.best_params_)
    print("Best RF CV macro-F1:", grid.best_score_)
    return grid.best_estimator_


# ---------------------------------------------------------------------------
# 8. Stacking ensemble (top performer across the reviewed studies)
# ---------------------------------------------------------------------------
def build_stacking_ensemble(X_train, y_train):
    estimators = [
        ("rf", RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE)),
        ("xgb", XGBClassifier(eval_metric="mlogloss", random_state=RANDOM_STATE)),
        ("svm", SVC(probability=True, random_state=RANDOM_STATE)),
        ("knn", KNeighborsClassifier(n_neighbors=7)),
    ]
    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5,
        n_jobs=-1,
    )
    stack.fit(X_train, y_train)
    return stack


# ---------------------------------------------------------------------------
# 9. Evaluation
# ---------------------------------------------------------------------------
def evaluate(model, X_test, y_test, label_encoder, tag: str):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro")
    report = classification_report(
        y_test, y_pred, target_names=label_encoder.classes_, output_dict=True
    )
    auc = None
    if y_proba is not None:
        try:
            auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
        except ValueError:
            auc = None

    print(f"\n=== {tag} — held-out test set ===")
    print(f"Accuracy: {acc:.4f} | Macro-F1: {f1m:.4f} | Macro-AUC: {auc}")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=label_encoder.classes_, cmap="Blues", ax=ax
    )
    ax.set_title(f"Confusion Matrix — {tag}")
    plt.tight_layout()
    fname = FIG_DIR / f"confusion_matrix_{tag.replace(' ', '_')}.png"
    plt.savefig(fname, dpi=150)
    plt.close()

    return {"accuracy": acc, "macro_f1": f1m, "macro_auc": auc, "report": report}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    df = clean_data(df)
    run_eda(df)

    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test, scaler, le = split_and_balance(X, y)

    print("\n--- Baseline model comparison (5-fold CV, macro-F1) ---")
    cv_results, _ = compare_models(X_train, y_train)

    print("\n--- Hyperparameter tuning: Random Forest ---")
    best_rf = tune_random_forest(X_train, y_train)
    rf_eval = evaluate(best_rf, X_test, y_test, le, "Tuned Random Forest")

    print("\n--- Stacking ensemble ---")
    stack_model = build_stacking_ensemble(X_train, y_train)
    stack_eval = evaluate(stack_model, X_test, y_test, le, "Stacking Ensemble")

    final_model, final_tag, final_eval = (
        (stack_model, "Stacking Ensemble", stack_eval)
        if stack_eval["macro_f1"] >= rf_eval["macro_f1"]
        else (best_rf, "Tuned Random Forest", rf_eval)
    )
    print(f"\nSelected final model: {final_tag} (macro-F1={final_eval['macro_f1']:.4f})")

    joblib.dump(
        {
            "model": final_model,
            "scaler": scaler,
            "label_encoder": le,
            "feature_columns": BASE_COLS + ENGINEERED_COLS,
            "model_name": final_tag,
        },
        MODEL_DIR / "best_model.joblib",
    )

    summary = {
        "cv_results": cv_results,
        "random_forest_test": {k: v for k, v in rf_eval.items() if k != "report"},
        "stacking_test": {k: v for k, v in stack_eval.items() if k != "report"},
        "final_model": final_tag,
    }
    with open(ROOT / "reports" / "results_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\nSaved final model -> models/best_model.joblib")
    print("Saved results summary -> reports/results_summary.json")
    print("Saved figures -> reports/figures/")


if __name__ == "__main__":
    main()
