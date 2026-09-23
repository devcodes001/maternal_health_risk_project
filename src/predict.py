"""
predict.py
----------
Load the trained pipeline and predict maternal health risk for one patient.

Usage:
    python predict.py --age 32 --systolic 140 --diastolic 95 --bs 12.0 --bodytemp 99.5 --heartrate 88
"""
import argparse
from pathlib import Path

import joblib
import pandas as pd

from features import add_clinical_features

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "best_model.joblib"


def predict(age, systolic, diastolic, bs, bodytemp, heartrate):
    bundle = joblib.load(MODEL_PATH)
    model, scaler, le, cols = (
        bundle["model"],
        bundle["scaler"],
        bundle["label_encoder"],
        bundle["feature_columns"],
    )

    raw = pd.DataFrame(
        [{
            "Age": age, "SystolicBP": systolic, "DiastolicBP": diastolic,
            "BS": bs, "BodyTemp": bodytemp, "HeartRate": heartrate,
        }]
    )
    feats = add_clinical_features(raw)[cols]
    scaled = scaler.transform(feats)

    pred = model.predict(scaled)[0]
    label = le.inverse_transform([pred])[0]

    proba = None
    if hasattr(model, "predict_proba"):
        proba = dict(zip(le.classes_, model.predict_proba(scaled)[0].round(3)))

    return label, proba


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Predict maternal health risk level")
    p.add_argument("--age", type=float, required=True)
    p.add_argument("--systolic", type=float, required=True, help="Systolic BP (mmHg)")
    p.add_argument("--diastolic", type=float, required=True, help="Diastolic BP (mmHg)")
    p.add_argument("--bs", type=float, required=True, help="Blood sugar (mmol/L)")
    p.add_argument("--bodytemp", type=float, required=True, help="Body temperature (F)")
    p.add_argument("--heartrate", type=float, required=True, help="Heart rate (bpm)")
    args = p.parse_args()

    label, proba = predict(args.age, args.systolic, args.diastolic, args.bs, args.bodytemp, args.heartrate)
    print(f"Predicted Risk Level: {label.upper()}")
    if proba:
        print("Class probabilities:", proba)
