"""
features.py
------------
Clinically-grounded feature engineering, based on techniques reported across
the maternal-health-risk ML literature (e.g. Mean Arterial Pressure, Shock
Index, Pulse Pressure, BP Ratio, and a composite risk score).
"""
import pandas as pd


def add_clinical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Mean Arterial Pressure: perfusion pressure, standard obstetric vital
    df["MAP"] = df["DiastolicBP"] + (df["SystolicBP"] - df["DiastolicBP"]) / 3

    # Shock Index: heart rate / systolic BP -- flags hemodynamic instability
    df["ShockIndex"] = df["HeartRate"] / df["SystolicBP"]

    # Pulse Pressure: systolic - diastolic -- arterial stiffness / preeclampsia marker
    df["PulsePressure"] = df["SystolicBP"] - df["DiastolicBP"]

    # BP Ratio: systolic / diastolic
    df["BPRatio"] = df["SystolicBP"] / df["DiastolicBP"]

    # Composite Risk Score: simple additive flag count across known risk thresholds
    df["CompositeRiskScore"] = (
        (df["Age"] >= 35).astype(int)
        + (df["Age"] < 18).astype(int)
        + (df["SystolicBP"] >= 140).astype(int)
        + (df["DiastolicBP"] >= 90).astype(int)
        + (df["BS"] >= 11).astype(int)
        + (df["BodyTemp"] >= 100).astype(int)
        + (df["HeartRate"] >= 90).astype(int)
    )

    return df


ENGINEERED_COLS = ["MAP", "ShockIndex", "PulsePressure", "BPRatio", "CompositeRiskScore"]
BASE_COLS = ["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]
