"""
get_data.py
-----------
Two ways to get the data used by this project:

1. download_real_dataset()  -> pulls the ACTUAL UCI "Maternal Health Risk" dataset
   (1,014 records, 6 features, collected via IoT risk-monitoring sensors from
   hospitals/clinics/maternity homes in rural Bangladesh; Ahmed et al., 2020).
   Requires internet access to archive.ics.uci.edu (works on your own machine;
   blocked in this sandbox).

2. generate_sample_dataset() -> builds a statistically-realistic SYNTHETIC stand-in
   with the same 6 columns, the same 3 risk classes, and feature distributions
   fit to the summary statistics reported across multiple published studies on
   this dataset. Use this only to demo/test the pipeline when you don't have
   internet access to UCI. Swap it out for the real CSV before drawing any
   real conclusions.
"""
import numpy as np
import pandas as pd


def download_real_dataset(save_path="data/maternal_health_risk.csv"):
    """Fetch the real dataset from the UCI ML Repository (id=863)."""
    from ucimlrepo import fetch_ucirepo

    ds = fetch_ucirepo(id=863)
    df = pd.concat([ds.data.features, ds.data.targets], axis=1)
    df.to_csv(save_path, index=False)
    print(f"Saved real UCI dataset ({len(df)} rows) to {save_path}")
    return df


def generate_sample_dataset(n=1014, seed=42, save_path=None):
    """
    Synthetic stand-in dataset, same schema as the UCI Maternal Health Risk set:
        Age, SystolicBP, DiastolicBP, BS (blood sugar, mmol/L),
        BodyTemp (F), HeartRate, RiskLevel {low risk, mid risk, high risk}
    Risk labels are generated from clinically-motivated thresholds (elevated
    BP / blood sugar / age raise risk) plus noise, so the classes are learnable
    but not trivial -- similar in spirit to the real data.
    """
    rng = np.random.default_rng(seed)

    age = rng.integers(13, 70, n).astype(float)
    systolic = np.clip(rng.normal(115, 18, n), 70, 160).round()
    diastolic = np.clip(rng.normal(76, 14, n), 49, 100).round()
    bs = np.clip(rng.normal(8.0, 3.2, n), 6.0, 19.0).round(1)
    body_temp = np.clip(rng.normal(98.4, 1.2, n), 98.0, 103.0).round(1)
    heart_rate = np.clip(rng.normal(74, 9, n), 7, 90).round().astype(int)
    heart_rate = np.where(heart_rate < 40, rng.integers(60, 90, n), heart_rate)  # guard sensor-noise outliers

    # Clinically-motivated composite risk score -> noisy thresholds -> 3 classes
    score = (
        (age > 35).astype(int) * 1.2
        + (age < 18).astype(int) * 0.8
        + (systolic >= 140).astype(int) * 2.0
        + (systolic >= 120).astype(int) * 0.8
        + (diastolic >= 90).astype(int) * 1.8
        + (bs >= 11).astype(int) * 2.2
        + (bs >= 8).astype(int) * 0.8
        + (body_temp >= 100).astype(int) * 1.0
        + (heart_rate >= 85).astype(int) * 0.5
        + rng.normal(0, 1.1, n)
    )

    risk = pd.cut(
        score,
        bins=[-np.inf, 1.6, 3.4, np.inf],
        labels=["low risk", "mid risk", "high risk"],
    )

    df = pd.DataFrame(
        {
            "Age": age.astype(int),
            "SystolicBP": systolic.astype(int),
            "DiastolicBP": diastolic.astype(int),
            "BS": bs,
            "BodyTemp": body_temp,
            "HeartRate": heart_rate.astype(int),
            "RiskLevel": risk.astype(str),
        }
    )

    if save_path:
        df.to_csv(save_path, index=False)
        print(f"Saved synthetic sample dataset ({len(df)} rows) to {save_path}")
    return df


if __name__ == "__main__":
    try:
        download_real_dataset("data/maternal_health_risk.csv")
    except Exception as e:
        print(f"Could not reach UCI ({e}). Falling back to synthetic sample data.")
        generate_sample_dataset(save_path="data/maternal_health_risk_SAMPLE.csv")
