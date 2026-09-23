"""
build_enhanced_dataset.py
--------------------------
Generates a comprehensive 1,500-row synthetic Maternal Health dataset
combining features from:
  - UCI Maternal Health Risk (Ahmed et al., 2020) — original 6 vitals
  - WHO Antenatal Care guidelines — weight/BMI, hemoglobin, urine protein
  - Preeclampsia clinical literature — oedema, headache, vision changes
  - Gestational Diabetes literature — HbA1c proxy via fasting BS
  - National Institute for Health and Care Excellence (NICE) guidelines

Total features: 14 (6 original + 8 extended clinical markers)
Target: 3-class RiskLevel (low risk / mid risk / high risk)
"""
import numpy as np
import pandas as pd
from pathlib import Path

def build_enhanced_dataset(n=1500, seed=42):
    rng = np.random.default_rng(seed)
    # ── Core vitals (same as UCI set) ──────────────────────────
    age        = rng.integers(14, 68, n).astype(float)
    systolic   = np.clip(rng.normal(116, 20, n), 70, 175).round()
    diastolic  = np.clip(rng.normal(76, 15, n), 49, 115).round()
    bs         = np.clip(rng.normal(7.8, 3.0, n), 4.0, 20.0).round(1)
    body_temp  = np.clip(rng.normal(98.4, 1.1, n), 97.0, 103.5).round(1)
    heart_rate = np.clip(rng.normal(74, 10, n), 40, 140).round().astype(int)

    # ── Extended clinical markers ──────────────────────────────
    height_cm   = np.clip(rng.normal(158, 8, n), 140, 185).round(1)
    weight_kg   = np.clip(rng.normal(62, 12, n), 40, 110).round(1)
    bmi         = (weight_kg / (height_cm/100)**2).round(1)
    hemoglobin  = np.clip(rng.normal(11.5, 1.8, n), 6.0, 16.0).round(1)   # g/dL; <11 = anaemia in pregnancy
    # Gestational week (12–40)
    gest_week   = rng.integers(12, 41, n)
    # Urine protein (0=negative,1=trace,2=+1,3=+2,4=+3)
    up_probs    = [0.55, 0.15, 0.15, 0.10, 0.05]
    urine_prot  = rng.choice([0,1,2,3,4], n, p=up_probs)
    # Binary symptoms
    swelling    = (rng.random(n) < 0.3).astype(int)
    headache    = (rng.random(n) < 0.25).astype(int)
    vision_chg  = (rng.random(n) < 0.15).astype(int)
    prev_preg   = rng.integers(0, 5, n)

    # ── Composite risk scoring (clinically motivated) ──────────
    score = (
        (age >= 35).astype(float)  * 1.5
      + (age < 18).astype(float)   * 1.2
      + (systolic >= 140).astype(float) * 2.5
      + (systolic >= 120).astype(float) * 0.8
      + (diastolic >= 90).astype(float) * 2.2
      + (diastolic >= 80).astype(float) * 0.5
      + (bs >= 11).astype(float)   * 2.5
      + (bs >= 7.8).astype(float)  * 0.7
      + (body_temp >= 100).astype(float) * 1.2
      + (heart_rate >= 100).astype(float) * 0.8
      + (hemoglobin < 9).astype(float)  * 2.0
      + (hemoglobin < 11).astype(float) * 0.8
      + urine_prot.astype(float)   * 0.9
      + swelling * 0.8
      + headache * 0.9
      + vision_chg * 1.5
      + (bmi >= 30).astype(float)  * 0.7
      + (bmi < 18.5).astype(float) * 0.5
      + rng.normal(0, 1.2, n)       # label noise
    )

    risk = pd.cut(score, bins=[-np.inf, 2.2, 4.8, np.inf],
                  labels=['low risk','mid risk','high risk'])

    df = pd.DataFrame({
        'Age':          age.astype(int),
        'SystolicBP':   systolic.astype(int),
        'DiastolicBP':  diastolic.astype(int),
        'BS':           bs,
        'BodyTemp':     body_temp,
        'HeartRate':    heart_rate,
        'HeightCM':     height_cm,
        'WeightKG':     weight_kg,
        'BMI':          bmi,
        'Hemoglobin':   hemoglobin,
        'GestationalWeek': gest_week,
        'UrineProtein': urine_prot,
        'Swelling':     swelling,
        'Headache':     headache,
        'VisionChange': vision_chg,
        'PreviousPregnancies': prev_preg,
        'RiskLevel':    risk.astype(str),
    })

    return df

if __name__ == '__main__':
    out = Path(__file__).parent.parent / 'data' / 'maternal_health_enhanced.csv'
    df  = build_enhanced_dataset(n=1500)
    df.to_csv(out, index=False)
    print(f"[OK] Saved enhanced dataset ({len(df)} rows, {df.shape[1]} features) -> {out}")
    print(df['RiskLevel'].value_counts())
    print(df.head(3))
