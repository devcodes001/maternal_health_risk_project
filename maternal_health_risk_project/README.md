# MaternalCare AI — Maternal Health Risk Prediction (50% Milestone Review)

An AI-driven clinical decision support and maternal health risk prediction system that classifies a pregnant woman's health risk level (**Low / Mid / High**) from six routine physiological vitals — synthesized from published literature on the UCI Maternal Health Risk dataset.

---

## 🎯 50% Project Milestone (First Review)

This repository is optimized for the **50% Project Review (Phase 1)** milestone, focusing on:
1. **Clinical Feature Engineering**: Adding Mean Arterial Pressure (MAP), Shock Index, Pulse Pressure, BP Ratio, and Composite Risk Score.
2. **Post-Split SMOTE Balancing**: Mitigating "mid risk" class imbalance without data leakage into the test set.
3. **Multi-Model Benchmarking**: 5-Fold Stratified Cross-Validation across 7 classifiers (Random Forest, XGBoost, SVM, Decision Tree, Logistic Regression, KNN, Naive Bayes).
4. **Interactive Streamlit Diagnostic Engine**: Single-command interactive interface with one-click demo presets, confidence breakdown, biomarker calculations, and patient screening logs.

---

## 🚀 How to Run the Applications

### Option A: Flask Full-Stack Web Application (Week 1–10 Scope)
```bash
# 1. Navigate to the webapp directory
cd webapp

# 2. Run the Flask application
python app.py
```
Open **`http://127.0.0.1:5000`** in your browser. 
- **Demo Patient**: Username: `demo` | Password: `password123` (or click **🤰 Demo Patient Account** on login).
- **Core Features**:
  1. **User Authentication**: Secure patient registration, login/logout, and session management.
  2. **Account Setup & Onboarding**: Gestational age calculator, LMP tracking, EDD detection, emergency contacts, medical history.
  3. **AI Risk Prediction Engine**: Classify maternal risk level (**Low / Mid / High**) with clinical biomarkers (MAP, Shock Index, Pulse Pressure, Composite Score).
  4. **Maternal Care Modules**: Pregnancy progress tracker, daily wellness & hydration logger, prenatal medication checklist, antenatal appointment manager, fetal kick counter, and clinical reports.
  5. **Settings & Profile Hub**: Dedicated hub (`/settings`) to manage passwords, configure alert preferences, and edit clinical profiles.

### Option B: Streamlit Diagnostic Engine (Original Interactive Tool)
```bash
# Launch Streamlit app
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

---

## 📊 Project Timeline & Gantt Chart
- **HTML Interactive Chart**: [gantt_chart.html](gantt_chart.html)
- **High-Resolution PNG**: [gantt_chart.png](gantt_chart.png)
- **Progress**: Weeks 1–10 completed (Green), Week 10–11 in progress (Gold), Weeks 11–14 upcoming (Gray).

---

## 📂 Project Structure

```
maternal_health_risk_project/
├── app.py                            # Streamlit interactive application (50% review UI)
├── requirements.txt                  # Project dependencies (Streamlit, Scikit-learn, etc.)
├── README.md                         # Project documentation
├── data/
│   ├── maternal_health_risk_SAMPLE.csv   # Stand-in benchmark dataset
│   └── maternal_health_enhanced.csv      # Extended clinical markers dataset
├── models/
│   └── best_model.joblib                 # Trained Tuned Random Forest + Scaler + Encoder
├── src/
│   ├── features.py                       # Clinical feature engineering (MAP, SI, PP, etc.)
│   ├── predict.py                        # CLI prediction tool
│   ├── train_pipeline.py                 # Full training & evaluation pipeline
│   ├── build_enhanced_dataset.py         # Synthetic data generation utility
│   └── get_data.py                       # UCI repo download script
├── notebooks/
│   └── maternal_health_risk_prediction.ipynb # Jupyter notebook walk-through
└── reports/
    ├── figures/                          # Heatmaps, boxplots, confusion matrices
    └── results_summary.json              # 5-fold CV & test metrics for 7 models
```

---

## 🩺 Application Modules

1. **Patient Risk Assessment (Live Predictor)**:
   - Input maternal vitals (Age, Systolic BP, Diastolic BP, Blood Glucose, Body Temp, Heart Rate).
   - Fast review presets (Normal Pregnancy, Gestational Diabetes, Severe Preeclampsia, Adolescent Tachycardia).
   - Instant 3-class risk classification with probability confidence bar.
   - Dynamic clinical biomarkers: Mean Arterial Pressure (MAP), Pulse Pressure, Shock Index, Composite Score.
   - Clinical action guidance & red-flag warning triggers.
2. **Maternal Wellness & Daily Tracking Hub**:
   - Gestational age calculator (Weeks & Days), trimester detection, estimated due date, and fetal milestone comparison.
   - **Hydration Tracker**: Daily water intake counter, +1/+2 cup log buttons, 7-day consistency chart, and ACOG hydration guidelines.
   - **Fetal Kick Counter**: Real-time movement logger, timestamp audit trail, and Count-to-10 clinical protocol guidance.
   - **Medications & Supplements**: Daily prenatal vitamin checklist (Iron/Folic Acid, Calcium, DHA) with adherence tracking.
   - **Symptoms & Mood Journal**: Daily mood assessment, pregnancy symptom tagging, and reflection journal.
   - **Antenatal Appointments**: Clinical visit scheduler and scan milestone tracker.
3. **Screening History Log**:
   - Session patient tracking table with filter by risk level.
   - One-click CSV export of assessment history.
4. **Dataset & EDA Explorer**:
   - UCI dataset summary, interactive distributions, and correlation heatmap.
5. **Model Benchmarking**:
   - 7-Model comparative leaderboard with 5-fold CV Macro-F1.
   - Confusion matrix and Random Forest feature importance rankings.
6. **Clinical Feature Engineering**:
   - Mathematical formulation and obstetric rationale for all engineered biomarkers.
   - Dynamic biomarker calculator sandbox.
7. **50% Project Milestone Dashboard**:
   - Clear breakdown of Phase 1 completed deliverables vs Phase 2 planned scope.
