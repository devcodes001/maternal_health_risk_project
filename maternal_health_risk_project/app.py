"""
MaternalCare AI — 50% Project Review Application (Streamlit)
-----------------------------------------------------------
An AI-Driven Maternal Health Risk Prediction & Clinical Decision Support System.
Converts patient physiological vitals into evidence-based risk levels (Low, Mid, High)
using a Tuned Random Forest model trained on the UCI Maternal Health Risk dataset
with clinical feature engineering (MAP, Shock Index, Pulse Pressure, BP Ratio, Composite Score).
"""

import sys
import json
import warnings
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Suppress sklearn unpickling warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PATHS & IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from features import add_clinical_features, BASE_COLS, ENGINEERED_COLS
except Exception:
    BASE_COLS = ["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]
    ENGINEERED_COLS = ["MAP", "ShockIndex", "PulsePressure", "BPRatio", "CompositeRiskScore"]
    def add_clinical_features(df):
        df = df.copy()
        df["MAP"] = df["DiastolicBP"] + (df["SystolicBP"] - df["DiastolicBP"]) / 3
        df["ShockIndex"] = df["HeartRate"] / df["SystolicBP"]
        df["PulsePressure"] = df["SystolicBP"] - df["DiastolicBP"]
        df["BPRatio"] = df["SystolicBP"] / df["DiastolicBP"]
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

MODEL_PATH = ROOT / "models" / "best_model.joblib"
DATA_SAMPLE_PATH = ROOT / "data" / "maternal_health_risk_SAMPLE.csv"
DATA_ENHANCED_PATH = ROOT / "data" / "maternal_health_enhanced.csv"
RESULTS_JSON_PATH = ROOT / "reports" / "results_summary.json"
FIGS_DIR = ROOT / "reports" / "figures"

# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT PAGE CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MaternalCare AI | 50% Project Review",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for clinical dashboard aesthetics
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1E3A8A 0%, #0284C7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #64748B;
        font-size: 1.02rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.03);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.06);
    }
    .badge-pill {
        display: inline-block;
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .badge-low {
        background-color: #DEF7EC;
        color: #03543F;
        border: 1px solid #BCF0DA;
    }
    .badge-mid {
        background-color: #FEF08A;
        color: #854D0E;
        border: 1px solid #FDE047;
    }
    .badge-high {
        background-color: #FDE8E8;
        color: #9B1C1C;
        border: 1px solid #F8B4B4;
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s;
    }
    .sidebar-brand {
        padding: 0.75rem 0;
        text-align: center;
        border-bottom: 1px solid #E2E8F0;
        margin-bottom: 1.25rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# MODEL & DATA CACHING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_trained_model():
    if not MODEL_PATH.exists():
        return None
    try:
        bundle = joblib.load(MODEL_PATH)
        return bundle
    except Exception as e:
        st.error(f"Error loading model bundle: {e}")
        return None

@st.cache_data
def load_sample_dataset():
    if DATA_SAMPLE_PATH.exists():
        return pd.read_csv(DATA_SAMPLE_PATH)
    elif DATA_ENHANCED_PATH.exists():
        return pd.read_csv(DATA_ENHANCED_PATH)
    return None

@st.cache_data
def load_results_summary():
    if RESULTS_JSON_PATH.exists():
        with open(RESULTS_JSON_PATH, "r") as f:
            return json.load(f)
    return None

model_bundle = load_trained_model()
dataset_df = load_sample_dataset()
results_summary = load_results_summary()

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE INITIALIZATION
# ─────────────────────────────────────────────────────────────────────────────
if "patient_records" not in st.session_state:
    st.session_state.patient_records = [
        {
            "timestamp": "2026-09-23 10:15",
            "name": "Ananya Sharma",
            "age": 28,
            "systolic": 115.0,
            "diastolic": 76.0,
            "bs": 6.8,
            "temp": 98.4,
            "hr": 74,
            "risk": "low risk",
            "confidence": "94.2%",
            "map": 89.0,
            "pulse_pressure": 39.0,
        },
        {
            "timestamp": "2026-09-23 11:30",
            "name": "Pooja Patel",
            "age": 34,
            "systolic": 138.0,
            "diastolic": 92.0,
            "bs": 11.2,
            "temp": 98.8,
            "hr": 86,
            "risk": "high risk",
            "confidence": "89.5%",
            "map": 107.3,
            "pulse_pressure": 46.0,
        },
    ]

# Preset values storage
if "input_age" not in st.session_state:
    st.session_state.input_age = 26
if "input_systolic" not in st.session_state:
    st.session_state.input_systolic = 115.0
if "input_diastolic" not in st.session_state:
    st.session_state.input_diastolic = 75.0
if "input_bs" not in st.session_state:
    st.session_state.input_bs = 6.8
if "input_temp" not in st.session_state:
    st.session_state.input_temp = 98.4
if "input_hr" not in st.session_state:
    st.session_state.input_hr = 74
if "input_patient_name" not in st.session_state:
    st.session_state.input_patient_name = "Patient #104"

# Maternal Wellness Tracking Session State
if "water_intake" not in st.session_state:
    st.session_state.water_intake = 6
if "water_goal" not in st.session_state:
    st.session_state.water_goal = 8
if "water_history" not in st.session_state:
    st.session_state.water_history = {"Mon": 7, "Tue": 8, "Wed": 6, "Thu": 8, "Fri": 9, "Sat": 7, "Sun": 6}
if "kick_count" not in st.session_state:
    st.session_state.kick_count = 8
if "kick_timestamps" not in st.session_state:
    st.session_state.kick_timestamps = [
        "10:14 AM", "10:22 AM", "10:35 AM", "11:02 AM", "11:18 AM", "11:42 AM", "12:05 PM", "12:20 PM"
    ]
if "medications" not in st.session_state:
    st.session_state.medications = [
        {"name": "Prenatal Multivitamin", "dosage": "1 Tablet", "timing": "08:30 AM (Breakfast)", "taken": True, "purpose": "Essential vitamins & folic acid"},
        {"name": "Iron & Folic Acid (IFA)", "dosage": "100mg Fe + 500mcg FA", "timing": "01:30 PM (Lunch)", "taken": True, "purpose": "Prevents maternal anemia"},
        {"name": "Calcium Carbonate + Vit D3", "dosage": "500mg", "timing": "08:00 PM (Dinner)", "taken": False, "purpose": "Fetal skeletal development"},
        {"name": "Omega-3 DHA", "dosage": "200mg", "timing": "09:00 PM (Night)", "taken": False, "purpose": "Fetal brain & eye development"},
    ]
if "journal_entries" not in st.session_state:
    st.session_state.journal_entries = [
        {
            "date": "2026-09-23",
            "week": 24,
            "mood": "😊 Happy & Energetic",
            "symptoms": ["Mild Swelling in Ankles"],
            "notes": "Felt active after morning stroll. Baby moved enthusiastically after lunch!",
        },
        {
            "date": "2026-09-22",
            "week": 24,
            "mood": "😴 Fatigued",
            "symptoms": ["Lower Back Pain", "Acid Reflux"],
            "notes": "Tired in the afternoon. Used pregnancy pillow for back relief. Drank tender coconut water.",
        },
    ]
if "appointments" not in st.session_state:
    st.session_state.appointments = [
        {
            "title": "2nd Trimester Ultrasound & Anomaly Scan",
            "doctor": "Dr. Sarah Rao (OB/GYN)",
            "clinic": "Maternity Care Clinic, Room 302",
            "date": "2026-09-28",
            "time": "10:30 AM",
            "type": "Ultrasound",
            "status": "Upcoming",
        },
        {
            "title": "Oral Glucose Tolerance Test (OGTT)",
            "doctor": "Clinical Pathology Lab",
            "clinic": "Main Diagnostic Center",
            "date": "2026-10-05",
            "time": "08:00 AM",
            "type": "Lab Test",
            "status": "Upcoming",
        },
        {
            "title": "Routine Antenatal Checkup #3",
            "doctor": "Dr. Sarah Rao (OB/GYN)",
            "clinic": "Maternity Care Clinic",
            "date": "2026-09-10",
            "time": "11:00 AM",
            "type": "Routine Checkup",
            "status": "Completed",
        },
    ]
if "lmp_date" not in st.session_state:
    st.session_state.lmp_date = "2026-04-08"

def calculate_pregnancy_timeline(lmp_date_val):
    try:
        if isinstance(lmp_date_val, str):
            lmp = datetime.strptime(lmp_date_val, "%Y-%m-%d").date()
        else:
            lmp = lmp_date_val
        today = datetime.now().date()
        days_pregnant = max((today - lmp).days, 0)
        weeks = days_pregnant // 7
        days = days_pregnant % 7
        edd = lmp + pd.Timedelta(days=280)
        
        if weeks <= 13:
            trimester = "1st Trimester"
            fruit = "Lime / Plum (~7.5 cm)"
            milestone = "Critical organogenesis; heartbeat detectable"
        elif weeks <= 27:
            trimester = "2nd Trimester"
            fruit = "Ear of Corn / Papaya (~30 cm, 600g)"
            milestone = "Hearing develops, active fetal kicks and sleep cycles"
        else:
            trimester = "3rd Trimester"
            fruit = "Pineapple / Watermelon (~48 cm, 2.5kg+)"
            milestone = "Rapid lung maturation and brain growth preparing for birth"
            
        progress = min(max(weeks / 40.0, 0.0), 1.0)
        return weeks, days, trimester, edd.strftime("%B %d, %Y"), fruit, milestone, progress
    except Exception:
        return 24, 2, "2nd Trimester", "January 13, 2027", "Ear of Corn (~30 cm)", "Active fetal movements", 0.60

def apply_preset(age, sbp, dbp, bs, temp, hr, name):
    st.session_state.input_age = age
    st.session_state.input_systolic = float(sbp)
    st.session_state.input_diastolic = float(dbp)
    st.session_state.input_bs = float(bs)
    st.session_state.input_temp = float(temp)
    st.session_state.input_hr = int(hr)
    st.session_state.input_patient_name = name

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <h2 style="margin:0; color:#1E3A8A; font-weight:800; font-size:1.4rem;">🩺 MaternalCare AI</h2>
            <span style="font-size:0.78rem; font-weight:600; color:#0284C7; letter-spacing:0.05em;">
                50% PROJECT MILESTONE (FIRST REVIEW)
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation Menu",
        [
            "🩺 Patient Risk Assessment",
            "🌸 Maternal Wellness Tracker",
            "📋 Screening History Log",
            "📊 Dataset & EDA Explorer",
            "🏆 Model Benchmarking",
            "🧪 Clinical Feature Engineering",
            "🎯 50% Project Milestone",
        ],
        index=0,
    )

    st.markdown("---")
    st.markdown("### 📌 Review Specifications")
    st.markdown(
        """
        - **Target:** 3-Class Risk (Low / Mid / High)
        - **Benchmark:** UCI Maternal Health Risk
        - **Final Model:** Tuned Random Forest (SMOTE)
        - **Macro-AUC:** **83.8%** | **F1:** **65.8%**
        - **Features:** 6 Base Vitals + 5 Clinical
        """
    )
    st.info("💡 **Review Mode:** Use presets on the assessment page for instant live evaluation demonstration.")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1: PATIENT RISK ASSESSMENT & LIVE INFERENCE
# ─────────────────────────────────────────────────────────────────────────────
if page == "🩺 Patient Risk Assessment":
    st.markdown('<div class="main-title">Maternal Health Risk Diagnostic Engine</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Real-time clinical inference powered by a Tuned Random Forest Classifier trained on UCI maternal physiological data.</div>',
        unsafe_allow_html=True,
    )

    # Quick Presets for Live Review Demonstration
    st.markdown("##### ⚡ Quick Review Presets (Click for Instant Simulation)")
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        if st.button("🟢 Normal / Low Risk", use_container_width=True):
            apply_preset(25, 110, 75, 6.5, 98.4, 72, "Patient (Low Risk Case)")
            st.rerun()
    with p2:
        if st.button("🟡 Gestational Diabetes (Mid)", use_container_width=True):
            apply_preset(31, 126, 82, 11.8, 98.6, 78, "Patient (Diabetes Case)")
            st.rerun()
    with p3:
        if st.button("🔴 Severe Preeclampsia (High)", use_container_width=True):
            apply_preset(36, 148, 96, 13.5, 99.4, 94, "Patient (Preeclampsia Case)")
            st.rerun()
    with p4:
        if st.button("🟠 Adolescent Tachycardia", use_container_width=True):
            apply_preset(17, 122, 82, 7.4, 100.2, 105, "Patient (Adolescent Case)")
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    col_in, col_out = st.columns([1.1, 1.2], gap="large")

    with col_in:
        st.subheader("1. Enter Patient Physiological Vitals")
        
        patient_name = st.text_input("Patient Identifier / Name", value=st.session_state.input_patient_name)
        
        c_age, c_hr = st.columns(2)
        with c_age:
            age = st.number_input("Maternal Age (Years)", min_value=12, max_value=65, value=st.session_state.input_age, step=1)
        with c_hr:
            heartrate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=160, value=st.session_state.input_hr, step=1)

        c_sbp, c_dbp = st.columns(2)
        with c_sbp:
            systolic = st.number_input("Systolic BP (mmHg)", min_value=60.0, max_value=200.0, value=st.session_state.input_systolic, step=1.0)
        with c_dbp:
            diastolic = st.number_input("Diastolic BP (mmHg)", min_value=40.0, max_value=130.0, value=st.session_state.input_diastolic, step=1.0)

        c_bs, c_temp = st.columns(2)
        with c_bs:
            bs = st.number_input("Blood Glucose - BS (mmol/L)", min_value=3.0, max_value=25.0, value=st.session_state.input_bs, step=0.1)
        with c_temp:
            bodytemp = st.number_input("Body Temperature (°F)", min_value=95.0, max_value=105.0, value=st.session_state.input_temp, step=0.1)

        predict_btn = st.button("🔍 Execute Risk Prediction", type="primary", use_container_width=True)

    with col_out:
        st.subheader("2. Diagnostic Analysis & Clinical Prediction")

        if model_bundle is None:
            st.warning("Model bundle `models/best_model.joblib` not found. Please train the model pipeline.")
        else:
            model = model_bundle["model"]
            scaler = model_bundle["scaler"]
            le = model_bundle["label_encoder"]
            cols = model_bundle["feature_columns"]

            # Construct patient dataframe and add clinical features
            raw_input = pd.DataFrame([{
                "Age": float(age),
                "SystolicBP": float(systolic),
                "DiastolicBP": float(diastolic),
                "BS": float(bs),
                "BodyTemp": float(bodytemp),
                "HeartRate": float(heartrate),
            }])
            
            feats_df = add_clinical_features(raw_input)
            scaled_feats = scaler.transform(feats_df[cols])

            pred_idx = model.predict(scaled_feats)[0]
            pred_label = le.inverse_transform([pred_idx])[0]
            
            # Predict probabilities
            proba_dict = {}
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(scaled_feats)[0]
                for cls_name, prob_val in zip(le.classes_, probs):
                    proba_dict[cls_name] = float(prob_val)

            top_confidence = proba_dict.get(pred_label, 1.0) * 100

            # Compute clinical markers
            map_val = feats_df["MAP"].iloc[0]
            pulse_press = feats_df["PulsePressure"].iloc[0]
            shock_idx = feats_df["ShockIndex"].iloc[0]
            bp_ratio = feats_df["BPRatio"].iloc[0]
            comp_score = feats_df["CompositeRiskScore"].iloc[0]

            # Display prediction banner
            if pred_label == "high risk":
                badge_class = "badge-high"
                badge_title = "🚨 HIGH MATERNAL HEALTH RISK"
                badge_desc = "Clinical urgency detected. High probability of preeclampsia, gestational diabetes, or severe hypertension. Immediate obstetrician referral recommended."
                color_hex = "#E63946"
            elif pred_label == "mid risk":
                badge_class = "badge-mid"
                badge_title = "⚠️ MID / MODERATE HEALTH RISK"
                badge_desc = "Elevated physiological indicators observed. Close antenatal monitoring and follow-up screening recommended within 48–72 hours."
                color_hex = "#F59E0B"
            else:
                badge_class = "badge-low"
                badge_title = "✅ LOW MATERNAL HEALTH RISK"
                badge_desc = "Physiological vitals are within normal obstetric baselines. Continue routine antenatal care and scheduled appointments."
                color_hex = "#10B981"

            st.markdown(
                f"""
                <div style="background-color: {color_hex}15; border-left: 5px solid {color_hex}; padding: 1.2rem; border-radius: 8px; margin-bottom: 1rem;">
                    <div style="font-size: 1.3rem; font-weight: 800; color: {color_hex};">{badge_title}</div>
                    <div style="font-size: 0.95rem; color: #334155; margin-top: 0.3rem;">{badge_desc}</div>
                    <div style="margin-top: 0.75rem; font-weight: 700; color: #1E293B;">
                        Model Confidence: <span style="color:{color_hex}; font-size:1.15rem;">{top_confidence:.1f}%</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Class Probability Gauge / Bar
            if proba_dict:
                prob_df = pd.DataFrame({
                    "Risk Category": [c.title() for c in proba_dict.keys()],
                    "Probability (%)": [v * 100 for v in proba_dict.values()],
                    "Color": ["#E63946" if "high" in c else "#F59E0B" if "mid" in c else "#10B981" for c in proba_dict.keys()]
                })
                
                fig = px.bar(
                    prob_df,
                    x="Probability (%)",
                    y="Risk Category",
                    orientation="h",
                    text=[f"{v:.1f}%" for v in prob_df["Probability (%)"]],
                    color="Risk Category",
                    color_discrete_map={
                        "High Risk": "#E63946",
                        "Mid Risk": "#F59E0B",
                        "Low Risk": "#10B981",
                    },
                    height=200,
                )
                fig.update_layout(
                    margin=dict(l=10, r=20, t=10, b=10),
                    showlegend=False,
                    xaxis=dict(range=[0, 100], title="Probability (%)"),
                    yaxis=dict(title=""),
                )
                st.plotly_chart(fig, use_container_width=True)

            # Clinical Biomarkers Metric Row
            st.markdown("##### 🧬 Computed Clinical Biomarkers")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("MAP", f"{map_val:.1f} mmHg", delta="Elevated" if map_val > 105 else "Normal")
            m2.metric("Pulse Press.", f"{pulse_press:.0f} mmHg", delta="Widened" if pulse_press > 50 else "Normal")
            m3.metric("Shock Index", f"{shock_idx:.2f}", delta="High" if shock_idx > 0.8 else "Normal")
            m4.metric("Risk Flags", f"{comp_score}/7", delta="Alert" if comp_score >= 2 else "Low")

            # Save to log button
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("💾 Save Assessment to Patient Screening Log", use_container_width=True):
                new_entry = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "name": patient_name or "Anonymous Patient",
                    "age": int(age),
                    "systolic": float(systolic),
                    "diastolic": float(diastolic),
                    "bs": float(bs),
                    "temp": float(bodytemp),
                    "hr": int(heartrate),
                    "risk": pred_label,
                    "confidence": f"{top_confidence:.1f}%",
                    "map": round(float(map_val), 1),
                    "pulse_pressure": round(float(pulse_press), 1),
                }
                st.session_state.patient_records.insert(0, new_entry)
                st.success(f"Successfully recorded assessment for {patient_name}!")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: MATERNAL WELLNESS TRACKER (WATER, KICKS, MEDS, JOURNAL, APPOINTMENTS)
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🌸 Maternal Wellness Tracker":
    st.markdown('<div class="main-title">Maternal Wellness & Daily Tracking Hub</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Comprehensive daily tracking for expectant mothers — hydration monitoring, fetal movement kick counter, prenatal vitamin schedules, and symptom journaling.</div>',
        unsafe_allow_html=True,
    )

    # Gestational Age & Pregnancy Timeline Banner
    weeks, days, trimester, edd_str, fruit, milestone, progress = calculate_pregnancy_timeline(st.session_state.lmp_date)

    with st.container():
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, #EFF6FF 0%, #F0FDF4 100%); border: 1px solid #BAE6FD; border-radius: 12px; padding: 1.25rem; margin-bottom: 1.25rem;">
                <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.5rem;">
                    <div>
                        <span style="background-color:#0284C7; color:#FFFFFF; padding:0.25rem 0.65rem; border-radius:9999px; font-size:0.78rem; font-weight:700;">{trimester}</span>
                        <h3 style="margin:0.4rem 0 0.1rem 0; color:#0F172A; font-weight:800;">Week {weeks}, Day {days} of Gestation</h3>
                        <p style="margin:0; color:#475569; font-size:0.92rem;">Estimated Due Date: <strong>{edd_str}</strong></p>
                    </div>
                    <div style="text-align:right;">
                        <span style="font-size:0.85rem; color:#64748B;">Baby Size Comparison</span>
                        <div style="font-size:1.15rem; font-weight:700; color:#0369A1;">👶 {fruit}</div>
                        <div style="font-size:0.8rem; color:#64748B;">{milestone}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(progress, text=f"Pregnancy Progress: {weeks}/40 Weeks ({int(progress*100)}%)")

    with st.expander("⚙️ Adjust Pregnancy Profile & LMP Date"):
        c_lmp1, c_lmp2 = st.columns([1, 2])
        with c_lmp1:
            try:
                curr_date = datetime.strptime(st.session_state.lmp_date, "%Y-%m-%d").date()
            except Exception:
                curr_date = datetime.now().date() - pd.Timedelta(weeks=24)
            new_lmp = st.date_input("Last Menstrual Period (LMP)", value=curr_date)
            if str(new_lmp) != st.session_state.lmp_date:
                st.session_state.lmp_date = str(new_lmp)
                st.rerun()
        with c_lmp2:
            st.info("💡 Tracking gestational age helps calibrate maternal blood pressure norms and expected fetal movement thresholds.")

    st.markdown("<br>", unsafe_allow_html=True)

    # 5 Dedicated Tracking Tabs
    tab_water, tab_kicks, tab_meds, tab_journal, tab_visits = st.tabs([
        "💧 Water & Hydration",
        "👶 Fetal Kick Counter",
        "💊 Medications & Vitamins",
        "📝 Symptoms & Mood Journal",
        "📅 Antenatal Appointments"
    ])

    # ── TAB 1: WATER & HYDRATION ─────────────────────────────────────────────
    with tab_water:
        st.markdown("### 💧 Daily Hydration Tracker")
        w_cur = st.session_state.water_intake
        w_goal = st.session_state.water_goal
        w_liters = w_cur * 0.25
        w_pct = min(w_cur / w_goal, 1.0)

        w_col1, w_col2 = st.columns([1.1, 1.4], gap="large")

        with w_col1:
            st.markdown(
                f"""
                <div class="metric-card" style="text-align:center; padding: 1.8rem 1rem;">
                    <div style="font-size: 2.8rem; font-weight:800; color:#0284C7;">{w_cur} <span style="font-size:1.4rem; color:#64748B;">/ {w_goal} Glasses</span></div>
                    <div style="font-size:1.05rem; font-weight:600; color:#334155; margin-top:0.2rem;">{w_liters:.2f} L of 2.0 L Goal</div>
                    <div style="margin-top:0.8rem; font-size:0.88rem; color:#0369A1; font-weight:700;">
                        {'🎉 Daily Hydration Goal Achieved!' if w_cur >= w_goal else f'{int(w_pct*100)}% of recommended intake'}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.progress(w_pct)

            st.markdown("##### Quick Water Log")
            btn1, btn2, btn3, btn4 = st.columns(4)
            with btn1:
                if st.button("➕ 1 Cup (250ml)", use_container_width=True):
                    st.session_state.water_intake += 1
                    st.rerun()
            with btn2:
                if st.button("➕ 2 Cups (500ml)", use_container_width=True):
                    st.session_state.water_intake += 2
                    st.rerun()
            with btn3:
                if st.button("➖ 1 Cup", use_container_width=True):
                    st.session_state.water_intake = max(0, st.session_state.water_intake - 1)
                    st.rerun()
            with btn4:
                if st.button("🔄 Reset", use_container_width=True):
                    st.session_state.water_intake = 0
                    st.rerun()

        with w_col2:
            st.markdown("##### 7-Day Hydration Consistency")
            hist = st.session_state.water_history
            df_hist = pd.DataFrame(list(hist.items()), columns=["Day", "Cups Consumed"])
            
            fig_w = px.bar(
                df_hist,
                x="Day",
                y="Cups Consumed",
                text="Cups Consumed",
                color="Cups Consumed",
                color_continuous_scale="Blues",
                height=240,
            )
            fig_w.add_hline(y=w_goal, line_dash="dash", line_color="#E11D48", annotation_text="Goal (8 Cups)")
            fig_w.update_layout(margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
            st.plotly_chart(fig_w, use_container_width=True)

            st.info("💡 **ACOG Obstetric Advice:** Maintaining 2.0–2.5 Liters of water daily supports amniotic fluid turnover, prevents maternal dehydration-induced contractions, and reduces urinary tract infections.")

    # ── TAB 2: FETAL KICK COUNTER ────────────────────────────────────────────
    with tab_kicks:
        st.markdown("### 👶 Fetal Movement & Kick Counter")
        st.markdown("Track your baby's kicks to monitor fetal neurological and physical wellbeing (recommended especially from Week 28 onwards).")

        k_col1, k_col2 = st.columns([1, 1.3], gap="large")

        with k_col1:
            kicks = st.session_state.kick_count
            st.markdown(
                f"""
                <div class="metric-card" style="text-align:center; padding: 1.5rem 1rem;">
                    <div style="font-size: 0.95rem; color:#64748B; font-weight:600;">MOVEMENTS LOGGED TODAY</div>
                    <div style="font-size: 3.5rem; font-weight:900; color:{'#059669' if kicks>=10 else '#D97706'};">{kicks}</div>
                    <div style="margin-top:0.3rem;">
                        <span class="{'badge-low' if kicks>=10 else 'badge-mid'}">
                            {'✅ Reassuring Fetal Activity (10+ Reached)' if kicks>=10 else '⏳ Monitoring in Progress'}
                        </span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🦶 Record Baby Kick / Movement", type="primary", use_container_width=True):
                st.session_state.kick_count += 1
                t_now = datetime.now().strftime("%I:%M %p")
                st.session_state.kick_timestamps.append(t_now)
                st.rerun()

            if st.button("🔄 Reset Kick Session", use_container_width=True):
                st.session_state.kick_count = 0
                st.session_state.kick_timestamps = []
                st.rerun()

        with k_col2:
            st.markdown("##### Session Movement Log")
            t_stamps = st.session_state.kick_timestamps
            if t_stamps:
                df_kicks = pd.DataFrame({
                    "Kick #": list(range(1, len(t_stamps) + 1)),
                    "Time Recorded": t_stamps,
                    "Status": ["Verified Movement"] * len(t_stamps),
                })
                st.dataframe(df_kicks, use_container_width=True, height=220)
            else:
                st.info("No kicks logged in current session. Press 'Record Baby Kick' when you feel movement.")

            st.markdown(
                """
                > **Clinical Protocol (Count-to-10 Method):**
                > Find a comfortable position resting on your left side. Note how long it takes to feel 10 distinct movements (kicks, rolls, flutters). 
                > If you do not feel 10 movements within 2 hours, drink cold water, rest, and contact your healthcare provider.
                """
            )

    # ── TAB 3: MEDICATIONS & VITAMINS ────────────────────────────────────────
    with tab_meds:
        st.markdown("### 💊 Prenatal Medications & Vitamin Schedule")
        meds = st.session_state.medications

        taken_cnt = sum(1 for m in meds if m.get("taken", False))
        total_meds = len(meds)
        med_pct = int((taken_cnt / total_meds * 100)) if total_meds else 0

        m_k1, m_k2 = st.columns([1, 3])
        with m_k1:
            st.metric("Adherence Today", f"{taken_cnt} / {total_meds}", delta=f"{med_pct}%")
        with m_k2:
            st.progress(med_pct / 100.0)

        st.markdown("##### Daily Prescriptions Checklist")
        for idx, med in enumerate(meds):
            m_c1, m_c2, m_c3 = st.columns([0.15, 0.65, 0.20])
            with m_c1:
                is_checked = st.checkbox("", value=med.get("taken", False), key=f"med_chk_{idx}")
                if is_checked != med.get("taken", False):
                    st.session_state.medications[idx]["taken"] = is_checked
                    st.rerun()
            with m_c2:
                st.markdown(f"**{med['name']}** ({med['dosage']}) — *{med['timing']}*")
                st.caption(f"Purpose: {med['purpose']}")
            with m_c3:
                status_text = "✅ Taken" if med.get("taken", False) else "⏳ Pending"
                st.markdown(f"<span style='font-weight:700; color:{'#059669' if med.get('taken') else '#D97706'};'>{status_text}</span>", unsafe_allow_html=True)
            st.markdown("<hr style='margin:0.4rem 0;'>", unsafe_allow_html=True)

        with st.expander("➕ Add New Medication / Supplement"):
            with st.form("new_med_form"):
                new_m_name = st.text_input("Medication / Vitamin Name", placeholder="e.g. Calcium + D3")
                new_m_dose = st.text_input("Dosage", placeholder="e.g. 500mg")
                new_m_time = st.text_input("Timing / Schedule", placeholder="e.g. 08:00 PM with dinner")
                new_m_purp = st.text_input("Clinical Purpose", placeholder="e.g. Bone strength")
                if st.form_submit_button("Add to Schedule"):
                    if new_m_name:
                        st.session_state.medications.append({
                            "name": new_m_name, "dosage": new_m_dose or "Standard dose",
                            "timing": new_m_time or "Daily", "taken": False,
                            "purpose": new_m_purp or "Prenatal support"
                        })
                        st.success(f"Added {new_m_name} to daily routine!")
                        st.rerun()

    # ── TAB 4: SYMPTOMS & MOOD JOURNAL ───────────────────────────────────────
    with tab_journal:
        st.markdown("### 📝 Daily Symptoms & Wellness Journal")
        
        j_col1, j_col2 = st.columns([1.2, 1.3], gap="large")

        with j_col1:
            st.markdown("##### Log Today's Entry")
            j_date = st.date_input("Entry Date", value=datetime.now().date())
            j_mood = st.selectbox(
                "Today's Mood / Energy",
                [
                    "😊 Happy & Energetic",
                    "😌 Calm & Relaxed",
                    "😴 Fatigued & Sleepy",
                    "😰 Anxious / Overwhelmed",
                    "🤢 Nauseous / Unwell"
                ],
            )
            j_symptoms = st.multiselect(
                "Observed Symptoms",
                [
                    "Mild Swelling in Ankles",
                    "Lower Back Pain",
                    "Acid Reflux / Heartburn",
                    "Morning Sickness / Nausea",
                    "Leg Cramps / Restless Legs",
                    "Headache",
                    "Braxton Hicks Contractions",
                    "Pelvic Pressure",
                    "Fatigue / Lethargy",
                ],
                default=["Mild Swelling in Ankles"],
            )
            j_notes = st.text_area("Personal Notes / Observations", placeholder="e.g., Felt active after breakfast. Baby had hiccups in the evening...")

            if st.button("💾 Save Daily Journal Entry", use_container_width=True):
                entry = {
                    "date": str(j_date),
                    "week": weeks,
                    "mood": j_mood,
                    "symptoms": j_symptoms,
                    "notes": j_notes or "No notes recorded.",
                }
                st.session_state.journal_entries.insert(0, entry)
                st.success("Daily journal entry saved successfully!")
                st.rerun()

        with j_col2:
            st.markdown("##### Recent Journal Reflections")
            entries = st.session_state.journal_entries
            if entries:
                for ent in entries:
                    syms_html = ''.join([f"<span class='badge-pill' style='background:#F1F5F9; color:#334155; margin-right:0.3rem; font-size:0.75rem;'>{s}</span>" for s in ent['symptoms']])
                    st.markdown(
                        f"""
                        <div class="metric-card" style="margin-bottom:0.8rem; padding:1rem;">
                            <div style="display:flex; justify-content:space-between;">
                                <strong style="color:#0F172A;">{ent['date']} (Week {ent['week']})</strong>
                                <span style="font-weight:700;">{ent['mood']}</span>
                            </div>
                            <div style="margin: 0.4rem 0;">
                                {syms_html}
                            </div>
                            <p style="margin:0.4rem 0 0 0; color:#475569; font-size:0.88rem;">{ent['notes']}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.info("No journal entries recorded yet.")

    # ── TAB 5: ANTENATAL APPOINTMENTS ────────────────────────────────────────
    with tab_visits:
        st.markdown("### 📅 Antenatal Appointments & Routine Clinical Scans")

        appts = st.session_state.appointments
        
        for idx, ap in enumerate(appts):
            a_col1, a_col2 = st.columns([3, 1])
            with a_col1:
                st.markdown(
                    f"""
                    <div class="metric-card" style="margin-bottom:0.8rem; padding:1rem;">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <h4 style="margin:0; color:#1E3A8A;">{ap['title']}</h4>
                            <span class="{'badge-low' if ap['status']=='Completed' else 'badge-mid'}">{ap['status']}</span>
                        </div>
                        <div style="margin-top:0.4rem; color:#475569; font-size:0.9rem;">
                            📅 <strong>{ap['date']}</strong> at ⏰ <strong>{ap['time']}</strong> | 🩺 {ap['doctor']}
                        </div>
                        <div style="color:#64748B; font-size:0.82rem; margin-top:0.2rem;">📍 {ap['clinic']} (Type: {ap['type']})</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with a_col2:
                btn_lbl = "Mark Done" if ap['status'] == "Upcoming" else "Mark Pending"
                if st.button(btn_lbl, key=f"btn_appt_{idx}", use_container_width=True):
                    new_status = "Completed" if ap['status'] == "Upcoming" else "Upcoming"
                    st.session_state.appointments[idx]["status"] = new_status
                    st.rerun()

        with st.expander("➕ Schedule New Clinical Visit / Test"):
            with st.form("new_appt_form"):
                ap_title = st.text_input("Visit Title", placeholder="e.g. 3rd Trimester Growth Scan")
                ap_doc = st.text_input("Doctor / Lab Name", placeholder="e.g. Dr. Sarah Rao (OB/GYN)")
                ap_clinic = st.text_input("Clinic / Hospital Location", placeholder="e.g. City Hospital, Wing B")
                c_d, c_t, c_ty = st.columns(3)
                with c_d:
                    ap_date = st.date_input("Date")
                with c_t:
                    ap_time = st.text_input("Time", value="10:00 AM")
                with c_ty:
                    ap_type = st.selectbox("Type", ["Routine Checkup", "Ultrasound Scan", "Lab Blood Test", "Specialist Consultation"])
                
                if st.form_submit_button("Save Appointment"):
                    if ap_title:
                        st.session_state.appointments.append({
                            "title": ap_title,
                            "doctor": ap_doc or "Attending Physician",
                            "clinic": ap_clinic or "Main Hospital",
                            "date": str(ap_date),
                            "time": ap_time,
                            "type": ap_type,
                            "status": "Upcoming",
                        })
                        st.success(f"Scheduled {ap_title}!")
                        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2: PATIENT SCREENING HISTORY LOG
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📋 Screening History Log":
    st.markdown('<div class="main-title">Patient Screening & Assessment Records</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Audit trail of all maternal risk assessments conducted during this session.</div>', unsafe_allow_html=True)

    records = st.session_state.patient_records
    
    # KPI metrics
    tot = len(records)
    high_cnt = sum(1 for r in records if r["risk"] == "high risk")
    mid_cnt = sum(1 for r in records if r["risk"] == "mid risk")
    low_cnt = sum(1 for r in records if r["risk"] == "low risk")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Patients Screened", tot)
    k2.metric("High Risk Identified", high_cnt, delta=f"{(high_cnt/tot*100):.0f}%" if tot else "0%")
    k3.metric("Mid Risk Identified", mid_cnt)
    k4.metric("Low Risk Identified", low_cnt)

    st.markdown("---")

    if records:
        df_records = pd.DataFrame(records)
        df_records.columns = [
            "Timestamp", "Patient Name", "Age", "Systolic BP", "Diastolic BP", 
            "Blood Sugar", "Body Temp", "Heart Rate", "Risk Level", "Confidence", "MAP", "Pulse Pressure"
        ]

        # Filter option
        filter_risk = st.multiselect(
            "Filter by Risk Level",
            options=["high risk", "mid risk", "low risk"],
            default=["high risk", "mid risk", "low risk"],
        )
        
        filtered_df = df_records[df_records["Risk Level"].isin(filter_risk)]
        st.dataframe(filtered_df, use_container_width=True, height=350)

        # Export CSV
        csv_data = filtered_df.to_csv(index=False).encode("utf-8")
        c_exp1, c_exp2 = st.columns([1, 4])
        with c_exp1:
            st.download_button(
                label="📥 Export to CSV",
                data=csv_data,
                file_name=f"maternal_screening_log_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with c_exp2:
            if st.button("🗑️ Clear Screening Records"):
                st.session_state.patient_records = []
                st.rerun()
    else:
        st.info("No patient assessments logged yet. Navigate to 'Patient Risk Assessment' to conduct and save screenings.")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3: DATASET & EDA EXPLORER
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📊 Dataset & EDA Explorer":
    st.markdown('<div class="main-title">UCI Dataset & Exploratory Data Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Statistical properties, class distributions, and correlation patterns across maternal health physiological parameters.</div>',
        unsafe_allow_html=True,
    )

    if dataset_df is not None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Records", f"{len(dataset_df):,}")
        c2.metric("Clinical Vitals", f"{len(BASE_COLS)}")
        c3.metric("Target Risk Classes", "3 (Low/Mid/High)")
        c4.metric("Benchmark Source", "UCI ML Repository")

        tab_data, tab_charts, tab_saved_figs = st.tabs(["📋 Data Viewer", "📈 Interactive Visualizations", "🖼️ Published Report Figures"])

        with tab_data:
            st.markdown("##### Raw Dataset Sample (First 15 Rows)")
            st.dataframe(dataset_df.head(15), use_container_width=True)

            st.markdown("##### Statistical Summary")
            st.dataframe(dataset_df.describe().round(2), use_container_width=True)

        with tab_charts:
            row1_1, row1_2 = st.columns(2)
            with row1_1:
                st.markdown("##### Class Distribution")
                if "RiskLevel" in dataset_df.columns:
                    class_counts = dataset_df["RiskLevel"].value_counts().reset_index()
                    class_counts.columns = ["RiskLevel", "Count"]
                    fig_pie = px.pie(
                        class_counts,
                        names="RiskLevel",
                        values="Count",
                        color="RiskLevel",
                        color_discrete_map={"high risk": "#E63946", "mid risk": "#F59E0B", "low risk": "#10B981"},
                        hole=0.45,
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                else:
                    st.info("RiskLevel column not available.")

            with row1_2:
                st.markdown("##### Systolic BP vs Blood Glucose by Risk Level")
                if "SystolicBP" in dataset_df.columns and "BS" in dataset_df.columns:
                    fig_scatter = px.scatter(
                        dataset_df,
                        x="SystolicBP",
                        y="BS",
                        color="RiskLevel" if "RiskLevel" in dataset_df.columns else None,
                        color_discrete_map={"high risk": "#E63946", "mid risk": "#F59E0B", "low risk": "#10B981"},
                        hover_data=["Age", "HeartRate"],
                        labels={"SystolicBP": "Systolic BP (mmHg)", "BS": "Blood Glucose (mmol/L)"},
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)

            st.markdown("##### Vital Distribution Boxplots")
            selected_vital = st.selectbox("Select Vital Parameter to inspect across Risk Levels:", BASE_COLS, index=1)
            if selected_vital in dataset_df.columns and "RiskLevel" in dataset_df.columns:
                fig_box = px.box(
                    dataset_df,
                    x="RiskLevel",
                    y=selected_vital,
                    color="RiskLevel",
                    color_discrete_map={"high risk": "#E63946", "mid risk": "#F59E0B", "low risk": "#10B981"},
                    points="outliers",
                    title=f"Distribution of {selected_vital} stratified by Maternal Health Risk",
                )
                st.plotly_chart(fig_box, use_container_width=True)

        with tab_saved_figs:
            st.markdown("##### Pipeline Evaluation Figures (from `reports/figures/`)")
            f_col1, f_col2 = st.columns(2)
            
            heatmap_path = FIGS_DIR / "correlation_heatmap.png"
            boxplots_path = FIGS_DIR / "feature_boxplots_by_risk.png"
            dist_path = FIGS_DIR / "class_distribution.png"

            with f_col1:
                if heatmap_path.exists():
                    st.image(str(heatmap_path), caption="Clinical Feature Correlation Heatmap", use_container_width=True)
                if dist_path.exists():
                    st.image(str(dist_path), caption="Class Imbalance & Distribution (Ahmed et al.)", use_container_width=True)
            with f_col2:
                if boxplots_path.exists():
                    st.image(str(boxplots_path), caption="Feature Boxplots Grouped by Risk Level", use_container_width=True)
    else:
        st.warning("Sample dataset not found at `data/maternal_health_risk_SAMPLE.csv`.")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4: MODEL BENCHMARKING & EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🏆 Model Benchmarking":
    st.markdown('<div class="main-title">Multi-Model Comparative Benchmarking</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Systematic 5-Fold Stratified Cross-Validation across 7 Machine Learning Classifiers with SMOTE Resampling.</div>',
        unsafe_allow_html=True,
    )

    if results_summary:
        cv_data = results_summary.get("cv_results", {})
        rf_test = results_summary.get("random_forest_test", {})
        stacking_test = results_summary.get("stacking_test", {})

        # Champion Model Spotlight
        st.markdown("### 🥇 Winning Architecture: Tuned Random Forest + SMOTE")
        st.markdown(
            "Synthesizing published literature (Nature *Sci Reports* 2024; Ahmed et al., 2020), Random Forest paired with engineered clinical markers and post-split SMOTE balances high-risk sensitivity with overall accuracy."
        )

        m_acc, m_f1, m_auc, m_stack = st.columns(4)
        m_acc.metric("Test Accuracy", f"{rf_test.get('accuracy', 0.665)*100:.1f}%")
        m_f1.metric("Macro-F1 Score", f"{rf_test.get('macro_f1', 0.658)*100:.1f}%")
        m_auc.metric("Macro One-vs-Rest AUC", f"{rf_test.get('macro_auc', 0.838)*100:.1f}%")
        m_stack.metric("Stacking Ensemble AUC", f"{stacking_test.get('macro_auc', 0.835)*100:.1f}%")

        st.markdown("---")

        # 7-Model Leaderboard
        st.markdown("### 📊 7-Classifier Cross-Validation Leaderboard (Macro-F1)")
        models_list = []
        for name, metrics in cv_data.items():
            models_list.append({
                "Algorithm": name,
                "Mean Macro-F1 (%)": round(metrics.get("mean_f1_macro", 0) * 100, 2),
                "Std Deviation (±%)": round(metrics.get("std_f1_macro", 0) * 100, 2),
            })
        
        models_df = pd.DataFrame(models_list).sort_values(by="Mean Macro-F1 (%)", ascending=False)
        
        col_table, col_bar = st.columns([1, 1.3], gap="large")
        with col_table:
            st.dataframe(models_df, use_container_width=True, hide_index=True)
            st.caption("Note: All classifiers evaluated on identical 5-fold stratified CV splits with SMOTE applied exclusively to the training fold.")
        
        with col_bar:
            fig_cv = px.bar(
                models_df,
                x="Mean Macro-F1 (%)",
                y="Algorithm",
                orientation="h",
                error_x="Std Deviation (±%)",
                color="Mean Macro-F1 (%)",
                color_continuous_scale="Teal",
                title="5-Fold Cross-Validation Performance Comparison",
            )
            fig_cv.update_layout(yaxis=dict(autorange="reversed"), height=320, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_cv, use_container_width=True)

        st.markdown("---")

        # Feature Importance & Confusion Matrix
        col_cm, col_imp = st.columns(2, gap="large")
        
        with col_cm:
            st.markdown("### 🎯 Model Confusion Matrix")
            cm_rf_path = FIGS_DIR / "confusion_matrix_Tuned_Random_Forest.png"
            if cm_rf_path.exists():
                st.image(str(cm_rf_path), caption="Tuned Random Forest Confusion Matrix (Holdout Test Set)", use_container_width=True)
            else:
                st.info("Confusion matrix figure not found.")

        with col_imp:
            st.markdown("### 🌲 Random Forest Feature Importances")
            if model_bundle and "model" in model_bundle and hasattr(model_bundle["model"], "feature_importances_"):
                m = model_bundle["model"]
                cols = model_bundle["feature_columns"]
                imp_df = pd.DataFrame({
                    "Feature": cols,
                    "Importance (%)": m.feature_importances_ * 100,
                }).sort_values(by="Importance (%)", ascending=True)

                fig_imp = px.bar(
                    imp_df,
                    x="Importance (%)",
                    y="Feature",
                    orientation="h",
                    color="Importance (%)",
                    color_continuous_scale="Blues",
                )
                fig_imp.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig_imp, use_container_width=True)
                st.caption("💡 Notice: Engineered features (Composite Risk Score & MAP) constitute over 26% of total model decision weight!")
            else:
                st.info("Model feature importances could not be extracted.")
    else:
        st.warning("Results summary not found at `reports/results_summary.json`.")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 5: CLINICAL FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🧪 Clinical Feature Engineering":
    st.markdown('<div class="main-title">Clinically-Grounded Feature Engineering</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Synthesizing clinical obstetric markers beyond raw sensor signals to enhance ML classification power.</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        Published comparative studies on maternal health datasets highlight that sensor readings alone lack physiological interaction context.
        To address this, our pipeline builds **5 domain-specific engineered features** prior to model training:
        """
    )

    f1, f2 = st.columns(2)
    with f1:
        st.markdown(
            """
            ### 1. Mean Arterial Pressure (MAP)
            $$\\text{MAP} = \\text{Diastolic BP} + \\frac{\\text{Systolic BP} - \\text{Diastolic BP}}{3}$$
            - **Clinical Rationale:** Reflects average arterial perfusion pressure during a single cardiac cycle.
            - **Obstetric Threshold:** MAP $> 105\\text{ mmHg}$ during 2nd/3rd trimester is a recognized risk marker for preeclampsia.
            
            ### 2. Shock Index (SI)
            $$\\text{SI} = \\frac{\\text{Heart Rate}}{\\text{Systolic BP}}$$
            - **Clinical Rationale:** Evaluates occult hemodynamic instability and circulatory compromise.
            - **Threshold:** Normal obstetric range is $0.5 - 0.7$. Values $\\ge 0.8$ flag hypovolemic risk or distress.
            """
        )
    with f2:
        st.markdown(
            """
            ### 3. Pulse Pressure (PP)
            $$\\text{PP} = \\text{Systolic BP} - \\text{Diastolic BP}$$
            - **Clinical Rationale:** Direct measure of arterial stiffness and vascular compliance.
            - **Threshold:** Normal is $30 - 50\\text{ mmHg}$. Widened pulse pressure indicates hypertensive disorder risk.

            ### 4. BP Ratio & Composite Risk Score
            $$\\text{BP Ratio} = \\frac{\\text{Systolic BP}}{\\text{Diastolic BP}}$$
            - **Composite Risk Score:** Additive integer count (0 to 7) of maternal warning flags across age, BP, glucose, body temperature, and pulse rate thresholds.
            """
        )

    st.markdown("---")
    st.markdown("### 🧮 Interactive Biomarker Sandbox")
    st.write("Adjust raw vitals below to observe dynamic recalculation of clinical biomarkers:")
    
    sb_col1, sb_col2 = st.columns(2)
    with sb_col1:
        test_sbp = st.slider("Systolic BP (mmHg)", 80, 180, 135)
        test_dbp = st.slider("Diastolic BP (mmHg)", 50, 120, 88)
    with sb_col2:
        test_hr = st.slider("Heart Rate (bpm)", 50, 140, 84)
        test_bs = st.slider("Blood Glucose (mmol/L)", 4.0, 18.0, 9.5)

    test_df = pd.DataFrame([{
        "Age": 28, "SystolicBP": test_sbp, "DiastolicBP": test_dbp,
        "BS": test_bs, "BodyTemp": 98.4, "HeartRate": test_hr
    }])
    calc_df = add_clinical_features(test_df)

    res1, res2, res3, res4 = st.columns(4)
    res1.metric("Calculated MAP", f"{calc_df['MAP'].iloc[0]:.1f} mmHg", delta="Elevated (>105)" if calc_df['MAP'].iloc[0] > 105 else "Normal")
    res2.metric("Pulse Pressure", f"{calc_df['PulsePressure'].iloc[0]:.0f} mmHg", delta="Widened (>50)" if calc_df['PulsePressure'].iloc[0] > 50 else "Normal")
    res3.metric("Shock Index", f"{calc_df['ShockIndex'].iloc[0]:.2f}", delta="Alert (≥0.8)" if calc_df['ShockIndex'].iloc[0] >= 0.8 else "Normal")
    res4.metric("Composite Score", f"{calc_df['CompositeRiskScore'].iloc[0]} / 7 Flags")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 6: 50% PROJECT MILESTONE & REVIEW DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🎯 50% Project Milestone":
    st.markdown('<div class="main-title">50% Project Review Milestone (Phase 1)</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Official milestone review documentation aligning with the First Review presentation requirements.</div>',
        unsafe_allow_html=True,
    )

    # Progress Bar
    st.markdown("##### 🚀 Project Completion Status: 50% (First Review Defense)")
    st.progress(0.50)

    st.markdown("<br>", unsafe_allow_html=True)

    r_col1, r_col2 = st.columns(2, gap="large")

    with r_col1:
        st.markdown(
            """
            ### ✅ Phase 1: Completed Deliverables (50%)
            1. **Problem Formulation & Literature Grounding:**
               - Synthesized findings across benchmark UCI studies (Ahmed et al., Nature *Sci Reports* 2024).
            2. **Sensor Data Ingestion & Quality Control:**
               - De-duplication and clinical outlier screening on 1,014 records.
            3. **Clinical Feature Engineering:**
               - Constructed Mean Arterial Pressure, Shock Index, Pulse Pressure, and Composite Risk Score.
            4. **Data Balancing via SMOTE:**
               - Addressed minority "mid risk" class distribution post-split without data leakage.
            5. **Multi-Model Benchmarking:**
               - Comparative 5-fold cross-validation across 7 algorithms.
            6. **Hyperparameter Optimization & Ensembling:**
               - GridSearchCV on Random Forest; evaluated Stacking ensemble meta-learners.
            7. **Streamlit Medical Decision Support App:**
               - Deployed interactive inference engine with probability calibration and audit log.
            """
        )

    with r_col2:
        st.markdown(
            """
            ### 🔜 Phase 2: Planned Final Scope (Remaining 50%)
            1. **IoT Wearable Hardware Ingestion:**
               - ESP32 / Arduino BLE integration with continuous pulse oximetry and automated NIBP cuffs.
            2. **Cloud EHR / Hospital Database Sync:**
               - Fast Healthcare Interoperability Resources (FHIR) / HL7 standard compliance.
            3. **Automated Emergency Dispatch Alerts:**
               - SMS / WhatsApp / Twilio urgent notifications to primary care doctors and ASHAs upon High Risk classification.
            4. **Explainable AI (XAI) with SHAP / LIME:**
               - Patient-specific Shapley force plots explaining vital contributions directly to clinical staff.
            5. **Multilingual Audio Support:**
               - Vernacular voice instructions for expectant mothers in rural healthcare settings.
            """
        )

    st.markdown("---")
    st.markdown("### 🏛️ System Architecture Flowchart")
    st.markdown(
        """
        ```mermaid
        graph TD
            A[Maternal Physiological Vitals] --> B[Data Cleaning & Validation]
            B --> C[Clinical Feature Engineering<br>MAP, Shock Index, Pulse Pressure]
            C --> D[StandardScaler Normalization]
            D --> E[Tuned Random Forest Model<br>Trained with SMOTE Resampling]
            E --> F{3-Class Risk Classification}
            F -->|Low Risk| G[Routine Antenatal Care Protocol]
            F -->|Mid Risk| H[48-72h Follow-up & Lab Screenings]
            F -->|High Risk| I[Immediate Obstetric Emergency Protocol]
            E --> J[Confidence Probabilities & Biomarker Analytics]
            J --> K[Streamlit Decision Support Interface]
        ```
        """
    )
    st.caption("Figure: MaternalCare AI Layered Machine Learning & Clinical Decision Architecture (50% Milestone).")

    st.markdown("---")
    st.markdown("### 📊 Project Implementation Timeline (Gantt Chart)")
    st.markdown(
        "Structured 14-week execution roadmap aligned with academic project milestones. "
        "Highlighting **Phase 1 (Completed — 50% Milestone)** and **Phase 2 (Planned / Future Scope)**."
    )

    # Gantt Chart Data tailored to MaternalCare AI
    gantt_tasks = [
        {"No": 1, "Task": "Project planning & clinical requirement analysis", "Start": 1.0, "End": 2.5, "Status": "Completed", "Weeks": "Week 1 - 2"},
        {"No": 2, "Task": "Literature review & clinical gap identification", "Start": 1.0, "End": 3.5, "Status": "Completed", "Weeks": "Week 1 - 3"},
        {"No": 3, "Task": "Problem definition, objectives & clinical scope", "Start": 2.0, "End": 3.5, "Status": "Completed", "Weeks": "Week 2 - 3"},
        {"No": 4, "Task": "Clinical dataset acquisition & sensor data preprocessing", "Start": 2.5, "End": 4.5, "Status": "Completed", "Weeks": "Week 2 - 4"},
        {"No": 5, "Task": "Exploratory Data Analysis (EDA) & physiological profiling", "Start": 3.5, "End": 5.5, "Status": "Completed", "Weeks": "Week 3 - 5"},
        {"No": 6, "Task": "Clinical feature engineering (MAP, Shock Index, PP)", "Start": 4.5, "End": 6.5, "Status": "Completed", "Weeks": "Week 4 - 6"},
        {"No": 7, "Task": "SMOTE class balancing & post-split data partitioning", "Start": 4.5, "End": 6.5, "Status": "Completed", "Weeks": "Week 4 - 6"},
        {"No": 8, "Task": "Multi-model comparative analysis (7 ML classifiers)", "Start": 5.5, "End": 7.5, "Status": "Completed", "Weeks": "Week 5 - 7"},
        {"No": 9, "Task": "Hyperparameter optimization (GridSearchCV) & Stacking", "Start": 6.5, "End": 8.0, "Status": "Completed", "Weeks": "Week 6 - 8"},
        {"No": 10, "Task": "Interactive Streamlit clinical decision support app", "Start": 6.5, "End": 8.5, "Status": "Completed", "Weeks": "Week 6 - 8"},
        {"No": 11, "Task": "Maternal daily tracking suite (Hydration, Kicks, Meds)", "Start": 7.5, "End": 9.5, "Status": "Completed", "Weeks": "Week 7 - 9"},
        {"No": 12, "Task": "Wearable IoT hardware telemetry & sensor interface", "Start": 9.5, "End": 11.5, "Status": "Planned / Pending", "Weeks": "Week 10 - 11"},
        {"No": 13, "Task": "Explainable AI (SHAP) & Emergency alert notifications", "Start": 11.5, "End": 13.0, "Status": "Planned / Pending", "Weeks": "Week 12 - 13"},
        {"No": 14, "Task": "End-to-end clinical validation & final defense prep", "Start": 12.5, "End": 14.0, "Status": "Planned / Pending", "Weeks": "Week 13 - 14"},
    ]

    df_gantt = pd.DataFrame(gantt_tasks)
    df_gantt["Duration"] = df_gantt["End"] - df_gantt["Start"]

    # Plotly Gantt Bar Chart
    fig_gantt = go.Figure()

    # Completed bars
    df_comp = df_gantt[df_gantt["Status"] == "Completed"]
    fig_gantt.add_trace(go.Bar(
        y=df_comp["Task"],
        x=df_comp["Duration"],
        base=df_comp["Start"],
        orientation='h',
        name="Completed (Phase 1)",
        marker=dict(color="#1D4ED8", line=dict(color="#1E40AF", width=1)),
        hovertemplate="<b>%{y}</b><br>Active: Week %{base:.1f} to %{x+base:.1f}<extra></extra>",
    ))

    # Planned bars
    df_plan = df_gantt[df_gantt["Status"] == "Planned / Pending"]
    fig_gantt.add_trace(go.Bar(
        y=df_plan["Task"],
        x=df_plan["Duration"],
        base=df_plan["Start"],
        orientation='h',
        name="Planned / Pending (Phase 2)",
        marker=dict(color="#94A3B8", line=dict(color="#64748B", width=1)),
        hovertemplate="<b>%{y}</b><br>Planned: Week %{base:.1f} to %{x+base:.1f}<extra></extra>",
    ))

    # Milestone Vertical Lines & Annotations
    fig_gantt.add_vline(x=8.5, line_width=2, line_dash="dash", line_color="#059669")
    fig_gantt.add_annotation(
        x=8.5, y=-0.5,
        text="◆ Current Progress: 50% Milestone (Week 8)",
        showarrow=True, arrowhead=2, arrowcolor="#059669",
        font=dict(color="#059669", size=11, family="sans-serif"),
        bgcolor="#DEF7EC", bordercolor="#059669"
    )

    fig_gantt.add_vline(x=14.0, line_width=2, line_dash="dash", line_color="#0284C7")
    fig_gantt.add_annotation(
        x=14.0, y=-0.5,
        text="◆ Project Completion / Final Defense (Week 14)",
        showarrow=True, arrowhead=2, arrowcolor="#0284C7",
        font=dict(color="#0284C7", size=11, family="sans-serif"),
        bgcolor="#E0F2FE", bordercolor="#0284C7"
    )

    fig_gantt.update_layout(
        height=520,
        margin=dict(l=10, r=20, t=30, b=50),
        xaxis=dict(
            title="Project Timeline (Weeks)",
            tickmode="linear",
            tick0=1,
            dtick=1,
            range=[0.5, 14.5],
            gridcolor="#E2E8F0"
        ),
        yaxis=dict(
            autorange="reversed",
            gridcolor="#F1F5F9"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        barmode='stack',
        plot_bgcolor="#FFFFFF",
    )

    st.plotly_chart(fig_gantt, use_container_width=True)

    # Detailed Gantt Breakdown Table
    st.markdown("##### 📋 Activity Schedule & Deliverables Table")
    df_table = df_gantt[["No", "Task", "Weeks", "Status"]].copy()
    st.dataframe(df_table, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #94A3B8; font-size: 0.85rem; padding: 1rem 0;">
        <strong>MaternalCare AI</strong> — An AI-Based Maternal Health Risk Prediction & Clinical Decision Support System.<br>
        50% Project Milestone Review Edition | Department of Computer Science & Engineering.
    </div>
    """,
    unsafe_allow_html=True,
)
