"""
Heart Disease Prediction System – Streamlit Web App
====================================================
Assignment 2 – Task 2 | Deployment

Uses 4 input attributes:
  1. age      – Age of the patient
  2. thalach  – Maximum heart rate achieved
  3. chol     – Serum cholesterol (mg/dl)
  4. cp       – Chest pain type (0-3)

The trained SVC model (heart_disease_svc_model.pkl) is loaded from the
same directory. Run locally with:  streamlit run heart_disease_app.py
"""

import io
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="♥",
    layout="centered",
)

# ── Inline SVG icons ──────────────────────────────────────────────
ICON_HEART = '<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.3 1.5 4.05 3 5.5l7 7Z"/></svg>'
ICON_HEART_PULSE = '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.3 1.5 4.05 3 5.5l7 7Z"/><path d="M3.22 12H9.5l.5-1 2 4.5 2-7 1.5 3.5h5.27"/></svg>'
ICON_CLIPBOARD = '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="8" height="4" x="8" y="2" rx="1" ry="1"/><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"/></svg>'
ICON_USER = '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>'
ICON_ACTIVITY = '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>'
ICON_DROPLET = '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22a7 7 0 0 0 7-7c0-2-1-3.9-3-5.5s-3.5-4-4-6.5c-.5 2.5-2 4.9-4 6.5C6 11.1 5 13 5 15a7 7 0 0 0 7 7z"/></svg>'
ICON_ZAP = '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>'
ICON_ALERT = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><path d="M12 9v4"/><path d="M12 17h.01"/></svg>'
ICON_CHECK = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><path d="m9 11 3 3L22 4"/></svg>'
ICON_INFO = '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/></svg>'
ICON_CHART = '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18"/><path d="m19 9-5 5-4-4-3 3"/></svg>'
ICON_UPLOAD = '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>'
ICON_GAUGE = '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m12 14 4-4"/><path d="M3.34 19a10 10 0 1 1 17.32 0"/></svg>'
ICON_SHIELD = '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/></svg>'

# ── Custom CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* ── Header ── */
.app-header {
    text-align: center;
    padding: 2rem 1rem 1rem;
}
.app-header .icon-wrap {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 56px; height: 56px;
    border-radius: 16px;
    background: linear-gradient(135deg, #ef4444, #f97316);
    color: #fff;
    margin-bottom: 0.75rem;
    box-shadow: 0 8px 24px rgba(239,68,68,0.3);
}
.app-header h1 {
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    margin: 0;
    background: linear-gradient(135deg, #e2e8f0, #f8fafc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.app-header p {
    font-size: 0.875rem;
    color: #94a3b8;
    margin-top: 0.25rem;
    font-weight: 400;
}

/* ── Section label ── */
.section-label {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-size: 0.8125rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: #94a3b8;
    margin-bottom: 0.75rem;
}
.section-label svg { color: #ef4444; }

/* ── Result cards ── */
.result-card {
    padding: 1.75rem 1.5rem;
    border-radius: 16px;
    text-align: center;
    animation: slideUp 0.45s cubic-bezier(0.16, 1, 0.3, 1);
    margin-top: 0.75rem;
}
.result-card.disease-positive {
    background: linear-gradient(135deg, #dc2626, #ef4444);
    border: 1px solid rgba(239,68,68,0.3);
    box-shadow: 0 8px 32px rgba(239,68,68,0.25);
}
.result-card.disease-negative {
    background: linear-gradient(135deg, #059669, #10b981);
    border: 1px solid rgba(16,185,129,0.3);
    box-shadow: 0 8px 32px rgba(16,185,129,0.25);
}
.result-card .result-icon { margin-bottom: 0.5rem; }
.result-card .result-icon svg { color: #fff; }
.result-card h2 {
    font-size: 1.375rem;
    font-weight: 700;
    color: #fff;
    margin: 0 0 0.25rem 0;
}
.result-card p {
    font-size: 0.875rem;
    color: rgba(255,255,255,0.85);
    margin: 0;
    font-weight: 400;
}

/* ── Confidence gauge ── */
.gauge-wrap {
    text-align: center;
    margin-top: 1rem;
    animation: slideUp 0.5s cubic-bezier(0.16, 1, 0.3, 1);
}
.gauge-wrap .gauge-label {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #94a3b8;
    margin-bottom: 0.5rem;
}
.gauge-wrap .gauge-label svg { color: #ef4444; }
.gauge-bar-track {
    height: 8px;
    background: rgba(148,163,184,0.15);
    border-radius: 99px;
    overflow: hidden;
    max-width: 360px;
    margin: 0 auto;
}
.gauge-bar-fill {
    height: 100%;
    border-radius: 99px;
    transition: width 0.8s cubic-bezier(0.16, 1, 0.3, 1);
}
.gauge-bar-fill.positive { background: linear-gradient(90deg, #dc2626, #f87171); }
.gauge-bar-fill.negative { background: linear-gradient(90deg, #059669, #34d399); }
.gauge-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: #e2e8f0;
    margin-top: 0.35rem;
}

/* ── Reference range card ── */
.ref-range {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.75rem;
    margin-top: 0.75rem;
}
.ref-range .ref-item {
    background: rgba(30, 41, 59, 0.6);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(148, 163, 184, 0.1);
    border-radius: 12px;
    padding: 0.875rem 1rem;
    animation: slideUp 0.4s cubic-bezier(0.16, 1, 0.3, 1);
}
.ref-range .ref-item .ref-label {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 0.6875rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: #94a3b8;
    margin-bottom: 0.25rem;
}
.ref-range .ref-item .ref-label svg { width: 14px; height: 14px; color: #ef4444; }
.ref-range .ref-item .ref-value {
    font-size: 0.8125rem;
    color: #cbd5e1;
    font-weight: 400;
}

/* ── Button overrides ── */
div.stButton > button {
    background: linear-gradient(135deg, #ef4444, #f97316) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    font-size: 0.9375rem !important;
    font-weight: 600 !important;
    padding: 0.7rem 2rem !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 14px rgba(239,68,68,0.3) !important;
    letter-spacing: 0.3px !important;
}
div.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(239,68,68,0.4) !important;
}
div.stButton > button:active {
    transform: translateY(0) !important;
}

/* ── Tabs ── */
div.stTabs [data-baseweb="tab-list"] {
    gap: 2px;
    background: rgba(30,41,59,0.5);
    border-radius: 12px;
    padding: 4px;
}
div.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    font-weight: 500;
    font-size: 0.8125rem;
    padding: 0.5rem 1rem;
}
div.stTabs [aria-selected="true"] {
    background: rgba(239,68,68,0.15) !important;
}

/* ── Footer ── */
.app-footer {
    text-align: center;
    padding: 1.5rem 0 0.5rem;
    font-size: 0.75rem;
    color: #64748b;
    border-top: 1px solid rgba(148,163,184,0.1);
    margin-top: 2rem;
}

/* ── Animations ── */
@keyframes slideUp {
    from { opacity: 0; transform: translateY(12px); }
    to { opacity: 1; transform: translateY(0); }
}
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────
st.markdown(f'''
<div class="app-header">
    <div class="icon-wrap">{ICON_HEART}</div>
    <h1>Heart Disease Predictor</h1>
    <p>Machine Learning Assignment 2 — Task 2 &nbsp;&bull;&nbsp; <strong>Muhammad Hassan Yousaf (FA24-BSE-082)</strong> &nbsp;&bull;&nbsp; Support Vector Classifier</p>
</div>
''', unsafe_allow_html=True)

# ── Load model ───────────────────────────────────────────────────
BASE = Path(__file__).parent

@st.cache_resource
def load_model():
    candidates = [
        BASE / "heart_disease_svc_model.pkl",
        BASE / "models" / "heart_disease_svc_model.pkl",
    ]
    for p in candidates:
        if p.exists():
            with open(p, "rb") as f:
                return pickle.load(f)
    raise FileNotFoundError("heart_disease_svc_model.pkl not found")

@st.cache_data
def load_dataset():
    candidates = [
        BASE / "heart.csv",
        BASE / "data" / "heart.csv",
    ]
    for p in candidates:
        if p.exists():
            return pd.read_csv(p)
    return None

try:
    model = load_model()
except FileNotFoundError:
    st.error("Model file not found. Please run Task2_Heart_Disease_Prediction.py first.")
    st.stop()

# ── Tabs ─────────────────────────────────────────────────────────
tab_predict, tab_batch, tab_model = st.tabs(["Predict", "Batch Predict", "Model Info"])

# Chest pain type mapping
CP_LABELS = {
    0: "Typical Angina",
    1: "Atypical Angina",
    2: "Non-Anginal Pain",
    3: "Asymptomatic"
}

# ═══════════════════════════════════════════════════════════════════
# TAB 1: SINGLE PREDICTION
# ═══════════════════════════════════════════════════════════════════
with tab_predict:
    st.markdown(f'<div class="section-label">{ICON_CLIPBOARD} Patient Information</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Age (years)",
            min_value=20, max_value=100, value=55, step=1,
            help="Patient age in years"
        )
        chol = st.number_input(
            "Serum Cholesterol (mg/dl)",
            min_value=100, max_value=600, value=230, step=1,
            help="Serum cholesterol measured in mg/dl. Normal: <200, Borderline: 200-239, High: 240+"
        )

    with col2:
        thalach = st.number_input(
            "Max Heart Rate Achieved",
            min_value=60, max_value=220, value=150, step=1,
            help="Maximum heart rate during exercise stress test"
        )
        cp = st.selectbox(
            "Chest Pain Type",
            options=[0, 1, 2, 3],
            format_func=lambda x: f"{x} — {CP_LABELS[x]}",
            help="Type of chest pain experienced by the patient"
        )

    # Reference ranges
    st.markdown(f'''
    <div class="ref-range">
        <div class="ref-item">
            <div class="ref-label">{ICON_DROPLET} Cholesterol Ranges</div>
            <div class="ref-value">Normal &lt;200 &bull; Borderline 200-239 &bull; High 240+</div>
        </div>
        <div class="ref-item">
            <div class="ref-label">{ICON_HEART_PULSE} Target Heart Rate</div>
            <div class="ref-value">Max HR = 220 - Age &bull; Target zone: 50-85%</div>
        </div>
    </div>
    ''', unsafe_allow_html=True)

    st.markdown("")
    predict_clicked = st.button("Predict Heart Disease Risk", key="predict_single",
                                use_container_width=True)

    if predict_clicked:
        feature_cols = ["age", "thalach", "chol", "cp"]
        features = pd.DataFrame([[age, thalach, chol, cp]], columns=feature_cols)
        prediction = model.predict(features)[0]

        # Compute confidence from decision_function
        try:
            distance = model.decision_function(features)[0]
            confidence = min(abs(distance) / 2.0, 1.0)
        except Exception:
            confidence = None

        if prediction == 1:
            st.markdown(f'''
            <div class="result-card disease-positive">
                <div class="result-icon">{ICON_ALERT}</div>
                <h2>HEART DISEASE DETECTED</h2>
                <p>The model indicates a higher risk of heart disease. Please consult a cardiologist.</p>
            </div>
            ''', unsafe_allow_html=True)
            gauge_cls = "positive"
        else:
            st.markdown(f'''
            <div class="result-card disease-negative">
                <div class="result-icon">{ICON_CHECK}</div>
                <h2>NO HEART DISEASE</h2>
                <p>The model indicates lower risk. Continue maintaining a healthy lifestyle.</p>
            </div>
            ''', unsafe_allow_html=True)
            gauge_cls = "negative"

        # Confidence gauge
        if confidence is not None:
            pct = int(confidence * 100)
            st.markdown(f'''
            <div class="gauge-wrap">
                <div class="gauge-label">{ICON_GAUGE} Model Confidence</div>
                <div class="gauge-bar-track">
                    <div class="gauge-bar-fill {gauge_cls}" style="width: {pct}%;"></div>
                </div>
                <div class="gauge-value">{pct}%</div>
            </div>
            ''', unsafe_allow_html=True)

        # Input summary
        st.markdown("")
        st.markdown(f'<div class="section-label">{ICON_CHART} Input Summary</div>',
                    unsafe_allow_html=True)
        summary_data = pd.DataFrame({
            "Feature": ["Age", "Max Heart Rate", "Cholesterol", "Chest Pain Type"],
            "Value": [
                f"{age} years",
                f"{thalach} bpm",
                f"{chol} mg/dl",
                f"{cp} — {CP_LABELS[cp]}"
            ],
            "Clinical Reference": [
                "Risk increases with age",
                f"Max theoretical: {220 - age} bpm",
                "Desirable: <200 mg/dl" if chol < 200 else ("Borderline: 200-239" if chol < 240 else "High: 240+"),
                "Asymptomatic pain is often most concerning"
            ]
        })
        st.dataframe(summary_data, use_container_width=True, hide_index=True)

        # Disclaimer
        st.caption("This is a machine learning prediction for educational purposes only. "
                   "Always consult a qualified healthcare provider for medical decisions.")

# ═══════════════════════════════════════════════════════════════════
# TAB 2: BATCH PREDICTION
# ═══════════════════════════════════════════════════════════════════
with tab_batch:
    st.markdown(f'<div class="section-label">{ICON_UPLOAD} Upload CSV for Batch Prediction</div>',
                unsafe_allow_html=True)
    st.caption("CSV must contain columns: `age`, `thalach`, `chol`, `cp`")

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"],
                                     label_visibility="collapsed", key="heart_upload")

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            required_cols = {"age", "thalach", "chol", "cp"}
            missing = required_cols - set(batch_df.columns.str.lower())
            if missing:
                st.error(f"Missing required columns: {', '.join(missing)}")
            else:
                batch_df.columns = batch_df.columns.str.lower()
                st.markdown(f'<div class="section-label">{ICON_CHART} Preview ({len(batch_df)} rows)</div>',
                            unsafe_allow_html=True)
                st.dataframe(batch_df.head(10), use_container_width=True, hide_index=True)

                if st.button("Run Batch Prediction", key="batch_predict_heart"):
                    X_batch = batch_df[["age", "thalach", "chol", "cp"]]
                    preds = model.predict(X_batch)
                    batch_df["Prediction"] = ["Heart Disease" if p == 1 else "No Heart Disease" for p in preds]

                    disease_count = sum(preds == 1)
                    total = len(preds)
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Total Patients", total)
                    c2.metric("Disease Detected", disease_count)
                    c3.metric("Detection Rate", f"{disease_count/total*100:.1f}%")

                    st.dataframe(batch_df, use_container_width=True, hide_index=True)

                    csv_buffer = batch_df.to_csv(index=False).encode("utf-8")
                    st.download_button("Download Results CSV", csv_buffer,
                                       "heart_disease_batch_predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"Error processing file: {e}")

    with st.expander("Download sample CSV template"):
        sample = pd.DataFrame({
            "age": [52, 67, 45, 38, 60],
            "thalach": [168, 108, 150, 170, 120],
            "chol": [212, 300, 225, 174, 280],
            "cp": [0, 1, 2, 3, 0]
        })
        st.dataframe(sample, use_container_width=True, hide_index=True)
        st.download_button("Download Template", sample.to_csv(index=False).encode("utf-8"),
                          "heart_disease_sample_template.csv", "text/csv", key="dl_template_heart")

# ═══════════════════════════════════════════════════════════════════
# TAB 3: MODEL INFO
# ═══════════════════════════════════════════════════════════════════
with tab_model:
    st.markdown(f'<div class="section-label">{ICON_INFO} Model Details</div>',
                unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Algorithm", "SVC (RBF)")
    c2.metric("Features", "4")
    c3.metric("Dataset", "UCI Cleveland")

    st.markdown("")
    st.markdown(f'<div class="section-label">{ICON_SHIELD} Selected Features (Assignment Constraint)</div>',
                unsafe_allow_html=True)

    feature_info = pd.DataFrame({
        "Feature": ["age", "thalach", "chol", "cp"],
        "Description": [
            "Age of the patient (years)",
            "Maximum heart rate achieved during stress test",
            "Serum cholesterol level (mg/dl)",
            "Chest pain type (0-3)"
        ],
        "Type": ["Numeric", "Numeric", "Numeric", "Categorical (0-3)"],
        "Clinical Significance": [
            "Primary risk factor; risk increases with age",
            "Lower max HR correlates with higher disease risk",
            "Higher cholesterol increases cardiovascular risk",
            "Asymptomatic (3) is often most concerning"
        ]
    })
    st.dataframe(feature_info, use_container_width=True, hide_index=True)

    # Show model performance
    df_data = load_dataset()
    if df_data is not None:
        with st.expander("Model Performance on Test Set"):
            try:
                from sklearn.preprocessing import LabelEncoder
                from sklearn.model_selection import train_test_split

                SELECTED = ["age", "thalach", "chol", "cp"]
                df_m = df_data[SELECTED + ["target"]].copy().dropna()
                le_cp = LabelEncoder()
                df_m["cp"] = le_cp.fit_transform(df_m["cp"])

                _, X_test, _, y_test = train_test_split(
                    df_m[SELECTED], df_m["target"], test_size=0.2, random_state=42
                )
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)

                st.metric("Test Accuracy", f"{acc*100:.2f}%")

                cm = confusion_matrix(y_test, y_pred)
                cm_df = pd.DataFrame(cm,
                    index=["Actual: No Disease", "Actual: Disease"],
                    columns=["Predicted: No Disease", "Predicted: Disease"]
                )
                st.markdown("**Confusion Matrix**")
                st.dataframe(cm_df, use_container_width=True)

                report = classification_report(y_test, y_pred, output_dict=True,
                                               target_names=["No Disease", "Disease"])
                report_df = pd.DataFrame(report).transpose()
                st.markdown("**Classification Report**")
                st.dataframe(report_df.style.format("{:.2f}"), use_container_width=True)
            except Exception as e:
                st.info(f"Could not compute performance metrics: {e}")

    with st.expander("About the UCI Heart Disease Dataset"):
        st.markdown("""
        **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/45/heart+disease)

        The Cleveland Heart Disease database is the most widely used subset
        of the original 14-attribute dataset. It contains 303 instances with
        13 features and a binary target variable indicating presence of
        heart disease.

        **Assignment Constraint:** Only 4 input attributes were selected from
        the full set of 13 features, following the project requirements.
        """)

# ── Footer ───────────────────────────────────────────────────────
st.markdown('''
<div class="app-footer">
    ML Assignment 2 — Task 2 &bull; Heart Disease Prediction &bull; UCI Cleveland Dataset &bull; SVC (RBF Kernel)
    <br>Built with Streamlit &bull; Deployed on Modal
</div>
''', unsafe_allow_html=True)
