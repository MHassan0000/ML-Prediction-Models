"""
ML Prediction Models — Unified Web Application
==============================================
Author: Muhammad Hassan Yousaf
Roll Number: FA24-BSE-082
Course: Machine Learning (Assignment 2)
GitHub: https://github.com/MHassan0000/ML-Prediction-Models
"""

import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ── Page Configuration ────────────────────────────────────────────
st.set_page_config(
    page_title="ML Prediction Hub — Hassan Yousaf",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ─────────────────────────────────────────────────────────
BASE = Path(__file__).parent


def find_file(filename: str, subfolder: str = ""):
    candidates = [
        BASE / subfolder / filename if subfolder else None,
        BASE / filename,
        BASE / "models" / filename,
        BASE / "data" / filename,
        BASE / "reports" / filename,
    ]
    for p in candidates:
        if p and p.exists():
            return p
    return BASE / filename


# ── Professional SVG File & System Icons (Zero Emojis) ─────────────
ICON_FILE = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/><path d="M14 2v4a2 2 0 0 0 2 2h4"/></svg>'
ICON_FILE_TEXT = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/><path d="M14 2v4a2 2 0 0 0 2 2h4"/><path d="M10 9H8"/><path d="M16 13H8"/><path d="M16 17H8"/></svg>'
ICON_FOLDER = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 20a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.9a2 2 0 0 1-1.69-.9L9.6 3.9A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2Z"/></svg>'
ICON_DATABASE = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/><path d="M3 12c0 1.66 4 3 9 3s9-1.34 9-3"/></svg>'
ICON_CPU = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="16" height="16" x="4" y="4" rx="2"/><rect width="6" height="6" x="9" y="9" rx="1"/><path d="M15 2v2"/><path d="M15 20v2"/><path d="M2 15h2"/><path d="M2 9h2"/><path d="M20 15h2"/><path d="M20 9h2"/><path d="M9 2v2"/><path d="M9 20v2"/></svg>'
ICON_USER = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>'
ICON_CHECK = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><path d="m9 11 3 3L22 4"/></svg>'
ICON_ALERT = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" x2="12" y1="8" y2="12"/><line x1="12" x2="12.01" y1="16" y2="16"/></svg>'
ICON_UPLOAD = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" x2="12" y1="3" y2="15"/></svg>'
ICON_DOWNLOAD = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" x2="12" y1="15" y2="3"/></svg>'
ICON_EXTERNAL = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M15 3h6v6"/><path d="M10 14 21 3"/><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/></svg>'
ICON_SLIDERS = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="4" x2="4" y1="21" y2="14"/><line x1="4" x2="4" y1="10" y2="3"/><line x1="12" x2="12" y1="21" y2="12"/><line x1="12" x2="12" y1="8" y2="3"/><line x1="20" x2="20" y1="21" y2="16"/><line x1="20" x2="20" y1="12" y2="3"/><line x1="1" x2="7" y1="14" y2="14"/><line x1="9" x2="15" y1="8" y2="8"/><line x1="17" x2="23" y1="16" y2="16"/></svg>'
ICON_ACTIVITY = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>'
ICON_CHART = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" x2="18" y1="20" y2="10"/><line x1="12" x2="12" y1="20" y2="4"/><line x1="6" x2="6" y1="20" y2="14"/></svg>'

# ── Global Styling ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background: #090d16 !important;
    border-right: 1px solid rgba(255,255,255,0.06);
}

.student-card {
    background: linear-gradient(135deg, rgba(99,102,241,0.08), rgba(16,185,129,0.05));
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 14px;
    padding: 1.15rem;
    margin-bottom: 1.25rem;
}
.student-card .badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #818cf8;
    background: rgba(99,102,241,0.15);
    padding: 3px 8px;
    border-radius: 6px;
    margin-bottom: 0.5rem;
}
.student-card h4 {
    margin: 0;
    font-size: 1.05rem;
    font-weight: 700;
    color: #f8fafc;
}
.student-card p {
    margin: 3px 0 0 0;
    font-size: 0.8rem;
    color: #94a3b8;
}

.nav-label {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #64748b;
    margin: 1.25rem 0 0.5rem 0;
}

/* Header Banner */
.portal-header {
    background: linear-gradient(135deg, rgba(15,23,42,0.95), rgba(30,41,59,0.75));
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 1.75rem 2rem;
    margin-bottom: 1.75rem;
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.28);
}
.portal-header h1 {
    font-size: 1.75rem;
    font-weight: 800;
    color: #f8fafc;
    margin: 0 0 0.35rem 0;
    letter-spacing: -0.02em;
}
.portal-header p {
    font-size: 0.9rem;
    color: #94a3b8;
    margin: 0;
}

/* Result Cards */
.result-card {
    padding: 1.5rem;
    border-radius: 14px;
    text-align: center;
    margin-top: 1rem;
    animation: fadeIn 0.4s ease-out;
}
.result-card.success {
    background: linear-gradient(135deg, rgba(16,185,129,0.15), rgba(5,150,105,0.25));
    border: 1px solid rgba(16,185,129,0.4);
}
.result-card.danger {
    background: linear-gradient(135deg, rgba(239,68,68,0.15), rgba(220,38,38,0.25));
    border: 1px solid rgba(239,68,68,0.4);
}
.result-card h3 {
    margin: 0.25rem 0 0.25rem 0;
    font-size: 1.35rem;
    font-weight: 700;
    color: #f8fafc;
}
.result-card p {
    margin: 0;
    font-size: 0.85rem;
    color: #cbd5e1;
}

/* Confidence Bar */
.confidence-box {
    margin: 1rem 0;
    text-align: center;
}
.confidence-bar-bg {
    height: 8px;
    background: rgba(255,255,255,0.08);
    border-radius: 99px;
    overflow: hidden;
    max-width: 320px;
    margin: 6px auto;
}
.confidence-bar-fill {
    height: 100%;
    border-radius: 99px;
}
.confidence-bar-fill.pos { background: linear-gradient(90deg, #10b981, #34d399); }
.confidence-bar-fill.neg { background: linear-gradient(90deg, #ef4444, #f87171); }

/* File Architecture Card */
.file-card {
    background: rgba(30,41,59,0.5);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 0.75rem;
}
.file-card .file-name {
    display: flex;
    align-items: center;
    gap: 8px;
    font-weight: 600;
    color: #f1f5f9;
    font-size: 0.9rem;
}
.file-card .file-desc {
    color: #94a3b8;
    font-size: 0.8rem;
    margin-top: 4px;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar: Student Credentials & Task Selector ───────────────────
with st.sidebar:
    st.markdown(f"""
    <div class="student-card">
        <div class="badge">{ICON_USER} Student Dossier</div>
        <h4>Muhammad Hassan Yousaf</h4>
        <p>Roll No: <strong>FA24-BSE-082</strong></p>
        <p>Program: BS Software Engineering</p>
        <p>Course: Machine Learning (CSC-461)</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f'<div class="nav-label">{ICON_FOLDER} Navigation Menu</div>', unsafe_allow_html=True)
    selected_view = st.radio(
        "Select Module:",
        [
            "Overview & Architecture",
            "Task 1 — Titanic Survival",
            "Task 2 — Heart Disease",
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown(f'<div class="nav-label">{ICON_EXTERNAL} Cloud Deployments</div>', unsafe_allow_html=True)
    st.markdown("""
    - [Titanic on Modal](https://mhassan0000--titanic-predictor-run.modal.run)
    - [Heart Disease on Modal](https://mhassan0000--heart-disease-predictor-run.modal.run)
    - [GitHub Repository](https://github.com/MHassan0000/ML-Prediction-Models)
    """)


# ═══════════════════════════════════════════════════════════════════
# VIEW 1: OVERVIEW & ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════
if selected_view == "Overview & Architecture":
    st.markdown(f"""
    <div class="portal-header">
        <h1>Machine Learning Prediction Systems</h1>
        <p>Assignment 2 &bull; Developed by Muhammad Hassan Yousaf (FA24-BSE-082) &bull; Support Vector Classification</p>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown(f"### {ICON_FILE_TEXT} Task 1: Titanic Survival Prediction", unsafe_allow_html=True)
        st.write("""
        - **Dataset**: Titanic Survival Dataset (891 passenger records)
        - **Model**: Support Vector Classifier (`SVC`) with Radial Basis Function (`rbf`) kernel
        - **Features (7)**: `pclass`, `sex`, `age`, `sibsp`, `parch`, `fare`, `embarked`
        - **Objective**: Predict binary survival status ($0 = \\text{Deceased}$, $1 = \\text{Survived}$)
        """)
        st.info("Live on Modal: `https://mhassan0000--titanic-predictor-run.modal.run`")

    with col_b:
        st.markdown(f"### {ICON_ACTIVITY} Task 2: Heart Disease Prediction", unsafe_allow_html=True)
        st.write("""
        - **Dataset**: UCI Heart Disease Cleveland Dataset (303 patient records)
        - **Model**: Support Vector Classifier (`SVC`) with Radial Basis Function (`rbf`) kernel
        - **Features (4 Constraint)**: `age`, `thalach` (Max HR), `chol` (Cholesterol), `cp` (Chest Pain Type)
        - **Objective**: Clinical risk detection ($0 = \\text{No Disease}$, $1 = \\text{Heart Disease}$)
        """)
        st.info("Live on Modal: `https://mhassan0000--heart-disease-predictor-run.modal.run`")

    st.markdown("---")
    st.markdown(f"### {ICON_FOLDER} Clean Project Structure", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="file-card">
            <div class="file-name">{ICON_DATABASE} data/</div>
            <div class="file-desc">Raw and preprocessed datasets: titanic.csv, heart.csv, encoded variants.</div>
        </div>
        <div class="file-card">
            <div class="file-name">{ICON_CPU} models/</div>
            <div class="file-desc">Serialised SVC pickle models and encoders: titanic_svc_model.pkl, heart_disease_svc_model.pkl.</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="file-card">
            <div class="file-name">{ICON_FILE_CODE} pipelines/</div>
            <div class="file-desc">Training scripts executing the 4-phase ML lifecycle end-to-end.</div>
        </div>
        <div class="file-card">
            <div class="file-name">{ICON_CHART} reports/</div>
            <div class="file-desc">Visual evaluation artifacts: EDA charts, correlation maps, confusion matrices.</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="file-card">
            <div class="file-name">{ICON_FILE_TEXT} notebooks/</div>
            <div class="file-desc">Academic submission notebooks with Hassan Yousaf's metadata and analysis.</div>
        </div>
        <div class="file-card">
            <div class="file-name">{ICON_EXTERNAL} modal/</div>
            <div class="file-desc">Modal serverless container deployment scripts for zero-latency inference.</div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# VIEW 2: TASK 1 — TITANIC SURVIVAL
# ═══════════════════════════════════════════════════════════════════
elif selected_view == "Task 1 — Titanic Survival":
    st.markdown(f"""
    <div class="portal-header">
        <h1>Titanic Passenger Survival Predictor</h1>
        <p>Task 1 &bull; Support Vector Machine (RBF Kernel) &bull; Muhammad Hassan Yousaf (FA24-BSE-082)</p>
    </div>
    """, unsafe_allow_html=True)

    # Load artifacts
    model_path = find_file("titanic_svc_model.pkl", "models")
    data_path = find_file("titanic.csv", "data")

    try:
        with open(model_path, "rb") as f:
            t_model = pickle.load(f)
        le_sex = LabelEncoder()
        le_sex.fit(["female", "male"])
        le_embarked = LabelEncoder()
        le_embarked.fit(["C", "Q", "S"])
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        st.stop()

    tab_pred, tab_batch, tab_dash = st.tabs(["Single Inference", "Batch CSV Prediction", "Model Dashboard"])

    with tab_pred:
        col1, col2 = st.columns(2)
        with col1:
            pclass = st.selectbox("Passenger Class", [1, 2, 3], format_func=lambda x: f"{x}st/nd/rd Class")
            sex = st.selectbox("Sex", ["female", "male"], format_func=lambda x: x.capitalize())
            age = st.slider("Age (Years)", 1, 80, 28)
            fare = st.number_input("Fare Paid ($)", 0.0, 600.0, 32.2, step=1.0)
        with col2:
            sibsp = st.number_input("Siblings / Spouses Aboard", 0, 8, 0)
            parch = st.number_input("Parents / Children Aboard", 0, 6, 0)
            embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"],
                                    format_func=lambda x: {"S": "Southampton (S)", "C": "Cherbourg (C)", "Q": "Queenstown (Q)"}[x])

        if st.button("Run Survival Prediction", use_container_width=True, type="primary"):
            feature_cols = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
            x_in = pd.DataFrame([[pclass, le_sex.transform([sex])[0], age, sibsp, parch, fare, le_embarked.transform([embarked])[0]]], columns=feature_cols)
            pred = t_model.predict(x_in)[0]
            try:
                dist = t_model.decision_function(x_in)[0]
                conf = min(abs(dist) / 2.0, 1.0)
            except Exception:
                conf = 0.85

            pct = int(conf * 100)
            if pred == 1:
                st.markdown(f"""
                <div class="result-card success">
                    {ICON_CHECK}
                    <h3>PASSENGER SURVIVED</h3>
                    <p>The model predicts this passenger would have survived the disaster.</p>
                </div>
                <div class="confidence-box">
                    <span style="font-size:0.8rem; color:#94a3b8;">Confidence Level: <strong>{pct}%</strong></span>
                    <div class="confidence-bar-bg"><div class="confidence-bar-fill pos" style="width:{pct}%;"></div></div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-card danger">
                    {ICON_ALERT}
                    <h3>PASSENGER DID NOT SURVIVE</h3>
                    <p>The model predicts this passenger would not have survived the disaster.</p>
                </div>
                <div class="confidence-box">
                    <span style="font-size:0.8rem; color:#94a3b8;">Confidence Level: <strong>{pct}%</strong></span>
                    <div class="confidence-bar-bg"><div class="confidence-bar-fill neg" style="width:{pct}%;"></div></div>
                </div>
                """, unsafe_allow_html=True)

    with tab_batch:
        st.markdown(f"#### {ICON_UPLOAD} Bulk CSV Inference", unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload CSV file with columns: pclass, sex, age, sibsp, parch, fare, embarked", type=["csv"], key="t_csv")
        if uploaded:
            bdf = pd.read_csv(uploaded)
            bdf.columns = bdf.columns.str.lower()
            req = {"pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"}
            if req.issubset(set(bdf.columns)):
                b_enc = bdf.copy()
                b_enc["sex"] = le_sex.transform(b_enc["sex"])
                b_enc["embarked"] = le_embarked.transform(b_enc["embarked"])
                preds = t_model.predict(b_enc[["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]])
                bdf["Prediction"] = ["Survived" if p == 1 else "Did Not Survive" for p in preds]
                st.success(f"Processed {len(bdf)} records successfully!")
                st.dataframe(bdf.head(15), use_container_width=True)
                csv_bytes = bdf.to_csv(index=False).encode("utf-8")
                st.download_button("Download Annotated CSV", csv_bytes, "titanic_predictions.csv", "text/csv")
            else:
                st.error(f"Missing required columns: {req - set(bdf.columns)}")

    with tab_dash:
        st.markdown(f"#### {ICON_CHART} Model Performance Metrics", unsafe_allow_html=True)
        if data_path.exists():
            df_t = pd.read_csv(data_path)
            f_cols = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
            df_sub = df_t[f_cols + ["survived"]].dropna().copy()
            df_sub["sex"] = le_sex.transform(df_sub["sex"])
            df_sub["embarked"] = le_embarked.transform(df_sub["embarked"])
            from sklearn.model_selection import train_test_split
            _, x_test, _, y_test = train_test_split(df_sub[f_cols], df_sub["survived"], test_size=0.2, random_state=42)
            y_pred = t_model.predict(x_test)

            m1, m2 = st.columns(2)
            m1.metric("Testing Accuracy", f"{accuracy_score(y_test, y_pred)*100:.2f}%")
            m2.metric("Total Test Evaluated", len(y_test))

            st.write("Confusion Matrix:")
            cm = confusion_matrix(y_test, y_pred)
            st.dataframe(pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# VIEW 3: TASK 2 — HEART DISEASE
# ═══════════════════════════════════════════════════════════════════
elif selected_view == "Task 2 — Heart Disease":
    st.markdown(f"""
    <div class="portal-header">
        <h1>Heart Disease Risk Prediction System</h1>
        <p>Task 2 &bull; 4 Clinical Attributes Constraint &bull; Muhammad Hassan Yousaf (FA24-BSE-082)</p>
    </div>
    """, unsafe_allow_html=True)

    # Load artifacts
    h_model_path = find_file("heart_disease_svc_model.pkl", "models")
    h_data_path = find_file("heart.csv", "data")

    try:
        with open(h_model_path, "rb") as f:
            h_model = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading Heart Disease model: {e}")
        st.stop()

    tab_pred2, tab_batch2, tab_dash2 = st.tabs(["Clinical Inference", "Batch CSV Prediction", "Model Dashboard"])

    with tab_pred2:
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider("Patient Age", 25, 80, 54)
            chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 230, step=1,
                                   help="Normal: <200, Borderline: 200-239, High: 240+")
        with col2:
            thalach = st.number_input("Max Heart Rate Achieved (bpm)", 60, 220, 150, step=1,
                                      help="Stress test max heart rate")
            cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3],
                              format_func=lambda x: {0: "0 — Typical Angina", 1: "1 — Atypical Angina", 2: "2 — Non-anginal Pain", 3: "3 — Asymptomatic"}[x])

        if st.button("Predict Clinical Risk", use_container_width=True, type="primary"):
            feature_cols = ["age", "thalach", "chol", "cp"]
            x_in = pd.DataFrame([[age, thalach, chol, cp]], columns=feature_cols)
            pred = h_model.predict(x_in)[0]
            try:
                dist = h_model.decision_function(x_in)[0]
                conf = min(abs(dist) / 2.0, 1.0)
            except Exception:
                conf = 0.85

            pct = int(conf * 100)
            if pred == 1:
                st.markdown(f"""
                <div class="result-card danger">
                    {ICON_ALERT}
                    <h3>HEART DISEASE DETECTED</h3>
                    <p>Clinical indications suggest elevated risk of cardiovascular disease.</p>
                </div>
                <div class="confidence-box">
                    <span style="font-size:0.8rem; color:#94a3b8;">Risk Certainty: <strong>{pct}%</strong></span>
                    <div class="confidence-bar-bg"><div class="confidence-bar-fill neg" style="width:{pct}%;"></div></div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-card success">
                    {ICON_CHECK}
                    <h3>NO HEART DISEASE DETECTED</h3>
                    <p>Parameters reside within normal cardiovascular bounds.</p>
                </div>
                <div class="confidence-box">
                    <span style="font-size:0.8rem; color:#94a3b8;">Negative Certainty: <strong>{pct}%</strong></span>
                    <div class="confidence-bar-bg"><div class="confidence-bar-fill pos" style="width:{pct}%;"></div></div>
                </div>
                """, unsafe_allow_html=True)

    with tab_batch2:
        st.markdown(f"#### {ICON_UPLOAD} Bulk Patient Assessment", unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload CSV file with columns: age, thalach, chol, cp", type=["csv"], key="h_csv")
        if uploaded:
            bdf = pd.read_csv(uploaded)
            bdf.columns = bdf.columns.str.lower()
            req = {"age", "thalach", "chol", "cp"}
            if req.issubset(set(bdf.columns)):
                preds = h_model.predict(bdf[["age", "thalach", "chol", "cp"]])
                bdf["Prediction"] = ["Heart Disease" if p == 1 else "No Disease" for p in preds]
                st.success(f"Processed {len(bdf)} patient records!")
                st.dataframe(bdf.head(15), use_container_width=True)
                csv_bytes = bdf.to_csv(index=False).encode("utf-8")
                st.download_button("Download Annotated CSV", csv_bytes, "heart_predictions.csv", "text/csv")
            else:
                st.error(f"Missing required columns: {req - set(bdf.columns)}")

    with tab_dash2:
        st.markdown(f"#### {ICON_CHART} Task 2 Performance & Model Info", unsafe_allow_html=True)
        if h_data_path.exists():
            df_h = pd.read_csv(h_data_path)
            f_cols = ["age", "thalach", "chol", "cp"]
            df_sub = df_h[f_cols + ["target"]].dropna().copy()
            from sklearn.model_selection import train_test_split
            _, x_test, _, y_test = train_test_split(df_sub[f_cols], df_sub["target"], test_size=0.2, random_state=42)
            y_pred = h_model.predict(x_test)

            m1, m2 = st.columns(2)
            m1.metric("Testing Accuracy", f"{accuracy_score(y_test, y_pred)*100:.2f}%")
            m2.metric("Features Constraint", "4 Features Adhered")

            st.write("Confusion Matrix:")
            cm = confusion_matrix(y_test, y_pred)
            st.dataframe(pd.DataFrame(cm, index=["Actual Healthy", "Actual Disease"], columns=["Pred Healthy", "Pred Disease"]), use_container_width=True)
