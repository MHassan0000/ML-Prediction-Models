"""
Titanic Passenger Survival Prediction – Streamlit Web App
==========================================================
Assignment 2 – Task 1 | Deployment

Run locally:  streamlit run titanic_app.py
"""

import os
import io
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="⛴",
    layout="centered",
)

# ── Inline SVG icons ──────────────────────────────────────────────
# Using Lucide-style inline SVGs for a clean professional look
ICON_SHIP = '<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 21c.6.5 1.2 1 2.5 1 2.5 0 2.5-2 5-2 1.3 0 1.9.5 2.5 1 .6.5 1.2 1 2.5 1 2.5 0 2.5-2 5-2 1.3 0 1.9.5 2.5 1"/><path d="M19.38 20A11.6 11.6 0 0 0 21 14l-9-4-9 4c0 2.9.94 5.34 2.81 7.76"/><path d="M19 13V7a2 2 0 0 0-2-2H7a2 2 0 0 0-2 2v6"/><path d="M12 1v4"/></svg>'
ICON_USER = '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>'
ICON_TICKET = '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 9a3 3 0 0 1 0 6v2a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-2a3 3 0 0 1 0-6V7a2 2 0 0 0-2-2H4a2 2 0 0 0-2 2Z"/><path d="M13 5v2"/><path d="M13 17v2"/><path d="M13 11v2"/></svg>'
ICON_ANCHOR = '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="5" r="3"/><line x1="12" y1="22" x2="12" y2="8"/><path d="M5 12H2a10 10 0 0 0 20 0h-3"/></svg>'
ICON_SEARCH = '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/></svg>'
ICON_CHECK = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><path d="m9 11 3 3L22 4"/></svg>'
ICON_X = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="m15 9-6 6"/><path d="m9 9 6 6"/></svg>'
ICON_INFO = '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/></svg>'
ICON_CLIPBOARD = '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="8" height="4" x="8" y="2" rx="1" ry="1"/><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"/></svg>'
ICON_DOLLAR = '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="2" x2="12" y2="22"/><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg>'
ICON_USERS = '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>'
ICON_UPLOAD = '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>'
ICON_CHART = '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18"/><path d="m19 9-5 5-4-4-3 3"/></svg>'
ICON_GAUGE = '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m12 14 4-4"/><path d="M3.34 19a10 10 0 1 1 17.32 0"/></svg>'

# ── Styling ───────────────────────────────────────────────────────
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
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: #fff;
    margin-bottom: 0.75rem;
    box-shadow: 0 8px 24px rgba(99,102,241,0.3);
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
.section-label svg { color: #6366f1; }

/* ── Cards ── */
.glass-card {
    background: rgba(30, 41, 59, 0.6);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(148, 163, 184, 0.1);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

/* ── Result cards ── */
.result-card {
    padding: 1.75rem 1.5rem;
    border-radius: 16px;
    text-align: center;
    animation: slideUp 0.45s cubic-bezier(0.16, 1, 0.3, 1);
    margin-top: 0.75rem;
}
.result-card.survived {
    background: linear-gradient(135deg, #059669, #10b981);
    border: 1px solid rgba(16,185,129,0.3);
    box-shadow: 0 8px 32px rgba(16,185,129,0.25);
}
.result-card.not-survived {
    background: linear-gradient(135deg, #dc2626, #ef4444);
    border: 1px solid rgba(239,68,68,0.3);
    box-shadow: 0 8px 32px rgba(239,68,68,0.25);
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
.gauge-wrap .gauge-label svg { color: #6366f1; }
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
.gauge-bar-fill.positive { background: linear-gradient(90deg, #059669, #34d399); }
.gauge-bar-fill.negative { background: linear-gradient(90deg, #dc2626, #f87171); }
.gauge-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: #e2e8f0;
    margin-top: 0.35rem;
}

/* ── Button overrides ── */
div.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    font-size: 0.9375rem !important;
    font-weight: 600 !important;
    padding: 0.7rem 2rem !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 14px rgba(99,102,241,0.3) !important;
    letter-spacing: 0.3px !important;
}
div.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(99,102,241,0.4) !important;
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
    background: rgba(99,102,241,0.2) !important;
}

/* ── Expander ── */
div.streamlit-expanderHeader {
    font-weight: 600;
    font-size: 0.875rem;
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
.app-footer a {
    color: #818cf8;
    text-decoration: none;
}

/* ── Animations ── */
@keyframes slideUp {
    from { opacity: 0; transform: translateY(12px); }
    to { opacity: 1; transform: translateY(0); }
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────
st.markdown(f'''
<div class="app-header">
    <div class="icon-wrap">{ICON_SHIP}</div>
    <h1>Titanic Survival Predictor</h1>
    <p>ML Assignment 2 — Task 1 &nbsp;&bull;&nbsp; Support Vector Classifier &nbsp;&bull;&nbsp; Titanic Dataset</p>
</div>
''', unsafe_allow_html=True)

# ── Load model & encoders ─────────────────────────────────────────
BASE = Path(__file__).parent

@st.cache_resource
def load_model():
    model_path = BASE / "titanic_svc_model.pkl"
    with open(model_path, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def build_encoders():
    le_sex = LabelEncoder()
    le_sex.fit(["female", "male"])          # 0=female, 1=male
    le_embarked = LabelEncoder()
    le_embarked.fit(["C", "Q", "S"])        # 0=C, 1=Q, 2=S
    return le_sex, le_embarked

@st.cache_data
def load_dataset():
    csv_path = BASE / "titanic.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    try:
        import seaborn as sns
        return sns.load_dataset("titanic")
    except Exception:
        return None

try:
    model = load_model()
    le_sex, le_embarked = build_encoders()
except FileNotFoundError:
    st.error("Model file not found (`titanic_svc_model.pkl`). "
             "Run `Task1_Titanic_Survival_Prediction.py` first to train and save the model.")
    st.stop()

# ── Tabs ──────────────────────────────────────────────────────────
tab_predict, tab_batch, tab_model = st.tabs(["Predict", "Batch Predict", "Model Info"])

# ═══════════════════════════════════════════════════════════════════
# TAB 1: SINGLE PREDICTION
# ═══════════════════════════════════════════════════════════════════
with tab_predict:
    st.markdown(f'<div class="section-label">{ICON_CLIPBOARD} Passenger Information</div>',
                unsafe_allow_html=True)

    CLASS_LABELS = {1: "1st Class — Upper", 2: "2nd Class — Middle", 3: "3rd Class — Lower"}
    PORT_LABELS  = {"S": "Southampton (S)", "C": "Cherbourg (C)", "Q": "Queenstown (Q)"}

    col1, col2 = st.columns(2)

    with col1:
        pclass = st.selectbox("Passenger Class", options=[1, 2, 3],
                              format_func=lambda x: CLASS_LABELS[x],
                              help="Ticket class: 1st (Upper), 2nd (Middle), 3rd (Lower)")
        sex    = st.selectbox("Sex", options=["female", "male"],
                              format_func=lambda x: x.capitalize(),
                              help="Biological sex of the passenger")
        age    = st.slider("Age (years)", min_value=1, max_value=80, value=28,
                          help="Age at time of voyage")
        fare   = st.number_input("Fare Paid ($)", min_value=0.0,
                                 max_value=600.0, value=30.0, step=0.5,
                                 help="Price of the ticket in US dollars")

    with col2:
        sibsp    = st.number_input("Siblings / Spouses Aboard", min_value=0, max_value=8, value=0,
                                   help="Number of siblings or spouse aboard the Titanic")
        parch    = st.number_input("Parents / Children Aboard", min_value=0, max_value=6, value=0,
                                   help="Number of parents or children aboard")
        embarked = st.selectbox("Port of Embarkation", options=["S", "C", "Q"],
                                format_func=lambda x: PORT_LABELS[x],
                                help="Port where the passenger boarded")

    st.markdown("")
    predict_clicked = st.button("Predict Survival", key="predict_single",
                                use_container_width=True)

    if predict_clicked:
        sex_enc      = le_sex.transform([sex])[0]
        embarked_enc = le_embarked.transform([embarked])[0]
        feature_cols = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
        X            = pd.DataFrame([[pclass, sex_enc, age, sibsp, parch, fare, embarked_enc]], columns=feature_cols)
        prediction   = model.predict(X)[0]

        # Compute confidence from decision_function (distance to hyperplane)
        try:
            distance  = model.decision_function(X)[0]
            confidence = min(abs(distance) / 2.0, 1.0)  # normalize roughly to 0-1
        except Exception:
            confidence = None

        if prediction == 1:
            st.markdown(f'''
            <div class="result-card survived">
                <div class="result-icon">{ICON_CHECK}</div>
                <h2>SURVIVED</h2>
                <p>The model predicts this passenger would likely have survived.</p>
            </div>
            ''', unsafe_allow_html=True)
            gauge_cls = "positive"
        else:
            st.markdown(f'''
            <div class="result-card not-survived">
                <div class="result-icon">{ICON_X}</div>
                <h2>DID NOT SURVIVE</h2>
                <p>The model predicts this passenger would likely not have survived.</p>
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
        summary = pd.DataFrame({
            "Feature": ["Class", "Sex", "Age", "Siblings/Spouses", "Parents/Children", "Fare", "Embarked"],
            "Value":   [CLASS_LABELS[pclass], sex.capitalize(), age, sibsp, parch, f"${fare:.2f}", PORT_LABELS[embarked]],
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════
# TAB 2: BATCH PREDICTION
# ═══════════════════════════════════════════════════════════════════
with tab_batch:
    st.markdown(f'<div class="section-label">{ICON_UPLOAD} Upload CSV for Batch Prediction</div>',
                unsafe_allow_html=True)
    st.caption("CSV must contain columns: `pclass`, `sex`, `age`, `sibsp`, `parch`, `fare`, `embarked`")

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"],
                                     label_visibility="collapsed")

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            required_cols = {"pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"}
            missing = required_cols - set(batch_df.columns.str.lower())
            if missing:
                st.error(f"Missing required columns: {', '.join(missing)}")
            else:
                batch_df.columns = batch_df.columns.str.lower()
                st.markdown(f'<div class="section-label">{ICON_CHART} Preview ({len(batch_df)} rows)</div>',
                            unsafe_allow_html=True)
                st.dataframe(batch_df.head(10), use_container_width=True, hide_index=True)

                if st.button("Run Batch Prediction", key="batch_predict"):
                    batch_encoded = batch_df.copy()
                    batch_encoded["sex"] = le_sex.transform(batch_encoded["sex"])
                    batch_encoded["embarked"] = le_embarked.transform(batch_encoded["embarked"])
                    X_batch = batch_encoded[["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]]
                    preds = model.predict(X_batch)
                    batch_df["Prediction"] = ["Survived" if p == 1 else "Did Not Survive" for p in preds]

                    survived_count = sum(preds == 1)
                    total = len(preds)
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Total Passengers", total)
                    c2.metric("Predicted Survived", survived_count)
                    c3.metric("Survival Rate", f"{survived_count/total*100:.1f}%")

                    st.dataframe(batch_df, use_container_width=True, hide_index=True)

                    csv_buffer = batch_df.to_csv(index=False).encode("utf-8")
                    st.download_button("Download Results CSV", csv_buffer,
                                       "titanic_batch_predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"Error processing file: {e}")

    # Provide sample template download
    with st.expander("Download sample CSV template"):
        sample = pd.DataFrame({
            "pclass": [1, 3, 2],
            "sex": ["female", "male", "female"],
            "age": [29, 22, 35],
            "sibsp": [0, 1, 1],
            "parch": [0, 0, 0],
            "fare": [211.0, 7.25, 26.0],
            "embarked": ["S", "S", "C"]
        })
        st.dataframe(sample, use_container_width=True, hide_index=True)
        st.download_button("Download Template", sample.to_csv(index=False).encode("utf-8"),
                          "titanic_sample_template.csv", "text/csv", key="dl_template_titanic")

# ═══════════════════════════════════════════════════════════════════
# TAB 3: MODEL INFO
# ═══════════════════════════════════════════════════════════════════
with tab_model:
    st.markdown(f'<div class="section-label">{ICON_INFO} Model Details</div>',
                unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Algorithm", "SVC (RBF)")
    c2.metric("Kernel", "RBF")
    c3.metric("Features", "7")

    st.markdown("")
    st.markdown(f'<div class="section-label">{ICON_CHART} Dataset & Feature Summary</div>',
                unsafe_allow_html=True)

    feature_info = pd.DataFrame({
        "Feature": ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"],
        "Description": [
            "Ticket class (1st, 2nd, 3rd)",
            "Biological sex",
            "Age in years",
            "Siblings / Spouses aboard",
            "Parents / Children aboard",
            "Fare paid ($)",
            "Port of embarkation"
        ],
        "Type": ["Categorical", "Categorical", "Numeric", "Numeric", "Numeric", "Numeric", "Categorical"],
        "Encoding": ["Ordinal (1,2,3)", "Label (0=F,1=M)", "None", "None", "None", "None", "Label (C=0,Q=1,S=2)"]
    })
    st.dataframe(feature_info, use_container_width=True, hide_index=True)

    # Show model performance if we can load the dataset
    df_data = load_dataset()
    if df_data is not None:
        with st.expander("Model Performance on Test Set"):
            try:
                features = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
                target = "survived"
                df_m = df_data[features + [target]].copy().dropna()
                df_m["sex"] = le_sex.transform(df_m["sex"])
                df_m["embarked"] = le_embarked.transform(df_m["embarked"])
                from sklearn.model_selection import train_test_split
                _, X_test, _, y_test = train_test_split(
                    df_m[features], df_m[target], test_size=0.2, random_state=42
                )
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                st.metric("Test Accuracy", f"{acc*100:.2f}%")

                cm = confusion_matrix(y_test, y_pred)
                cm_df = pd.DataFrame(cm,
                    index=["Actual: Did Not Survive", "Actual: Survived"],
                    columns=["Predicted: Did Not Survive", "Predicted: Survived"]
                )
                st.markdown("**Confusion Matrix**")
                st.dataframe(cm_df, use_container_width=True)

                report = classification_report(y_test, y_pred, output_dict=True,
                                               target_names=["Did Not Survive", "Survived"])
                report_df = pd.DataFrame(report).transpose()
                st.markdown("**Classification Report**")
                st.dataframe(report_df.style.format("{:.2f}"), use_container_width=True)
            except Exception as e:
                st.info(f"Could not compute performance metrics: {e}")

# ── Footer ────────────────────────────────────────────────────────
st.markdown('''
<div class="app-footer">
    ML Assignment 2 — Task 1 &bull; Titanic Survival Prediction &bull; SVC (RBF Kernel)
    <br>Built with Streamlit &bull; Deployed on Modal
</div>
''', unsafe_allow_html=True)
