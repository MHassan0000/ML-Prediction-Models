import pickle
import re
from pathlib import Path

import numpy as np
import streamlit as st

st.set_page_config(
    page_title="Review Sentiment Analyzer",
    page_icon="https://cdn-icons-png.flaticon.com/512/5968/5968217.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Font Awesome 6 ────────────────────────────────────────────────────────────
st.markdown(
    '<link rel="stylesheet" '
    'href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css" '
    'crossorigin="anonymous">',
    unsafe_allow_html=True,
)

# ── Global dark-theme CSS ─────────────────────────────────────────────────────
st.markdown(
    """
<style>

/* ─ Typography stack ─ */
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI",
                 Roboto, "Helvetica Neue", Arial, sans-serif;
}

/* ─ Root backgrounds ─ */
.stApp,
.main .block-container,
[data-testid="stMain"] {
    background: #07090f !important;
}

/* Force text colours inside Streamlit wrappers */
.stApp, .stApp * {
    color: #e6eaf0;
}

/* ─ Sidebar ─ */
[data-testid="stSidebar"] {
    background: #0a0d14 !important;
    border-right: 1px solid rgba(255,255,255,0.055) !important;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 1.4rem; }

/* ─ Scrollbar ─ */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #07090f; }
::-webkit-scrollbar-thumb { background: rgba(255,153,0,0.25); border-radius: 10px; }

/* ─ App header ─ */
.app-header {
    background: linear-gradient(135deg, #0d1117 0%, #090c14 100%);
    border: 1px solid rgba(255,153,0,0.18);
    border-left: 3px solid #FF9900;
    border-radius: 14px;
    padding: 2rem 2.4rem;
    margin-bottom: 1.6rem;
    box-shadow: 0 0 60px rgba(255,153,0,0.04), 0 2px 20px rgba(0,0,0,0.5);
}
.header-top {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 0.6rem;
}
.header-icon-box {
    background: rgba(255,153,0,0.12);
    border: 1px solid rgba(255,153,0,0.3);
    color: #FF9900;
    width: 46px; height: 46px;
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.25rem;
    flex-shrink: 0;
}
.app-header h1 {
    color: #f0f4f8;
    font-size: 1.7rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.4px;
}
.app-header .sub {
    color: #5a6478;
    font-size: 0.86rem;
    margin: 0 0 1rem 0;
    padding-left: 0;
}
.meta-strip { display: flex; gap: 0.45rem; flex-wrap: wrap; }
.meta-pill {
    background: rgba(255,153,0,0.07);
    border: 1px solid rgba(255,153,0,0.2);
    color: rgba(255,153,0,0.85);
    border-radius: 6px;
    padding: 0.18rem 0.6rem;
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.2px;
}

/* ─ Section label ─ */
.sect {
    display: flex;
    align-items: center;
    gap: 0.45rem;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #3d4555;
    margin-bottom: 0.65rem;
}
.sect i { color: #FF9900; }

/* ─ Glass card ─ */
.gcard {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.065);
    border-radius: 12px;
    padding: 1.4rem 1.5rem;
    margin-bottom: 0.9rem;
}

/* ─ Result card — positive ─ */
.res-pos {
    background: radial-gradient(ellipse at top, rgba(34,197,94,0.1) 0%, rgba(34,197,94,0.03) 70%),
                rgba(255,255,255,0.02);
    border: 1px solid rgba(34,197,94,0.3);
    border-radius: 14px;
    padding: 2.2rem 2rem;
    text-align: center;
    box-shadow: 0 0 50px rgba(34,197,94,0.1), 0 2px 20px rgba(0,0,0,0.4);
    animation: surfaceIn 0.4s ease;
}
/* ─ Result card — negative ─ */
.res-neg {
    background: radial-gradient(ellipse at top, rgba(248,81,73,0.1) 0%, rgba(248,81,73,0.03) 70%),
                rgba(255,255,255,0.02);
    border: 1px solid rgba(248,81,73,0.3);
    border-radius: 14px;
    padding: 2.2rem 2rem;
    text-align: center;
    box-shadow: 0 0 50px rgba(248,81,73,0.1), 0 2px 20px rgba(0,0,0,0.4);
    animation: surfaceIn 0.4s ease;
}
.res-icon {
    font-size: 2.4rem;
    margin-bottom: 0.8rem;
    display: block;
    filter: drop-shadow(0 0 12px currentColor);
}
.res-icon-pos { color: #22c55e; }
.res-icon-neg { color: #f85149; }
.res-label {
    font-size: 1.6rem;
    font-weight: 800;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.res-label-pos { color: #4ade80; }
.res-label-neg { color: #fc8181; }
.res-stars { font-size: 1.1rem; letter-spacing: 3px; margin-bottom: 0.5rem; }
.star-on  { color: #FF9900; }
.star-off { color: rgba(255,255,255,0.12); }
.res-hint { font-size: 0.8rem; color: #4a5568; }

/* ─ Confidence strip ─ */
.conf-strip {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.065);
    border-radius: 10px;
    padding: 1rem 1.3rem;
    margin-top: 0.6rem;
}
.conf-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
}
.conf-label-left {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #3d4555;
}
.conf-label-left i { color: #FF9900; }
.conf-pct { font-size: 1.15rem; font-weight: 800; color: #e6eaf0; }
.conf-track {
    background: rgba(255,255,255,0.06);
    border-radius: 8px; height: 7px; overflow: hidden;
}
.conf-fill { height: 100%; border-radius: 8px; transition: width 0.6s ease; }

/* ─ Stat tiles ─ */
.stat-tile {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.065);
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
    margin-bottom: 0.6rem;
}
.stat-lbl {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.35rem;
    font-size: 0.65rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #3d4555;
    margin-bottom: 0.3rem;
}
.stat-lbl i { color: #FF9900; }
.stat-val { font-size: 1.2rem; font-weight: 800; color: #e6eaf0; }

/* ─ Example cards ─ */
.ex-card {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.055);
    border-radius: 10px;
    padding: 0.85rem 1.1rem;
    margin-bottom: 0.4rem;
    border-left-width: 3px;
}
.ex-pos { border-left-color: rgba(34,197,94,0.5); }
.ex-neg { border-left-color: rgba(248,81,73,0.5); }
.ex-head {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.78rem;
    font-weight: 700;
    color: #c8cdd8;
    margin-bottom: 0.3rem;
}
.ex-head .fa-thumbs-up   { color: #22c55e; }
.ex-head .fa-thumbs-down { color: #f85149; }
.ex-body { font-size: 0.78rem; color: #3d4555; line-height: 1.45; }

/* ─ Sidebar sections ─ */
.sb-logo {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding-bottom: 1.2rem;
    border-bottom: 1px solid rgba(255,255,255,0.055);
    margin-bottom: 1.2rem;
}
.sb-logo-box {
    background: rgba(255,153,0,0.1);
    border: 1px solid rgba(255,153,0,0.2);
    color: #FF9900;
    width: 36px; height: 36px;
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem;
}
.sb-name { font-size: 0.88rem; font-weight: 700; color: #e6eaf0; line-height: 1.2; }
.sb-desc { font-size: 0.68rem; color: #3d4555; }

.sb-grp { margin-bottom: 1.3rem; }
.sb-grp-title {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 1.3px;
    text-transform: uppercase;
    color: #3d4555;
    margin-bottom: 0.6rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid rgba(255,255,255,0.045);
}
.sb-grp-title i { color: #FF9900; }
.sb-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.35rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.03);
    font-size: 0.8rem;
}
.sb-row:last-child { border: none; }
.sb-k { color: #3d4555; }
.sb-v { color: #c8cdd8; font-weight: 600; }
.badge {
    background: rgba(34,197,94,0.1);
    border: 1px solid rgba(34,197,94,0.25);
    color: #4ade80;
    border-radius: 5px;
    padding: 0.1rem 0.5rem;
    font-size: 0.68rem;
    font-weight: 600;
}
.badge-warn {
    background: rgba(255,153,0,0.1);
    border-color: rgba(255,153,0,0.25);
    color: #FF9900;
}

.how-row {
    display: flex;
    align-items: flex-start;
    gap: 0.55rem;
    padding: 0.3rem 0;
    font-size: 0.78rem;
    color: #4a5568;
}
.how-num {
    background: rgba(255,153,0,0.12);
    border: 1px solid rgba(255,153,0,0.2);
    color: #FF9900;
    width: 20px; height: 20px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.62rem; font-weight: 800;
    flex-shrink: 0; margin-top: 1px;
}

/* ─ Buttons ─ */
.stButton > button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.86rem !important;
    height: 2.6rem !important;
    letter-spacing: 0.2px !important;
    transition: all 0.2s ease !important;
}
.stButton > button[kind="primary"] {
    background: #FF9900 !important;
    color: #07090f !important;
    border: none !important;
    font-weight: 700 !important;
}
.stButton > button[kind="primary"]:hover {
    background: #e68900 !important;
    box-shadow: 0 0 20px rgba(255,153,0,0.3) !important;
}
.stButton > button[kind="secondary"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    color: #8b949e !important;
}
.stButton > button[kind="secondary"]:hover {
    background: rgba(255,255,255,0.07) !important;
    color: #e6eaf0 !important;
}

/* ─ Textarea ─ */
.stTextArea > label { display: none !important; }
.stTextArea textarea {
    background: #0d1117 !important;
    color: #e6eaf0 !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    font-size: 0.9rem !important;
    line-height: 1.65 !important;
    caret-color: #FF9900 !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
.stTextArea textarea:focus {
    border-color: rgba(255,153,0,0.5) !important;
    box-shadow: 0 0 0 3px rgba(255,153,0,0.08) !important;
    outline: none !important;
}
.stTextArea textarea::placeholder { color: #2a3040 !important; }

/* ─ Warning / info ─ */
[data-testid="stAlert"] {
    background: rgba(255,153,0,0.07) !important;
    border: 1px solid rgba(255,153,0,0.2) !important;
    border-radius: 8px !important;
    color: rgba(255,153,0,0.9) !important;
}

/* ─ Divider ─ */
.hr { border: none; border-top: 1px solid rgba(255,255,255,0.055); margin: 1.4rem 0; }

/* ─ Footer ─ */
.footer {
    text-align: center;
    color: #232a35;
    font-size: 0.72rem;
    padding: 0.5rem 0 1rem;
}

footer[class] { visibility: hidden !important; }

@keyframes surfaceIn {
    from { opacity: 0; transform: translateY(-8px); }
    to   { opacity: 1; transform: translateY(0);    }
}
</style>
""",
    unsafe_allow_html=True,
)

# ── Load model ────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    with open(BASE / "amazon_best_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(BASE / "amazon_vectorizer.pkl", "rb") as f:
        raw = pickle.load(f)
    vectorizer = raw["vectorizer"] if isinstance(raw, dict) else raw
    return model, vectorizer


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_confidence(model, X) -> float:
    try:
        score = float(np.atleast_1d(model.decision_function(X))[0])
        return float(1 / (1 + np.exp(-abs(score))))
    except Exception:
        return 0.5


model, vectorizer = load_model()

# ── Session state ─────────────────────────────────────────────────────────────
if "preset" not in st.session_state:
    st.session_state.preset = ""

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
<div class="sb-logo">
    <div class="sb-logo-box"><i class="fa-solid fa-chart-line"></i></div>
    <div>
        <div class="sb-name">Review Analyzer</div>
        <div class="sb-desc">ML Sentiment Engine</div>
    </div>
</div>

<div class="sb-grp">
    <div class="sb-grp-title"><i class="fa-solid fa-robot"></i> Model</div>
    <div class="sb-row"><span class="sb-k">Algorithm</span><span class="sb-v">Linear SVC</span></div>
    <div class="sb-row"><span class="sb-k">CV Accuracy</span><span class="sb-v">~89%</span></div>
    <div class="sb-row"><span class="sb-k">Features</span><span class="sb-v">TF-IDF 5,000</span></div>
    <div class="sb-row"><span class="sb-k">N-grams</span><span class="sb-v">Uni + Bi</span></div>
    <div class="sb-row"><span class="sb-k">Evaluation</span><span class="sb-v">10-Fold CV</span></div>
</div>

<div class="sb-grp">
    <div class="sb-grp-title"><i class="fa-solid fa-database"></i> Dataset</div>
    <div class="sb-row"><span class="sb-k">Source</span><span class="sb-v">Amazon Polarity</span></div>
    <div class="sb-row">
        <span class="sb-k">Size</span>
        <span class="sb-v"><span class="badge badge-warn">25,000</span></span>
    </div>
    <div class="sb-row">
        <span class="sb-k">Balance</span>
        <span class="sb-v"><span class="badge">50 / 50</span></span>
    </div>
    <div class="sb-row"><span class="sb-k">Total pool</span><span class="sb-v">3.6M reviews</span></div>
</div>

<div class="sb-grp">
    <div class="sb-grp-title"><i class="fa-solid fa-gears"></i> Pipeline</div>
    <div class="how-row"><div class="how-num">1</div><span>Clean &amp; normalise raw text</span></div>
    <div class="how-row"><div class="how-num">2</div><span>TF-IDF encodes word importance</span></div>
    <div class="how-row"><div class="how-num">3</div><span>Linear SVC classifies the vector</span></div>
    <div class="how-row"><div class="how-num">4</div><span>Confidence from decision score</span></div>
</div>

<div class="sb-grp">
    <div class="sb-grp-title"><i class="fa-solid fa-user-graduate"></i> Project</div>
    <div class="sb-row"><span class="sb-k">Student</span><span class="sb-v">M. Hassan Yousaf</span></div>
    <div class="sb-row"><span class="sb-k">Roll No.</span><span class="sb-v">FA24-BSE-082</span></div>
    <div class="sb-row"><span class="sb-k">Course</span><span class="sb-v">AIC354</span></div>
    <div class="sb-row"><span class="sb-k">Assignment</span><span class="sb-v">4 · Task 2</span></div>
</div>
""",
        unsafe_allow_html=True,
    )

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="app-header">
    <div class="header-top">
        <div class="header-icon-box">
            <i class="fa-solid fa-magnifying-glass-chart"></i>
        </div>
        <h1>Amazon Review Sentiment Analyzer</h1>
    </div>
    <p class="sub">
        Classify any product review as positive or negative —
        trained on 25,000 real Amazon reviews using 10-fold cross validation.
    </p>
    <div class="meta-strip">
        <span class="meta-pill"><i class="fa-solid fa-layer-group"></i>&nbsp; Linear SVC</span>
        <span class="meta-pill"><i class="fa-solid fa-rotate"></i>&nbsp; 10-Fold CV</span>
        <span class="meta-pill"><i class="fa-solid fa-bullseye"></i>&nbsp; ~89% Accuracy</span>
        <span class="meta-pill"><i class="fa-brands fa-amazon"></i>&nbsp; Amazon Polarity</span>
        <span class="meta-pill"><i class="fa-solid fa-code-branch"></i>&nbsp; Assignment 4 · Task 2</span>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# ── Input ─────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="sect"><i class="fa-solid fa-pen-to-square"></i> Product Review Input</div>',
    unsafe_allow_html=True,
)

col_in, col_tip = st.columns([3, 1], gap="medium")

with col_in:
    review_input = st.text_area(
        "review",
        value=st.session_state.preset,
        height=160,
        placeholder=(
            "Paste or type any Amazon product review here...\n\n"
            'e.g. "This blender is outstanding. Quiet, easy to clean, '
            "and produces perfectly smooth results every time.\""
        ),
    )
    col_a, col_c = st.columns([3, 1], gap="small")
    with col_a:
        analyze = st.button(
            "Analyze Review",
            type="primary",
            use_container_width=True,
        )
    with col_c:
        if st.button("Clear", use_container_width=True):
            st.session_state.preset = ""
            st.rerun()

with col_tip:
    st.markdown(
        """
<div class="gcard" style="height:100%;box-sizing:border-box;">
    <div class="sect" style="margin-bottom:0.8rem;">
        <i class="fa-solid fa-lightbulb"></i> Tips
    </div>
    <ul style="margin:0;padding-left:1rem;font-size:0.79rem;color:#3d4555;line-height:2.1;">
        <li>Use the full review for best accuracy</li>
        <li>Short phrases work too</li>
        <li>Mixed languages may reduce accuracy</li>
        <li>Try the examples below to get started</li>
    </ul>
</div>
""",
        unsafe_allow_html=True,
    )

# ── Prediction ────────────────────────────────────────────────────────────────
if analyze:
    if not review_input.strip():
        st.warning("Please enter a review before analyzing.")
    else:
        cleaned = clean_text(review_input)
        X_in    = vectorizer.transform([cleaned])
        pred    = int(model.predict(X_in)[0])
        conf    = get_confidence(model, X_in)
        pct     = int(conf * 100)
        is_pos  = pred == 1

        col_res, col_data = st.columns([2, 1], gap="medium")

        with col_res:
            if is_pos:
                st.markdown(
                    """
<div class="res-pos">
    <span class="res-icon res-icon-pos">
        <i class="fa-solid fa-circle-check"></i>
    </span>
    <div class="res-label res-label-pos">Positive Review</div>
    <div class="res-stars">
        <i class="fa-solid fa-star star-on"></i>
        <i class="fa-solid fa-star star-on"></i>
        <i class="fa-solid fa-star star-on"></i>
        <i class="fa-solid fa-star star-on"></i>
        <i class="fa-solid fa-star star-on"></i>
    </div>
    <div class="res-hint">Model classified this as a satisfied customer review.</div>
</div>
""",
                    unsafe_allow_html=True,
                )
                fill = "#22c55e"
            else:
                st.markdown(
                    """
<div class="res-neg">
    <span class="res-icon res-icon-neg">
        <i class="fa-solid fa-circle-xmark"></i>
    </span>
    <div class="res-label res-label-neg">Negative Review</div>
    <div class="res-stars">
        <i class="fa-solid fa-star star-on"></i>
        <i class="fa-solid fa-star star-on"></i>
        <i class="fa-solid fa-star star-off"></i>
        <i class="fa-solid fa-star star-off"></i>
        <i class="fa-solid fa-star star-off"></i>
    </div>
    <div class="res-hint">Model classified this as a dissatisfied customer review.</div>
</div>
""",
                    unsafe_allow_html=True,
                )
                fill = "#f85149"

            st.markdown(
                f"""
<div class="conf-strip">
    <div class="conf-row">
        <div class="conf-label-left">
            <i class="fa-solid fa-gauge-high"></i> Confidence
        </div>
        <div class="conf-pct">{pct}%</div>
    </div>
    <div class="conf-track">
        <div class="conf-fill" style="width:{pct}%;background:{fill};
             box-shadow:0 0 10px {fill}55;"></div>
    </div>
</div>
""",
                unsafe_allow_html=True,
            )

        with col_data:
            wc  = len(review_input.split())
            cc  = len(review_input)
            lbl = "Positive" if is_pos else "Negative"
            ic  = "fa-thumbs-up" if is_pos else "fa-thumbs-down"
            lc  = "#4ade80" if is_pos else "#fc8181"

            st.markdown(
                f"""
<div class="stat-tile">
    <div class="stat-lbl"><i class="fa-solid {ic}"></i> Verdict</div>
    <div class="stat-val" style="color:{lc}; font-size:1rem;">{lbl}</div>
</div>
<div class="stat-tile">
    <div class="stat-lbl"><i class="fa-solid fa-gauge-high"></i> Confidence</div>
    <div class="stat-val">{pct}%</div>
</div>
<div class="stat-tile">
    <div class="stat-lbl"><i class="fa-solid fa-align-left"></i> Words</div>
    <div class="stat-val">{wc:,}</div>
</div>
<div class="stat-tile">
    <div class="stat-lbl"><i class="fa-solid fa-text-width"></i> Characters</div>
    <div class="stat-val">{cc:,}</div>
</div>
""",
                unsafe_allow_html=True,
            )

st.markdown('<hr class="hr">', unsafe_allow_html=True)

# ── Examples ──────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="sect"><i class="fa-solid fa-list-check"></i> Example Reviews</div>',
    unsafe_allow_html=True,
)

positive_examples = {
    "Outstanding blender": (
        "This blender is hands down the best kitchen appliance I have ever bought. "
        "Super quiet motor, easy to clean, and makes perfectly smooth shakes every time. "
        "Worth every penny. Arrived two days early too."
    ),
    "Excellent headphones": (
        "Sound quality is phenomenal. The noise cancellation is so effective I cannot "
        "hear anything around me. Very comfortable for long sessions and battery lasts "
        "all day. Highly recommend to anyone looking for premium audio."
    ),
    "Perfect birthday gift": (
        "Bought this as a birthday gift and my friend absolutely loved it. "
        "Packaging was beautiful, product quality is top notch, and delivery was fast. "
        "Will definitely order from this seller again."
    ),
}

negative_examples = {
    "Poor build quality": (
        "Complete waste of money. Stopped working after just one week. "
        "The build quality feels very cheap and the materials are flimsy. "
        "Customer support never responded to my emails. Total disappointment."
    ),
    "Wrong item received": (
        "Nothing like what was shown in the product photos. The color is completely "
        "different and the size is way off. The return process has been a nightmare. "
        "Avoid this seller at all costs."
    ),
    "Arrived damaged": (
        "Item arrived with a visible crack down the side. Clearly was not packaged "
        "properly at all. Contacted the seller three times with no reply. Had to open "
        "a dispute with Amazon. Very frustrating from start to finish."
    ),
}

col_p, col_n = st.columns(2, gap="medium")

with col_p:
    st.markdown(
        '<div class="sect"><i class="fa-solid fa-thumbs-up" style="color:#22c55e;"></i>'
        " Positive Examples</div>",
        unsafe_allow_html=True,
    )
    for title, text in positive_examples.items():
        st.markdown(
            f"""
<div class="ex-card ex-pos">
    <div class="ex-head">
        <i class="fa-solid fa-thumbs-up"></i> {title}
    </div>
    <div class="ex-body">{text[:125]}...</div>
</div>
""",
            unsafe_allow_html=True,
        )
        if st.button(f"Use — {title}", key=f"p_{title}", use_container_width=True):
            st.session_state.preset = text
            st.rerun()

with col_n:
    st.markdown(
        '<div class="sect"><i class="fa-solid fa-thumbs-down" style="color:#f85149;"></i>'
        " Negative Examples</div>",
        unsafe_allow_html=True,
    )
    for title, text in negative_examples.items():
        st.markdown(
            f"""
<div class="ex-card ex-neg">
    <div class="ex-head">
        <i class="fa-solid fa-thumbs-down"></i> {title}
    </div>
    <div class="ex-body">{text[:125]}...</div>
</div>
""",
            unsafe_allow_html=True,
        )
        if st.button(f"Use — {title}", key=f"n_{title}", use_container_width=True):
            st.session_state.preset = text
            st.rerun()

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown('<hr class="hr">', unsafe_allow_html=True)
st.markdown(
    """
<div class="footer">
    <i class="fa-solid fa-graduation-cap"></i>&nbsp;
    Muhammad Hassan Yousaf &nbsp;&middot;&nbsp; FA24-BSE-082 &nbsp;&middot;&nbsp;
    Machine Learning Fundamentals (AIC354) &nbsp;&middot;&nbsp;
    Assignment 4, Task 2 &nbsp;&middot;&nbsp;
    COMSATS University Islamabad
</div>
""",
    unsafe_allow_html=True,
)
