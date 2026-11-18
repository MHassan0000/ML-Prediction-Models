import pickle
import re
from pathlib import Path

import numpy as np
import streamlit as st

st.set_page_config(
    page_title="Amazon Review Analyzer",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* ── Global ── */
.stApp { background-color: #f4f6f9; }

/* ── Header banner ── */
.header-wrap {
    background: linear-gradient(135deg, #FF9900 0%, #e67e00 100%);
    padding: 2rem 2.5rem 1.8rem;
    border-radius: 18px;
    margin-bottom: 1.8rem;
    box-shadow: 0 6px 24px rgba(255,153,0,0.28);
}
.header-wrap h1 {
    color: white; font-size: 2.1rem; margin: 0 0 0.3rem 0;
    font-weight: 800; letter-spacing: -0.5px;
}
.header-wrap p {
    color: rgba(255,255,255,0.88); margin: 0; font-size: 1rem;
}
.header-badge {
    display: inline-block; background: rgba(255,255,255,0.2);
    border-radius: 20px; padding: 0.15rem 0.75rem;
    font-size: 0.78rem; color: white; margin-top: 0.6rem;
    border: 1px solid rgba(255,255,255,0.35);
}

/* ── Section label ── */
.section-lbl {
    font-size: 0.75rem; font-weight: 700; letter-spacing: 1.2px;
    color: #888; text-transform: uppercase; margin-bottom: 0.5rem;
}

/* ── Input card ── */
.input-card {
    background: white; border-radius: 16px; padding: 1.6rem;
    box-shadow: 0 2px 14px rgba(0,0,0,0.07); margin-bottom: 1.2rem;
}

/* ── Result cards ── */
.result-pos {
    background: linear-gradient(135deg, #2ecc71, #27ae60);
    color: white; padding: 2rem 2.5rem; border-radius: 16px;
    text-align: center; margin: 1rem 0;
    box-shadow: 0 10px 30px rgba(46,204,113,0.32);
    animation: popIn 0.4s cubic-bezier(0.175,0.885,0.32,1.275);
}
.result-neg {
    background: linear-gradient(135deg, #e74c3c, #c0392b);
    color: white; padding: 2rem 2.5rem; border-radius: 16px;
    text-align: center; margin: 1rem 0;
    box-shadow: 0 10px 30px rgba(231,76,60,0.32);
    animation: popIn 0.4s cubic-bezier(0.175,0.885,0.32,1.275);
}
.result-emoji { font-size: 3.6rem; line-height: 1; }
.result-label {
    font-size: 2rem; font-weight: 800; letter-spacing: 1.5px;
    margin: 0.4rem 0 0.2rem;
}
.result-stars { font-size: 1.9rem; letter-spacing: 4px; margin-bottom: 0.4rem; }
.result-sub   { font-size: 0.9rem; opacity: 0.88; }

/* ── Confidence bar ── */
.conf-wrap { margin: 0.8rem 0; }
.conf-label {
    display: flex; justify-content: space-between;
    font-size: 0.82rem; color: #555; margin-bottom: 0.3rem;
}
.conf-bar-bg {
    background: #e9ecef; border-radius: 10px; height: 10px; overflow: hidden;
}
.conf-bar-fill {
    height: 100%; border-radius: 10px; transition: width 0.6s ease;
}

/* ── Stat chips ── */
.chip-row { display: flex; gap: 0.5rem; flex-wrap: wrap; margin: 0.6rem 0; }
.chip {
    background: #fff3cd; border: 1px solid #ffc107; border-radius: 20px;
    padding: 0.2rem 0.75rem; font-size: 0.78rem; font-weight: 600; color: #856404;
}
.chip-blue {
    background: #cfe2ff; border-color: #9ec5fe; color: #084298;
}
.chip-green {
    background: #d1e7dd; border-color: #a3cfbb; color: #0f5132;
}

/* ── Info card ── */
.info-card {
    background: white; border-radius: 12px; padding: 1.1rem 1.2rem;
    box-shadow: 0 2px 10px rgba(0,0,0,0.06); margin: 0.5rem 0;
    font-size: 0.88rem; line-height: 1.6;
}
.info-card b { color: #2c3e50; }

/* ── Example cards ── */
.ex-card {
    background: white; border-radius: 10px; padding: 0.75rem 1rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05); margin: 0.3rem 0;
    font-size: 0.85rem; color: #444; border-left: 4px solid #dee2e6;
    line-height: 1.4;
}
.ex-pos { border-left-color: #2ecc71; }
.ex-neg { border-left-color: #e74c3c; }

/* ── How it works steps ── */
.how-step {
    display: flex; align-items: flex-start; gap: 0.7rem;
    margin: 0.4rem 0; font-size: 0.85rem; color: #444;
}
.how-num {
    background: #FF9900; color: white; border-radius: 50%;
    width: 1.4rem; height: 1.4rem; min-width: 1.4rem;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.72rem; font-weight: 700; margin-top: 1px;
}

/* ── Divider ── */
.divider { border: none; border-top: 1px solid #e9ecef; margin: 1.4rem 0; }

/* ── Sidebar ── */
[data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #f0f0f0; }
[data-testid="stSidebar"] .stMarkdown h3 {
    font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px;
    color: #888; margin-bottom: 0.4rem;
}

/* ── Buttons ── */
.stButton > button {
    border-radius: 10px; font-weight: 600;
    height: 2.8rem; font-size: 0.95rem;
}

/* ── Text area ── */
.stTextArea > label { display: none; }
.stTextArea textarea {
    border-radius: 12px; border: 2px solid #e9ecef;
    font-size: 0.95rem; line-height: 1.6;
    transition: border-color 0.2s;
}
.stTextArea textarea:focus { border-color: #FF9900 !important; box-shadow: none !important; }

footer { visibility: hidden; }

@keyframes popIn {
    from { opacity: 0; transform: scale(0.9); }
    to   { opacity: 1; transform: scale(1);   }
}
</style>
""",
    unsafe_allow_html=True,
)

# ── Load model ────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent


@st.cache_resource
def load_model():
    model_path = BASE / "amazon_best_model.pkl"
    vec_path = BASE / "amazon_vectorizer.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(vec_path, "rb") as f:
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


def confidence_from_svc(model, X):
    """Return a 0-1 confidence using sigmoid of the decision function."""
    try:
        scores = model.decision_function(X)
        score = float(np.atleast_1d(scores)[0])
        return float(1 / (1 + np.exp(-abs(score))))
    except Exception:
        return None


model, vectorizer = load_model()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🤖 Model Info")
    st.markdown(
        """
<div class="info-card">
    <b>Algorithm</b><br>Linear SVC<br><br>
    <b>CV Accuracy</b><br>~94%+ (10-fold)<br><br>
    <b>Feature Method</b><br>TF-IDF Vectorizer<br>
    <small>Uni-grams + Bi-grams, 5000 features</small>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("### 📦 Dataset")
    st.markdown(
        """
<div class="info-card">
    <b>Amazon Polarity</b><br>
    <small>via HuggingFace Datasets</small><br><br>
    <div class="chip-row">
        <span class="chip">25K reviews</span>
        <span class="chip-green">50 / 50 balanced</span>
    </div>
    <br>
    <small style="color:#888">Source: 3.6M+ Amazon product reviews<br>
    Positive = 4-5 stars &nbsp;|&nbsp; Negative = 1-2 stars</small>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("### 👨‍💻 Project")
    st.markdown(
        """
<div class="info-card">
    <b>Muhammad Hassan Yousaf</b><br>
    <small style="color:#888">FA24-BSE-082</small><br><br>
    Machine Learning Fundamentals<br>
    <small style="color:#888">AIC354 &nbsp;|&nbsp; Assignment 4, Task 2</small><br><br>
    <small style="color:#888">Instructor: Dr. Rao Muhammad Adeel Nawab<br>
    COMSATS University Islamabad</small>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("### ⚙️ How it works")
    for num, step in enumerate(
        [
            "Text is lowercased and cleaned",
            "TF-IDF converts words to numbers",
            "Linear SVC classifies the review",
            "Confidence shown via decision score",
        ],
        1,
    ):
        st.markdown(
            f'<div class="how-step"><div class="how-num">{num}</div><span>{step}</span></div>',
            unsafe_allow_html=True,
        )

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="header-wrap">
    <h1>🛍️ Amazon Review Sentiment Analyzer</h1>
    <p>Paste any product review and find out instantly whether it's positive or negative.</p>
    <span class="header-badge">📊 Assignment 4 · Task 2 · ML Fundamentals AIC354</span>
</div>
""",
    unsafe_allow_html=True,
)

# ── Session state for example injection ───────────────────────────────────────
if "preset" not in st.session_state:
    st.session_state.preset = ""

# ── Input section ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-lbl">📝 Product Review</div>', unsafe_allow_html=True)

review_input = st.text_area(
    "review",
    value=st.session_state.preset,
    height=165,
    placeholder=(
        'Paste or type any Amazon product review here...\n\n'
        'e.g. "This blender is fantastic. Super quiet, easy to clean, '
        'and makes the smoothest shakes. Worth every penny!"'
    ),
)

col_analyze, col_clear = st.columns([3, 1])
with col_analyze:
    analyze = st.button("🔍  Analyze Review", type="primary", use_container_width=True)
with col_clear:
    if st.button("✕  Clear", use_container_width=True):
        st.session_state.preset = ""
        st.rerun()

# ── Prediction ────────────────────────────────────────────────────────────────
if analyze:
    if not review_input.strip():
        st.warning("Please enter a review before analyzing.")
    else:
        cleaned = clean_text(review_input)
        X_in = vectorizer.transform([cleaned])
        pred = int(model.predict(X_in)[0])
        conf = confidence_from_svc(model, X_in)

        is_positive = pred == 1

        if is_positive:
            st.markdown(
                """
<div class="result-pos">
    <div class="result-emoji">😄</div>
    <div class="result-label">POSITIVE</div>
    <div class="result-stars">★ ★ ★ ★ ★</div>
    <div class="result-sub">The model classified this as a positive review.</div>
</div>
""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
<div class="result-neg">
    <div class="result-emoji">😞</div>
    <div class="result-label">NEGATIVE</div>
    <div class="result-stars">★ ★ ☆ ☆ ☆</div>
    <div class="result-sub">The model classified this as a negative review.</div>
</div>
""",
                unsafe_allow_html=True,
            )

        # Confidence bar
        if conf is not None:
            pct = int(conf * 100)
            fill_color = "#2ecc71" if is_positive else "#e74c3c"
            st.markdown(
                f"""
<div class="conf-wrap">
    <div class="conf-label">
        <span>Model confidence</span>
        <span><b>{pct}%</b></span>
    </div>
    <div class="conf-bar-bg">
        <div class="conf-bar-fill" style="width:{pct}%; background:{fill_color};"></div>
    </div>
</div>
""",
                unsafe_allow_html=True,
            )

        # Quick stats
        word_count = len(review_input.split())
        char_count = len(review_input)
        c1, c2, c3 = st.columns(3)
        c1.metric("Words", f"{word_count:,}")
        c2.metric("Characters", f"{char_count:,}")
        c3.metric("Prediction", "Positive ✅" if is_positive else "Negative ❌")

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── Example reviews ───────────────────────────────────────────────────────────
st.markdown('<div class="section-lbl">🎯 Try an Example</div>', unsafe_allow_html=True)

positive_examples = {
    "⭐ Great blender": (
        "This blender is hands down the best kitchen appliance I have ever bought. "
        "Super quiet motor, easy to clean, and makes perfectly smooth shakes every time. "
        "Worth every single penny. Arrived two days early too!"
    ),
    "⭐ Excellent headphones": (
        "Sound quality is phenomenal. The noise cancellation is so good I can't hear "
        "anything around me. Very comfortable for long sessions. Battery lasts all day. "
        "Highly recommend to anyone looking for premium audio."
    ),
    "⭐ Perfect gift": (
        "Bought this as a birthday gift and my friend absolutely loved it. "
        "Packaging was beautiful, product quality is top notch, and delivery was fast. "
        "Will definitely order from this seller again."
    ),
}

negative_examples = {
    "👎 Cheap quality": (
        "Complete waste of money. Stopped working after just one week. "
        "The build quality feels very cheap and the materials are flimsy. "
        "Customer support never responded to my emails. Total disappointment."
    ),
    "👎 Wrong item": (
        "Nothing like what was shown in the pictures. The color is completely different "
        "and the size is way off. Tried to return it and the process has been a nightmare. "
        "Avoid this seller at all costs."
    ),
    "👎 Arrived broken": (
        "Item arrived with a crack right down the side. Clearly was not packaged properly. "
        "Contacted the seller three times with no reply. Had to open a dispute with Amazon. "
        "Very frustrating experience from start to finish."
    ),
}

col_p, col_n = st.columns(2)

with col_p:
    st.markdown("**Positive examples** ✅")
    for label, text in positive_examples.items():
        st.markdown(
            f'<div class="ex-card ex-pos"><b>{label}</b><br><small>{text[:110]}...</small></div>',
            unsafe_allow_html=True,
        )
        if st.button(f"Use this →", key=f"p_{label}", use_container_width=True):
            st.session_state.preset = text
            st.rerun()

with col_n:
    st.markdown("**Negative examples** ❌")
    for label, text in negative_examples.items():
        st.markdown(
            f'<div class="ex-card ex-neg"><b>{label}</b><br><small>{text[:110]}...</small></div>',
            unsafe_allow_html=True,
        )
        if st.button(f"Use this →", key=f"n_{label}", use_container_width=True):
            st.session_state.preset = text
            st.rerun()

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.caption(
    "Muhammad Hassan Yousaf (FA24-BSE-082)  |  "
    "Machine Learning Fundamentals (AIC354)  |  "
    "Assignment 4, Task 2  |  "
    "COMSATS University Islamabad, Lahore Campus"
)
