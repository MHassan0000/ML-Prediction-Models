import pickle
import re
from pathlib import Path

import numpy as np
import streamlit as st

st.set_page_config(page_title="SMS Spam Detection", page_icon="📱", layout="wide")

BASE = Path(__file__).parent


@st.cache_resource
def load_model():
    model_path = BASE / "spam_best_model.pkl"
    vec_path = BASE / "spam_vectorizer.pkl"
    if not model_path.exists() or not vec_path.exists():
        model_path = BASE.parent.parent / "models" / "spam_best_model.pkl"
        vec_path = BASE.parent.parent / "models" / "spam_vectorizer.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(vec_path, "rb") as f:
        vec_data = pickle.load(f)
    if isinstance(vec_data, dict):
        vectorizer = vec_data["vectorizer"]
    else:
        vectorizer = vec_data
    return model, vectorizer


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


model, vectorizer = load_model()

st.title("📱 SMS Spam Detection")
st.write("**Assignment 4, Task 2** — Muhammad Hassan Yousaf (FA24-BSE-082)")
st.write(
    "This app classifies SMS messages as **Spam** or **Ham** (legitimate) "
    "using a machine learning model trained on the UCI SMS Spam Collection dataset."
)

st.write("")

message = st.text_area(
    "Enter an SMS message to classify:",
    height=120,
    placeholder="e.g. Congratulations! You've won a $1000 gift card. Call now to claim!",
)

if st.button("Predict", type="primary", use_container_width=True):
    if not message.strip():
        st.warning("Please enter a message first.")
    else:
        cleaned = clean_text(message)
        X = vectorizer.transform([cleaned])
        prediction = model.predict(X)[0]

        label = str(prediction).lower()
        is_spam = label in ("spam", "1")

        if is_spam:
            st.error("### SPAM detected!")
        else:
            st.success("### HAM — This message looks legitimate.")

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            confidence = float(np.max(proba)) * 100
            st.caption(f"Confidence: {confidence:.1f}%  |  Cleaned: {cleaned}")
        else:
            st.caption(f"Cleaned text: {cleaned}")

st.write("")
st.subheader("Example Messages to Try")

examples = {
    "Spam 1": "Congratulations! You've won a $1000 gift card. Call now to claim!",
    "Spam 2": "URGENT: Your account has been compromised. Click here immediately",
    "Spam 3": "FREE entry in a weekly competition. Text WIN to 80085 now!",
    "Ham 1": "Hey, are you coming to the party tonight?",
    "Ham 2": "Can you pick up some milk on the way home?",
    "Ham 3": "I'll call you later when I'm free. See you!",
}

col1, col2 = st.columns(2)

with col1:
    for label, text in list(examples.items())[:3]:
        st.markdown(f"**{label}**")
        st.code(text, language=None)

with col2:
    for label, text in list(examples.items())[3:]:
        st.markdown(f"**{label}**")
        st.code(text, language=None)

st.write("")
st.caption(
    "Muhammad Hassan Yousaf (FA24-BSE-082)  |  "
    "Machine Learning Fundamentals (AIC354)  |  Assignment 4, Task 2"
)
