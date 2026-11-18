"""
Trains the Amazon Review Sentiment model locally.
Saves: amazon_best_model.pkl and amazon_vectorizer.pkl

Run: python train_local.py
"""

import os
import re
import sys
import pickle

try:
    from datasets import load_dataset
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "datasets", "-q"], check=True)
    from datasets import load_dataset

import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

SAVE_DIR    = os.path.dirname(os.path.abspath(__file__))
N_PER_CLASS = 12500


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# 1. Stream dataset
print(f"Streaming {N_PER_CLASS:,} reviews per class from Amazon Polarity...")
ds = load_dataset("fancyzhx/amazon_polarity", split="train", streaming=True)

positives, negatives = [], []
for item in ds:
    text = ((item.get("title") or "") + " " + (item.get("content") or "")).strip()
    if not text:
        continue
    if item["label"] == 1 and len(positives) < N_PER_CLASS:
        positives.append(text)
    elif item["label"] == 0 and len(negatives) < N_PER_CLASS:
        negatives.append(text)
    total = len(positives) + len(negatives)
    if total % 2000 == 0 and total > 0:
        print(f"  Collected: {len(positives):,} positive, {len(negatives):,} negative")
    if len(positives) >= N_PER_CLASS and len(negatives) >= N_PER_CLASS:
        break

# Build plain numpy arrays — no PyArrow backing
texts  = np.array(positives + negatives, dtype=object)
labels = np.array([1] * len(positives) + [0] * len(negatives), dtype=np.int32)

# Shuffle
rng  = np.random.default_rng(42)
idx  = rng.permutation(len(texts))
texts  = texts[idx]
labels = labels[idx]

print(f"Dataset: {len(texts):,} reviews | Positive: {labels.sum():,} | Negative: {(1-labels).sum():,}")

# 2. Clean
print("Cleaning text...")
clean = np.array([clean_text(t) for t in texts], dtype=object)

# 3. TF-IDF on full dataset
print("Fitting TF-IDF (max_features=5000, ngram_range=(1,2))...")
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X = tfidf.fit_transform(clean)

# 4. Quick accuracy check on a held-out slice
split = int(len(texts) * 0.9)
model_check = LinearSVC(max_iter=2000, random_state=42)
model_check.fit(X[:split], labels[:split])
acc = accuracy_score(labels[split:], model_check.predict(X[split:]))
print(f"Quick validation accuracy (last 10%): {acc * 100:.2f}%")

# 5. Retrain on full dataset for the app
print("Training final LinearSVC on all 25,000 reviews...")
tfidf_final = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_all = tfidf_final.fit_transform(clean)
model_final = LinearSVC(max_iter=2000, random_state=42)
model_final.fit(X_all, labels)

# 6. Save
model_path = os.path.join(SAVE_DIR, "amazon_best_model.pkl")
vec_path   = os.path.join(SAVE_DIR, "amazon_vectorizer.pkl")

with open(model_path, "wb") as f:
    pickle.dump(model_final, f)

with open(vec_path, "wb") as f:
    pickle.dump({"vectorizer": tfidf_final}, f)

print(f"\nSaved: {model_path}  ({os.path.getsize(model_path) / 1024:.1f} KB)")
print(f"Saved: {vec_path}  ({os.path.getsize(vec_path) / 1024:.1f} KB)")
print("\nDone. Model files are ready for the Streamlit app.")
