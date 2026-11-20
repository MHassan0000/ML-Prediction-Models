"""
Generates amazon_sentiment_10fold.ipynb
Run: python generate_notebook.py
"""

import json
import os

TARGET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "amazon_sentiment_10fold.ipynb")


def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": [source]}


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [source]}


cells = []

# Title
cells.append(md(
    "# Amazon Product Review Sentiment Analysis (10-Fold Cross Validation)\n"
    "\n"
    "**Deployment Link:** [https://amazon-sentiment-hassan.streamlit.app](https://amazon-sentiment-hassan.streamlit.app)\n"
    "\n"
    "**Name:** Muhammad Hassan Yousaf  \n"
    "**Roll Number:** FA24-BSE-082  \n"
    "**Course:** Machine Learning Fundamentals (AIC354), Assignment 4, Task 2  \n"
    "**Instructor:** Dr. Rao Muhammad Adeel Nawab  \n"
    "**University:** COMSATS University Islamabad, Lahore Campus  \n"
    "**GitHub:** [https://github.com/MHassan0000/ML-Prediction-Models](https://github.com/MHassan0000/ML-Prediction-Models)\n"
    "\n"
    "Customer reviews shape what millions of people buy on Amazon every day. The goal here is to build a classifier that reads a product review and decides whether the customer was satisfied or not, based only on the words they wrote. No star rating is used as a feature - just the text.\n"
    "\n"
    "**Dataset:** Amazon Polarity (HuggingFace) - 25,000 reviews, balanced 50/50 positive and negative  \n"
    "**Feature Extraction:** TF-IDF Vectorizer (uni-grams and bi-grams, 5000 features)  \n"
    "**Evaluation:** 10-Fold Cross Validation, Accuracy metric\n"
    "\n"
    "**6 Algorithms compared:**\n"
    "1. Logistic Regression\n"
    "2. Multinomial Naive Bayes\n"
    "3. Random Forest\n"
    "4. Linear SVC\n"
    "5. k-Nearest Neighbors\n"
    "6. Decision Tree\n"
    "\n"
    "To run on Google Colab: open this notebook and run all cells from top to bottom. The dataset streams automatically through HuggingFace - no manual upload needed."
))

# Step 1
cells.append(md("## Step 1: Import Libraries"))

cells.append(code(
    "import os\n"
    "import re\n"
    "import pickle\n"
    "import warnings\n"
    "warnings.filterwarnings('ignore')\n"
    "\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import sklearn\n"
    "import matplotlib\n"
    "import matplotlib.pyplot as plt\n"
    "\n"
    "from sklearn.base import clone\n"
    "from sklearn.model_selection import KFold\n"
    "from sklearn.metrics import accuracy_score, classification_report\n"
    "from sklearn.feature_extraction.text import TfidfVectorizer\n"
    "from sklearn.preprocessing import LabelEncoder\n"
    "\n"
    "from sklearn.linear_model import LogisticRegression\n"
    "from sklearn.naive_bayes import MultinomialNB\n"
    "from sklearn.ensemble import RandomForestClassifier\n"
    "from sklearn.svm import LinearSVC\n"
    "from sklearn.neighbors import KNeighborsClassifier\n"
    "from sklearn.tree import DecisionTreeClassifier\n"
    "\n"
    "from tqdm import tqdm\n"
    "\n"
    "pd.set_option('display.float_format', lambda v: f'{v:.4f}')\n"
    "\n"
    "print(f'NumPy        : {np.__version__}')\n"
    "print(f'Pandas       : {pd.__version__}')\n"
    "print(f'Scikit-learn : {sklearn.__version__}')\n"
    "print(f'Matplotlib   : {matplotlib.__version__}')"
))

# Step 2
cells.append(md(
    "## Step 2: Load Dataset\n"
    "\n"
    "The Amazon Polarity dataset contains over 3.6 million product reviews from Amazon, labelled positive (4-5 star) or negative (1-2 star). "
    "To keep training times reasonable I stream and sample 12,500 reviews from each class, giving a balanced 25,000-review working dataset. "
    "The dataset is available on HuggingFace as `fancyzhx/amazon_polarity` and downloads automatically - no Kaggle account or manual file upload needed."
))

cells.append(code(
    "try:\n"
    "    from datasets import load_dataset\n"
    "except ImportError:\n"
    "    import subprocess, sys\n"
    "    subprocess.run([sys.executable, '-m', 'pip', 'install', 'datasets', '-q'], check=True)\n"
    "    from datasets import load_dataset\n"
    "\n"
    "N_PER_CLASS = 12500\n"
    "CSV_PATH    = 'amazon_reviews_sample.csv'\n"
    "\n"
    "if os.path.exists(CSV_PATH):\n"
    "    print(f'Loading from local CSV: {CSV_PATH}')\n"
    "    df = pd.read_csv(CSV_PATH)\n"
    "    df['label'] = df['label'].astype(int)\n"
    "    print(f'Loaded {len(df):,} reviews from CSV.')\n"
    "else:\n"
    "    print(f'Streaming Amazon Polarity dataset ({N_PER_CLASS:,} reviews per class)...')\n"
    "    ds = load_dataset('fancyzhx/amazon_polarity', split='train', streaming=True)\n"
    "\n"
    "    positives_text, negatives_text = [], []\n"
    "    positives_lbl,  negatives_lbl  = [], []\n"
    "\n"
    "    for item in ds:\n"
    "        text = ((item.get('title') or '') + ' ' + (item.get('content') or '')).strip()\n"
    "        if not text:\n"
    "            continue\n"
    "        if item['label'] == 1 and len(positives_text) < N_PER_CLASS:\n"
    "            positives_text.append(text); positives_lbl.append(1)\n"
    "        elif item['label'] == 0 and len(negatives_text) < N_PER_CLASS:\n"
    "            negatives_text.append(text); negatives_lbl.append(0)\n"
    "        total = len(positives_text) + len(negatives_text)\n"
    "        if total % 5000 == 0 and total > 0:\n"
    "            print(f'  Collected: {len(positives_text):,} positive, {len(negatives_text):,} negative')\n"
    "        if len(positives_text) >= N_PER_CLASS and len(negatives_text) >= N_PER_CLASS:\n"
    "            break\n"
    "\n"
    "    # Build with explicit types so there is no PyArrow backing\n"
    "    all_texts = positives_text + negatives_text\n"
    "    all_labels = positives_lbl + negatives_lbl\n"
    "    all_sentiments = ['positive'] * len(positives_text) + ['negative'] * len(negatives_text)\n"
    "    df = pd.DataFrame({'text': all_texts, 'label': all_labels, 'sentiment': all_sentiments})\n"
    "    df['label'] = df['label'].astype(int)\n"
    "    df = df.sample(frac=1, random_state=42).reset_index(drop=True)\n"
    "    print(f'\\nDataset ready: {len(df):,} reviews')\n"
    "\n"
    "print(f'Shape: {df.shape}')\n"
    "df.head(5)"
))

# Save CSV cell
cells.append(md(
    "### Save Dataset as CSV\n"
    "\n"
    "The cell below saves a local copy of the 25,000-review dataset. "
    "On subsequent runs the notebook loads from this file instead of streaming again, which is much faster. "
    "The download trigger works in Google Colab."
))

cells.append(code(
    "CSV_PATH = 'amazon_reviews_sample.csv'\n"
    "df[['text', 'label', 'sentiment']].to_csv(CSV_PATH, index=False)\n"
    "print(f'Saved {CSV_PATH} ({len(df):,} rows, {os.path.getsize(CSV_PATH) / 1024:.1f} KB)')\n"
    "\n"
    "try:\n"
    "    from google.colab import files\n"
    "    files.download(CSV_PATH)\n"
    "    print('Download started in browser.')\n"
    "except ImportError:\n"
    "    print(f'Running locally - file saved to: {os.path.abspath(CSV_PATH)}')"
))

# Step 3
cells.append(md("## Step 3: Understand the Data"))

cells.append(code(
    "class_counts = df['sentiment'].value_counts()\n"
    "class_pct    = df['sentiment'].value_counts(normalize=True) * 100\n"
    "\n"
    "print('Class distribution:')\n"
    "for label in class_counts.index:\n"
    "    print(f'  {label:>8}: {class_counts[label]:>6,} reviews ({class_pct[label]:.1f}%)')\n"
    "print(f'\\nTotal reviews : {len(df):,}')\n"
    "print(f'Missing values: {df.isnull().sum().sum()}')"
))

cells.append(code(
    "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n"
    "\n"
    "colors = ['#2ecc71', '#e74c3c']\n"
    "bars = axes[0].bar(class_counts.index, class_counts.values, color=colors, edgecolor='white', width=0.5)\n"
    "for bar, count, pct in zip(bars, class_counts.values, class_pct.values):\n"
    "    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 100,\n"
    "                 f'{count:,}\\n({pct:.1f}%)', ha='center', fontweight='bold')\n"
    "axes[0].set_title('Review Sentiment Distribution', fontsize=13)\n"
    "axes[0].set_xlabel('Sentiment')\n"
    "axes[0].set_ylabel('Number of Reviews')\n"
    "axes[0].spines[['top', 'right']].set_visible(False)\n"
    "\n"
    "df['text_length'] = df['text'].apply(len)\n"
    "for label, color in zip(['positive', 'negative'], colors):\n"
    "    axes[1].hist(df[df['sentiment'] == label]['text_length'], bins=60,\n"
    "                 alpha=0.7, label=label, color=color, edgecolor='white')\n"
    "axes[1].set_title('Review Length Distribution', fontsize=13)\n"
    "axes[1].set_xlabel('Number of Characters')\n"
    "axes[1].set_ylabel('Frequency')\n"
    "axes[1].legend()\n"
    "axes[1].spines[['top', 'right']].set_visible(False)\n"
    "\n"
    "plt.tight_layout()\n"
    "plt.show()"
))

cells.append(code(
    "print('Sample POSITIVE reviews:')\n"
    "for _, row in df[df['sentiment'] == 'positive'].head(3).iterrows():\n"
    "    print(f'  \"{row[\"text\"][:120]}\"')\n"
    "    print()\n"
    "\n"
    "print('Sample NEGATIVE reviews:')\n"
    "for _, row in df[df['sentiment'] == 'negative'].head(3).iterrows():\n"
    "    print(f'  \"{row[\"text\"][:120]}\"')\n"
    "    print()\n"
    "\n"
    "print(f'Average review length: {df[\"text_length\"].mean():.0f} characters')\n"
    "print(f'Longest review       : {df[\"text_length\"].max()} characters')\n"
    "print(f'Shortest review      : {df[\"text_length\"].min()} characters')"
))

# Step 4
cells.append(md(
    "## Step 4: Pre-processing\n"
    "\n"
    "Each review goes through the same cleaning steps: lowercase conversion, URL removal, digit stripping, and punctuation removal. "
    "Amazon reviews are mostly well-formed English so the main job is normalizing case and stripping noise so TF-IDF can focus on actual words."
))

cells.append(code(
    "def clean_text(text):\n"
    "    text = str(text).lower()\n"
    "    text = re.sub(r'http\\S+|www\\.\\S+', '', text)\n"
    "    text = re.sub(r'\\d+', '', text)\n"
    "    text = re.sub(r'[^a-z\\s]', '', text)\n"
    "    text = re.sub(r'\\s+', ' ', text).strip()\n"
    "    return text\n"
    "\n"
    "print('Before and after cleaning:')\n"
    "for idx in [0, 2, 5]:\n"
    "    orig    = df.loc[idx, 'text']\n"
    "    cleaned = clean_text(orig)\n"
    "    label   = df.loc[idx, 'sentiment'].upper()\n"
    "    print(f'\\n  [{label}]')\n"
    "    print(f'  BEFORE: \"{orig[:100]}\"')\n"
    "    print(f'  AFTER : \"{cleaned[:100]}\"')\n"
    "\n"
    "df['clean_text'] = df['text'].apply(clean_text)\n"
    "print(f'\\nCleaned {len(df):,} reviews.')"
))

# Step 5
cells.append(md(
    "## Step 5: Feature Extraction\n"
    "\n"
    "TF-IDF converts each review into a numerical vector. Each dimension corresponds to a word or two-word phrase (bigram), "
    "and its value reflects how distinctive that term is for a particular review. Using bigrams alongside unigrams lets the model "
    "pick up phrases like 'not good' or 'highly recommend' which carry clearer sentiment than either word alone."
))

cells.append(code(
    "tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))\n"
    "\n"
    "X = tfidf.fit_transform(df['clean_text'])\n"
    "y = df['label'].values\n"
    "\n"
    "print(f'TF-IDF matrix shape : {X.shape}')\n"
    "print(f'Vocabulary size     : {len(tfidf.vocabulary_):,} features')\n"
    "print(f'Non-zero entries    : {X.nnz:,}')\n"
    "print(f'Sparsity            : {(1 - X.nnz / (X.shape[0] * X.shape[1])) * 100:.2f}%')"
))

cells.append(code(
    "feature_names = tfidf.get_feature_names_out()\n"
    "mean_scores   = X.mean(axis=0).A1\n"
    "top_indices   = mean_scores.argsort()[::-1][:20]\n"
    "\n"
    "print('Top 20 features by mean TF-IDF score:')\n"
    "for rank, idx in enumerate(top_indices, 1):\n"
    "    print(f'  {rank:>2}. {feature_names[idx]:<20} {mean_scores[idx]:.6f}')"
))

# Step 6
cells.append(md("## Step 6: Define 6 Algorithms"))

cells.append(code(
    "algorithms = {\n"
    "    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),\n"
    "    'Multinomial NB':      MultinomialNB(),\n"
    "    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),\n"
    "    'Linear SVC':          LinearSVC(max_iter=2000, random_state=42),\n"
    "    'KNN':                 KNeighborsClassifier(n_neighbors=5),\n"
    "    'Decision Tree':       DecisionTreeClassifier(random_state=42),\n"
    "}\n"
    "\n"
    "print(f'{len(algorithms)} algorithms ready:')\n"
    "for i, name in enumerate(algorithms, 1):\n"
    "    print(f'  {i}. {name}')"
))

# Step 7
cells.append(md("## Step 7: Set Up 10-Fold Cross Validation"))

cells.append(code(
    "K  = 10\n"
    "kf = KFold(n_splits=K, shuffle=True, random_state=42)\n"
    "\n"
    "print(f'K = {K} folds | Total reviews: {X.shape[0]:,}')\n"
    "print(f'Training per fold : ~{X.shape[0] - X.shape[0] // K:,}')\n"
    "print(f'Test per fold     : ~{X.shape[0] // K:,}')\n"
    "\n"
    "print(f'\\n  {\"Fold\":<6} {\"Train\":>8} {\"Test\":>8}')\n"
    "for i, (tr, te) in enumerate(kf.split(X), 1):\n"
    "    print(f'  {i:<6} {len(tr):>8,} {len(te):>8,}')"
))

# Step 8
cells.append(md(
    "## Step 8: Run 10-Fold Cross Validation\n"
    "\n"
    "The TF-IDF vectorizer is fitted on the training portion of each fold only, then applied to the test portion. "
    "This prevents any test vocabulary from leaking into the feature representation. Each model is cloned and trained fresh for every fold."
))

cells.append(code(
    "results          = {}\n"
    "first_fold_preds = {}\n"
    "total_tasks      = len(algorithms) * K\n"
    "\n"
    "print(f'Running {K}-fold CV for {len(algorithms)} algorithms ({total_tasks} total fits)...')\n"
    "\n"
    "# Convert to numpy array so KFold integer index arrays work correctly\n"
    "X_texts = np.array(df['clean_text'].tolist(), dtype=object)\n"
    "y       = df['label'].astype(int).values\n"
    "with tqdm(total=total_tasks, desc='Cross Validation', unit='fit') as pbar:\n"
    "    for algo_name, algo in algorithms.items():\n"
    "        fold_accs = []\n"
    "        for fold_i, (tr, te) in enumerate(kf.split(X_texts), 1):\n"
    "            vec  = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))\n"
    "            X_tr = vec.fit_transform(X_texts[tr])\n"
    "            X_te = vec.transform(X_texts[te])\n"
    "\n"
    "            m = clone(algo)\n"
    "            m.fit(X_tr, y[tr])\n"
    "            y_pred = m.predict(X_te)\n"
    "            fold_accs.append(accuracy_score(y[te], y_pred))\n"
    "\n"
    "            if fold_i == 1:\n"
    "                first_fold_preds[algo_name] = (y[te], y_pred)\n"
    "            pbar.update(1)\n"
    "\n"
    "        results[algo_name] = fold_accs\n"
    "        print(f'  {algo_name:<22} -> {np.mean(fold_accs)*100:.2f}% +/- {np.std(fold_accs)*100:.2f}%')\n"
    "\n"
    "print('\\nDone!')"
))

# Step 9
cells.append(md("## Step 9: Compare Results\n\n### Per-Fold Accuracy"))

cells.append(code(
    "fold_df = pd.DataFrame(results, index=[f'Fold {i}' for i in range(1, K + 1)]).T\n"
    "print('Per-fold accuracy for each algorithm:')\n"
    "(fold_df * 100).round(2)"
))

cells.append(md("### Summary Table"))

cells.append(code(
    "summary = pd.DataFrame({\n"
    "    'Mean Accuracy (%)': [np.mean(v) * 100 for v in results.values()],\n"
    "    'Std Dev (%)':       [np.std(v)  * 100 for v in results.values()],\n"
    "    'Min Accuracy (%)':  [np.min(v)  * 100 for v in results.values()],\n"
    "    'Max Accuracy (%)':  [np.max(v)  * 100 for v in results.values()],\n"
    "}, index=results.keys()).sort_values('Mean Accuracy (%)', ascending=False)\n"
    "\n"
    "summary.insert(0, 'Rank', range(1, len(summary) + 1))\n"
    "print('Algorithm comparison sorted by mean accuracy:')\n"
    "summary"
))

cells.append(md("### Charts"))

cells.append(code(
    "COLORS       = ['#2ecc71', '#3498db', '#9b59b6', '#e67e22', '#e74c3c', '#1abc9c']\n"
    "sorted_algos = summary.index.tolist()\n"
    "sorted_means = [np.mean(results[a]) * 100 for a in sorted_algos]\n"
    "sorted_stds  = [np.std(results[a])  * 100 for a in sorted_algos]\n"
    "\n"
    "fig, axes = plt.subplots(1, 3, figsize=(20, 6))\n"
    "\n"
    "bars = axes[0].barh(sorted_algos, sorted_means, xerr=sorted_stds,\n"
    "                    color=COLORS, edgecolor='white', capsize=4, height=0.6)\n"
    "axes[0].set_yticks(range(len(sorted_algos)))\n"
    "axes[0].set_yticklabels(sorted_algos)\n"
    "axes[0].set_xlabel('Accuracy (%)')\n"
    "axes[0].set_title('Mean Accuracy with Std Dev', fontsize=13)\n"
    "axes[0].invert_yaxis()\n"
    "axes[0].spines[['top', 'right']].set_visible(False)\n"
    "for i, (mean, std) in enumerate(zip(sorted_means, sorted_stds)):\n"
    "    axes[0].text(mean + std + 0.2, i, f'{mean:.2f}%', va='center', fontsize=9, fontweight='bold')\n"
    "\n"
    "box_data = [np.array(results[a]) * 100 for a in sorted_algos]\n"
    "bp = axes[1].boxplot(box_data, vert=True, patch_artist=True, labels=sorted_algos)\n"
    "for patch, c in zip(bp['boxes'], COLORS):\n"
    "    patch.set_facecolor(c)\n"
    "    patch.set_alpha(0.7)\n"
    "axes[1].set_ylabel('Accuracy (%)')\n"
    "axes[1].set_title('Accuracy Distribution per Algorithm', fontsize=13)\n"
    "axes[1].tick_params(axis='x', rotation=45)\n"
    "axes[1].spines[['top', 'right']].set_visible(False)\n"
    "\n"
    "for i, algo in enumerate(sorted_algos):\n"
    "    axes[2].plot(range(1, K + 1), np.array(results[algo]) * 100, marker='o',\n"
    "                 label=algo, color=COLORS[i], linewidth=2)\n"
    "axes[2].set_xlabel('Fold')\n"
    "axes[2].set_ylabel('Accuracy (%)')\n"
    "axes[2].set_title('Accuracy Across All 10 Folds', fontsize=13)\n"
    "axes[2].legend(fontsize=8)\n"
    "axes[2].set_xticks(range(1, K + 1))\n"
    "axes[2].spines[['top', 'right']].set_visible(False)\n"
    "\n"
    "plt.tight_layout()\n"
    "plt.show()"
))

# Step 10
cells.append(md("## Step 10: Observations"))

cells.append(code(
    "mean_accs  = {n: np.mean(v) * 100 for n, v in results.items()}\n"
    "best_algo  = max(mean_accs, key=mean_accs.get)\n"
    "worst_algo = min(mean_accs, key=mean_accs.get)\n"
    "\n"
    "print(f'Best  algorithm : {best_algo}')\n"
    "print(f'  Mean Accuracy : {mean_accs[best_algo]:.2f}%')\n"
    "print(f'  Std Deviation : {np.std(results[best_algo]) * 100:.2f}%')\n"
    "print()\n"
    "print(f'Worst algorithm : {worst_algo}')\n"
    "print(f'  Mean Accuracy : {mean_accs[worst_algo]:.2f}%')\n"
    "print()\n"
    "print(f'Accuracy gap    : {mean_accs[best_algo] - mean_accs[worst_algo]:.2f}%')\n"
    "print()\n"
    "\n"
    "y_test_b, y_pred_b = first_fold_preds[best_algo]\n"
    "print(f'Classification Report for {best_algo} (Fold 1):')\n"
    "print(classification_report(y_test_b, y_pred_b, target_names=['negative', 'positive'], digits=4))"
))

# Step 11
cells.append(md(
    "## Step 11: Train Final Model\n"
    "\n"
    "The best algorithm is now trained on the full 25,000-review dataset. "
    "The model and vectorizer are saved as pickle files for use in the web application."
))

cells.append(code(
    "print(f'Training final model: {best_algo}')\n"
    "\n"
    "final_tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))\n"
    "X_all = final_tfidf.fit_transform(df['clean_text'])\n"
    "\n"
    "final_model = clone(algorithms[best_algo])\n"
    "final_model.fit(X_all, y)\n"
    "\n"
    "print(f'Trained on {X_all.shape[0]:,} reviews with {X_all.shape[1]:,} features.')\n"
    "\n"
    "with open('amazon_best_model.pkl', 'wb') as f:\n"
    "    pickle.dump(final_model, f)\n"
    "\n"
    "with open('amazon_vectorizer.pkl', 'wb') as f:\n"
    "    pickle.dump({'vectorizer': final_tfidf}, f)\n"
    "\n"
    "print(f'Saved amazon_best_model.pkl  ({os.path.getsize(\"amazon_best_model.pkl\") / 1024:.1f} KB)')\n"
    "print(f'Saved amazon_vectorizer.pkl  ({os.path.getsize(\"amazon_vectorizer.pkl\") / 1024:.1f} KB)')"
))

# Step 12
cells.append(md("## Step 12: Test Predictions"))

cells.append(code(
    "with open('amazon_best_model.pkl', 'rb') as f:\n"
    "    loaded_model = pickle.load(f)\n"
    "with open('amazon_vectorizer.pkl', 'rb') as f:\n"
    "    loaded_vec = pickle.load(f)['vectorizer']\n"
    "\n"
    "print('Model loaded successfully!')\n"
    "\n"
    "test_reviews = [\n"
    "    'This product is absolutely amazing. Best purchase I have ever made. Highly recommend!',\n"
    "    'Complete waste of money. Broke after two days. Terrible quality, do not buy.',\n"
    "    'Exactly as described, fast shipping and great quality for the price.',\n"
    "    'Very disappointed. The color looked nothing like the pictures. Returning it.',\n"
    "    'My kids love this toy. Kept them busy for hours. Great gift idea.',\n"
    "    'Poor customer service and the item arrived damaged. Very frustrating.',\n"
    "]\n"
    "\n"
    "label_map = {0: 'NEGATIVE', 1: 'POSITIVE'}\n"
    "icons     = {0: '---', 1: '+++'}\n"
    "\n"
    "print(f'\\nTest predictions ({best_algo}):')\n"
    "print('=' * 70)\n"
    "for review in test_reviews:\n"
    "    cleaned = clean_text(review)\n"
    "    pred    = loaded_model.predict(loaded_vec.transform([cleaned]))[0]\n"
    "    label   = label_map[pred]\n"
    "    icon    = icons[pred]\n"
    "    print(f'  [{icon}]  {label}  -  \"{review[:60]}...\"')"
))

# Step 13
cells.append(md("## Step 13: Ready for Deployment"))

cells.append(code(
    "print('Amazon Review Sentiment Analysis - Project Summary')\n"
    "print('=' * 55)\n"
    "print(f'  Problem         : Amazon Product Review Sentiment (Binary Text Classification)')\n"
    "print(f'  Dataset         : Amazon Polarity - {len(df):,} reviews (balanced)')\n"
    "print(f'  Feature method  : TF-IDF Vectorizer (max_features=5000, ngrams=(1,2))')\n"
    "print(f'  Evaluation      : 10-Fold Cross Validation')\n"
    "print(f'  Best algorithm  : {best_algo}')\n"
    "print(f'  CV Accuracy     : {mean_accs[best_algo]:.2f}%')\n"
    "print(f'  Saved model     : amazon_best_model.pkl')\n"
    "print(f'  Saved vectorizer: amazon_vectorizer.pkl')\n"
    "print()\n"
    "print(f'  Deployment link : https://amazon-sentiment-hassan.streamlit.app')\n"
    "print(f'  GitHub          : https://github.com/MHassan0000/ML-Prediction-Models')"
))

# Serialize
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.11.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}

with open(TARGET, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Notebook saved: {TARGET}")
print(f"Total cells   : {len(cells)}")
