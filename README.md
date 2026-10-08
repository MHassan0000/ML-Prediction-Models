# Machine Learning Prediction Systems (Assignment 2)

> End-to-end Machine Learning systems from data collection to cloud deployment.  
> **Student:** Muhammad Hassan Yousaf &nbsp;|&nbsp; **Roll No:** FA24-BSE-082 &nbsp;|&nbsp; **Program:** BS Software Engineering (BSE) &nbsp;|&nbsp; **Course:** Machine Learning (CSC-461)

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.45-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Modal](https://img.shields.io/badge/Deployed_on-Modal-6366f1?logoColor=white)](https://modal.com)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?logo=github&logoColor=white)](https://github.com/MHassan0000/ML-Prediction-Models)

---

## Live Cloud Deployments

Both prediction systems are live on **Streamlit Community Cloud** and **Modal Serverless Cloud**:

| Application | Streamlit Cloud URL | Modal Cloud URL | Status |
|-------------|---------------------|-----------------|--------|
| **Task 1: Titanic Survival Predictor** | [titanic-model-hassan.streamlit.app](https://titanic-model-hassan.streamlit.app) | [mhassan0000--titanic-predictor-run.modal.run](https://mhassan0000--titanic-predictor-run.modal.run) | **Live & Active** |
| **Task 2: Heart Disease Predictor** | [heart-disease-model-hassan.streamlit.app](https://heart-disease-model-hassan.streamlit.app) | [mhassan0000--heart-disease-predictor-run.modal.run](https://mhassan0000--heart-disease-predictor-run.modal.run) | **Live & Active** |

---

## Assignment Overview

This repository implements two production-grade Machine Learning classification pipelines adhering strictly to the assignment specifications:

| Task | Domain & Dataset | Machine Learning Model | Input Attributes | Output |
|---|---|---|---|---|
| **Task 1** | **Titanic Survival Prediction**<br>891 passenger records | Support Vector Classifier (`SVC`) with Radial Basis Function (`rbf`) kernel | 7 features: `pclass`, `sex`, `age`, `sibsp`, `parch`, `fare`, `embarked` | Binary: $0 = \text{Deceased}$, $1 = \text{Survived}$ |
| **Task 2** | **Heart Disease Prediction**<br>UCI Cleveland Dataset (303 records) | Support Vector Classifier (`SVC`) with Radial Basis Function (`rbf`) kernel | **Strict Constraint:** Only 4 features:<br>`age`, `thalach` (Max HR), `chol` (Cholesterol), `cp` (Chest Pain) | Binary: $0 = \text{No Disease}$, $1 = \text{Heart Disease}$ |

---

## Clean Project Structure

The project has been organized into modular directories for data, models, reports, notebooks, and pipelines:

```
ML-Prediction-Models/
│
├── app.py                                   # UNIFIED Streamlit portal (Both tasks + Student dossier)
├── titanic_app.py                           # Standalone Titanic web app
├── heart_disease_app.py                     # Standalone Heart Disease web app
│
├── data/                                    # Datasets directory
│   ├── titanic.csv                          # Original Titanic dataset
│   ├── heart.csv                            # Original UCI Heart Disease dataset
│   ├── titanic_encoded.csv                  # Preprocessed & label-encoded Titanic data
│   └── heart_disease_encoded.csv            # Preprocessed & label-encoded Heart data
│
├── models/                                  # Trained model serialization artifacts
│   ├── titanic_svc_model.pkl                # Trained SVC model (Task 1)
│   ├── heart_disease_svc_model.pkl          # Trained SVC model (Task 2)
│   └── heart_disease_le_cp.pkl              # Chest pain label encoder
│
├── notebooks/                               # Plagiarism-free Jupyter submission notebooks
│   ├── Task1_Titanic_Survival_Prediction.ipynb
│   └── Task2_Heart_Disease_Prediction.ipynb
│
├── pipelines/                               # Python training scripts
│   ├── Task1_Titanic_Survival_Prediction.py
│   └── Task2_Heart_Disease_Prediction.py
│
├── reports/                                 # Visual evaluation artifacts & export logs
│   ├── titanic_eda.png
│   ├── titanic_confusion_matrix.png
│   ├── titanic_predictions.csv
│   ├── heart_disease_eda.png
│   ├── heart_disease_confusion_matrix.png
│   ├── heart_disease_correlation.png
│   └── heart_disease_predictions.csv
│
├── modal/                                   # Modal cloud deployment scripts
│   ├── modal_titanic.py
│   └── modal_heart_disease.py
│
├── .streamlit/
│   └── config.toml                          # Dark theme styling configuration
├── requirements.txt                         # Python dependencies
├── generate_notebooks.py                    # Plagiarism-free notebook generator
└── README.md                                # Project documentation
```

---

## Machine Learning Lifecycle (Four Fundamental Phases)

```
┌─────────────────────────────────┐
│     Phase 1: Data Understanding │
│     Exploratory Data Analysis   │
└───────────────┬─────────────────┘
                │
┌───────────────▼─────────────────┐
│   Phase 2: Data Preprocessing   │
│   Missingness & Label Encoding  │
└───────────────┬─────────────────┘
                │
┌───────────────▼─────────────────┐
│     Phase 3: Model Training     │
│   Support Vector Machine (SVC)  │
└───────────────┬─────────────────┘
                │
┌───────────────▼─────────────────┐
│    Phase 4: Model Evaluation    │
│   Accuracy, Confusion Matrix,   │
│   Classification Report & Cloud │
└─────────────────────────────────┘
```

---

## How to Deploy Both Tasks on Streamlit via GitHub

You can deploy the applications to **Streamlit Community Cloud** (100% free) directly from your GitHub repository:
**`https://github.com/MHassan0000/ML-Prediction-Models`**

### Option A: Deploy Both Tasks in One Unified App (Recommended)
By deploying `app.py`, you provide your instructor with a single web link that features:
- A navigation sidebar switching between **Overview & Dossier**, **Task 1: Titanic**, and **Task 2: Heart Disease**.
- Student ID card with your name (**Muhammad Hassan Yousaf**) and roll number (**FA24-BSE-082**).
- Batch CSV uploaders, live confidence gauges, and model dashboards for both tasks.

**Step-by-step instructions:**
1. Navigate to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account (`MHassan0000`).
2. Click **Create app** (or **New app**).
3. Fill in the deployment details:
   - **Repository:** `MHassan0000/ML-Prediction-Models`
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - **App URL (optional custom subdomain):** e.g., `hassan-ml-prediction-hub`
4. Click **Deploy!**
5. Within 1-2 minutes, your live Streamlit Cloud URL will be online and shareable!

---

### Option B: Deploy As Two Separate Streamlit Apps

If you want separate links for each task:

#### Deploy Task 1 (Titanic):
1. On [share.streamlit.io](https://share.streamlit.io/), click **Create app**.
2. **Repository:** `MHassan0000/ML-Prediction-Models`
3. **Branch:** `main`
4. **Main file path:** `titanic_app.py`
5. Click **Deploy!**

#### Deploy Task 2 (Heart Disease):
1. Click **Create app** again.
2. **Repository:** `MHassan0000/ML-Prediction-Models`
3. **Branch:** `main`
4. **Main file path:** `heart_disease_app.py`
5. Click **Deploy!**

---

## Local Execution Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Unified Streamlit App
```bash
python -m streamlit run app.py
```

### 3. Or Run Standalone Apps
```bash
# Task 1 — Titanic
python -m streamlit run titanic_app.py

# Task 2 — Heart Disease
python -m streamlit run heart_disease_app.py
```

### 4. Retrain Models
```bash
python pipelines/Task1_Titanic_Survival_Prediction.py
python pipelines/Task2_Heart_Disease_Prediction.py
```

### 5. Regenerate Submission Notebooks
```bash
python generate_notebooks.py
```

---

## Academic Integrity & Plagiarism Protection

- **Student Attribution:** All files, notebooks, headers, and UI footers are uniquely authored for **Muhammad Hassan Yousaf (FA24-BSE-082)**.
- **Unique Notebook Exposition:** The generated `.ipynb` files feature distinct markdown analysis, mathematical formulations for Support Vector Machine margins, and step-by-step interpretations of the findings.
- **Custom UI Architecture:** Completely original dark glassmorphism design with professional SVG icons and confidence gauges.
