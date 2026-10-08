# ML Prediction Models

> Machine Learning Assignment 2 — End-to-end ML systems from data collection to cloud deployment.

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.45-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Modal](https://img.shields.io/badge/Deployed_on-Modal-6366f1?logo=data:image/svg+xml;base64,PHN2Zz48L3N2Zz4=&logoColor=white)](https://modal.com)

---

## Live Deployments on Modal

| Application | Live URL | Status |
|-------------|----------|--------|
| **Titanic Survival Predictor** | [mhassan0000--titanic-predictor-run.modal.run](https://mhassan0000--titanic-predictor-run.modal.run) | Active |
| **Heart Disease Predictor** | [mhassan0000--heart-disease-predictor-run.modal.run](https://mhassan0000--heart-disease-predictor-run.modal.run) | Active |

---

## Overview

This project implements two complete Machine Learning prediction systems following the full ML lifecycle:

| # | Task | Dataset | Algorithm | Features |
|---|------|---------|-----------|----------|
| 1 | **Titanic Survival Prediction** | Titanic (seaborn built-in, 891 rows) | SVC (RBF) | 7 input attributes |
| 2 | **Heart Disease Prediction** | UCI Heart Disease Cleveland (303 rows) | SVC (RBF) | 4 input attributes |

Each system covers: Data Collection → EDA → Preprocessing → Label Encoding → Model Training → Evaluation → Deployment

## Architecture

```
ML-Prediction-Models/
├── Task1_Titanic_Survival_Prediction.py     # Training pipeline (Task 1)
├── Task1_Titanic_Survival_Prediction.ipynb  # Jupyter notebook (Task 1)
├── Task2_Heart_Disease_Prediction.py        # Training pipeline (Task 2)
├── Task2_Heart_Disease_Prediction.ipynb     # Jupyter notebook (Task 2)
├── titanic_app.py                           # Streamlit web app (Task 1)
├── heart_disease_app.py                     # Streamlit web app (Task 2)
├── modal_titanic.py                         # Modal cloud deployment (Task 1)
├── modal_heart_disease.py                   # Modal cloud deployment (Task 2)
├── titanic_svc_model.pkl                    # Trained SVC model (Task 1)
├── heart_disease_svc_model.pkl              # Trained SVC model (Task 2)
├── heart_disease_le_cp.pkl                  # Label encoder for chest pain
├── titanic.csv                              # Titanic dataset
├── heart.csv                                # Heart Disease dataset
├── .streamlit/config.toml                   # Streamlit theme configuration
├── requirements.txt                         # Python dependencies
└── generate_notebooks.py                    # Notebook generator utility
```

## ML Lifecycle (Both Tasks)

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│    1. Data    │───▶│   2. Data    │───▶│ 3. Preprocess│───▶│  4. Model    │
│  Collection  │    │Understanding │    │  & Encoding  │    │  Training    │
└──────────────┘    └──────────────┘    └──────────────┘    └──────┬───────┘
                                                                   │
┌──────────────┐    ┌──────────────┐    ┌──────────────┐           │
│ 7. Cloud     │◀───│6. Prediction │◀───│ 5. Model     │◀──────────┘
│  Deployment  │    │   & Export   │    │  Evaluation  │
└──────────────┘    └──────────────┘    └──────────────┘
```

## Features

### Web Application
- **Dark mode UI** with glassmorphism design and inline SVG icons
- **Confidence gauge** showing model certainty via `decision_function` distance
- **Batch prediction** — upload CSV files for bulk predictions with downloadable results
- **Model dashboard** — live accuracy, confusion matrix, and classification report
- **Clinical reference ranges** (Heart Disease app) for cholesterol and heart rate
- **Responsive design** with smooth CSS animations

### ML Pipeline
- **Support Vector Classifier** with RBF kernel
- **Label Encoding** for categorical features
- **Train/Test split** (80/20, random_state=42)
- **Evaluation metrics**: Accuracy, Confusion Matrix, Classification Report
- **EDA visualizations** saved as PNG

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train Models (generates `.pkl` files)

```bash
python Task1_Titanic_Survival_Prediction.py
python Task2_Heart_Disease_Prediction.py
```

### 3. Run Streamlit Apps Locally

```bash
# Task 1 — Titanic
streamlit run titanic_app.py

# Task 2 — Heart Disease
streamlit run heart_disease_app.py
```

### 4. Deploy to Modal

```bash
pip install modal
modal setup                          # one-time auth

# Deploy permanently
modal deploy modal_titanic.py
modal deploy modal_heart_disease.py

# Or test with ephemeral dev server
modal serve modal_titanic.py
```

### 5. Deploy to Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set main file to `titanic_app.py` or `heart_disease_app.py`
5. Deploy

## Datasets

### Task 1: Titanic
- **Source:** seaborn built-in dataset
- **Size:** 891 passengers
- **Features used:** pclass, sex, age, sibsp, parch, fare, embarked
- **Target:** survived (0/1)

### Task 2: Heart Disease
- **Source:** [UCI ML Repository](https://archive.ics.uci.edu/dataset/45/heart+disease)
- **Size:** 303 patients (Cleveland subset)
- **Features used (4 only — assignment constraint):** age, thalach, chol, cp
- **Target:** target (0=No Disease, 1=Disease)

## Requirements

```
scikit-learn==1.9.1
numpy==2.5.3
scipy==1.18.1
pandas==2.2.3
seaborn==0.13.2
matplotlib==3.11.2
streamlit==1.45.1
```

## License

This project is an academic assignment. All code is original work.
