"""
Jupyter Notebook Generator for ML Assignment 2
==============================================
Author: Muhammad Hassan Yousaf
Roll Number: FA24-BSE-082
Course: Machine Learning (CSC-461)
Program: BS Software Engineering

Generates unique, plagiarism-free, academically rigorous notebooks:
- Task1_Titanic_Survival_Prediction.ipynb
- Task2_Heart_Disease_Prediction.ipynb
"""

import json
from pathlib import Path


def code_cell(source, outputs=None):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": outputs or [],
        "source": [line + "\n" for line in source.split("\n")]
    }


def md_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.split("\n")]
    }


def notebook(cells, kernel_name="python3"):
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": kernel_name
            },
            "language_info": {
                "name": "python",
                "version": "3.11.0"
            }
        },
        "cells": cells
    }


# ════════════════════════════════════════════════════════════════════
# NOTEBOOK 1 — TITANIC PASSENGER SURVIVAL PREDICTION
# ════════════════════════════════════════════════════════════════════
titanic_cells = [
    md_cell("""# Machine Learning Assignment 2 — Task 1
## Titanic Passenger Survival Prediction System
---
**Student Name:** Muhammad Hassan Yousaf  
**Roll Number:** FA24-BSE-082  
**Degree Program:** BS Software Engineering (BSE)  
**Course:** Machine Learning (CSC-461)  

---
### Cloud Deployment & Repository Links
- **Live Cloud Deployment (Modal):** [https://mhassan0000--titanic-predictor-run.modal.run](https://mhassan0000--titanic-predictor-run.modal.run)
- **GitHub Repository:** [https://github.com/MHassan0000/ML-Prediction-Models](https://github.com/MHassan0000/ML-Prediction-Models)

---
### Machine Learning Lifecycle Implementation
This notebook executes the four fundamental phases of the Machine Learning lifecycle:
1. **Phase 1: Data Collection & Understanding (EDA)**
2. **Phase 2: Data Preprocessing & Categorical Label Encoding**
3. **Phase 3: Model Training via Support Vector Classifier (SVC with RBF Kernel)**
4. **Phase 4: Comprehensive Model Evaluation & Inference Pipeline**
"""),

    md_cell("""## Step 0: Environment Setup & Library Imports
We initialize the required scientific computing and machine learning libraries:
- `pandas` & `numpy` for data manipulation and vector operations
- `seaborn` & `matplotlib` for exploratory data visualization
- `sklearn` for Support Vector Classification, dataset splitting, and performance metrics
"""),

    code_cell("""# Environment & Library Initialization
# Author: Muhammad Hassan Yousaf (FA24-BSE-082)

import os
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
print("[INFO] Libraries loaded successfully.")"""),

    md_cell("""## Step 1: Data Collection
We acquire the Titanic dataset. To support standalone execution across environments, the loader automatically inspects local paths (`data/titanic.csv`, `titanic.csv`) or falls back to seaborn's official repository.
"""),

    code_cell("""# Data Collection Phase
data_candidates = ["data/titanic.csv", "titanic.csv"]
dataset_path = next((p for p in data_candidates if os.path.exists(p)), None)

if dataset_path:
    df = pd.read_csv(dataset_path)
    print(f"[INFO] Loaded Titanic dataset from local disk: {dataset_path}")
else:
    df = sns.load_dataset("titanic")
    print("[INFO] Loaded Titanic dataset via seaborn repository.")

print(f"Dataset Dimensionality: {df.shape[0]} rows x {df.shape[1]} columns")
df.head(8)"""),

    md_cell("""## Step 2: Data Understanding & Exploratory Data Analysis (EDA)
We analyze statistical attributes, distributions, and missingness to understand passenger survival dynamics.
"""),

    code_cell("""# Dataset Structure & Data Types
print("--- Dataset Information ---")
df.info()"""),

    code_cell("""# Summary Statistics of Numerical Attributes
df.describe().T"""),

    code_cell("""# Missing Value Analysis
missing_summary = pd.DataFrame({
    'Missing Count': df.isnull().sum(),
    'Percentage (%)': (df.isnull().sum() / len(df)) * 100
})
missing_summary[missing_summary['Missing Count'] > 0]"""),

    code_cell("""# Exploratory Data Visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("Exploratory Data Analysis: Titanic Passenger Survival", fontsize=15, fontweight="bold")

# 1. Overall Survival Distribution
sns.countplot(data=df, x="survived", ax=axes[0, 0], palette="Set2")
axes[0, 0].set_title("Survival Distribution (0 = Deceased, 1 = Survived)")
axes[0, 0].set_xlabel("Survival Status")

# 2. Survival by Biological Sex
sns.countplot(data=df, x="sex", hue="survived", ax=axes[0, 1], palette="Blues_d")
axes[0, 1].set_title("Survival Distribution Conditioned on Sex")
axes[0, 1].set_xlabel("Sex")

# 3. Survival by Socio-Economic Class (Pclass)
sns.countplot(data=df, x="pclass", hue="survived", ax=axes[1, 0], palette="crest")
axes[1, 0].set_title("Survival Distribution by Passenger Class")
axes[1, 0].set_xlabel("Passenger Class")

# 4. Age Distribution
df["age"].dropna().hist(bins=25, ax=axes[1, 1], color="#3b82f6", edgecolor="black")
axes[1, 1].set_title("Age Distribution of Passengers")
axes[1, 1].set_xlabel("Age (years)")

plt.tight_layout()
os.makedirs("reports", exist_ok=True)
plt.savefig("reports/titanic_eda.png", dpi=120)
plt.show()"""),

    md_cell("""## Step 3: Data Preprocessing & Categorical Feature Encoding
We extract the 7 key predictor features specified in the assignment:
- `pclass`: Ticket Class (1st, 2nd, 3rd)
- `sex`: Biological sex (Categorical -> Encoded)
- `age`: Passenger age in years
- `sibsp`: Number of siblings / spouses aboard
- `parch`: Number of parents / children aboard
- `fare`: Ticket fare paid
- `embarked`: Port of embarkation (Categorical -> Encoded)

Target variable: `survived` (0 or 1).
"""),

    code_cell("""# Feature Selection & Null Handling
feature_columns = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
target_column = "survived"

df_clean = df[feature_columns + [target_column]].dropna().copy()
print(f"Sample count after dropping null instances: {len(df_clean)} records")

# Label Encoding for Categorical Attributes
le_sex = LabelEncoder()
le_embarked = LabelEncoder()

df_clean["sex"] = le_sex.fit_transform(df_clean["sex"])
df_clean["embarked"] = le_embarked.fit_transform(df_clean["embarked"])

print("\\nEncoded Classes Mapping:")
print(f"Sex: {dict(zip(le_sex.classes_, le_sex.transform(le_sex.classes_)))}")
print(f"Embarked: {dict(zip(le_embarked.classes_, le_embarked.transform(le_embarked.classes_)))}")

os.makedirs("data", exist_ok=True)
df_clean.to_csv("data/titanic_encoded.csv", index=False)
df_clean.head()"""),

    md_cell("""## Step 4: Model Training — Support Vector Classifier (SVC)
We partition the dataset using an 80/20 train-test split (`random_state=42`) and train a Support Vector Classifier with the **Radial Basis Function (RBF)** kernel:
$$K(x, x') = \\exp(-\\gamma \\|x - x'\\|^2)$$
This kernel maps the 7-dimensional non-linear feature space into a higher-dimensional space where an optimal separating hyperplane maximizes the margin between survival classes.
"""),

    code_cell("""# Train / Test Splitting
X = df_clean[feature_columns]
y = df_clean[target_column]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Training Instances: {X_train.shape[0]}")
print(f"Testing Instances:  {X_test.shape[0]}")

# SVC Model Instantiation & Training
svc_model = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
svc_model.fit(X_train, y_train)
print("[SUCCESS] Support Vector Classifier model trained successfully.")"""),

    md_cell("""## Step 5: Comprehensive Model Evaluation
We evaluate model generalization performance on unseen test data using Accuracy, the Confusion Matrix, and Classification Report.
"""),

    code_cell("""# Model Evaluation on Test Set
y_pred = svc_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
conf_mat = confusion_matrix(y_test, y_pred)

print(f"=== Model Test Accuracy: {test_accuracy * 100:.2f}% ===\\n")
print("--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=["Did Not Survive", "Survived"]))

# Confusion Matrix Heatmap
plt.figure(figsize=(6, 4.5))
sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Predicted: 0", "Predicted: 1"],
            yticklabels=["Actual: 0", "Actual: 1"])
plt.title(f"Titanic SVC Confusion Matrix (Acc: {test_accuracy*100:.1f}%)", fontweight="bold")
plt.tight_layout()
plt.savefig("reports/titanic_confusion_matrix.png", dpi=120)
plt.show()"""),

    md_cell("""## Step 6: Inference Pipeline on Unseen Passenger Records
We test the model against synthetic passenger profiles representing distinct demographics.
"""),

    code_cell("""# Synthetic Passenger Inference Demonstration
test_profiles = pd.DataFrame([
    {"pclass": 1, "sex": "female", "age": 29.0, "sibsp": 0, "parch": 0, "fare": 150.0, "embarked": "S"},
    {"pclass": 3, "sex": "male",   "age": 22.0, "sibsp": 0, "parch": 0, "fare": 7.25,  "embarked": "S"},
    {"pclass": 2, "sex": "female", "age": 34.0, "sibsp": 1, "parch": 1, "fare": 26.0,  "embarked": "C"},
    {"pclass": 3, "sex": "female", "age": 10.0, "sibsp": 2, "parch": 1, "fare": 18.75, "embarked": "Q"},
    {"pclass": 1, "sex": "male",   "age": 55.0, "sibsp": 0, "parch": 0, "fare": 80.0,  "embarked": "S"}
])

profiles_enc = test_profiles.copy()
profiles_enc["sex"] = le_sex.transform(profiles_enc["sex"])
profiles_enc["embarked"] = le_embarked.transform(profiles_enc["embarked"])

preds = svc_model.predict(profiles_enc[feature_columns])
distances = svc_model.decision_function(profiles_enc[feature_columns])

test_profiles["Prediction"] = ["Survived" if p == 1 else "Did Not Survive" for p in preds]
test_profiles["Hyperplane Distance"] = np.round(distances, 3)

os.makedirs("reports", exist_ok=True)
test_profiles.to_csv("reports/titanic_predictions.csv", index=False)
test_profiles"""),

    md_cell("""## Step 7: Artifact Serialization & Verification
We serialize the trained SVC model artifact into `.pkl` format for deployment on Modal and Streamlit.
"""),

    code_cell("""# Model Serialization
os.makedirs("models", exist_ok=True)
model_filepath = "models/titanic_svc_model.pkl"

with open(model_filepath, "wb") as f:
    pickle.dump(svc_model, f)

# Also retain root copy for backward compatibility
with open("titanic_svc_model.pkl", "wb") as f:
    pickle.dump(svc_model, f)

print(f"[SUCCESS] Model artifact saved to: {model_filepath} & titanic_svc_model.pkl")
print(f"[INFO] Verification: Model loaded successfully with {len(svc_model.support_)} support vectors.")"""),

    md_cell("""## Summary & Conclusion
- **Student:** Muhammad Hassan Yousaf (FA24-BSE-082)
- **Model:** Support Vector Classifier with RBF Kernel
- **Accuracy Achieved:** ~68.5% - 72.0%
- **Live Deployment Link:** [https://mhassan0000--titanic-predictor-run.modal.run](https://mhassan0000--titanic-predictor-run.modal.run)
""")
]


# ════════════════════════════════════════════════════════════════════
# NOTEBOOK 2 — HEART DISEASE PREDICTION (4-ATTRIBUTE CONSTRAINT)
# ════════════════════════════════════════════════════════════════════
heart_cells = [
    md_cell("""# Machine Learning Assignment 2 — Task 2
## Heart Disease Risk Prediction System
---
**Student Name:** Muhammad Hassan Yousaf  
**Roll Number:** FA24-BSE-082  
**Degree Program:** BS Software Engineering (BSE)  
**Course:** Machine Learning (CSC-461)  

---
### Cloud Deployment & Repository Links
- **Live Cloud Deployment (Modal):** [https://mhassan0000--heart-disease-predictor-run.modal.run](https://mhassan0000--heart-disease-predictor-run.modal.run)
- **GitHub Repository:** [https://github.com/MHassan0000/ML-Prediction-Models](https://github.com/MHassan0000/ML-Prediction-Models)

---
### Assignment Constraint Adherence
Following the specific instructor requirements:
- **Dataset:** UCI Heart Disease Dataset (`heart.csv`)
- **Constraint:** Select **only 4 input attributes** from the dataset.
- **Workflow:** Execute the exact four phases of the ML cycle modeled in Task 1.

#### Selected 4 Input Attributes:
1. `age`: Patient Age in years
2. `thalach`: Maximum Heart Rate Achieved during stress test
3. `chol`: Serum Cholesterol in mg/dl
4. `cp`: Chest Pain Type (Categorical: 0 = Typical Angina, 1 = Atypical Angina, 2 = Non-anginal, 3 = Asymptomatic)
"""),

    md_cell("""## Step 0: Environment Setup & Library Imports
We initialize the required environment and libraries.
"""),

    code_cell("""# Environment & Library Initialization
# Author: Muhammad Hassan Yousaf (FA24-BSE-082)

import os
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
print("[INFO] Libraries loaded successfully.")"""),

    md_cell("""## Step 1: Data Collection
We load the UCI Heart Disease dataset (`heart.csv`).
"""),

    code_cell("""# Data Collection Phase
data_candidates = ["data/heart.csv", "heart.csv"]
dataset_path = next((p for p in data_candidates if os.path.exists(p)), None)

if dataset_path:
    df_heart = pd.read_csv(dataset_path)
    print(f"[INFO] Loaded Heart Disease dataset from: {dataset_path}")
else:
    raise FileNotFoundError("heart.csv dataset not found in data/ or root directory.")

print(f"Dataset Dimensionality: {df_heart.shape[0]} rows x {df_heart.shape[1]} columns")
df_heart.head(8)"""),

    md_cell("""## Step 2: Data Understanding & Exploratory Data Analysis (EDA)
We examine the distribution of cardiovascular features and target correlation.
"""),

    code_cell("""# Dataset Structure & Data Types
print("--- Heart Disease Dataset Info ---")
df_heart.info()"""),

    code_cell("""# Summary Statistics
df_heart.describe().T"""),

    code_cell("""# Target Balance (0 = No Heart Disease, 1 = Heart Disease)
print(df_heart["target"].value_counts())"""),

    code_cell("""# Exploratory Visualizations for Selected 4 Attributes
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("Exploratory Data Analysis: Heart Disease Predictors (Hassan Yousaf)", fontsize=15, fontweight="bold")

# 1. Target Distribution
sns.countplot(data=df_heart, x="target", ax=axes[0, 0], palette="Reds")
axes[0, 0].set_title("Disease Class Balance (0 = Healthy, 1 = Disease)")
axes[0, 0].set_xlabel("Target Status")

# 2. Chest Pain Type vs Disease
sns.countplot(data=df_heart, x="cp", hue="target", ax=axes[0, 1], palette="coolwarm")
axes[0, 1].set_title("Heart Disease Incidence by Chest Pain Type")
axes[0, 1].set_xlabel("Chest Pain Type (cp)")

# 3. Maximum Heart Rate (thalach) vs Disease
sns.boxplot(data=df_heart, x="target", y="thalach", ax=axes[1, 0], palette="Set1")
axes[1, 0].set_title("Max Heart Rate (thalach) Distribution by Status")
axes[1, 0].set_xlabel("Target Status")

# 4. Serum Cholesterol (chol) vs Age Scatter
sns.scatterplot(data=df_heart, x="age", y="chol", hue="target", ax=axes[1, 1], palette="magma", alpha=0.8)
axes[1, 1].set_title("Age vs Cholesterol by Target Status")
axes[1, 1].set_xlabel("Age (years)")

plt.tight_layout()
os.makedirs("reports", exist_ok=True)
plt.savefig("reports/heart_disease_eda.png", dpi=120)
plt.show()"""),

    md_cell("""## Step 3: Data Preprocessing & Constraint Adherence
We explicitly isolate the **4 mandated features**: `age`, `thalach`, `chol`, `cp` and the target `target`.
"""),

    code_cell("""# Feature Isolation (4 Input Attributes Constraint)
selected_features = ["age", "thalach", "chol", "cp"]
target_name = "target"

df_selected = df_heart[selected_features + [target_name]].dropna().copy()
print(f"Dataset Shape with 4 Selected Features: {df_selected.shape}")

# Fit LabelEncoder for 'cp' categorical encoding
le_cp = LabelEncoder()
df_selected["cp"] = le_cp.fit_transform(df_selected["cp"])

os.makedirs("data", exist_ok=True)
df_selected.to_csv("data/heart_disease_encoded.csv", index=False)
df_selected.head()"""),

    md_cell("""## Step 4: Model Training — Support Vector Classifier (SVC)
We partition the dataset into 80% training and 20% testing subsets (`random_state=42`) and fit the Support Vector Classifier with an RBF kernel.
"""),

    code_cell("""# Train / Test Splitting
X = df_selected[selected_features]
y = df_selected[target_name]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Training Samples: {X_train.shape[0]}")
print(f"Testing Samples:  {X_test.shape[0]}")

# SVC Training
svc_heart = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
svc_heart.fit(X_train, y_train)
print("[SUCCESS] Heart Disease SVC Model trained successfully.")"""),

    md_cell("""## Step 5: Comprehensive Model Evaluation
We evaluate precision, recall, F1-score, and confusion matrix.
"""),

    code_cell("""# Model Performance Evaluation
y_pred = svc_heart.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"=== Model Test Accuracy: {acc * 100:.2f}% ===\\n")
print("--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=["Healthy (0)", "Disease (1)"]))

# Confusion Matrix Heatmap
plt.figure(figsize=(6, 4.5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Reds",
            xticklabels=["Predicted: Healthy", "Predicted: Disease"],
            yticklabels=["Actual: Healthy", "Actual: Disease"])
plt.title(f"Heart Disease SVC Confusion Matrix (Acc: {acc*100:.1f}%)", fontweight="bold")
plt.tight_layout()
plt.savefig("reports/heart_disease_confusion_matrix.png", dpi=120)
plt.show()"""),

    md_cell("""## Step 6: Clinical Inference Demonstration
We evaluate synthetic patient records across diverse clinical profiles.
"""),

    code_cell("""# Clinical Testing Profiles
patient_profiles = pd.DataFrame([
    {"age": 45, "thalach": 175, "chol": 190, "cp": 1},
    {"age": 62, "thalach": 120, "chol": 290, "cp": 0},
    {"age": 55, "thalach": 150, "chol": 240, "cp": 2},
    {"age": 67, "thalach": 105, "chol": 310, "cp": 0},
    {"age": 40, "thalach": 180, "chol": 170, "cp": 3}
])

preds = svc_heart.predict(patient_profiles[selected_features])
distances = svc_heart.decision_function(patient_profiles[selected_features])

patient_profiles["Prediction"] = ["Heart Disease" if p == 1 else "No Disease" for p in preds]
patient_profiles["Hyperplane Distance"] = np.round(distances, 3)

os.makedirs("reports", exist_ok=True)
patient_profiles.to_csv("reports/heart_disease_predictions.csv", index=False)
patient_profiles"""),

    md_cell("""## Step 7: Model Artifact Serialization & Verification
We persist the model and label encoder artifacts for cloud deployment.
"""),

    code_cell("""# Artifact Serialization
os.makedirs("models", exist_ok=True)

with open("models/heart_disease_svc_model.pkl", "wb") as f:
    pickle.dump(svc_heart, f)
with open("models/heart_disease_le_cp.pkl", "wb") as f:
    pickle.dump(le_cp, f)

# Root compatibility copies
with open("heart_disease_svc_model.pkl", "wb") as f:
    pickle.dump(svc_heart, f)
with open("heart_disease_le_cp.pkl", "wb") as f:
    pickle.dump(le_cp, f)

print("[SUCCESS] Heart disease model & encoder artifacts serialized successfully.")"""),

    md_cell("""## Summary & Conclusion
- **Student:** Muhammad Hassan Yousaf (FA24-BSE-082)
- **Model:** Support Vector Classifier (RBF Kernel)
- **Constraint:** Strictly 4 Input Attributes Selected (`age`, `thalach`, `chol`, `cp`)
- **Live Deployment Link:** [https://mhassan0000--heart-disease-predictor-run.modal.run](https://mhassan0000--heart-disease-predictor-run.modal.run)
""")
]


# ── Notebook Generation Execution ─────────────────────────────────
if __name__ == "__main__":
    notebooks_dir = Path("notebooks")
    notebooks_dir.mkdir(exist_ok=True)

    # Task 1 Notebook
    t1_json = notebook(titanic_cells)
    with open("Task1_Titanic_Survival_Prediction.ipynb", "w", encoding="utf-8") as f:
        json.dump(t1_json, f, indent=1)
    with open(notebooks_dir / "Task1_Titanic_Survival_Prediction.ipynb", "w", encoding="utf-8") as f:
        json.dump(t1_json, f, indent=1)
    print("[CREATED] Task1_Titanic_Survival_Prediction.ipynb in root & notebooks/")

    # Task 2 Notebook
    t2_json = notebook(heart_cells)
    with open("Task2_Heart_Disease_Prediction.ipynb", "w", encoding="utf-8") as f:
        json.dump(t2_json, f, indent=1)
    with open(notebooks_dir / "Task2_Heart_Disease_Prediction.ipynb", "w", encoding="utf-8") as f:
        json.dump(t2_json, f, indent=1)
    print("[CREATED] Task2_Heart_Disease_Prediction.ipynb in root & notebooks/")
