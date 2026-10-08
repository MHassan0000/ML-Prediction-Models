"""
Jupyter Notebook Generator for Machine Learning Assignment 2
Author: Muhammad Hassan Yousaf
Roll Number: FA24-BSE-082
Course: Machine Learning (CSC-461)
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


# ====================================================================
# TASK 1: TITANIC SURVIVAL PREDICTION
# ====================================================================
titanic_cells = [
    md_cell("""# Task 1: Titanic Passenger Survival Prediction System

**Name:** Muhammad Hassan Yousaf  
**Roll Number:** FA24-BSE-082  
**Course:** Machine Learning (CSC-461)  

### Live Cloud Deployment
- **Streamlit App:** https://titanic-model-hassan.streamlit.app
- **Modal App:** https://mhassan0000--titanic-predictor-run.modal.run
- **GitHub Repository:** https://github.com/MHassan0000/ML-Prediction-Models

---
### Overview
In this task, we build an end to end machine learning pipeline to predict passenger survival on the Titanic using a Support Vector Classifier (SVC). The implementation follows the standard ML lifecycle steps:
1. Data Collection
2. Data Understanding and EDA
3. Data Preprocessing and Label Encoding
4. Model Training (SVC with RBF kernel)
5. Model Evaluation
6. Predictions on Sample Data
7. Saving the Trained Model
"""),

    md_cell("## 0. Import Libraries\nWe import pandas, numpy, seaborn, matplotlib, and scikit-learn modules."),

    code_cell("""import os
import pickle
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

warnings.filterwarnings('ignore')
print("Libraries imported successfully.")"""),

    md_cell("## 1. Data Collection\nLoad the Titanic dataset from local data folder or seaborn."),

    code_cell("""# Load dataset
data_path = "data/titanic.csv" if os.path.exists("data/titanic.csv") else "titanic.csv"

if os.path.exists(data_path):
    df = pd.read_csv(data_path)
    print(f"Loaded dataset from {data_path}")
else:
    df = sns.load_dataset("titanic")
    print("Loaded dataset from seaborn")

print("Dataset shape:", df.shape)
df.head()"""),

    md_cell("## 2. Data Understanding and Exploratory Data Analysis\nReview data structure, column types, statistics, missing values, and visual distributions."),

    code_cell("""# Column info and summary statistics
print("Column Information:")
df.info()

print("\\nDescriptive Statistics:")
df.describe()"""),

    code_cell("""# Check missing values
print("Missing values per column:")
print(df.isnull().sum())

print("\\nSurvival distribution:")
print(df['survived'].value_counts())"""),

    code_cell("""# EDA plots
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# 1. Overall survival count
sns.countplot(data=df, x='survived', ax=axes[0, 0], palette='Set2')
axes[0, 0].set_title('Survival Count (0 = Died, 1 = Survived)')

# 2. Survival by sex
sns.countplot(data=df, x='sex', hue='survived', ax=axes[0, 1], palette='Set1')
axes[0, 1].set_title('Survival by Gender')

# 3. Survival by passenger class
sns.countplot(data=df, x='pclass', hue='survived', ax=axes[1, 0], palette='Set3')
axes[1, 0].set_title('Survival by Passenger Class')

# 4. Age distribution
df['age'].dropna().hist(bins=25, ax=axes[1, 1], color='steelblue', edgecolor='black')
axes[1, 1].set_title('Age Distribution')

plt.tight_layout()
os.makedirs("reports", exist_ok=True)
plt.savefig("reports/titanic_eda.png", dpi=100)
plt.show()"""),

    md_cell("## 3. Data Preprocessing and Label Encoding\nSelect the 7 required features: pclass, sex, age, sibsp, parch, fare, and embarked. Handle missing values and encode categorical attributes."),

    code_cell("""# Feature selection
features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
target = 'survived'

df_clean = df[features + [target]].dropna().copy()
print("Shape after dropping nulls:", df_clean.shape)

# Encode sex and embarked columns
le_sex = LabelEncoder()
le_embarked = LabelEncoder()

df_clean['sex'] = le_sex.fit_transform(df_clean['sex'])
df_clean['embarked'] = le_embarked.fit_transform(df_clean['embarked'])

# Save encoded dataset
os.makedirs("data", exist_ok=True)
df_clean.to_csv("data/titanic_encoded.csv", index=False)
df_clean.head()"""),

    md_cell("## 4. Model Training (Support Vector Classifier)\nSplit data into 80% training and 20% testing sets, then fit an SVC model with RBF kernel."),

    code_cell("""# Train-test split
X = df_clean[features]
y = df_clean[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Train SVC
svc_model = SVC(kernel='rbf', random_state=42)
svc_model.fit(X_train, y_train)
print("SVC model trained successfully.")"""),

    md_cell("## 5. Model Evaluation\nCalculate accuracy, generate confusion matrix, and print classification report."),

    code_cell("""# Evaluate model
y_pred = svc_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"Test Accuracy: {acc * 100:.2f}%\\n")
print("Confusion Matrix:")
print(cm)

print("\\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Died", "Survived"]))

# Plot confusion matrix
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Died', 'Survived'],
            yticklabels=['Died', 'Survived'])
plt.title(f"Confusion Matrix (Accuracy: {acc*100:.1f}%)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("reports/titanic_confusion_matrix.png", dpi=100)
plt.show()"""),

    md_cell("## 6. Testing Predictions\nTest model predictions on sample passenger profiles."),

    code_cell("""# Sample predictions
sample_data = pd.DataFrame([
    {"pclass": 1, "sex": "female", "age": 29.0, "sibsp": 0, "parch": 0, "fare": 211.0, "embarked": "S"},
    {"pclass": 3, "sex": "male",   "age": 22.0, "sibsp": 1, "parch": 0, "fare": 7.25,  "embarked": "S"},
    {"pclass": 2, "sex": "female", "age": 35.0, "sibsp": 1, "parch": 0, "fare": 26.0,  "embarked": "C"}
])

sample_encoded = sample_data.copy()
sample_encoded['sex'] = le_sex.transform(sample_encoded['sex'])
sample_encoded['embarked'] = le_embarked.transform(sample_encoded['embarked'])

predictions = svc_model.predict(sample_encoded[features])
sample_data['Predicted_Survival'] = ['Survived' if p == 1 else 'Did Not Survive' for p in predictions]

os.makedirs("reports", exist_ok=True)
sample_data.to_csv("reports/titanic_predictions.csv", index=False)
sample_data"""),

    md_cell("## 7. Saving the Trained Model\nSerialize the model using pickle for deployment."),

    code_cell("""# Save model
os.makedirs("models", exist_ok=True)
with open("models/titanic_svc_model.pkl", "wb") as f:
    pickle.dump(svc_model, f)

print("Model saved to models/titanic_svc_model.pkl")""")
]


# ====================================================================
# TASK 2: HEART DISEASE PREDICTION (4-ATTRIBUTE CONSTRAINT)
# ====================================================================
heart_cells = [
    md_cell("""# Task 2: Heart Disease Prediction System

**Name:** Muhammad Hassan Yousaf  
**Roll Number:** FA24-BSE-082  
**Course:** Machine Learning (CSC-461)  

### Live Cloud Deployment
- **Streamlit App:** https://heart-disease-model-hassan.streamlit.app
- **Modal App:** https://mhassan0000--heart-disease-predictor-run.modal.run
- **GitHub Repository:** https://github.com/MHassan0000/ML-Prediction-Models

---
### Assignment Constraint
As specified in the assignment instructions, we select **only 4 input attributes** from the UCI Heart Disease dataset:
1. `age`: Patient age in years
2. `thalach`: Maximum heart rate achieved during exercise stress test
3. `chol`: Serum cholesterol in mg/dl
4. `cp`: Chest pain type (0 to 3)

We apply the exact same ML lifecycle steps as Task 1.
"""),

    md_cell("## 0. Import Libraries\nImport libraries for analysis, visualization, and modeling."),

    code_cell("""import os
import pickle
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

warnings.filterwarnings('ignore')
print("Libraries imported successfully.")"""),

    md_cell("## 1. Data Collection\nLoad the UCI Heart Disease dataset (`heart.csv`)."),

    code_cell("""# Load heart disease dataset
data_path = "data/heart.csv" if os.path.exists("data/heart.csv") else "heart.csv"

if os.path.exists(data_path):
    df_heart = pd.read_csv(data_path)
    print(f"Loaded dataset from {data_path}")
else:
    raise FileNotFoundError("heart.csv not found in data/ or root")

print("Dataset shape:", df_heart.shape)
df_heart.head()"""),

    md_cell("## 2. Data Understanding and Exploratory Data Analysis\nInspect dataset summary, target balance, and feature distributions."),

    code_cell("""# Dataset info and statistics
print("Dataset Info:")
df_heart.info()

print("\\nDescriptive Statistics:")
df_heart.describe()"""),

    code_cell("""# Target distribution (0 = No Disease, 1 = Disease)
print("Target counts:")
print(df_heart['target'].value_counts())"""),

    code_cell("""# EDA plots for the 4 attributes
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# 1. Target count
sns.countplot(data=df_heart, x='target', ax=axes[0, 0], palette='Reds')
axes[0, 0].set_title('Heart Disease Target Distribution')

# 2. Chest pain type vs target
sns.countplot(data=df_heart, x='cp', hue='target', ax=axes[0, 1], palette='coolwarm')
axes[0, 1].set_title('Heart Disease by Chest Pain Type')

# 3. Max heart rate boxplot
sns.boxplot(data=df_heart, x='target', y='thalach', ax=axes[1, 0], palette='Set2')
axes[1, 0].set_title('Max Heart Rate vs Target')

# 4. Cholesterol vs Age
sns.scatterplot(data=df_heart, x='age', y='chol', hue='target', ax=axes[1, 1], palette='magma')
axes[1, 1].set_title('Age vs Cholesterol')

plt.tight_layout()
os.makedirs("reports", exist_ok=True)
plt.savefig("reports/heart_disease_eda.png", dpi=100)
plt.show()"""),

    code_cell("""# Feature correlation heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(df_heart.corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title("Heart Disease Feature Correlation")
plt.tight_layout()
plt.savefig("reports/heart_disease_correlation.png", dpi=100)
plt.show()"""),

    md_cell("## 3. Data Preprocessing and 4-Attribute Selection\nIsolate the 4 required attributes: `age`, `thalach`, `chol`, and `cp`. Encode `cp` and verify missing values."),

    code_cell("""# Select the 4 mandated features
selected_features = ['age', 'thalach', 'chol', 'cp']
target_col = 'target'

df_sub = df_heart[selected_features + [target_col]].dropna().copy()
print("Selected dataset shape:", df_sub.shape)

# Encode chest pain type (cp)
le_cp = LabelEncoder()
df_sub['cp'] = le_cp.fit_transform(df_sub['cp'])

# Save encoded dataset
os.makedirs("data", exist_ok=True)
df_sub.to_csv("data/heart_disease_encoded.csv", index=False)
df_sub.head()"""),

    md_cell("## 4. Model Training (SVC with RBF Kernel)\nSplit the data into train (80%) and test (20%) sets, then train the Support Vector Classifier."),

    code_cell("""# Train-test split
X = df_sub[selected_features]
y = df_sub[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Train SVC
svc_heart = SVC(kernel='rbf', random_state=42)
svc_heart.fit(X_train, y_train)
print("SVC model trained successfully.")"""),

    md_cell("## 5. Model Evaluation\nEvaluate performance using accuracy score, confusion matrix, and classification report."),

    code_cell("""# Evaluate model
y_pred = svc_heart.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"Test Accuracy: {acc * 100:.2f}%\\n")
print("Confusion Matrix:")
print(cm)

print("\\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["No Disease", "Disease"]))

# Plot confusion matrix
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
            xticklabels=['No Disease', 'Disease'],
            yticklabels=['No Disease', 'Disease'])
plt.title(f"Confusion Matrix (Accuracy: {acc*100:.1f}%)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("reports/heart_disease_confusion_matrix.png", dpi=100)
plt.show()"""),

    md_cell("## 6. Testing Predictions\nRun predictions on sample patient data with different clinical attributes."),

    code_cell("""# Sample patient testing
patients = pd.DataFrame([
    {"age": 52, "thalach": 168, "chol": 212, "cp": 0},
    {"age": 67, "thalach": 108, "chol": 300, "cp": 1},
    {"age": 45, "thalach": 150, "chol": 225, "cp": 2},
    {"age": 38, "thalach": 170, "chol": 174, "cp": 3},
    {"age": 60, "thalach": 120, "chol": 280, "cp": 0}
])

patients_enc = patients.copy()
patients_enc['cp'] = le_cp.transform(patients_enc['cp'])

preds = svc_heart.predict(patients_enc[selected_features])
patients['Prediction'] = ['Heart Disease' if p == 1 else 'No Heart Disease' for p in preds]

os.makedirs("reports", exist_ok=True)
patients.to_csv("reports/heart_disease_predictions.csv", index=False)
patients"""),

    md_cell("## 7. Saving the Trained Model\nSave the trained SVC model and chest pain label encoder."),

    code_cell("""# Save model and label encoder
os.makedirs("models", exist_ok=True)

with open("models/heart_disease_svc_model.pkl", "wb") as f:
    pickle.dump(svc_heart, f)

with open("models/heart_disease_le_cp.pkl", "wb") as f:
    pickle.dump(le_cp, f)

print("Saved model to models/heart_disease_svc_model.pkl")
print("Saved encoder to models/heart_disease_le_cp.pkl")""")
]


# ====================================================================
# WRITE NOTEBOOKS
# ====================================================================
if __name__ == "__main__":
    notebooks_dir = Path("notebooks")
    notebooks_dir.mkdir(exist_ok=True)

    # Task 1 Notebook
    t1_json = notebook(titanic_cells)
    with open("Task1_Titanic_Survival_Prediction.ipynb", "w", encoding="utf-8") as f:
        json.dump(t1_json, f, indent=1)
    with open(notebooks_dir / "Task1_Titanic_Survival_Prediction.ipynb", "w", encoding="utf-8") as f:
        json.dump(t1_json, f, indent=1)
    print("Created Task1_Titanic_Survival_Prediction.ipynb (root and notebooks/)")

    # Task 2 Notebook
    t2_json = notebook(heart_cells)
    with open("Task2_Heart_Disease_Prediction.ipynb", "w", encoding="utf-8") as f:
        json.dump(t2_json, f, indent=1)
    with open(notebooks_dir / "Task2_Heart_Disease_Prediction.ipynb", "w", encoding="utf-8") as f:
        json.dump(t2_json, f, indent=1)
    print("Created Task2_Heart_Disease_Prediction.ipynb (root and notebooks/)")
