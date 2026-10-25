# ML Prediction Models

Muhammad Hassan Yousaf — FA24-BSE-082  
Machine Learning (CSC-461)

## Assignment 2

| Task | Problem | Algorithm | Accuracy |
|------|---------|-----------|----------|
| 1 | Titanic Survival Prediction | SVC | ~80% |
| 2 | Heart Disease Prediction | SVC | ~82% |

**App:** [Streamlit](https://titanic-model-hassan.streamlit.app)

## Assignment 3

### Task 1: Car Price Prediction (Instructor Provided)

The instructor's notebooks for Car Price Prediction using K-Fold CV are in `Assignment3/Task1_Car_Price_Prediction/`. Algorithms used: Linear Regression, KNN, Decision Tree, Random Forest, Gradient Boosting.

### Task 2: House Price Prediction (My Work)

Predicts California house prices using 5 **different** algorithms with 10-Fold CV.

**Dataset:** California Housing (20,640 instances)  
**Algorithms:** Ridge, Lasso, ElasticNet, LinearSVR, XGBoost  
**Best Model:** XGBoost (RMSE = 0.4463)  
**App:** [Streamlit](https://house-price-hassan.streamlit.app)

Files are in `Assignment3/Task2_House_Price_Prediction/`.

## How to Run

```bash
pip install -r requirements.txt

# run notebook on Google Colab (upload housing.csv when prompted)

# run web app locally
streamlit run Assignment3/Task2_House_Price_Prediction/house_price_app.py
```
