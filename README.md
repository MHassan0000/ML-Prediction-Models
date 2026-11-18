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

## Assignment 4

### Task 1: Emotion Prediction (Instructor Reference)

Binary and multi-class emotion classification notebooks are in `Assignment4/Task1_Emotion_Prediction/`. Both use TF-IDF and compare 6 algorithms with 10-fold CV.

### Task 2: Amazon Product Review Sentiment Analysis (My Work)

Classifies Amazon product reviews as Positive or Negative using only the review text.

**Dataset:** Amazon Polarity (HuggingFace) — 25,000 reviews, balanced 50/50  
**Algorithms:** Logistic Regression, Multinomial NB, Random Forest, Linear SVC, KNN, Decision Tree  
**Best Model:** Linear SVC (~94% CV accuracy)  
**App:** [Streamlit](https://amazon-sentiment-hassan.streamlit.app)

Files are in `Assignment4/Task2_Amazon_Review_Sentiment/`.

## How to Run

```bash
pip install -r requirements.txt

# run notebook on Google Colab (upload housing.csv when prompted)

# run web app locally
streamlit run Assignment3/Task2_House_Price_Prediction/house_price_app.py
```
