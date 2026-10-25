"""
House Price Prediction - Streamlit App
Muhammad Hassan Yousaf (FA24-BSE-082)
Assignment 3, Task 2
"""

import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="House Price Prediction", page_icon="🏠", layout="wide")

BASE = Path(__file__).parent

# load model
@st.cache_resource
def load_model():
    model_path = BASE / "house_price_best_model.pkl"
    prep_path = BASE / "house_price_preprocessor.pkl"
    if not model_path.exists() or not prep_path.exists():
        # try models/ folder
        model_path = BASE.parent / "models" / "house_price_best_model.pkl"
        prep_path = BASE.parent / "models" / "house_price_preprocessor.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(prep_path, "rb") as f:
        prep = pickle.load(f)
    return model, prep

model, prep = load_model()

st.title("🏠 California House Price Prediction")
st.write("**Assignment 3, Task 2** — Muhammad Hassan Yousaf (FA24-BSE-082)")
st.write("Predicts median house value using XGBoost trained on the California Housing dataset (20,640 instances).")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    med_inc = st.number_input("Median Income (×$10k)", min_value=0.5, max_value=15.0, value=3.87, step=0.1)
    house_age = st.number_input("House Age (years)", min_value=1.0, max_value=52.0, value=29.0, step=1.0)
    ave_rooms = st.number_input("Avg Rooms per Household", min_value=1.0, max_value=15.0, value=5.43, step=0.1)
    ave_bedrms = st.number_input("Avg Bedrooms per Household", min_value=0.3, max_value=5.0, value=1.10, step=0.1)

with col2:
    population = st.number_input("Population", min_value=3.0, max_value=35682.0, value=1425.0, step=10.0)
    ave_occup = st.number_input("Avg Occupancy", min_value=1.0, max_value=10.0, value=3.07, step=0.1)
    latitude = st.number_input("Latitude", min_value=32.0, max_value=42.0, value=35.63, step=0.01)
    longitude = st.number_input("Longitude", min_value=-124.5, max_value=-114.0, value=-119.57, step=0.01)

if st.button("Predict Price", type="primary", use_container_width=True):
    inp = pd.DataFrame([{
        "MedInc": med_inc, "HouseAge": house_age, "AveRooms": ave_rooms,
        "AveBedrms": ave_bedrms, "Population": population, "AveOccup": ave_occup,
        "Latitude": latitude, "Longitude": longitude,
    }])

    prediction = model.predict(prep.transform(inp))[0]
    price = prediction * 100_000

    st.success(f"### Predicted House Value: **${price:,.0f}**")
    st.caption(f"Raw output: {prediction:.4f} (×$100k) | Model: XGBoost")

st.markdown("---")
st.caption("Muhammad Hassan Yousaf (FA24-BSE-082) — ML Assignment 3")
