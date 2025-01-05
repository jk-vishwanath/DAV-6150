import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
from io import BytesIO

# Load the model from GitHub URL
@st.cache_resource
def load_model():
    url = "https://github.com/jk-vishwanath/DAV-6150/blob/main/heart_failure_model.pkl?raw=true"
    response = requests.get(url)
    if response.status_code == 200:
        return pickle.load(BytesIO(response.content))
    else:
        st.error("Failed to load the model from GitHub.")
        return None

model = load_model()

# Streamlit app
st.title("Heart Failure Prediction App")
st.write("This app predicts the likelihood of heart failure based on patient data.")

# Input fields
age = st.number_input("Age", min_value=1, max_value=120, value=30)
sex = st.selectbox("Sex", ["M", "F"])
chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
resting_bp = st.number_input("Resting Blood Pressure", min_value=50, max_value=250, value=120)
cholesterol = st.number_input("Cholesterol", min_value=50, max_value=600, value=200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
exercise_angina = st.selectbox("Exercise Induced Angina", ["Y", "N"])
oldpeak = st.number_input("Oldpeak (ST depression)", min_value=0.0, max_value=10.0, value=1.0)
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# Predict button
if st.button("Predict"):
    if model is not None:
        input_data = pd.DataFrame({
            "Age": [age],
            "Sex": [sex],
            "ChestPainType": [chest_pain],
            "RestingBP": [resting_bp],
            "Cholesterol": [cholesterol],
            "FastingBS": [fasting_bs],
            "RestingECG": [resting_ecg],
            "MaxHR": [max_hr],
            "ExerciseAngina": [exercise_angina],
            "Oldpeak": [oldpeak],
            "ST_Slope": [st_slope]
        })

        # Make prediction
        prediction = model.predict(input_data)
        result = "Heart Disease Detected" if prediction[0] == 1 else "No Heart Disease Detected"
        st.success(f"Prediction: {result}")
    else:
        st.error("Model could not be loaded.")
