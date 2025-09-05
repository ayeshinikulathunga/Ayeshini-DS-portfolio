import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("heart_disease_rf_model.pkl")
scaler = joblib.load("heart_scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.title("Heart Disease Predictor")
st.write("Provide patient details to estimate the risk of heart disease.")

#user inputs
age = st.number_input("Age (years)", min_value=20, max_value=100, value=50)
sex = st.selectbox("Sex", options={0: "Female", 1: "Male"}.keys(), format_func=lambda x: "Female" if x==0 else "Male")
cp = st.selectbox("Chest Pain Type", options={
    0: "Typical Angina",
    1: "Atypical Angina",
    2: "Non-anginal Pain",
    3: "Asymptomatic"
}.keys(), format_func=lambda x: {0:"Typical Angina",1:"Atypical Angina",2:"Non-anginal Pain",3:"Asymptomatic"}[x])

trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options={0: "No", 1: "Yes"}.keys(), format_func=lambda x: "Yes" if x==1 else "No")
restecg = st.selectbox("Resting ECG Results", options={
    0: "Normal",
    1: "ST-T Wave Abnormality",
    2: "Left Ventricular Hypertrophy"
}.keys(), format_func=lambda x: {0:"Normal",1:"ST-T Wave Abnormality",2:"Left Ventricular Hypertrophy"}[x])

thalach = st.number_input("Maximum Heart Rate Achieved", min_value=70, max_value=220, value=150)
exang = st.selectbox("Exercise Induced Angina", options={0: "No", 1: "Yes"}.keys(), format_func=lambda x: "Yes" if x==1 else "No")
oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
slope = st.selectbox("Slope of Peak Exercise ST Segment", options={0:"Upsloping",1:"Flat",2:"Downsloping"}.keys(), format_func=lambda x: {0:"Upsloping",1:"Flat",2:"Downsloping"}[x])
ca = st.number_input("Number of Major Vessels (0-3) Colored by Fluoroscopy", min_value=0, max_value=3, value=0)
thal = st.selectbox("Thalassemia", options={0:"Normal",1:"Fixed Defect",2:"Reversible Defect"}.keys(), format_func=lambda x: {0:"Normal",1:"Fixed Defect",2:"Reversible Defect"}[x])

# Convert to feature vector 
# Start with zeros for all one-hot encoded features
user_data = {col: 0 for col in feature_columns}

user_data["age"] = age
user_data["sex"] = sex
user_data["trestbps"] = trestbps
user_data["chol"] = chol
user_data["fbs"] = fbs
user_data["thalach"] = thalach
user_data["exang"] = exang
user_data["oldpeak"] = oldpeak
user_data["ca"] = ca

user_data[f"cp_{cp}"] = 1
user_data[f"restecg_{restecg}"] = 1
user_data[f"slope_{slope}"] = 1
user_data[f"thal_{thal}"] = 1

if st.button("Predict"):
    input_df = pd.DataFrame([user_data])[feature_columns]
    input_scaled = scaler.transform(input_df)

    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    if pred == 1:
        st.error(f"⚠️ High risk of heart disease detected. (Probability: {prob*100:.2f}%)")
    else:
        st.success(f"✅ No heart disease detected. (Probability: {prob*100:.2f}%)")
