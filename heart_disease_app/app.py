import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("heart_model.pkl")
model_columns = joblib.load("model_columns.pkl")

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️")

st.title("❤️ Heart Disease Risk Prediction System")
st.write("Enter patient clinical data:")

# Inputs
age = st.number_input("Age", 1, 120)
bp = st.number_input("Resting Blood Pressure")
chol = st.number_input("Cholesterol")
maxhr = st.number_input("Max Heart Rate")
oldpeak = st.number_input("Oldpeak", step=0.1)
fastingbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])

chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
exercise_angina = st.selectbox("Exercise Angina", ["N", "Y"])
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

if st.button("Predict Risk"):
    input_df = pd.DataFrame(0, index=[0], columns=model_columns)

    input_df['Age'] = age
    input_df['RestingBP'] = bp
    input_df['Cholesterol'] = chol
    input_df['MaxHR'] = maxhr
    input_df['Oldpeak'] = oldpeak
    input_df['FastingBS'] = fastingbs

    if f'ChestPainType_{chest_pain}' in input_df.columns:
        input_df[f'ChestPainType_{chest_pain}'] = 1

    if exercise_angina == "Y":
        input_df['ExerciseAngina_Y'] = 1

    if f'ST_Slope_{st_slope}' in input_df.columns:
        input_df[f'ST_Slope_{st_slope}'] = 1

    prob = model.predict_proba(input_df)[0][1]

    if prob >= 0.35:
        st.error(f"⚠ HIGH RISK ({prob:.2%})")
    else:
        st.success(f"✅ LOW RISK ({prob:.2%})")
