import streamlit as st
import pandas as pd
import joblib
import sqlite3

# -------------------------------------------------
# Load Model & Columns (Loaded ONCE)
# -------------------------------------------------
model = joblib.load("heart_model.pkl")
model_columns = joblib.load("model_columns.pkl")

# -------------------------------------------------
# Database Functions (SQLite)
# -------------------------------------------------
def init_db():
    conn = sqlite3.connect("heart_logs.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            age INTEGER,
            sex TEXT,
            chest_pain TEXT,
            resting_bp INTEGER,
            cholesterol INTEGER,
            fasting_bs INTEGER,
            max_hr INTEGER,
            exercise_angina TEXT,
            oldpeak REAL,
            st_slope TEXT,
            risk_score REAL
        )
    """)
    conn.commit()
    conn.close()


def log_prediction(data, score):
    conn = sqlite3.connect("heart_logs.db")
    c = conn.cursor()
    c.execute("""
        INSERT INTO predictions VALUES (?,?,?,?,?,?,?,?,?,?,?)
    """, (
        data["Age"],
        data["Sex"],
        data["ChestPainType"],
        data["RestingBP"],
        data["Cholesterol"],
        data["FastingBS"],
        data["MaxHR"],
        data["ExerciseAngina"],
        data["Oldpeak"],
        data["ST_Slope"],
        score
    ))
    conn.commit()
    conn.close()


# Initialize database ONCE
init_db()

# -------------------------------------------------
# Streamlit UI
# -------------------------------------------------
st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️")

st.title("❤️ Heart Disease Risk Prediction System")
st.warning("⚠ This tool is for educational purposes only, not for medical diagnosis.")

st.subheader("Enter Patient Details")

# Input Fields
age = st.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.selectbox("Sex", ["M", "F"])
chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
resting_bp = st.number_input("Resting Blood Pressure", value=120)
cholesterol = st.number_input("Cholesterol", value=200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
max_hr = st.number_input("Max Heart Rate", value=150)
exercise_angina = st.selectbox("Exercise Angina", ["N", "Y"])
oldpeak = st.number_input("Oldpeak", step=0.1, value=0.0)
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# -------------------------------------------------
# Prediction Button
# -------------------------------------------------
if st.button("Predict Risk"):
    # Create empty dataframe with correct columns
    input_df = pd.DataFrame(0, index=[0], columns=model_columns)

    # Fill numerical values
    input_df["Age"] = age
    input_df["RestingBP"] = resting_bp
    input_df["Cholesterol"] = cholesterol
    input_df["FastingBS"] = fasting_bs
    input_df["MaxHR"] = max_hr
    input_df["Oldpeak"] = oldpeak

    # Handle categorical features
    if f"Sex_{sex}" in input_df.columns:
        input_df[f"Sex_{sex}"] = 1

    if f"ChestPainType_{chest_pain}" in input_df.columns:
        input_df[f"ChestPainType_{chest_pain}"] = 1

    if exercise_angina == "Y" and "ExerciseAngina_Y" in input_df.columns:
        input_df["ExerciseAngina_Y"] = 1

    if f"ST_Slope_{st_slope}" in input_df.columns:
        input_df[f"ST_Slope_{st_slope}"] = 1

    # Predict probability
    probability = model.predict_proba(input_df)[0][1]

    # Medical threshold
    threshold = 0.35

    if probability >= threshold:
        st.error(f"⚠ HIGH RISK of Heart Disease\n\nRisk Probability: {probability:.2%}")
    else:
        st.success(f"✅ LOW RISK of Heart Disease\n\nRisk Probability: {probability:.2%}")

    # Log prediction to database
    patient_data = {
        "Age": age,
        "Sex": sex,
        "ChestPainType": chest_pain,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "MaxHR": max_hr,
        "ExerciseAngina": exercise_angina,
        "Oldpeak": oldpeak,
        "ST_Slope": st_slope
    }

    log_prediction(patient_data, probability)

# -------------------------------------------------
# Show Prediction History (Optional)
# -------------------------------------------------
st.divider()

if st.checkbox("Show Prediction History"):
    conn = sqlite3.connect("heart_logs.db")
    df = pd.read_sql("SELECT * FROM predictions", conn)
    conn.close()
    st.dataframe(df)
