import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sqlite3

from sklearn.metrics import confusion_matrix

# =================================================
# PAGE CONFIG
# =================================================
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="wide"
)

# =================================================
# LOAD MODEL & FILES (ONCE)
# =================================================
model = joblib.load("heart_model.pkl")
model_columns = joblib.load("model_columns.pkl")

# Optional evaluation files
try:
    X_test = joblib.load("X_test.pkl")
    y_test = joblib.load("y_test.pkl")
except:
    X_test, y_test = None, None

# =================================================
# DATABASE FUNCTIONS
# =================================================
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

init_db()

# =================================================
# HEADER
# =================================================
st.title("❤️ Heart Disease Risk Prediction System")
st.warning("⚠ Educational use only. Not a medical diagnosis.")

# =================================================
# INPUT SECTION
# =================================================
st.subheader("🧑 Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 1, 120, 50)
    sex = st.selectbox("Sex", ["M", "F"])
    chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
    resting_bp = st.number_input("Resting Blood Pressure", value=120)
    cholesterol = st.number_input("Cholesterol", value=200)

with col2:
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    max_hr = st.number_input("Max Heart Rate", value=150)
    exercise_angina = st.selectbox("Exercise Angina", ["N", "Y"])
    oldpeak = st.number_input("Oldpeak", step=0.1, value=0.0)
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# =================================================
# PREDICTION
# =================================================
if st.button("🔍 Predict Risk"):

    input_df = pd.DataFrame(0, index=[0], columns=model_columns)

    input_df["Age"] = age
    input_df["RestingBP"] = resting_bp
    input_df["Cholesterol"] = cholesterol
    input_df["FastingBS"] = fasting_bs
    input_df["MaxHR"] = max_hr
    input_df["Oldpeak"] = oldpeak

    if f"Sex_{sex}" in input_df.columns:
        input_df[f"Sex_{sex}"] = 1

    if f"ChestPainType_{chest_pain}" in input_df.columns:
        input_df[f"ChestPainType_{chest_pain}"] = 1

    if exercise_angina == "Y" and "ExerciseAngina_Y" in input_df.columns:
        input_df["ExerciseAngina_Y"] = 1

    if f"ST_Slope_{st_slope}" in input_df.columns:
        input_df[f"ST_Slope_{st_slope}"] = 1

    probability = model.predict_proba(input_df)[0][1]
    threshold = 0.35

    if probability >= threshold:
        st.error(f"⚠ HIGH RISK\nRisk Probability: {probability:.2%}")
    else:
        st.success(f"✅ LOW RISK\nRisk Probability: {probability:.2%}")

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

# =================================================
# CONSULTANT DASHBOARD
# =================================================
st.divider()
st.header("📊 Consultant’s View (Analytics Dashboard)")

conn = sqlite3.connect("heart_logs.db")
df = pd.read_sql("SELECT * FROM predictions", conn)
conn.close()

# ---------------- SQL ANALYSIS ----------------
st.subheader("📌 SQL Deep Analysis")

if not df.empty:
    avg_risk = df["risk_score"].mean()
    high_risk_count = (df["risk_score"] >= 0.35).sum()

    col1, col2 = st.columns(2)
    col1.metric("Average Risk Score", f"{avg_risk:.2%}")
    col2.metric("Total High Risk Cases", high_risk_count)

    st.subheader("🕒 5 Most Recent Predictions")
    st.dataframe(df.tail(5))

else:
    st.info("No records available.")

# ---------------- EDA ----------------
st.subheader("📈 Risk Distribution (EDA)")

if not df.empty:
    st.bar_chart(df["risk_score"])

# ---------------- FEATURE IMPORTANCE ----------------
st.subheader("🧬 Feature Importance (Why the Model Predicts This)")

if hasattr(model, "feature_importances_"):
    importance_df = pd.DataFrame({
        "Feature": model_columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=True)

    st.bar_chart(importance_df.set_index("Feature"))

# ---------------- CONFUSION MATRIX ----------------
st.subheader("🧪 Model Evaluation – Confusion Matrix")

if X_test is not None and y_test is not None:
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    cm_df = pd.DataFrame(
        cm,
        index=["Actual No Disease", "Actual Disease"],
        columns=["Predicted No Disease", "Predicted Disease"]
    )

    st.dataframe(cm_df)
else:
    st.warning("Confusion matrix files not found.")
