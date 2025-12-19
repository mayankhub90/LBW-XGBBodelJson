import streamlit as st
import joblib
from preprocessing import preprocess_input
from pathlib import Path

st.set_page_config(page_title="LBW Risk Prediction", layout="centered")

model = joblib.load(ARTIFACTS_DIR / "xgb_model.pkl")

st.title("Low Birth Weight Risk Prediction")

# Optional identifier
st.text_input("Beneficiary Name (optional)")

# Core inputs
age = st.number_input("Beneficiary age", 15, 50)
hb_risk = st.selectbox("Hb Risk", [0, 1])
parity = st.number_input("Child order / parity", 0, 6)
living_children = st.number_input("Number of living children", 0, 6)
month_conception = st.selectbox("Month of Conception", list(range(1, 13)))

anc = st.selectbox("No of ANCs completed", [0, 1, 2, 3, 4])

# Dynamic BMI inputs
bmi_pw1 = bmi_pw2 = bmi_pw3 = bmi_pw4 = 0.0
if anc >= 1:
    bmi_pw1 = st.number_input("BMI at PW1", 10.0, 40.0)
if anc >= 2:
    bmi_pw2 = st.number_input("BMI at PW2", 10.0, 40.0)
if anc >= 3:
    bmi_pw3 = st.number_input("BMI at PW3", 10.0, 40.0)
if anc == 4:
    bmi_pw4 = st.number_input("BMI at PW4", 10.0, 40.0)

if st.button("Predict Risk"):
    raw = {
        "age": age,
        "hb_risk": hb_risk,
        "parity": parity,
        "living_children": living_children,
        "month_conception": month_conception,
        "anc": anc,
        "bmi_pw1": bmi_pw1,
        "bmi_pw2": bmi_pw2,
        "bmi_pw3": bmi_pw3,
        "bmi_pw4": bmi_pw4,
        "reg_days": 120,
        "counselling_gap": 10,
        "lmp1": 30,
        "lmp2": 90,
        "lmp3": 150,
        "tobacco": 0,
        "chew": 0,
        "alcohol": 0,
        "tt": 1,
        "ifa": 3.0,
        "calcium": 3.0,
        "food": 1,
        "assets": 2.5,
        "toilet": 1,
        "water": 1,
        "education": 2,
        "social": 1,
        "jsy": 1,
        "rajhsri": 0,
        "pmmvy_inst": 1,
        "jsy_inst": 1
    }

    X = preprocess_input(raw)
    prob = model.predict_proba(X)[0][1]

    st.success(f"Predicted LBW Risk Probability: **{prob:.3%}**")
