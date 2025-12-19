import streamlit as st
from pathlib import Path
from preprocessing import preprocess_input

# -------------------------------------------------
# Streamlit config
# -------------------------------------------------
st.set_page_config(
    page_title="LBW Risk Prediction",
    layout="centered"
)

st.title("Low Birth Weight Risk Prediction")

# -------------------------------------------------
# Load model safely from artifacts/
# -------------------------------------------------
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "xgb_model.pkl"

if not MODEL_PATH.exists():
    st.error("xgb_model.pkl not found in artifacts/")
    st.stop()

model = joblib.load(MODEL_PATH)

# -------------------------------------------------
# Optional identifier (not used by model)
# -------------------------------------------------
st.text_input("Beneficiary Name (optional)")

# -------------------------------------------------
# Core inputs
# -------------------------------------------------
age = st.number_input("Beneficiary age", min_value=15, max_value=50)
hb_risk = st.selectbox("Hb Risk (0 = Normal, 1 = Risk)", [0, 1])
parity = st.number_input("Child order / parity", min_value=0, max_value=6)
living_children = st.number_input("Number of living children", min_value=0, max_value=6)
month_conception = st.selectbox("Month of Conception", list(range(1, 13)))

anc = st.selectbox("No of ANCs completed", [0, 1, 2, 3, 4])

# -------------------------------------------------
# BMI progression (ANC-dependent)
# -------------------------------------------------
bmi_pw1 = bmi_pw2 = bmi_pw3 = bmi_pw4 = 0.0

if anc >= 1:
    bmi_pw1 = st.number_input("BMI at PW1", min_value=10.0, max_value=40.0)
if anc >= 2:
    bmi_pw2 = st.number_input("BMI at PW2", min_value=10.0, max_value=40.0)
if anc >= 3:
    bmi_pw3 = st.number_input("BMI at PW3", min_value=10.0, max_value=40.0)
if anc == 4:
    bmi_pw4 = st.number_input("BMI at PW4", min_value=10.0, max_value=40.0)

# -------------------------------------------------
# Prediction
# -------------------------------------------------
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

    try:
        X = preprocess_input(raw)
        prob = model.predict_proba(X)[0][1]
        st.success(f"Predicted LBW Risk Probability: **{prob:.2%}**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
