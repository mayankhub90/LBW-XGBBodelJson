import streamlit as st
import json
import pandas as pd
import xgboost as xgb

from preprocessing import preprocess_payload

st.set_page_config(page_title="LBW Risk Assessment", layout="wide")

# -------------------------------
# Load artifacts
# -------------------------------
@st.cache_resource
def load_artifacts():
    with open("artifacts/features.json") as f:
        FEATURES = json.load(f)

    booster = xgb.Booster()
    booster.load_model("artifacts/xgb_model.json")

    return booster, FEATURES

model, FEATURES = load_artifacts()

# -------------------------------
# UI
# -------------------------------
st.title("🤰 Low Birth Weight (LBW) Risk Assessment")

beneficiary_name = st.text_input("Beneficiary Name (for record only)")

with st.form("lbw_form"):
    st.subheader("👩 Background")

    c1, c2, c3 = st.columns(3)
    age = c1.number_input("Beneficiary age", 15, 45, 25)
    living_children = c2.number_input("Number of living child at now", 0, 6, 0)
    parity = c3.number_input("Child order / parity", living_children + 1, 10, living_children + 1)

    month_conception = st.selectbox(
        "Month of Conception",
        ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    )
    month_map = {m: i+1 for i,m in enumerate(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])}

    st.subheader("🩺 Clinical & Anthropometry")
    height = st.number_input("Height (cm)", 130, 200, 155)
    weight_pw2 = st.number_input("Weight PW2 (kg)", 30.0, 90.0, 50.0)
    hb = st.number_input("Hemoglobin (g/dL)", 4.0, 16.0, 11.0)
    anc = st.number_input("No of ANCs completed", 0, 4, 2)

    st.subheader("🥗 Nutrition")
    ifa = st.number_input("IFA tablets (last month)", 0, 120, 30)
    calcium = st.number_input("Calcium tablets (last month)", 0, 120, 30)
    food_group = st.selectbox("Food Groups Category", [1,2,3,4,5])

    st.subheader("🏠 Household")
    toilet = st.selectbox("Toilet type", [
        "Improved toilet",
        "Pit latrine (basic)",
        "Unimproved / unknown",
        "No facility / open defecation"
    ])

    water = st.selectbox("Water source", [
        "Piped supply (home/yard/stand)",
        "Groundwater – handpump/borewell",
        "Protected well",
        "Surface/Unprotected source",
        "Delivered / other"
    ])

    education = st.selectbox("Education level", [
        "No schooling",
        "Primary (1–5)",
        "Middle (6–8)",
        "Secondary (9–12)",
        "Graduate & above"
    ])

    social_media = st.selectbox("Social media exposure", [0,1])

    wm = st.checkbox("Has Washing Machine")
    ac = st.checkbox("Has AC / Cooler")

    submitted = st.form_submit_button("🔍 Predict Risk")

# -------------------------------
# Prediction
# -------------------------------
if submitted:
    payload = {
        "Beneficiary age": age,
        "Hemoglobin": hb,
        "Child order/parity": parity,
        "Number of living child at now": living_children,
        "MonthConception": month_map[month_conception],
        "Height_cm": height,
        "Weight_PW2": weight_pw2,
        "No of ANCs completed": anc,
        "IFA_tablets": ifa,
        "Calcium_tablets": calcium,
        "Food_Groups_Category": food_group,
        "toilet_type_clean": toilet,
        "water_source_clean": water,
        "education_clean": education,
        "Social_Media_Category": social_media,
        "Has_Washing_Machine": int(wm),
        "Has_AC_or_Cooler": int(ac),
        "consume_tobacco": 0,
        "Status of current chewing of tobacco": 0,
        "consume_alcohol": 0,
        "Registered for cash transfer scheme: JSY": 0,
        "Registered for cash transfer scheme: RAJHSRI": 0,
        "PMMVY-Number of installment received": 0,
        "JSY-Number of installment received": 0,
        "RegistrationBucket": 0,
        "counselling_gap_days": 0,
        "ANCBucket": 0,
        "LMPtoINST1": 0,
        "LMPtoINST2": 0,
        "LMPtoINST3": 0,
        "Service received during last ANC: TT Injection given": 0
    }

    X = preprocess_payload(payload, FEATURES)
    dmat = xgb.DMatrix(X)

    prob = float(model.predict(dmat)[0])

    st.success(f"LBW Risk Probability: **{prob:.2%}**")
