# app.py
import streamlit as st
import json
import pandas as pd
import numpy as np
import xgboost as xgb
import shap

from preprocessing import preprocess_payload

st.set_page_config(
    page_title="LBW Risk Predictor",
    layout="wide"
)

# -------------------------------
# Load artifacts
# -------------------------------
@st.cache_resource
def load_artifacts():
    with open("artifacts/features.json", "r") as f:
        FEATURES = json.load(f)

    booster = xgb.Booster()
    booster.load_model("artifacts/xgb_model.json")

    background = pd.read_csv("artifacts/background.csv")

    return booster, FEATURES, background

booster, FEATURES, BACKGROUND = load_artifacts()

st.title("🤰 Low Birth Weight (LBW) Risk Predictor")

# -------------------------------
# INPUT FORM (baseline subset)
# -------------------------------
with st.form("lbw_form"):

    st.subheader("Beneficiary Background")

    age = st.number_input("Beneficiary age", 15, 45, 25)
    hb = st.number_input("Haemoglobin (g/dL)", 4.0, 16.0, 11.0)

    living = st.number_input("Number of living children", 0, 6, 0)
    miscarriages = st.number_input("Previous miscarriages / abortions", 0, 6, 0)

    parity = living + miscarriages + 1
    st.info(f"Calculated parity: {parity}")

    tobacco = st.selectbox("Consumes tobacco", ["No", "Yes"])
    alcohol = st.selectbox("Consumes alcohol", ["No", "Yes"])

    submit = st.form_submit_button("🔍 Predict LBW Risk")

# -------------------------------
# PREDICTION
# -------------------------------
if submit:
    payload = {
        "Beneficiary age": age,
        "measured_HB": hb,
        "Number of living child at now": living,
        "Child order/parity": parity,
        "consume_tobacco": 1 if tobacco == "Yes" else 0,
        "consume_alcohol": 1 if alcohol == "Yes" else 0,
    }

    # Preprocess
    X = preprocess_payload(payload, FEATURES)

    # Safety check
    if list(X.columns) != FEATURES:
        st.error("Feature mismatch between UI and model")
        st.stop()

    # Predict
    dmat = xgb.DMatrix(X, feature_names=FEATURES)
    prob = float(booster.predict(dmat)[0])

    st.metric("LBW Risk Probability", f"{prob:.2%}")

    # ---------------------------
    # SHAP (Top drivers)
    # ---------------------------
    explainer = shap.TreeExplainer(booster)
    shap_vals = explainer.shap_values(X)

    shap_df = (
        pd.DataFrame({
            "Feature": FEATURES,
            "Impact": shap_vals[0]
        })
        .sort_values("Impact", key=abs, ascending=False)
        .head(10)
    )

    st.subheader("Top Risk Drivers")
    st.bar_chart(shap_df.set_index("Feature"))
