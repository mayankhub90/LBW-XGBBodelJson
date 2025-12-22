import streamlit as st
import pandas as pd
from datetime import date

st.set_page_config(page_title="LBW Risk – Data Entry", layout="wide")

st.title("📋 Beneficiary Data Entry Form (UI Only)")

# ---------------------------
# BASIC DETAILS
# ---------------------------
st.subheader("Basic Beneficiary Details")

col1, col2, col3 = st.columns(3)

with col1:
    beneficiary_age = st.number_input("Beneficiary age (years)", 10, 60)

with col2:
    height = st.number_input("Height (cm)", 120, 200)

with col3:
    month_conception = st.selectbox(
        "Month of Conception",
        list(range(1, 13))
    )

# ---------------------------
# PREGNANCY & REGISTRATION
# ---------------------------
st.subheader("Pregnancy & Registration Details")

col1, col2, col3 = st.columns(3)

with col1:
    lmp = st.date_input("LMP Date")

with col2:
    registration_date = st.date_input("Registration Date")

with col3:
    registration_bucket = st.selectbox(
        "Registration Bucket",
        ["Early", "On-time", "Late"]
    )

# ---------------------------
# PARITY & HOUSEHOLD
# ---------------------------
st.subheader("Parity & Household Information")

col1, col2, col3 = st.columns(3)

with col1:
    parity = st.number_input("Child order / parity", 0, 10)

with col2:
    living_children = st.number_input("Number of living children", 0, 10)

with col3:
    household_assets = st.number_input(
        "Household Assets Score (log1p)", 0.0
    )

# ---------------------------
# BMI PROGRESSION
# ---------------------------
st.subheader("BMI Progression")

col1, col2, col3, col4 = st.columns(4)

with col1:
    bmi_pw1 = st.number_input("BMI PW1")

with col2:
    bmi_pw2 = st.number_input("BMI PW2")

with col3:
    bmi_pw3 = st.number_input("BMI PW3")

with col4:
    bmi_pw4 = st.number_input("BMI PW4")

# ---------------------------
# TOBACCO & ALCOHOL
# ---------------------------
st.subheader("Substance Use")

col1, col2, col3 = st.columns(3)

with col1:
    consume_tobacco = st.selectbox("Consume tobacco", ["No", "Yes"])

with col2:
    chewing_status = st.selectbox(
        "Status of current chewing of tobacco",
        ["Never", "Stopped", "Currently chewing"]
    )

with col3:
    consume_alcohol = st.selectbox("Consume alcohol", ["No", "Yes"])

# ---------------------------
# ANC DETAILS
# ---------------------------
st.subheader("ANC Details")

anc_completed = st.selectbox("No. of ANCs completed", [0, 1, 2, 3, 4])

anc_data = {}

for i in range(1, anc_completed + 1):
    st.markdown(f"**ANC {i}**")
    col1, col2 = st.columns(2)

    with col1:
        anc_date = st.date_input(f"ANC {i} Date", key=f"anc_date_{i}")

    with col2:
        anc_weight = st.number_input(
            f"ANC {i} Weight (kg)", key=f"anc_weight_{i}"
        )

    anc_data[f"ANC{i}_Date"] = anc_date
    anc_data[f"ANC{i}_Weight"] = anc_weight

# ---------------------------
# SERVICES & SUPPLEMENTS
# ---------------------------
st.subheader("Services & Supplements")

col1, col2, col3 = st.columns(3)

with col1:
    tt_given = st.selectbox(
        "TT Injection given in last ANC", ["No", "Yes"]
    )

with col2:
    ifa_tabs = st.number_input(
        "IFA tablets received (last month, log1p)", 0.0
    )

with col3:
    calcium_tabs = st.number_input(
        "Calcium tablets consumed (last month, log1p)", 0.0
    )

# ---------------------------
# SOCIAL & ENVIRONMENT
# ---------------------------
st.subheader("Social & Environmental Factors")

col1, col2, col3 = st.columns(3)

with col1:
    food_group = st.selectbox(
        "Food Groups Category",
        ["Low", "Medium", "High"]
    )

with col2:
    toilet_type = st.selectbox(
        "Toilet Type", ["Improved", "Unimproved"]
    )

with col3:
    water_source = st.selectbox(
        "Water Source", ["Improved", "Unimproved"]
    )

col1, col2, col3 = st.columns(3)

with col1:
    education = st.selectbox(
        "Education Level",
        ["None", "Primary", "Secondary", "Higher"]
    )

with col2:
    social_media = st.selectbox(
        "Social Media Category",
        ["None", "WhatsApp", "Facebook", "YouTube", "Multiple"]
    )

with col3:
    social_media_type = st.multiselect(
        "Type of Social Media Enrolled In",
        ["WhatsApp", "Facebook", "YouTube", "Instagram"]
    )

# ---------------------------
# CASH TRANSFER SCHEMES
# ---------------------------
st.subheader("Cash Transfer Schemes")

col1, col2, col3 = st.columns(3)

with col1:
    jsy_registered = st.selectbox("Registered for JSY", ["No", "Yes"])
    jsy_inst = st.number_input("JSY installments received", 0, 5)

with col2:
    rajhsri_registered = st.selectbox("Registered for RAJHSRI", ["No", "Yes"])

with col3:
    pmmvy_inst = st.number_input(
        "PMMVY installments received", 0, 3
    )
    pmmvy_inst_date = st.date_input("PMMVY Installment Date")

# ---------------------------
# FINAL DATA TABLE
# ---------------------------
if st.button("➕ Add Beneficiary Record"):
    record = {
        "Beneficiary age": beneficiary_age,
        "height": height,
        "MonthConception": month_conception,
        "LMP": lmp,
        "Registration Date": registration_date,
        "RegistrationBucket": registration_bucket,
        "Child order/parity": parity,
        "Number of living child at now": living_children,
        "BMI_PW1_Prog": bmi_pw1,
        "BMI_PW2_Prog": bmi_pw2,
        "BMI_PW3_Prog": bmi_pw3,
        "BMI_PW4_Prog": bmi_pw4,
        "consume_tobacco": consume_tobacco,
        "Status of current chewing of tobacco": chewing_status,
        "consume_alcohol": consume_alcohol,
        "No of ANCs completed": anc_completed,
        "Service received during last ANC: TT Injection given": tt_given,
        "No. of IFA tablets received/procured in last one month_log1p": ifa_tabs,
        "No. of calcium tablets consumed in last one month_log1p": calcium_tabs,
        "Food_Groups_Category": food_group,
        "Household_Assets_Score_log1p": household_assets,
        "toilet_type_clean": toilet_type,
        "water_source_clean": water_source,
        "education_clean": education,
        "Social_Media_Category": social_media,
        "Type of Social Media Enrolled In": ",".join(social_media_type),
        "Registered for cash transfer scheme: JSY": jsy_registered,
        "JSY-Number of installment received": jsy_inst,
        "Registered for cash transfer scheme: RAJHSRI": rajhsri_registered,
        "PMMVY-Number of installment received": pmmvy_inst,
        "PMMVY Instalment Date": pmmvy_inst_date,
    }

    record.update(anc_data)

    df = pd.DataFrame([record])
    st.success("Record added successfully")
    st.dataframe(df)
