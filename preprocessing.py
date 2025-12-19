import pandas as pd
import numpy as np

def preprocess_payload(raw: dict, FEATURES: list) -> pd.DataFrame:
    df = pd.DataFrame([raw])

    # -------------------------------
    # Derived variables
    # -------------------------------

    # Hemoglobin → risk bin
    hb = df["Hemoglobin"].iloc[0]
    df["measured_HB_risk_bin"] = 0 if hb >= 11 else 1

    # BMI from height + weight PW2
    height_m = df["Height_cm"].iloc[0] / 100
    df["BMI_PW2_Prog"] = df["Weight_PW2"].iloc[0] / (height_m ** 2)

    # Parity validation
    if df["Child order/parity"].iloc[0] <= df["Number of living child at now"].iloc[0]:
        raise ValueError("Parity must be greater than living children")

    # Log transforms
    df["No. of IFA tablets received/procured in last one month_log1p"] = \
        np.log1p(df["IFA_tablets"].iloc[0])

    df["No. of calcium tablets consumed in last one month_log1p"] = \
        np.log1p(df["Calcium_tablets"].iloc[0])

    df["Household_Assets_Score_log1p"] = np.log1p(
        df["Has_Washing_Machine"].iloc[0] +
        df["Has_AC_or_Cooler"].iloc[0] +
        df["Social_Media_Category"].iloc[0]
    )

    # -------------------------------
    # Final alignment
    # -------------------------------
    df_final = df.reindex(columns=FEATURES, fill_value=0)

    return df_final
