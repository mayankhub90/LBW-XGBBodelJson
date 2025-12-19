# preprocessing.py
import pandas as pd
import numpy as np

def preprocess_payload(payload: dict, FEATURES: list) -> pd.DataFrame:
    """
    Converts raw UI payload into model-ready dataframe
    Ensures all FEATURES exist and are in correct order
    """

    # Create base DF
    df = pd.DataFrame([payload])

    # -----------------------------
    # Derived features
    # -----------------------------

    # Hemoglobin risk bin
    if "measured_HB" in df.columns:
        df["measured_HB_risk_bin"] = pd.cut(
            df["measured_HB"],
            bins=[-np.inf, 6, 8, 11, np.inf],
            labels=[0, 1, 2, 3]   # numeric bins as used in model
        ).astype(float)

    # Log1p transforms (if raw values present)
    log_features = {
        "No. of IFA tablets received/procured in last one month":
            "No. of IFA tablets received/procured in last one month_log1p",
        "No. of calcium tablets consumed in last one month":
            "No. of calcium tablets consumed in last one month_log1p",
        "Household_Assets_Score":
            "Household_Assets_Score_log1p"
    }

    for raw_col, log_col in log_features.items():
        if raw_col in df.columns:
            df[log_col] = np.log1p(df[raw_col])

    # -----------------------------
    # GUARANTEE feature completeness
    # -----------------------------
    for f in FEATURES:
        if f not in df.columns:
            df[f] = 0.0

    # -----------------------------
    # Final strict ordering
    # -----------------------------
    df_final = df[FEATURES].astype(float)

    return df_final
