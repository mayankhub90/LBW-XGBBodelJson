import numpy as np
import json
from pathlib import Path

# -------------------------------------------------
# Resolve artifacts directory safely
# -------------------------------------------------
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"

FEATURES_PATH = ARTIFACTS_DIR / "features.json"

if not FEATURES_PATH.exists():
    raise FileNotFoundError("features.json not found inside artifacts/")

with open(FEATURES_PATH, "r") as f:
    FEATURE_ORDER = json.load(f)

# -------------------------------------------------
# Bucket logic (must match training notebook)
# -------------------------------------------------
def anc_bucket(anc: int) -> int:
    if anc <= 1:
        return 0
    elif anc <= 3:
        return 1
    else:
        return 2

def registration_bucket(days: int) -> int:
    if days <= 90:
        return 0
    elif days <= 180:
        return 1
    else:
        return 2

# -------------------------------------------------
# Main preprocessing function
# -------------------------------------------------
def preprocess_input(raw: dict) -> np.ndarray:
    """
    Converts raw UI inputs into a model-ready NumPy array
    in the exact feature order used during training.
    """

    f = {}

    # -----------------------------
    # Core demographics
    # -----------------------------
    f["Beneficiary age"] = raw["age"]
    f["measured_HB_risk_bin"] = raw["hb_risk"]
    f["Child order/parity"] = raw["parity"]
    f["Number of living child at now"] = raw["living_children"]
    f["MonthConception"] = raw["month_conception"]

    # -----------------------------
    # ANC & BMI progression
    # -----------------------------
    anc = raw["anc"]
    f["No of ANCs completed"] = anc

    f["BMI_PW1_Prog"] = raw["bmi_pw1"] if anc >= 1 else 0
    f["BMI_PW2_Prog"] = raw["bmi_pw2"] if anc >= 2 else 0
    f["BMI_PW3_Prog"] = raw["bmi_pw3"] if anc >= 3 else 0
    f["BMI_PW4_Prog"] = raw["bmi_pw4"] if anc >= 4 else 0

    f["ANCBucket"] = anc_bucket(anc)

    # -----------------------------
    # Registration & counselling
    # -----------------------------
    f["RegistrationBucket"] = registration_bucket(raw["reg_days"])
    f["counselling_gap_days"] = raw["counselling_gap"]

    # -----------------------------
    # LMP → installment gaps
    # -----------------------------
    f["LMPtoINST1"] = raw["lmp1"]
    f["LMPtoINST2"] = raw["lmp2"]
    f["LMPtoINST3"] = raw["lmp3"]

    # -----------------------------
    # Behavioural indicators
    # -----------------------------
    f["consume_tobacco"] = raw["tobacco"]
    f["Status of current chewing of tobacco"] = raw["chew"]
    f["consume_alcohol"] = raw["alcohol"]

    # -----------------------------
    # Health services & supplements
    # -----------------------------
    f["Service received during last ANC: TT Injection given"] = raw["tt"]
    f["No. of IFA tablets received/procured in last one month_log1p"] = raw["ifa"]
    f["No. of calcium tablets consumed in last one month_log1p"] = raw["calcium"]

    # -----------------------------
    # Household & social context
    # -----------------------------
    f["Food_Groups_Category"] = raw["food"]
    f["Household_Assets_Score_log1p"] = raw["assets"]
    f["toilet_type_clean"] = raw["toilet"]
    f["water_source_clean"] = raw["water"]
    f["education_clean"] = raw["education"]
    f["Social_Media_Category"] = raw["social"]

    # -----------------------------
    # Cash transfer schemes
    # -----------------------------
    f["Registered for cash transfer scheme: JSY"] = raw["jsy"]
    f["Registered for cash transfer scheme: RAJHSRI"] = raw["rajhsri"]
    f["PMMVY-Number of installment received"] = raw["pmmvy_inst"]
    f["JSY-Number of installment received"] = raw["jsy_inst"]

    # -----------------------------
    # Final ordered model input
    # -----------------------------
    try:
        X = np.array([f[col] for col in FEATURE_ORDER], dtype=float).reshape(1, -1)
    except KeyError as e:
        raise KeyError(f"Missing feature during preprocessing: {e}")

    return X
