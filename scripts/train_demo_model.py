from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# --------------------------------------------------
# Synthetic demonstration dataset
# --------------------------------------------------

RANDOM_STATE = 42
N_SAMPLES = 10000

rng = np.random.default_rng(RANDOM_STATE)

GENDERS = ["M", "F"]

ADMISSION_TYPES = [
    "EW EMER.",
    "URGENT",
    "OBSERVATION ADMIT",
    "SURGICAL SAME DAY ADMISSION",
    "ELECTIVE",
    "DIRECT EMER.",
    "EU OBSERVATION",
    "DIRECT OBSERVATION",
    "AMBULATORY OBSERVATION",
]

ADMISSION_LOCATIONS = [
    "AMBULATORY SURGERY TRANSFER",
    "CLINIC REFERRAL",
    "EMERGENCY ROOM",
    "INFORMATION NOT AVAILABLE",
    "INTERNAL TRANSFER TO OR FROM PSYCH",
    "PACU",
    "PHYSICIAN REFERRAL",
    "PROCEDURE SITE",
    "TRANSFER FROM HOSPITAL",
    "TRANSFER FROM SKILLED NURSING FACILITY",
    "WALK-IN/SELF REFERRAL",
]

FIRST_CAREUNITS = [
    "Cardiac Vascular Intensive Care Unit (CVICU)",
    "Coronary Care Unit (CCU)",
    "Intensive Care Unit (ICU)",
    "Med/Surg",
    "Medical Intensive Care Unit (MICU)",
    "Medical/Surgical Intensive Care Unit (MICU/SICU)",
    "Medicine",
    "Medicine/Cardiology Intermediate",
    "Neuro Intermediate",
    "Neuro Stepdown",
    "Neuro Surgical Intensive Care Unit (Neuro SICU)",
    "Neurology",
    "PACU",
    "Surgery/Trauma",
    "Surgery/Vascular/Intermediate",
    "Surgical Intensive Care Unit (SICU)",
    "Trauma SICU (TSICU)",
]

# --------------------------------------------------
# Generate synthetic numerical features
# --------------------------------------------------

synthetic_data = pd.DataFrame({
    "age": rng.integers(18, 101, N_SAMPLES),
    "heart_rate_mean": rng.uniform(55, 130, N_SAMPLES),
    "mbp_mean": rng.uniform(50, 110, N_SAMPLES),
    "resp_rate_mean": rng.uniform(10, 35, N_SAMPLES),
    "spo2_mean": rng.uniform(88, 100, N_SAMPLES),
    "hemoglobin_min": rng.uniform(7, 16, N_SAMPLES),
    "platelets_min": rng.uniform(50, 400, N_SAMPLES),
    "wbc_max": rng.uniform(3, 25, N_SAMPLES),
    "bicarbonate_min": rng.uniform(12, 32, N_SAMPLES),
    "bun_max": rng.uniform(5, 70, N_SAMPLES),
    "creatinine_max": rng.uniform(0.4, 4.0, N_SAMPLES),
    "glucose_max": rng.uniform(70, 300, N_SAMPLES),
    "sodium_min": rng.uniform(125, 150, N_SAMPLES),
    "potassium_max": rng.uniform(3, 6, N_SAMPLES),
    "inr_max": rng.uniform(0.9, 3.0, N_SAMPLES),
    "gcs_min": rng.integers(3, 16, N_SAMPLES),
    "urine_output_24h": rng.uniform(200, 4000, N_SAMPLES),

    "invasive_vent_24h": rng.integers(0, 2, N_SAMPLES),
    "invasive_vent_at_24h": rng.integers(0, 2, N_SAMPLES),
    "noninvasive_vent_24h": rng.integers(0, 2, N_SAMPLES),
    "hfnc_24h": rng.integers(0, 2, N_SAMPLES),
    "tracheostomy_24h": rng.integers(0, 2, N_SAMPLES),
    "any_vasoactive_24h": rng.integers(0, 2, N_SAMPLES),
    "vasoactive_agent_count_24h": rng.integers(0, 5, N_SAMPLES),

    "gender": rng.choice(GENDERS, N_SAMPLES),
    "admission_type": rng.choice(ADMISSION_TYPES, N_SAMPLES),
    "admission_location": rng.choice(ADMISSION_LOCATIONS, N_SAMPLES),
    "first_careunit": rng.choice(FIRST_CAREUNITS, N_SAMPLES),
})

print("Synthetic dataset shape:", synthetic_data.shape)

# --------------------------------------------------
# Make synthetic support variables internally consistent
# --------------------------------------------------

# A patient ventilated at hour 24 must have received
# invasive ventilation during the first 24 hours.
synthetic_data["invasive_vent_at_24h"] = (
    synthetic_data["invasive_vent_at_24h"]
    * synthetic_data["invasive_vent_24h"]
)

# Set vasoactive agent count to zero when no
# vasoactive medication was administered.
synthetic_data["vasoactive_agent_count_24h"] = np.where(
    synthetic_data["any_vasoactive_24h"] == 1,
    np.maximum(synthetic_data["vasoactive_agent_count_24h"], 1),
    0
)

# Verify the relationships.
assert (
    synthetic_data.loc[
        synthetic_data["invasive_vent_at_24h"] == 1,
        "invasive_vent_24h"
    ] == 1
).all()

assert (
    synthetic_data.loc[
        synthetic_data["any_vasoactive_24h"] == 0,
        "vasoactive_agent_count_24h"
    ] == 0
).all()

print("Synthetic support variables validated successfully.")

# --------------------------------------------------
# Generate an artificial demonstration outcome
# --------------------------------------------------

# These coefficients are arbitrary and are NOT
# clinical risk estimates.

demo_score = (
    -2.0
    + 0.015 * (synthetic_data["age"] - 60)
    + 0.7 * synthetic_data["invasive_vent_24h"]
    + 0.5 * synthetic_data["any_vasoactive_24h"]
    + 0.15 * synthetic_data["vasoactive_agent_count_24h"]
    + 0.04 * (15 - synthetic_data["gcs_min"])
    + 0.02 * (synthetic_data["bun_max"] - 20)
)

# Convert the artificial score into a probability
# used only to generate synthetic labels.
demo_probability = 1 / (1 + np.exp(-demo_score))

synthetic_data["demo_outcome"] = rng.binomial(
    1,
    demo_probability
)

print(
    "Artificial positive outcome rate:",
    f"{synthetic_data['demo_outcome'].mean():.1%}"
)

# --------------------------------------------------
# Prepare features and preprocessing
# --------------------------------------------------

X = synthetic_data.drop(columns=["demo_outcome"])
y = synthetic_data["demo_outcome"]

categorical_features = [
    "gender",
    "admission_type",
    "admission_location",
    "first_careunit",
]

numerical_features = [
    col for col in X.columns
    if col not in categorical_features
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numerical",
            "passthrough",
            numerical_features
        ),
    ]
)

demo_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                min_samples_leaf=5,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    ]
)

print("Demonstration pipeline created successfully.")

# --------------------------------------------------
# Train the demonstration model
# --------------------------------------------------

print("Training synthetic demonstration model...")

demo_pipeline.fit(X, y)

print("Training completed successfully.")


# --------------------------------------------------
# Save the trained pipeline
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "synthetic_demo_pipeline.pkl"

joblib.dump(demo_pipeline, MODEL_PATH)

print(f"Demonstration model saved to: {MODEL_PATH}")

# --------------------------------------------------
# Test the saved model
# --------------------------------------------------

loaded_model = joblib.load(MODEL_PATH)

# Use one synthetic observation as a test input.
test_patient = X.iloc[[0]]

test_probability = loaded_model.predict_proba(test_patient)[0, 1]

print("Model test successful.")
print(f"Illustrative demonstration score: {test_probability:.1%}")
print(f"Number of input features: {test_patient.shape[1]}")
