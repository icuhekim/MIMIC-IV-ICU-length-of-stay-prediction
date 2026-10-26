import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="ICU Length of Stay | Synthetic ML Demo",
    page_icon="🏥",
    layout="wide"
)


# --------------------------------------------------
# Load model
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "synthetic_demo_pipeline.pkl"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("🏥 ICU Length of Stay — Interactive ML Demonstration")

st.write(
    """
    Explore how a machine learning application processes clinical inputs
    and generates an illustrative classification score.

    This application is inspired by a research project predicting
    prolonged ICU length of stay using MIMIC-IV data.
    The publicly deployed model is trained entirely on synthetic data.
    """
)

st.warning(
    """
    **SYNTHETIC DATA DEMONSTRATION — NOT FOR CLINICAL USE**

    **Active model:** Random Forest trained on 10,000 synthetic observations.

    **Outcome:** Artificial labels generated using an illustrative formula.

    Scores are not clinically validated probabilities of prolonged ICU stay.
    Do not enter real patient information or use this application
    for clinical decisions.
    """
)


# --------------------------------------------------
# Patient / admission information
# --------------------------------------------------

st.header("1. Patient and Admission Information")

st.caption(
    "Enter basic demographic and admission characteristics."
)

col1, col2, col3 = st.columns(3)

with col1:
    age = int(
        st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=65,
            step=1
        )
    )

    gender = st.selectbox(
        "Gender",
        ["M", "F"]
    )


with col2:
    admission_type = st.selectbox(
        "Admission Type",
        [
            "EW EMER.",
            "URGENT",
            "OBSERVATION ADMIT",
            "SURGICAL SAME DAY ADMISSION",
            "ELECTIVE",
            "DIRECT EMER.",
            "EU OBSERVATION",
            "DIRECT OBSERVATION",
            "AMBULATORY OBSERVATION"
        ]
    )

    admission_location = st.selectbox(
        "Admission Location",
        [
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
            "WALK-IN/SELF REFERRAL"
        ]
    )


with col3:
    first_careunit = st.selectbox(
        "First ICU Care Unit",
        [
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
            "Trauma SICU (TSICU)"
        ]
    )


# --------------------------------------------------
# Vital signs
# --------------------------------------------------

st.header("2. First-Day Vital Signs")

st.caption(
    "Enter the mean vital-sign values recorded during the first 24 hours."
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    heart_rate_mean = int(
        st.number_input(
            "Mean Heart Rate (bpm)",
            min_value=20,
            max_value=250,
            value=85,
            step=1
        )
    )

with col2:
    mbp_mean = int(
        st.number_input(
            "Mean Arterial Pressure (mmHg)",
            min_value=20,
            max_value=180,
            value=75,
            step=1
        )
    )

with col3:
    resp_rate_mean = int(
        st.number_input(
            "Mean Respiratory Rate (breaths/min)",
            min_value=5,
            max_value=60,
            value=18,
            step=1
        )
    )

with col4:
    spo2_mean = int(
        st.number_input(
            "Mean SpO₂ (%)",
            min_value=50,
            max_value=100,
            value=96,
            step=1
        )
    )


# --------------------------------------------------
# Laboratory values
# --------------------------------------------------

st.header("3. First-Day Laboratory Values")

st.caption(
    "Enter the minimum or maximum value observed during the first 24 hours, "
    "as specified for each variable."
)

col1, col2, col3 = st.columns(3)

with col1:
    hemoglobin_min = st.number_input(
        "Minimum Hemoglobin (g/dL)",
        min_value=2.0,
        max_value=25.0,
        value=10.0,
        step=0.1
    )

    bicarbonate_min = int(
        st.number_input(
            "Minimum Bicarbonate (mEq/L)",
            min_value=1,
            max_value=60,
            value=22,
            step=1
        )
    )

    creatinine_max = st.number_input(
        "Maximum Creatinine (mg/dL)",
        min_value=0.1,
        max_value=30.0,
        value=1.0,
        step=0.1
    )

    sodium_min = int(
        st.number_input(
            "Minimum Sodium (mEq/L)",
            min_value=90,
            max_value=180,
            value=138,
            step=1
        )
    )


with col2:
    platelets_min = int(
        st.number_input(
            "Minimum Platelet Count",
            min_value=1,
            max_value=2000,
            value=200,
            step=1
        )
    )

    bun_max = int(
        st.number_input(
            "Maximum BUN (mg/dL)",
            min_value=1,
            max_value=300,
            value=20,
            step=1
        )
    )

    glucose_max = int(
        st.number_input(
            "Maximum Glucose (mg/dL)",
            min_value=20,
            max_value=2000,
            value=150,
            step=1
        )
    )

    potassium_max = st.number_input(
        "Maximum Potassium (mEq/L)",
        min_value=1.0,
        max_value=12.0,
        value=4.5,
        step=0.1
    )


with col3:
    wbc_max = st.number_input(
        "Maximum WBC Count",
        min_value=0.1,
        max_value=500.0,
        value=10.0,
        step=0.1
    )

    inr_max = st.number_input(
        "Maximum INR",
        min_value=0.5,
        max_value=10.0,
        value=1.1,
        step=0.1
    )

    gcs_min = int(
        st.number_input(
            "Minimum GCS",
            min_value=3,
            max_value=15,
            value=15,
            step=1
        )
    )

    urine_output_24h = int(
        st.number_input(
            "Urine Output in First 24 Hours (mL)",
            min_value=0,
            max_value=20000,
            value=1500,
            step=50
        )
    )


# --------------------------------------------------
# ICU support
# --------------------------------------------------

st.header("4. ICU Support During First 24 Hours")

st.caption(
    "Enter respiratory and circulatory support information from the first "
    "24 hours, including ventilation status at the 24-hour prediction landmark."
)

col1, col2, col3 = st.columns(3)

with col1:
    invasive_vent_24h = st.checkbox(
        "Invasive Mechanical Ventilation During First 24 Hours"
    )

    invasive_vent_at_24h = st.checkbox(
        "Still Invasively Ventilated at 24 Hours",
        value=False,
        disabled=not invasive_vent_24h,
        help=(
            "Select this only if the patient remains invasively ventilated "
            "at the 24-hour prediction landmark."
        )
    )


with col2:
    noninvasive_vent_24h = st.checkbox(
        "Non-invasive Ventilation"
    )

    hfnc_24h = st.checkbox(
        "High-Flow Nasal Cannula"
    )


with col3:
    tracheostomy_24h = st.checkbox(
        "Tracheostomy"
    )

    any_vasoactive_24h = st.checkbox(
        "Any Vasoactive Medication"
    )


# --------------------------------------------------
# Vasoactive consistency
# --------------------------------------------------

if any_vasoactive_24h:
    vasoactive_agent_count_24h = int(
        st.number_input(
            "Number of Different Vasoactive Agents",
            min_value=1,
            max_value=7,
            value=1,
            step=1,
            help=(
                "Count the number of different vasoactive agents used "
                "during the first 24 hours."
            )
        )
    )
else:
    vasoactive_agent_count_24h = 0

    st.caption(
        "Number of vasoactive agents is automatically set to 0 when "
        "no vasoactive medication is selected."
    )


# --------------------------------------------------
# Create input dataframe
# --------------------------------------------------

input_data = pd.DataFrame(
    {
        "age": [age],
        "heart_rate_mean": [heart_rate_mean],
        "mbp_mean": [mbp_mean],
        "resp_rate_mean": [resp_rate_mean],
        "spo2_mean": [spo2_mean],
        "hemoglobin_min": [hemoglobin_min],
        "platelets_min": [platelets_min],
        "wbc_max": [wbc_max],
        "bicarbonate_min": [bicarbonate_min],
        "bun_max": [bun_max],
        "creatinine_max": [creatinine_max],
        "glucose_max": [glucose_max],
        "sodium_min": [sodium_min],
        "potassium_max": [potassium_max],
        "inr_max": [inr_max],
        "gcs_min": [gcs_min],
        "urine_output_24h": [urine_output_24h],
        "invasive_vent_24h": [int(invasive_vent_24h)],
        "invasive_vent_at_24h": [int(invasive_vent_at_24h)],
        "noninvasive_vent_24h": [int(noninvasive_vent_24h)],
        "hfnc_24h": [int(hfnc_24h)],
        "tracheostomy_24h": [int(tracheostomy_24h)],
        "any_vasoactive_24h": [int(any_vasoactive_24h)],
        "vasoactive_agent_count_24h": [vasoactive_agent_count_24h],
        "gender": [gender],
        "admission_type": [admission_type],
        "admission_location": [admission_location],
        "first_careunit": [first_careunit]
    }
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

st.divider()

st.header("5. Synthetic Model Demonstration")

st.caption(
    "Generate an illustrative classification score from the selected inputs. "
    "This score does not represent actual ICU length-of-stay risk."
)

if st.button(
    "Generate Demonstration Score",
    type="primary",
    use_container_width=True
):
    try:
        probability = model.predict_proba(input_data)[0, 1]

        st.metric(
            "Illustrative demonstration score",
            f"{probability:.1%}"
        )

        st.info(
            "This score represents the model's estimated probability "
            "of a fictional outcome generated for software testing. "
            "It is not a clinically validated prediction."
        )

        st.caption(
            "The demonstration uses a Random Forest trained on synthetic "
            "observations. The original MIMIC-IV-trained research model "
            "is not used or distributed in this application."
        )

    except Exception as e:
        st.error(
            "The demonstration score could not be generated. "
            "Please verify that the model and input features match."
        )
        st.exception(e)


# --------------------------------------------------
# About the model
# --------------------------------------------------

with st.expander("About this demonstration"):

    st.write(
        """
        This application demonstrates a machine-learning workflow
        inspired by a research project on prolonged ICU length of stay.

        **Public demonstration model**
        - Algorithm: Random Forest classifier
        - Training data: 10,000 artificially generated observations
        - Features: 28 synthetic clinical and admission variables
        - Target: Fictional binary outcome generated using an
          illustrative mathematical formula

        The model uses a preprocessing pipeline that one-hot encodes
        categorical features and processes numerical features.

        **Important distinction:** The original research project used
        MIMIC-IV data and a separately trained XGBoost model.
        That research model is not included in this application.

        Demonstration scores do not represent clinically meaningful
        probabilities and should not be used for patient care.
        """
    )
