# Predicting Prolonged ICU Length of Stay Using First-24-Hour Clinical Data

This project develops a machine-learning model to predict whether an ICU stay will exceed 5 days using clinical information available during the first 24 hours of ICU admission.

The project was developed as a capstone for the 4Geeks Academy Data Science and Machine Learning Bootcamp.

## Project Objective

The goal is to estimate the probability of prolonged ICU length of stay among patients who remain in the ICU at the 24-hour prediction landmark.

The binary outcome is defined as:

- **0:** ICU length of stay of 1–5 days
- **1:** ICU length of stay greater than 5 days

This is an educational machine-learning project and is not intended for clinical decision-making.

---

## Data Source

Data were obtained from **MIMIC-IV v3.1**.

MIMIC-IV is a large, de-identified critical care database developed by the MIT Laboratory for Computational Physiology and distributed through PhysioNet.

Access to MIMIC-IV requires completion of the appropriate credentialing, training, and data-use requirements.

Because the underlying dataset is credentialed, raw MIMIC-derived CSV files are not included in this repository.

---

## Cohort

The final analytic cohort contains:

- **59,809 ICU stays**
- **47,974 unique patients**
- **11,687 prolonged ICU stays**
- **19.5% positive outcome prevalence**

Patients with ICU stays shorter than 24 hours were excluded because the prediction is made at the 24-hour landmark.

Each row represents one ICU stay.

---

## Predictors

The final model uses 28 predictors derived from the first 24 hours of ICU admission.

### Demographic and Admission Variables

- Age
- Gender
- Admission type
- Admission location
- First ICU care unit

### Vital Signs

- Mean heart rate
- Mean arterial pressure
- Mean respiratory rate
- Mean oxygen saturation

### Laboratory Variables

- Minimum hemoglobin
- Minimum platelet count
- Maximum white blood cell count
- Minimum bicarbonate
- Maximum BUN
- Maximum creatinine
- Maximum glucose
- Minimum sodium
- Maximum potassium
- Maximum INR

### Other Clinical Variables

- Minimum Glasgow Coma Scale
- Urine output during the first 24 hours
- Invasive ventilation during the first 24 hours
- Invasive ventilation at the 24-hour landmark
- Non-invasive ventilation
- High-flow nasal cannula
- Tracheostomy
- Any vasoactive medication
- Number of different vasoactive agents

---

## Data Preparation

Several preprocessing steps were performed before model development.

### Missing Data

Numerical variables were imputed using median values.

Categorical variables were imputed using the most frequent category.

All imputation was performed inside the machine-learning pipeline to reduce the risk of information leakage.

### Numerical Variables

Continuous variables were standardized using `StandardScaler`.

### Categorical Variables

Categorical variables were encoded using one-hot encoding with:

```python
OneHotEncoder(handle_unknown="ignore")
```

### Patient-Level Data Splitting

The dataset was split using patient identifiers rather than individual ICU stays.

This prevents different ICU stays from the same patient from appearing in both the training and test sets.

A `GroupShuffleSplit` strategy was used for the final train/test split.

Model development and hyperparameter optimization used `StratifiedGroupKFold`.

This preserved patient-level grouping while approximately maintaining class balance across folds.

---

## Machine-Learning Models

Three classification models were evaluated:

1. Logistic Regression
2. Random Forest
3. XGBoost

Cross-validation results were approximately:

| Model | ROC-AUC | PR-AUC | Recall | Precision | F1 |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.802 | 0.557 | 0.341 | 0.652 | 0.448 |
| Random Forest | 0.818 | 0.577 | 0.334 | 0.688 | 0.449 |
| XGBoost | 0.820 | 0.583 | 0.370 | 0.669 | 0.477 |

XGBoost showed the strongest overall cross-validated performance and was selected for hyperparameter optimization.

---

## Final Model

The final model is a tuned XGBoost classifier.

Performance on the held-out test cohort was approximately:

- **ROC-AUC:** 0.824
- **PR-AUC:** 0.586
- **Accuracy:** 0.837
- **Recall:** 0.351
- **Precision:** 0.666
- **F1 score:** 0.459
- **Brier score:** 0.117

The Brier score was lower than the baseline prevalence-based Brier score, indicating that the model provides more informative probability estimates than assigning the same baseline probability to every patient.

The default classification threshold of 0.50 is used for classification metrics.

No classification threshold was optimized using the held-out test set.

---

## Model Interpretation

SHAP was used to evaluate how predictors contributed to model predictions.

Important predictors included:

- Invasive ventilation at the 24-hour landmark
- Initial ICU care unit
- Invasive ventilation during the first 24 hours
- Respiratory rate
- Glucose
- Vasoactive-agent burden
- Heart rate
- Oxygen saturation
- Glasgow Coma Scale
- Renal-function markers

SHAP values describe how the fitted model uses each variable and should not be interpreted as causal effects.

---

## Streamlit Application

A Streamlit application was developed as part of this capstone project to demonstrate how the trained XGBoost model could be incorporated into an interactive prediction interface.

The application accepts clinical information from the first 24 hours of an ICU admission and generates an estimated probability of ICU length of stay exceeding 5 days.

The prediction framework applies to patients who remain in the ICU at the 24-hour landmark.

**Model availability:** The trained model was developed using credentialed-access MIMIC-IV data and is not distributed in this public repository. The Streamlit application source code is retained to demonstrate the implementation, but running the application requires a compatible trained model file.

This application is an educational proof of concept. It has not undergone external or prospective clinical validation and must not be used for patient care.

---
---

## Project Structure

```text
Capstone-project/
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── models/
│   └── prolonged_icu_xgb_pipeline.pkl
│
├── src/
│   ├── app.py
│   ├── explore.ipynb
│   └── utils.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

Raw MIMIC-derived datasets are excluded from version control because of data-access restrictions.

Trained model files are also excluded from version control in the current repository configuration.

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
```

Navigate to the project directory:

```bash
cd Capstone-project
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run src/app.py
```

The trained model file must be available locally at:

```text
models/prolonged_icu_xgb_pipeline.pkl
```

for the application to run.

---

## Technologies Used

- Python
- pandas
- NumPy
- scikit-learn
- XGBoost
- SHAP
- Streamlit
- Matplotlib
- Google BigQuery
- Google Cloud Platform
- MIMIC-IV
- Git
- GitHub

---

## Limitations

This project has several important limitations:

- The model was developed using retrospective MIMIC-IV data.
- The model has not been externally validated.
- ICU practice patterns and patient populations may differ across institutions.
- Some predictors, particularly ICU care-unit categories, may reflect institution-specific workflows.
- Predictions represent statistical associations and should not be interpreted as causal effects.
- Model performance may differ in other patient populations or healthcare systems.
- The application is not intended for clinical decision-making.

---

## Disclaimer

This project is for educational and research demonstration purposes only.

The model and application are not validated medical devices and must not be used to guide patient care.

---

## Author

Developed as a capstone project for the **4Geeks Academy Data Science and Machine Learning Bootcamp**.

##