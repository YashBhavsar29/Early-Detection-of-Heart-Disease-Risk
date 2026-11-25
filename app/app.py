"""Educational Streamlit demonstration for the saved heart disease pipeline.

Run from the repository root:
    streamlit run app/app.py
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "heart_disease_logistic_pipeline.joblib"
FEATURE_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]


@st.cache_resource
def load_model():
    """Load the complete preprocessing and prediction pipeline once."""
    return joblib.load(MODEL_PATH)


def coded_selectbox(label: str, choices: dict[str, int]) -> int:
    """Show readable labels while returning the numeric code expected by the pipeline."""
    selected_label = st.selectbox(label, options=list(choices))
    return choices[selected_label]


def main() -> None:
    st.set_page_config(page_title="Heart Disease Risk Demo", page_icon="❤")
    st.title("Heart Disease Risk Prediction")
    st.caption("Educational decision-support demonstration using a Logistic Regression pipeline.")
    st.warning(
        "This application is for learning only. It is not clinically validated and must not be used "
        "for diagnosis, treatment, or medical decisions. Consult a qualified clinician for health advice."
    )

    if not MODEL_PATH.exists():
        st.error(
            "Saved model not found. Run `python src/models/train_final_model.py` from the project root first."
        )
        st.stop()

    model = load_model()

    with st.form("patient_inputs"):
        st.subheader("Clinical attributes")
        left, right = st.columns(2)

        with left:
            age = st.number_input("Age (years)", min_value=1, max_value=120, value=54, step=1)
            sex = coded_selectbox("Sex", {"Female": 0, "Male": 1})
            cp = coded_selectbox(
                "Chest pain type",
                {
                    "Typical angina": 1,
                    "Atypical angina": 2,
                    "Non-anginal pain": 3,
                    "Asymptomatic": 4,
                },
            )
            trestbps = st.number_input(
                "Resting blood pressure (mm Hg)", min_value=50, max_value=300, value=130, step=1
            )
            chol = st.number_input("Serum cholesterol (mg/dl)", min_value=50, max_value=700, value=246, step=1)
            fbs = coded_selectbox(
                "Fasting blood sugar greater than 120 mg/dl", {"No": 0, "Yes": 1}
            )
            restecg = coded_selectbox(
                "Resting ECG result",
                {
                    "Normal": 0,
                    "ST-T wave abnormality": 1,
                    "Probable or definite left ventricular hypertrophy": 2,
                },
            )

        with right:
            thalach = st.number_input("Maximum heart rate achieved", min_value=40, max_value=250, value=150, step=1)
            exang = coded_selectbox("Exercise-induced angina", {"No": 0, "Yes": 1})
            oldpeak = st.number_input(
                "ST depression induced by exercise relative to rest",
                min_value=0.0,
                max_value=10.0,
                value=1.0,
                step=0.1,
            )
            slope = coded_selectbox(
                "Peak exercise ST-segment slope",
                {"Upsloping": 1, "Flat": 2, "Downsloping": 3},
            )
            ca = st.selectbox("Number of major vessels coloured by fluoroscopy", options=[0, 1, 2, 3])
            thal = coded_selectbox(
                "Thallium test result",
                {"Normal": 3, "Fixed defect": 6, "Reversible defect": 7},
            )

        submitted = st.form_submit_button("Estimate model prediction")

    if submitted:
        patient = pd.DataFrame(
            [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]],
            columns=FEATURE_COLUMNS,
        )
        probability = float(model.predict_proba(patient)[0, 1])
        prediction = int(model.predict(patient)[0])

        st.subheader("Model output")
        st.metric("Estimated probability of the dataset's disease-present class", f"{probability:.1%}")
        if prediction == 1:
            st.error("Model classification: disease-present class")
        else:
            st.success("Model classification: no-disease class")

        with st.expander("View encoded values sent to the model"):
            st.dataframe(patient, hide_index=True, use_container_width=True)

    st.divider()
    st.caption(
        "Target definition: the model groups original UCI `num` values 1–4 as disease present and `num = 0` as no disease. "
        "Predictions reflect patterns in a small historical dataset, not an individual clinical assessment."
    )


if __name__ == "__main__":
    main()
