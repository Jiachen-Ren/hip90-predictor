# -*- coding: utf-8 -*-
"""HIP90 research-only calculator. Run with: streamlit run HIP90_streamlit_calculator.py"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
import streamlit as st

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "hip90_M2_pipeline.joblib"
MANIFEST_PATH = HERE / "model_manifest.json"

st.set_page_config(page_title="HIP90 | 90-Day Ambulation", page_icon="🩺",
                   layout="centered")
st.markdown("""
<style>
.block-container {max-width: 860px; padding-top: 2.4rem}
h1 {color:#183D4B; letter-spacing:-.03em}
[data-testid="stMetricValue"] {color:#196A83}
</style>
""", unsafe_allow_html=True)

st.caption("HIP90  /  POSTOPERATIVE FUNCTIONAL PROGNOSIS  /  RESEARCH PROTOTYPE")
st.title("90-Day Independent Ambulation Predictor")
st.write("Estimate the probability of independent walking 90 days after hip fracture surgery "
         "using the final wearable-augmented model (M2).")
st.warning("Research demonstration only. Not validated for clinical decisions. "
           "Do not enter identifiable patient information.")

if not MODEL_PATH.exists() or not MANIFEST_PATH.exists():
    st.error("The fitted model is not available. Run HIP90_with_web_calculator.py "
             "first, then launch the copied app inside the web_calculator folder.")
    st.stop()

# Only load model files generated locally by the trusted training script.
@st.cache_resource
def load_artifacts():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    model = joblib.load(MODEL_PATH)
    return model, manifest

model, manifest = load_artifacts()
features = manifest["feature_order"]
codebook = manifest["codebook"]
if len(features) != 13:
    st.error("Unexpected model schema; expected exactly 13 predictors.")
    st.stop()

st.subheader("Clinical characteristics")
with st.form("hip90"):
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age (years)", min_value=65, max_value=110, value=79, step=1)
        bmi = st.number_input("BMI (kg/m²)", min_value=10.0, max_value=60.0, value=23.0, step=.1)
        cci = st.number_input("Charlson Comorbidity Index", min_value=0, max_value=30, value=2)
        gender = st.selectbox("Sex", list(codebook["gender"].values()), index=1)
    with c2:
        fracture = st.selectbox("Fracture type", list(codebook["type"].values()))
        asa = st.selectbox("ASA grade", list(codebook["ASA"].values()), index=2)
        prewalk = st.selectbox("Prefracture ambulation",
                               list(codebook["Preoperative_ambulation_ability"].values()))
        surgery = st.selectbox("Surgical procedure",
                               list(codebook["Surgical_method"].values()))
    st.subheader("Wearable-derived measures")
    st.caption("Seven consecutive days beginning with the first postoperative out-of-bed mobilization.")
    c3, c4 = st.columns(2)
    with c3:
        steps = st.number_input("Mean daily steps (steps/day)", min_value=0.0,
                                max_value=50000.0, value=800.0, step=50.0)
        active = st.number_input("Mean daily activity time (min/day)", min_value=0.0,
                                 max_value=1440.0, value=45.0, step=5.0)
        speed = st.number_input("Mean walking speed (m/s)", min_value=0.0,
                                max_value=3.0, value=.35, step=.01)
    with c4:
        sleep = st.number_input("Mean sleep duration (h/night)", min_value=0.0,
                                max_value=24.0, value=6.5, step=.1)
        continuity = st.number_input("Sleep continuity (0–1)", min_value=0.0,
                                     max_value=1.0, value=.75, step=.01)
    submitted = st.form_submit_button("Calculate predicted probability", type="primary",
                                       use_container_width=True)

if submitted:
    values = {
        "age":float(age), "gender":gender, "BMI（kg/m²）":float(bmi),
        "type":fracture, "ASA":asa,
        "Preoperative_ambulation_ability":prewalk,
        "CCI":float(cci), "Surgical_method":surgery,
        "7-day_average_daily_steps(steps/day)":float(steps),
        "7-day_average_daily_activity_time(minutes/day)":float(active),
        "7-day_average_walking_speed(meters/second)":float(speed),
        "7-day_average_sleep_duration(hours/night)":float(sleep),
        "Sleep_continuity_from_day_1_to_day_7":float(continuity),
    }
    frame = pd.DataFrame([{key:values[key] for key in features}], columns=features)
    try:
        p = float(model.predict_proba(frame)[0, 1])
        if not np.isfinite(p) or not (0 <= p <= 1):
            raise ValueError("Model returned an invalid probability")
        st.divider()
        st.metric("Predicted probability of independent ambulation at 90 days",
                  f"{p:.1%}")
        st.progress(p)
        st.caption("The complementary probability of non-independent ambulation is "
                   f"{1-p:.1%}. Neither value represents a causal effect or an "
                   "individually validated prognosis.")
        st.info("The calculator applies the saved original preprocessing pipeline and "
                "the fitted M2 estimator. It does not retrain or recalibrate the model.")
    except Exception as exc:
        st.error(f"Prediction failed: {type(exc).__name__}: {exc}")

st.divider()
st.caption("Model: " + manifest["algorithm"] + " | Outcome: FIM-L ≥ 6 at day 90. "
           "The model was fitted using development-cohort data. "
           "Input ranges are interface guardrails, not validated clinical eligibility criteria.")
