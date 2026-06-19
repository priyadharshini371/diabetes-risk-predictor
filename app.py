import streamlit as st
import pickle
import numpy as np

# -------------------------
# PAGE SETTINGS
# -------------------------

st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="centered"
)

# -------------------------
# LOAD MODEL
# -------------------------

try:
    with open("diabetes_model.pkl", "rb") as f:
        model = pickle.load(f)

except:
    st.error("Model file not found")
    st.stop()

# -------------------------
# HERO SECTION
# -------------------------

st.title("🩺 Diabetes Risk Predictor")

st.markdown("""
Predict Early • Stay Healthy

This application predicts diabetes risk based on health details.
""")

st.info(
    "Educational Purpose Only"
)

st.divider()

# -------------------------
# INPUTS
# -------------------------

col1, col2 = st.columns(2)

with col1:

    pregnancies = st.number_input(
        "Pregnancies",
        min_value=0,
        max_value=20,
        value=0
    )

    glucose = st.number_input(
        "Glucose",
        min_value=0,
        max_value=250,
        value=100
    )

    bmi = st.number_input(
        "BMI",
        min_value=0.0,
        max_value=70.0,
        value=25.0
    )

    insulin = st.number_input(
        "Insulin",
        min_value=0,
        max_value=900,
        value=80
    )

with col2:

    bp = st.number_input(
        "Blood Pressure",
        min_value=0,
        max_value=200,
        value=70
    )

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=100,
        value=30
    )

    skin = st.number_input(
        "Skin Thickness",
        min_value=0,
        max_value=100,
        value=20
    )

    dpf = st.number_input(
        "Diabetes Pedigree",
        min_value=0.0,
        max_value=3.0,
        value=0.5
    )

# -------------------------
# PREDICT
# -------------------------

if st.button("🔍 Predict Risk"):

    try:

        input_data = np.array([[
            pregnancies,
            glucose,
            bp,
            skin,
            insulin,
            bmi,
            dpf,
            age
        ]])

        prediction = model.predict(
            input_data
        )[0]

        probability = model.predict_proba(
            input_data
        )[0][1]

        st.divider()

        st.subheader(
            "Prediction Result"
        )

        if prediction == 1:

            st.error(
                "🔴 High Diabetes Risk"
            )

        else:

            st.success(
                "🟢 Low Diabetes Risk"
            )

        st.metric(
            "Risk Probability",
            f"{probability*100:.1f}%"
        )

        st.progress(
            float(probability)
        )

        st.subheader(
            "Suggestions"
        )

        if probability > 0.7:

            st.warning("""
• Exercise Daily

• Reduce Sugar

• Drink Water
""")

        else:

            st.success("""
• Maintain Healthy Lifestyle
""")

    except Exception as e:

        st.error(
            f"Prediction Error: {e}"
        )

# -------------------------

st.divider()

st.caption(
    "Developed by R.PRIYADHARSHINI"
)
