import streamlit as st
import pickle
import numpy as np

# -------------------------
# PAGE
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
    st.error("Model file missing")
    st.stop()

# -------------------------
# HEADER
# -------------------------

st.title("🩺 Diabetes Risk Predictor")

st.subheader(
    "Predict Early • Stay Healthy"
)

st.write(
"""
This application predicts diabetes risk based on patient health details.
"""
)

st.divider()

# -------------------------
# INPUTS
# -------------------------

col1, col2 = st.columns(2)

with col1:

    pregnancies = st.number_input(
        "Pregnancies",
        0,
        20,
        0
    )

    glucose = st.number_input(
        "Glucose",
        0,
        250,
        100
    )

    bmi = st.number_input(
        "BMI",
        0.0,
        70.0,
        25.0
    )

    insulin = st.number_input(
        "Insulin",
        0,
        900,
        80
    )

with col2:

    bp = st.number_input(
        "Blood Pressure",
        0,
        200,
        70
    )

    age = st.number_input(
        "Age",
        1,
        100,
        30
    )

    skin = st.number_input(
        "Skin Thickness",
        0,
        100,
        20
    )

    dpf = st.number_input(
        "Diabetes Pedigree",
        0.0,
        3.0,
        0.5
    )

# -------------------------
# PREDICT
# -------------------------

if st.button("🔍 Predict Risk"):

    try:

        data = np.array([[
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
            data
        )[0]

        probability = model.predict_proba(
            data
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

        # ----------------

        st.subheader(
            "Patient Summary"
        )

        st.info(
f"""
Age: {age}

Glucose: {glucose}

BMI: {bmi}

Blood Pressure: {bp}
"""
)

        # ----------------

        st.subheader(
            "Health Status"
        )

        c1, c2, c3 = st.columns(3)

        with c1:

            if bmi < 18.5:
                st.metric(
                    "BMI",
                    "Low"
                )

            elif bmi < 25:
                st.metric(
                    "BMI",
                    "Normal"
                )

            else:
                st.metric(
                    "BMI",
                    "High"
                )

        with c2:

            if glucose < 100:

                st.metric(
                    "Glucose",
                    "Normal"
                )

            else:

                st.metric(
                    "Glucose",
                    "High"
                )

        with c3:

            if bp < 120:

                st.metric(
                    "BP",
                    "Healthy"
                )

            else:

                st.metric(
                    "BP",
                    "High"
                )

        # ----------------

        st.subheader(
            "Suggestions"
        )

        if probability > 0.7:

            st.warning("""
• Exercise Daily

• Reduce Sugar

• Drink Water
""")

        elif probability > 0.4:

            st.info("""
• Maintain Healthy Diet
""")

        else:

            st.success("""
• Continue Healthy Lifestyle
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
