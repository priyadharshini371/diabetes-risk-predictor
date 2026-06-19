import streamlit as st
import pickle
import numpy as np

# -------------------
# PAGE CONFIG
# -------------------

st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide"
)

# -------------------
# LOAD MODEL
# -------------------

with open("diabetes_model.pkl", "rb") as f:
    model = pickle.load(f)

# -------------------
# TITLE
# -------------------

# -------------------------
# HERO SECTION
# -------------------------

st.markdown("""
<div style='
padding:40px;
background:linear-gradient(135deg,#E8F7FF,#FFFFFF);
border-radius:20px;
text-align:center;
'>

<h1 style='color:#1E3A5F;'>
🩺 Diabetes Risk Predictor
</h1>

<h3 style='color:#4B5563;'>
Predict Early. Stay Healthy.
</h3>

<p style='font-size:18px;color:#6B7280;'>

AI-powered system that analyzes patient health information
and estimates diabetes risk instantly.

</p>

</div>
""", unsafe_allow_html=True)

st.write("")

col1, col2 = st.columns(2)

with col1:
    st.info(
"""
🧠 AI Prediction

Get diabetes risk instantly
"""
)

with col2:
    st.info(
"""
📊 Health Insights

Understand patient health
"""
)

st.divider()

tab1, tab2, tab3 = st.tabs([
    "🏠 Home",
    "🩺 Predict",
    "📊 About"
])

# ===================
# HOME
# ===================

with tab1:

    st.header(
        "AI-Based Diabetes Risk Prediction"
    )

    st.write("""
This project predicts diabetes risk using Machine Learning.

Features:
- Diabetes Prediction
- Risk Probability
- Health Suggestions
- Interactive UI
""")

# ===================
# PREDICT
# ===================

with tab2:

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

    if st.button(
        "Predict"
    ):

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
• Exercise regularly

• Reduce sugar

• Stay hydrated

• Consult doctor
""")

            else:

                st.success("""
• Maintain healthy lifestyle
""")

        except Exception as e:

            st.error(
                f"Error: {e}"
            )

# ===================
# ABOUT
# ===================

with tab3:

    st.header(
        "Project Information"
    )

    st.write("""
Dataset:
Pima Indians Diabetes Dataset

Algorithm:
Logistic Regression

""")

# -------------------

st.markdown("---")


