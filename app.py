import streamlit as st
import pandas as pd
import numpy as np
import pickle

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
model = pickle.load(open("diabetes_model.pkl", "rb"))

# -------------------
# CUSTOM CSS
# -------------------
st.markdown("""
<style>

.stButton>button{
width:100%;
background:linear-gradient(90deg,#4F46E5,#7C3AED);
color:white;
font-size:18px;
border-radius:10px;
height:55px;
border:none;
}

.result{
padding:20px;
border-radius:12px;
background:#f2f6ff;
}

</style>
""", unsafe_allow_html=True)

# -------------------
# TITLE
# -------------------
st.title("🩺 Diabetes Risk Predictor")

tab1, tab2 = st.tabs(["🔎 Predict","📊 Dataset Insights"])

# ==========================
# TAB 1
# ==========================
with tab1:

    st.write(
        "Enter patient's details below."
    )

    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input(
            "Pregnancies",
            0,
            20,
            0
        )

        glucose = st.number_input(
            "Glucose Level",
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
            "Diabetes Pedigree Function",
            0.0,
            3.0,
            0.50
        )

    if st.button("🔍 Predict"):

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

        prediction = model.predict(input_data)[0]

        probability = model.predict_proba(
            input_data
        )[0][1]

        st.divider()

        st.subheader("Prediction Result")

        if prediction == 1:

            st.error(
                f"🔴 High Diabetes Risk ({probability*100:.1f}%)"
            )

        else:

            st.success(
                f"🟢 Low Diabetes Risk ({probability*100:.1f}%)"
            )

        st.progress(
            float(probability)
        )

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

        st.subheader(
            "Health Suggestions"
        )

        if probability > 0.7:

            st.warning("""
• Exercise regularly

• Reduce sugar intake

• Drink more water

• Consult doctor
""")

        elif probability > 0.4:

            st.info("""
• Maintain healthy diet

• Monitor glucose
""")

        else:

            st.success("""
• Continue healthy habits
""")

# ==========================
# TAB 2
# ==========================

with tab2:

    st.subheader(
        "Dataset Insights"
    )

    st.write("""
Dataset:
Pima Indians Diabetes Dataset

Target:
0 → No Diabetes
1 → Diabetes
""")

# FOOTER

st.divider()

st.caption(
"Educational Purpose Only • Developed by R.PRIYADHARSHINI"
)
