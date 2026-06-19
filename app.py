import streamlit as st
import pickle
import numpy as np

# ------------------------
# PAGE CONFIG
# ------------------------

st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide"
)

# ------------------------
# LOAD MODEL
# ------------------------

with open("diabetes_model.pkl", "rb") as f:
    model = pickle.load(f)

# ------------------------
# CUSTOM CSS
# ------------------------

st.markdown("""
<style>

/* Background */

.stApp{
background:#EAF6FF;
}

/* Title */

h1{
color:#0B2447 !important;
text-align:center;
font-size:58px;
font-weight:700;
}

/* Text */

p,label,h2,h3{
color:#0F172A !important;
}

/* Inputs */

.stNumberInput input{

background:white !important;

color:black !important;

border:1px solid #BFDFFF !important;

border-radius:12px;

}

/* Tabs */

button[data-baseweb="tab"]{

background:white !important;

color:#0F172A !important;

border-radius:10px;

padding:10px;

}

button[aria-selected="true"]{

background:#BFDBFE !important;

}

/* Button */

.stButton>button{

width:100%;

height:60px;

background:#60A5FA !important;

color:white !important;

font-size:18px;

font-weight:bold;

border:none;

border-radius:14px;

}

/* Alerts */

div[data-testid="stAlert"]{

background:white !important;

color:black !important;

border-radius:15px;

}

</style>
""", unsafe_allow_html=True)

# ------------------------

st.title("🩺 Diabetes Risk Predictor")

tab1, tab2 = st.tabs([
    "🔎 Predict",
    "📊 Dataset Insights"
])

# ==========================
# PREDICT
# ==========================

with tab1:

    st.subheader(
        "Enter Patient Details"
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
            "Diabetes Pedigree Function",
            0.0,
            3.0,
            0.50
        )

    if st.button(
        "🔍 Predict Risk"
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
                "Suggestions"
            )

            if probability > 0.7:

                st.warning("""
• Exercise Daily

• Reduce Sugar

• Stay Hydrated

• Consult Doctor
""")

            elif probability > 0.4:

                st.info("""
• Maintain Healthy Diet

• Monitor Health
""")

            else:

                st.success("""
• Continue Healthy Lifestyle
""")

        except Exception as e:

            st.error(
                f"Error: {e}"
            )

# ==========================
# DATASET
# ==========================

with tab2:

    st.subheader(
        "Dataset Information"
    )

    st.info("""
Dataset:
Pima Indians Diabetes Dataset

Target:

0 → Non Diabetic

1 → Diabetic
""")

# ------------------------

st.divider()

st.caption(
"Developed by R.PRIYADHARSHINI • Educational Purpose Only"
)
