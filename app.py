import streamlit as st
import pickle
import numpy as np

# ------------------------
# PAGE
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
# LIGHT BLUE + WHITE THEME
# ------------------------

st.markdown("""
<style>

/* Background */

.stApp{
background:linear-gradient(
135deg,
#E6F7FF,
#FFFFFF
);
}

/* Title */

h1{
text-align:center;
color:#0F172A;
}

/* Inputs */

.stNumberInput input{

background:white;

color:black;

border-radius:12px;

}

/* Button */

.stButton>button{

width:100%;

height:55px;

background:linear-gradient(
90deg,
#60A5FA,
#BFDBFE
);

color:#0F172A;

font-size:18px;

font-weight:bold;

border:none;

border-radius:15px;

}

/* Result */

div[data-testid="stAlert"]{

border-radius:15px;

}

/* Tabs */

button[data-baseweb="tab"]{

background:white;

border-radius:10px;

}

</style>
""", unsafe_allow_html=True)

# ------------------------

st.title("🩺 Diabetes Risk Predictor")

tab1, tab2 = st.tabs([
"🔎 Predict",
"📊 Dataset Insights"
])

# ======================
# TAB 1
# ======================

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
                    f"🔴 High Diabetes Risk"
                )

            else:

                st.success(
                    f"🟢 Low Diabetes Risk"
                )

            st.write(
                f"Risk Probability: {probability*100:.2f}%"
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
Reduce sugar intake

Exercise daily

Drink water

Consult doctor
""")

            elif probability > 0.4:

                st.info("""
Maintain healthy diet

Monitor glucose
""")

            else:

                st.success("""
Healthy lifestyle maintained
""")

        except Exception as e:

            st.error(
                f"Error: {e}"
            )

# ======================
# TAB 2
# ======================

with tab2:

    st.subheader(
        "Dataset Insights"
    )

    st.write("""
Dataset:
Pima Indians Diabetes Dataset

Outcome:

0 → Non Diabetic

1 → Diabetic
""")

# ------------------------

st.divider()

st.caption(
"Educational Purpose Only • Developed by R.PRIYADHARSHINI"
)
