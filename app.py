import streamlit as st
import pickle
import numpy as np

# --------------------
# PAGE
# --------------------

st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide"
)

# --------------------
# LOAD MODEL + SCALER
# --------------------

with open("diabetes_model.pkl","rb") as f:
    model = pickle.load(f)

with open("scaler.pkl","rb") as f:
    scaler = pickle.load(f)

# --------------------
# UI
# --------------------

st.markdown("""
<style>

.stApp{
background:#EAF6FF;
}

h1{
text-align:center;
color:#0F172A;
}

h2,h3,label,p{
color:#1E293B !important;
}

.stNumberInput input{
background:white !important;
color:black !important;
border-radius:12px;
}

.stButton>button{

width:100%;

height:55px;

background:#60A5FA;

color:white;

font-size:18px;

font-weight:bold;

border:none;

border-radius:14px;

}

button[data-baseweb="tab"]{

background:white;

color:black;

border-radius:10px;

}

</style>
""", unsafe_allow_html=True)

# --------------------

st.title(
"🩺 Diabetes Risk Predictor"
)

tab1, tab2 = st.tabs([
"🔎 Predict",
"📊 Dataset Insights"
])

# ====================

with tab1:

    st.subheader(
        "Enter Patient Details"
    )

    c1, c2 = st.columns(2)

    with c1:

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

    with c2:

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

            scaled = scaler.transform(
                data
            )

            prediction = model.predict(
                scaled
            )[0]

            probability = model.predict_proba(
                scaled
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
Reduce sugar

Exercise

Drink water

Consult doctor
""")

            elif probability > 0.4:

                st.info("""
Maintain healthy diet
""")

            else:

                st.success("""
Healthy lifestyle maintained
""")

        except Exception as e:

            st.error(
                str(e)
            )

# ====================

with tab2:

    st.subheader(
        "Dataset Insights"
    )

    st.info("""
Dataset:
Pima Indians Diabetes Dataset

Outcome:

0 → Non Diabetic

1 → Diabetic
""")

# --------------------

st.divider()

st.caption(
"Developed by R.PRIYADHARSHINI"
)
