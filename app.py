import streamlit as st
import pickle
import numpy as np

# -----------------
# PAGE
# -----------------

st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="centered"
)

# -----------------
# LOAD MODEL
# -----------------

with open("diabetes_model.pkl","rb") as f:
    model = pickle.load(f)

# -----------------
# CLEAN UI
# -----------------

st.markdown("""
<style>

.stApp{
background-color:#F6FBFF;
}

h1{
color:#1E3A5F !important;
text-align:center;
font-size:48px !important;
}

p,label{
color:#374151 !important;
}

.stNumberInput input{
background:white !important;
color:black !important;
border:1px solid #D6EAF8 !important;
border-radius:10px;
}

.stButton>button{

width:100%;

background:#4A90E2;

color:white;

border:none;

height:50px;

font-size:18px;

border-radius:10px;

}

div[data-testid="stAlert"]{
border-radius:10px;
}

</style>
""", unsafe_allow_html=True)

# -----------------

st.title("🩺 Diabetes Risk Predictor")

st.markdown(
"Enter patient details to estimate diabetes risk."
)

# -----------------

col1,col2=st.columns(2)

with col1:

    pregnancies=st.number_input(
        "Pregnancies",
        0,
        20,
        0
    )

    glucose=st.number_input(
        "Glucose",
        0,
        250,
        100
    )

    bmi=st.number_input(
        "BMI",
        0.0,
        70.0,
        25.0
    )

    insulin=st.number_input(
        "Insulin",
        0,
        900,
        80
    )

with col2:

    bp=st.number_input(
        "Blood Pressure",
        0,
        200,
        70
    )

    age=st.number_input(
        "Age",
        1,
        100,
        30
    )

    skin=st.number_input(
        "Skin Thickness",
        0,
        100,
        20
    )

    dpf=st.number_input(
        "Diabetes Pedigree",
        0.0,
        3.0,
        0.5
    )

# -----------------

if st.button("Predict Risk"):

    try:

        data=np.array([[
            pregnancies,
            glucose,
            bp,
            skin,
            insulin,
            bmi,
            dpf,
            age
        ]])

        prediction=model.predict(
            data
        )[0]

        probability=model.predict_proba(
            data
        )[0][1]

        st.divider()

        st.subheader(
            "Result"
        )

        if prediction==1:

            st.error(
                f"🔴 High Risk ({probability*100:.1f}%)"
            )

        else:

            st.success(
                f"🟢 Low Risk ({probability*100:.1f}%)"
            )

        st.progress(
            float(probability)
        )

    except Exception:

        st.error(
            "Prediction failed"
        )

# -----------------

st.markdown("---")

st.caption(
"Developed by R.PRIYADHARSHINI"
)
