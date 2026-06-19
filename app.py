import streamlit as st
import pickle
import numpy as np

# PAGE

st.set_page_config(
page_title="Diabetes Risk Predictor",
page_icon="🩺",
layout="wide"
)

# LOAD MODEL

with open(
"diabetes_model.pkl",
"rb"
) as f:

    model = pickle.load(f)

# UI

st.markdown("""
<style>

.stApp{
background:#EAF6FF;
}

h1{
color:#0B2447;
text-align:center;
}

label,p,h2,h3{
color:#1E293B !important;
}

.stNumberInput input{
background:white !important;
color:black !important;
}

.stButton>button{

width:100%;

height:55px;

background:#60A5FA;

color:white;

border:none;

border-radius:14px;

font-size:18px;

}

button[data-baseweb="tab"]{

background:white;

color:black;

}

</style>
""",
unsafe_allow_html=True)

# TITLE

st.title(
"🩺 Diabetes Risk Predictor"
)

tab1, tab2 = st.tabs([
"🔎 Predict",
"📊 Dataset"
])

# PREDICT

with tab1:

    c1,c2 = st.columns(2)

    with c1:

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

    with c2:

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
        "Diabetes Pedigree Function",
        0.0,
        3.0,
        0.5
        )

    if st.button(
    "🔍 Predict Risk"
    ):

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

            prob=model.predict_proba(
            data
            )[0][1]

            if prediction==1:

                st.error(
                "🔴 High Diabetes Risk"
                )

            else:

                st.success(
                "🟢 Low Diabetes Risk"
                )

            st.metric(
            "Risk %",
            f"{prob*100:.1f}%"
            )

            st.progress(
            float(prob)
            )

        except Exception as e:

            st.error(
            str(e)
            )

# DATASET

with tab2:

    st.info("""
Pima Indians Dataset

0 → Non Diabetic

1 → Diabetic
""")

st.caption(
"Developed by R.PRIYADHARSHINI"
)
