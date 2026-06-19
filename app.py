import streamlit as st
import pickle
import numpy as np
import pandas as pd

# -------------------
# PAGE
# -------------------

st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide"
)

# -------------------
# MODEL
# -------------------

with open("diabetes_model.pkl","rb") as f:
    model=pickle.load(f)

# -------------------
# STYLE
# -------------------

st.markdown("""
<style>

.stApp{
background:#F5FAFF;
}

h1,h2,h3{
color:#1E3A5F;
}

.stButton>button{
width:100%;
height:50px;
background:#4F8EF7;
color:white;
border:none;
border-radius:12px;
font-size:18px;
}

.stNumberInput input{
background:white;
color:black;
border-radius:10px;
}

</style>
""",unsafe_allow_html=True)

# -------------------

st.title("🩺 Diabetes Risk Predictor")

pages=st.tabs([
"🏠 Home",
"🩺 Predict",
"📊 Dataset"
])

# ===================
# HOME
# ===================

with pages[0]:

    st.subheader(
        "AI Based Diabetes Risk Prediction"
    )

    st.write("""
Predict diabetes risk using patient health inputs.

Features:
- Risk Prediction
- Probability Score
- Health Suggestions
- Dataset Information
""")

    st.info(
        "Educational Purpose Only"
    )

# ===================
# PREDICT
# ===================

with pages[1]:

    c1,c2=st.columns(2)

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
            "Diabetes Pedigree",
            0.0,
            3.0,
            0.5
        )

    if st.button(
        "🔍 Predict"
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

            pred=model.predict(
                data
            )[0]

            prob=model.predict_proba(
                data
            )[0][1]

            st.divider()

            if pred==1:

                st.error(
                    "🔴 High Risk"
                )

            else:

                st.success(
                    "🟢 Low Risk"
                )

            st.metric(
                "Risk %",
                f"{prob*100:.1f}%"
            )

            st.progress(
                float(prob)
            )

            st.subheader(
                "Suggestions"
            )

            if prob>0.7:

                st.warning("""
Exercise Daily

Reduce Sugar

Consult Doctor
""")

            else:

                st.success("""
Maintain Healthy Lifestyle
""")

        except:

            st.error(
                "Prediction Failed"
            )

# ===================
# DATASET
# ===================

with pages[2]:

    st.subheader(
        "Dataset Overview"
    )

    data={
        "Class":[
            "Non Diabetic",
            "Diabetic"
        ],
        "Value":[
            500,
            268
        ]
    }

    st.dataframe(
        pd.DataFrame(
            data
        )
    )

    st.info("""
Dataset:
Pima Indians Diabetes Dataset

768 Records
""")

# -------------------

st.markdown("---")
