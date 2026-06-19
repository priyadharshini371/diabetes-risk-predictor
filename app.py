import streamlit as st
import joblib
import numpy as np

# Page config
st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺", layout="centered")

# Custom styling — fixed text colors
st.markdown("""
    <style>
    .stApp {
        background-color: #f5f7fa;
    }
    h1, h2, h3, p, label, .stMarkdown {
        color: #1a1a2e !important;
    }
    </style>
""", unsafe_allow_html=True)

model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')

# Default values for removed inputs (medians from training data)
DEFAULT_SKIN_THICKNESS = 29
DEFAULT_BMI = 32.0
DEFAULT_INSULIN = 125

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.write("This app predicts diabetes risk based on patient health metrics using a Logistic Regression model.")
    st.write("**Dataset:** Pima Indians Diabetes Dataset")
    st.write("**Model:** Logistic Regression")
    st.markdown("---")
    st.caption("Built as a mini data science project")

# Main title
st.title("🩺 Diabetes Risk Predictor")
st.write("Enter the patient's health details below to predict diabetes risk.")
st.markdown("---")

# Inputs in columns
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", 0, 20, 0)
    glucose = st.number_input("Glucose Level", 0, 300, 100)

with col2:
    bp = st.number_input("Blood Pressure", 0, 200, 70)
    age = st.number_input("Age", 0, 120, 30)

dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)

st.markdown("---")

predict_clicked = st.button("🔍 Predict", use_container_width=True)

if predict_clicked:
    input_data = np.array([[pregnancies, glucose, bp, DEFAULT_SKIN_THICKNESS,
                             DEFAULT_INSULIN, DEFAULT_BMI, dpf, age]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    st.markdown("### Result")
    if prediction == 1:
        st.error("⚠️ **High Risk of Diabetes**")
        st.write("Please consult a healthcare professional for further evaluation.")
    else:
        st.success("✅ **Low Risk of Diabetes**")
        st.write("Keep maintaining a healthy lifestyle!")
