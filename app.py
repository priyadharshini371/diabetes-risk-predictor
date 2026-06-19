import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Page config
st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺", layout="centered")

# Custom styling - gradient background + readable text
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
    }
    h1, h2, h3, p, label, .stMarkdown {
        color: #1a1a2e !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255, 255, 255, 0.6);
        border-radius: 8px 8px 0 0;
        padding: 8px 16px;
    }
    </style>
""", unsafe_allow_html=True)

# Load model, scaler, and dataset
model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')
df = pd.read_csv('diabetes.csv')

# Default values for inputs removed from the UI (medians from training data)
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

tab1, tab2 = st.tabs(["🔍 Predict", "📊 Dataset Insights"])

# ---------------- TAB 1: PREDICTION ----------------
with tab1:
    st.write("Enter the patient's health details below to predict diabetes risk.")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        pregnancies = st.number_input("Pregnancies", 0, 20, 0)
        glucose = st.number_input("Glucose Level", 0, 300, 100)
    with col2:
        bp = st.number_input("Blood Pressure", 0, 200, 70)
        age = st.number_input("Age", 0, 120, 30)

    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
    st.markdown("---")

    if st.button("🔍 Predict", use_container_width=True):
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

# ---------------- TAB 2: DATASET INSIGHTS ----------------
with tab2:
    st.subheader("Dataset Overview")
    st.write(f"Total records: {df.shape[0]}")
    st.dataframe(df.head())

    st.subheader("Diabetic vs Non-Diabetic Count")
    fig1, ax1 = plt.subplots()
    sns.countplot(x='Outcome', data=df, ax=ax1)
    ax1.set_xticklabels(['Non-Diabetic', 'Diabetic'])
    st.pyplot(fig1)

    st.subheader("Glucose Levels by Outcome")
    fig2, ax2 = plt.subplots()
    sns.boxplot(x='Outcome', y='Glucose', data=df, ax=ax2)
    ax2.set_xticklabels(['Non-Diabetic', 'Diabetic'])
    st.pyplot(fig2)

    st.subheader("Feature Correlation Heatmap")
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax3)
    st.pyplot(fig3)
