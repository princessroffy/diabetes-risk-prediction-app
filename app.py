import streamlit as st
import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Page settings
st.set_page_config(
    page_title="Diabetes Risk Prediction App",
    page_icon="🩺",
    layout="centered"
)

MODEL_PATH = Path("model.pkl")
DATA_PATH = Path("data.csv")


def load_or_train_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)

    data = pd.read_csv(DATA_PATH)
    X = data.drop("Outcome", axis=1)
    y = data["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_PATH)
    return model


model = load_or_train_model()

# Custom styling
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .title {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.3rem;
    }
    .subtitle {
        font-size: 1rem;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }
    .section-title {
        font-size: 1.2rem;
        font-weight: 600;
        margin-top: 1.2rem;
        margin-bottom: 0.8rem;
        color: #111827;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">Diabetes Risk Prediction App</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Fill in the patient health details below to estimate diabetes risk using a machine learning model.</div>',
    unsafe_allow_html=True
)

st.info("This app is for educational purposes only and is not a medical diagnostic tool.")

st.markdown('<div class="section-title">Patient Information</div>',
            unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input(
        "Pregnancies", min_value=0, max_value=20, value=1)
    glucose = st.number_input("Glucose", min_value=0, max_value=300, value=100)
    blood_pressure = st.number_input(
        "Blood Pressure", min_value=0, max_value=200, value=70)
    skin_thickness = st.number_input(
        "Skin Thickness", min_value=0, max_value=100, value=20)

with col2:
    insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
    dpf = st.number_input("Diabetes Pedigree Function",
                          min_value=0.0, max_value=3.0, value=0.5)
    age = st.number_input("Age", min_value=1, max_value=120, value=30)

if st.button("Predict Risk", use_container_width=True):
    input_data = pd.DataFrame([{
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.markdown('<div class="section-title">Prediction Result</div>',
                unsafe_allow_html=True)

    if prediction == 1:
        st.error("The model predicts a higher diabetes risk.")
    else:
        st.success("The model predicts a lower diabetes risk.")

    st.metric("Risk Probability", f"{probability:.2%}")

    with st.expander("View Entered Patient Data"):
        st.dataframe(input_data, use_container_width=True)

st.markdown("---")
st.caption("Built with Python, scikit-learn, and Streamlit.")
