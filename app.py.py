import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺")

# Load the trained model and the scaler (both saved with joblib in the notebook)
model = joblib.load("logistic_regression_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🩺 Diabetes Prediction")
st.write("Enter the patient's health details to estimate diabetes risk.")

# ---- Inputs ----
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male", "Other"])
    age = st.number_input("Age", min_value=1, max_value=100, value=35)
    bmi = st.number_input("BMI", min_value=10.0, max_value=70.0, value=30.0, step=0.1)
    hba1c = st.number_input("HbA1c Level", min_value=3.5, max_value=9.0, value=7.0, step=0.1)

with col2:
    smoking = st.selectbox(
        "Smoking History",
        ["never", "current", "former", "ever", "not current", "No Info"],
    )
    glucose = st.number_input("Blood Glucose Level", min_value=80, max_value=300, value=180)
    hypertension = st.selectbox("Hypertension", ["No", "Yes"])
    heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])

# ---- Encode inputs exactly like the LabelEncoder in the notebook ----
gender_map = {"Female": 0, "Male": 1, "Other": 2}
smoking_map = {"No Info": 0, "current": 1, "ever": 2, "former": 3, "never": 4, "not current": 5}

features = ["gender", "age", "hypertension", "heart_disease",
            "smoking_history", "bmi", "HbA1c_level", "blood_glucose_level"]

input_df = pd.DataFrame(
    [[gender_map[gender], age, int(hypertension == "Yes"), int(heart_disease == "Yes"),
      smoking_map[smoking], bmi, hba1c, glucose]],
    columns=features,
)

# ---- Predict ----
if st.button("Predict"):
    scaled = scaler.transform(input_df)
    prediction = model.predict(scaled)[0]
    probability = model.predict_proba(scaled)[0][1]

    if prediction == 1:
        st.error(f"🔴 Higher risk of diabetes (model probability: {probability:.1%})")
    else:
        st.success(f"🟢 Low risk of diabetes (model probability: {probability:.1%})")

    st.caption("This is a machine-learning prediction, not a medical diagnosis.")
