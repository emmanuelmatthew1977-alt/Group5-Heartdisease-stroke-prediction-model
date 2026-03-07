import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="CVD Risk Predictor", layout="centered")
st.title("🫀 Heart Disease & Stroke Risk Predictor")
st.markdown("**Ridge Regression model** predicting if a patient has heart disease, stroke, or both.")

# Load model
@st.cache_resource
def load_model():
    return joblib.load('final_ridge_cvd_model.pkl')

model = load_model()

# Simple input form
st.sidebar.header("Enter Patient Details")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
age = st.sidebar.slider("Age", 10, 100, 50)
hypertension = st.sidebar.selectbox("Hypertension (1=Yes, 0=No)", [0, 1])
ever_married = st.sidebar.selectbox("Ever Married", ["Yes", "No"])
work_type = st.sidebar.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
residence_type = st.sidebar.selectbox("Residence Type", ["Urban", "Rural"])
avg_glucose_level = st.sidebar.slider("Avg Glucose Level", 50.0, 300.0, 100.0)
bmi = st.sidebar.slider("BMI", 10.0, 60.0, 25.0)
smoking_status = st.sidebar.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])

if st.sidebar.button("Predict Risk"):
    input_data = pd.DataFrame({
        'gender': [gender],
        'age': [age],
        'hypertension': [hypertension],
        'ever_married': [ever_married],
        'work_type': [work_type],
        'Residence_type': [residence_type],
        'avg_glucose_level': [avg_glucose_level],
        'bmi': [bmi],
        'smoking_status': [smoking_status],
        'age_group': [pd.cut([age], bins=[-np.inf, 18, 40, 60, np.inf], labels=['child', 'young_adult', 'middle_age', 'senior'])[0]],
        'glucose_group': [pd.cut([avg_glucose_level], bins=[-np.inf, 100, 126, 200, np.inf], labels=['normal', 'prediabetes', 'diabetes', 'high'])[0]],
        'bmi_group': [pd.cut([bmi], bins=[-np.inf, 18.5, 25, 30, np.inf], labels=['underweight', 'normal', 'overweight', 'obese'])[0]]
    })

    # Ensure string types
    for col in ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status', 'age_group', 'glucose_group', 'bmi_group']:
        input_data[col] = input_data[col].astype(str)

    prediction = model.predict(input_data)[0]
    prob = model.decision_function(input_data)[0]

    if prediction == 1:
        st.error(f"**HIGH RISK** (Decision score: {prob:.2f})")
        st.write("This patient may have heart disease, stroke, or both. Recommend immediate medical consultation.")
    else:
        st.success(f"**Low Risk** (Decision score: {prob:.2f})")
        st.write("Low likelihood of heart disease or stroke based on the model.")

    st.info("Note: Model trained on Ridge Regression with balanced class weights. Always consult a doctor for real diagnosis.")
