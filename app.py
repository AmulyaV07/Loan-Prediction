
import streamlit as st
import pickle
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="centered"
)

# Load saved model and columns
with open("loan_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("loan_columns.pkl", "rb") as f:
    columns = pickle.load(f)

# Application title
st.title("🏦 Loan Approval Prediction")
st.write("Enter applicant details to predict loan approval status.")

st.divider()

# Personal Information
st.subheader("👤 Personal Information")

gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])

# Financial Information
st.subheader("💰 Financial Information")

applicant_income = st.number_input(
    "Applicant Income (Rs in thousands)", min_value=0, value=5000
)

coapplicant_income = st.number_input(
    "Coapplicant Income (Rs in thousands)", min_value=0.0, value=0.0
)

loan_amount = st.number_input(
    "Loan Amount (in thousands)", min_value=0.0, value=150.0
)

loan_term = st.number_input(
    "Loan Amount Term (in days)", min_value=0.0, value=360.0
)

credit_history = st.selectbox("Credit History", [1.0, 0.0])

property_area = st.selectbox(
    "Property Area", ["Urban", "Semiurban", "Rural"]
)

st.divider()

# Prediction
if st.button("🔍 Predict Loan Status", use_container_width=True):

    applicant = pd.DataFrame([{
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": credit_history,
        "Property_Area": property_area
    }])

    # One-hot encoding
    applicant = pd.get_dummies(
        applicant,
        drop_first=True,
        dtype=int
    )

    # Match training feature columns
    applicant = applicant.reindex(
        columns=columns,
        fill_value=0
    )

    # Prediction
    prediction = model.predict(applicant)[0]

    st.subheader("📊 Prediction Result")

    if prediction == 1:
        st.success("🎉 Congratulations! Loan Approved!")
    else:
        st.error("❌ Sorry! Loan Rejected!")

    st.caption(
        "Educational demonstration only. "
        "This is not an actual lending decision."
    )
