import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('loan_model.pkl')

st.set_page_config(page_title="Loan Predictor", layout="centered")

st.title("🏦 Loan Approval Prediction App")
st.markdown("Enter applicant details to check eligibility.")

# Create the form for user input
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Married", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed", ["Yes", "No"])
        
    with col2:
        applicant_income = st.number_input("Applicant Income ($)", min_value=0)
        coapplicant_income = st.number_input("Co-applicant Income ($)", min_value=0)
        loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0)
        loan_term = st.number_input("Term (Days)", min_value=0, value=360)
        credit_history = st.selectbox("Credit History", ["Clear (1.0)", "Debts (0.0)"])
        property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

    submit = st.form_submit_button("Predict Status")

if submit:
    # --- Pre-processing the input to match the model's training format ---
    # (Mapping labels back to the numbers the LabelEncoder used)
    gen = 1 if gender == "Male" else 0
    mar = 1 if married == "Yes" else 0
    dep = 3 if dependents == "3+" else int(dependents)
    edu = 0 if education == "Graduate" else 1
    emp = 1 if self_employed == "Yes" else 0
    cred = 1.0 if "Clear" in credit_history else 0.0
    prop = {"Rural": 0, "Semiurban": 1, "Urban": 2}[property_area]
    
    # Arrange features in a 2D array
    features = np.array([[gen, mar, dep, edu, emp, applicant_income, 
                          coapplicant_income, loan_amount, loan_term, cred, prop]])
    
    # Make Prediction
    prediction = model.predict(features)
    
    if prediction[0] == 1:
        st.success("✅ Congratulations! Your Loan is likely to be **APPROVED**.")
    else:
        st.error("❌ Sorry, your Loan application is likely to be **REJECTED**.")