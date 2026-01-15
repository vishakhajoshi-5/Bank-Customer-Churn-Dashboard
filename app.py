import streamlit as st
import pandas as pd
import pickle
import os

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Bank Customer Churn Prediction",
    page_icon="🏦",
    layout="centered"
)

# ------------------ TITLE ------------------
st.title("🏦 Bank Customer Churn Prediction Dashboard")
st.caption("Predict customer churn risk using a trained ML model")

# ------------------ LOAD MODEL ------------------
MODEL_PATH = "models/churn_model.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("❌ Trained model not found. Please check model path.")
    st.stop()

model = pickle.load(open(MODEL_PATH, "rb"))

# ------------------ USER INPUTS ------------------
st.subheader("📋 Customer Details")

credit_score = st.slider("Credit Score", 350, 850, 650)
age = st.slider("Age", 18, 80, 35)
tenure = st.selectbox("Tenure (years)", list(range(0, 11)))
balance = st.number_input("Account Balance", min_value=0.0, value=50000.0)
salary = st.number_input("Estimated Salary", min_value=1000.0, value=60000.0)

num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
has_card = st.selectbox("Has Credit Card", [0, 1])
is_active = st.selectbox("Is Active Member", [0, 1])

gender = st.selectbox("Gender", ["Male", "Female"])
geo = st.selectbox("Geography", ["France", "Germany", "Spain"])

# ------------------ PREDICTION ------------------
if st.button("🔮 Predict Churn"):

    # -------- Base Numerical Features --------
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_products],
        'HasCrCard': [has_card],
        'IsActiveMember': [is_active],
        'EstimatedSalary': [salary]
    })

    # -------- Feature Engineering --------
    input_data['BalanceSalaryRatio'] = min(balance / salary, 5)
    input_data['IsHighValueCustomer'] = int(balance > 100000)

    # -------- Gender Encoding --------
    input_data['Gender_Male'] = 1 if gender == "Male" else 0

    # -------- Geography Encoding --------
    input_data['Geography_Germany'] = 1 if geo == "Germany" else 0
    input_data['Geography_Spain'] = 1 if geo == "Spain" else 0

    # -------- Add Missing Training Columns --------
    for col in model.feature_names_in_:
        if col not in input_data.columns:
            input_data[col] = 0

    # -------- Reorder Columns --------
    input_data = input_data[model.feature_names_in_]

    # -------- Predict Probability --------
    churn_prob = model.predict_proba(input_data)[0][1]

    # ------------------ OUTPUT ------------------
    st.subheader("📊 Prediction Result")
    st.metric("Churn Probability", f"{churn_prob:.1%}")

    if churn_prob >= 0.65:
        st.error("⚠ High Risk of Churn")
    elif churn_prob >= 0.45:
        st.warning("⚠ Medium Risk of Churn")
    else:
        st.success("✅ Low Risk of Churn")

    # ------------------ FIX 3: RISK EXPLANATION ------------------
    reasons = []

    if num_products <= 1:
        reasons.append("Low number of products used")

    if tenure <= 2:
        reasons.append("Short relationship with the bank")

    if is_active == 0:
        reasons.append("Customer is not an active member")

    if credit_score < 600:
        reasons.append("Low credit score")

    if balance < 10000:
        reasons.append("Low account balance")

    if churn_prob >= 0.45 and reasons:
        st.subheader("🔍 Why is the churn risk elevated?")
        for r in reasons:
            st.write(f"• {r}")

    # ------------------ BUSINESS NOTE ------------------
    st.info(
        "💡 Churn risk is influenced by customer engagement, tenure, "
        "product usage, financial stability, and activity level."
    )
