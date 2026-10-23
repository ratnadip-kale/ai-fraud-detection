import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="AI Fraud Detection",
    page_icon="🔍",
    layout="centered"
)

st.title("🔍 AI Fraud Detection System")
st.write("Enter transaction details to check the model prediction.")

# Load the trained model
try:
    model = joblib.load("fraud_model.pkl")
except FileNotFoundError:
    st.error("Model file 'fraud_model.pkl' not found. Run train_model.py first.")
    st.stop()

# Inputs
amount = st.number_input(
    "Transaction Amount",
    min_value=0.0,
    value=1000.0
)

transaction_hour = st.slider(
    "Transaction Hour",
    min_value=0,
    max_value=23,
    value=12
)

distance_from_home = st.number_input(
    "Distance from Home (km)",
    min_value=0.0,
    value=5.0
)

transactions_last_24h = st.number_input(
    "Transactions in Last 24 Hours",
    min_value=0,
    value=2,
    step=1
)

# Prediction
if st.button("Check Transaction"):
    input_data = pd.DataFrame([{
        "amount": amount,
        "transaction_hour": transaction_hour,
        "distance_from_home": distance_from_home,
        "transactions_last_24h": transactions_last_24h
    }])

    # Same column order as training
    input_data = input_data[list(model.feature_names_in_)]

    prediction = model.predict(input_data)[0]
    fraud_probability = model.predict_proba(input_data)[0][1] * 100

    if prediction == 1:
        st.error("⚠️ This transaction looks FRAUDULENT.")
    else:
        st.success("✅ This transaction looks legitimate.")

    st.metric("Fraud Probability", f"{fraud_probability:.1f}%")
    st.progress(int(fraud_probability))