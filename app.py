import streamlit as st
import pandas as pd
import joblib

from sklearn.metrics import mean_squared_error, r2_score

# 🔐 Login Page
USERNAME = "admin"
PASSWORD = "1234"

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("🔐 Login Required")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    login_btn = st.button("Login")

    if login_btn:
        if username == USERNAME and password == PASSWORD:
            st.session_state.logged_in = True
            st.experimental_rerun()
        else:
            st.error("Invalid credentials ❌")
    st.stop()

# ✅ Main App Starts After Login
st.title("🔍 Energy Consumption Prediction")

# Upload dataset
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Select columns used for prediction
    selected_cols = [
        'Access_to_electricity_of_population',
        'GDP per capita',
        'Financial_flows_to_developing_countries_US',
        'Renewable electricity Generating Capacity per capita',
        'Electricity_from_fossil_fuels_TWh'
    ]

    target_col = 'Primary_energy_consumption_per_capita_kWh_person'

    # Load model and preprocessing objects
    rf_model = joblib.load("rf_model.pkl")
    imputer = joblib.load("imputer.pkl")
    scaler = joblib.load("scaler.pkl")

    # Prepare data
    data = df[selected_cols]
    data_imputed = imputer.transform(data)
    data_scaled = scaler.transform(data_imputed)

    # Make predictions
    predictions = rf_model.predict(data_scaled)
    df['Predicted_Energy_Consumption'] = predictions

    st.subheader("📊 Prediction Results")
    st.write(df[[*selected_cols, 'Predicted_Energy_Consumption']])
