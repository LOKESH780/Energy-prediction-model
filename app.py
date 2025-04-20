import streamlit as st
import joblib
import numpy as np

# Load trained components
model = joblib.load('rf_model.pkl')
scaler = joblib.load('scaler.pkl')
imputer = joblib.load('imputer.pkl')

st.title("🔌 Global Energy Consumption Predictor")

st.markdown("Enter the values below to predict energy consumption per capita (kWh/person):")

# Input form
electricity = st.number_input("Access to Electricity (% population)", min_value=0.0, max_value=100.0)
gdp_per_capita = st.number_input("GDP per Capita", min_value=0.0)
financial_flows = st.number_input("Financial Flows to Developing Countries (USD)", min_value=0.0)
renewable_capacity = st.number_input("Renewable Electricity Generating Capacity per Capita", min_value=0.0)
fossil_fuel_electricity = st.number_input("Electricity from Fossil Fuels (TWh)", min_value=0.0)

if st.button("Predict"):
    user_input = [[
        electricity,
        gdp_per_capita,
        financial_flows,
        renewable_capacity,
        fossil_fuel_electricity
    ]]

    # Preprocessing
    input_imputed = imputer.transform(user_input)
    input_scaled = scaler.transform(input_imputed)

    # Prediction
    prediction = model.predict(input_scaled)[0]
    st.success(f"🌍 Predicted Energy Consumption per Capita: {prediction:.2f} kWh/person")
