
import streamlit as st
import joblib
import numpy as np
from login import login


import base64

def set_background(image_path):
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{encoded}");
            background-size: cover;
            background-position: center;
        }}
        h1, h2, h3 {{
            color: #003366;
        }}
        </style>
    """, unsafe_allow_html=True)

set_background("background.png")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login()
    st.stop()

model = joblib.load('rf_model.pkl')
scaler = joblib.load('scaler.pkl')
imputer = joblib.load('imputer.pkl')

st.title("🔌 Global Energy Consumption Predictor")
st.markdown("Enter the values below to predict energy consumption per capita (kWh/person):")

col1, col2 = st.columns(2)
with col1:
    electricity = st.number_input("⚡ Access to Electricity (% population)", min_value=0.0, max_value=100.0)
    gdp_per_capita = st.number_input("💰 GDP per Capita", min_value=0.0)
    financial_flows = st.number_input("🌍 Financial Flows to Developing Countries (USD)", min_value=0.0)
with col2:
    renewable_capacity = st.number_input("🔋 Renewable Electricity Capacity per Capita", min_value=0.0)
    fossil_fuel_electricity = st.number_input("🔥 Electricity from Fossil Fuels (TWh)", min_value=0.0)

if st.button("Predict 🔮"):
    user_input = [[
        electricity,
        gdp_per_capita,
        financial_flows,
        renewable_capacity,
        fossil_fuel_electricity
    ]]
    input_imputed = imputer.transform(user_input)
    input_scaled = scaler.transform(input_imputed)
    prediction = model.predict(input_scaled)[0]
    st.success(f"🌟 Predicted Energy Consumption per Capita: {prediction:.2f} kWh/person")
