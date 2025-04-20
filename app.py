
import streamlit as st
import joblib
import numpy as np
from login import login

# Add background image
st.markdown("""
    <style>
    .stApp {
        background-image: url("https://www.google.com/imgres?q=energy%20and%20utilities%20industry&imgurl=https%3A%2F%2Fi0.wp.com%2Fwww.maiervidorno.com%2Fwp-content%2Fuploads%2F2022%2F10%2FLinkedin-Events-2021-1920-%25C3%2597-1080-px-18.png%3Ffit%3D1920%252C1080%26ssl%3D1&imgrefurl=https%3A%2F%2Fwww.maiervidorno.com%2Findustry-expertise%2Fenergy-utilities-industry%2F&docid=dsd2h2uMeZ8TUM&tbnid=ElouVagUD9SazM&vet=12ahUKEwj3062HueaMAxWv4zgGHXt3G_kQM3oECBkQAA..i&w=1920&h=1080&hcb=2&ved=2ahUKEwj3062HueaMAxWv4zgGHXt3G_kQM3oECBkQAA");
        background-size: cover;
    }
    h1, h2, h3 {
        color: #003366;
    }
    </style>
""", unsafe_allow_html=True)

# Login check
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login()
    st.stop()

# Load model and preprocessors
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
