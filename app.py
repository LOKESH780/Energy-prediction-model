import streamlit as st
import joblib
import numpy as np
import pandas as pd
from login import login
from io import BytesIO
import base64

# === Background (optional) ===

def add_bg_from_local(image_file):
    with open(image_file, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()
    st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
        }}
        h1, h2, h3 {{
            color: #ffffff;
        }}
        .stButton > button {{
            color: white;
            background-color: #0066cc;
            border-radius: 8px;
            padding: 0.5em 1em;
            min-width: 100px;
            text-align: center;
            font-weight: bold;
        }}
        </style>
    """, unsafe_allow_html=True)

add_bg_from_local("wallpaper.png")

# === Login Logic ===
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login()
    st.stop()

# === Logout Button on Top-Right ===
col1, col2, col3 = st.columns([6, 1, 1])
with col3:
    if st.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.rerun()

# === Load Model and Scaler Only ===
model = joblib.load('rf_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("🔌 Global Energy Consumption Predictor")

input_method = st.radio("Select Input Method:", ["Manual Input", "Upload CSV File"])

feature_cols = [
    "Access_to_electricity_of_population",
    "gdp_per_capita",
    "Financial_flows_to_developing_countries_US",
    "Renewable_electricity_generating_capacity_per_capita",
    "Electricity_from_fossil_fuels_TWh"
]

# === Manual Input Mode ===
if input_method == "Manual Input":
    st.subheader("📝 Enter values manually")
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
        input_df = pd.DataFrame(user_input, columns=feature_cols)
        input_filled = input_df.fillna(0)
        input_scaled = scaler.transform(input_filled)
        prediction = model.predict(input_scaled)[0]
        st.success(f"🌟 Predicted Energy Consumption per Capita: {prediction:.2f} kWh/person")

# === CSV Upload Mode ===
elif input_method == "Upload CSV File":
    st.subheader("📁 Upload CSV for batch prediction")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("✅ File Uploaded Successfully!")
            #st.dataframe(df.head())

            if not all(col in df.columns for col in feature_cols):
                st.error(f"❌ CSV must contain the following columns: {', '.join(feature_cols)}")
            else:
                X = df[feature_cols].fillna(0)
                X_scaled = scaler.transform(X)
                predictions = model.predict(X_scaled)
                df["Predicted Energy Consumption"] = predictions
                st.success("✅ Prediction Completed!")
                st.dataframe(df)

                csv_buffer = BytesIO()
                df.to_csv(csv_buffer, index=False)
                csv_data = csv_buffer.getvalue()

                st.download_button(
                    label="📥 Download Predictions as CSV",
                    data=csv_data,
                    file_name="predicted_energy_consumption.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Something went wrong while reading the file: {e}")
