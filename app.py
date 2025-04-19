import streamlit as st
import pandas as pd
import joblib
from login import login
from sklearn.metrics import mean_squared_error, r2_score

# Handle login state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login()
    st.stop()

# Title after successful login
st.title("🔍 Global Energy Consumption Prediction")
st.write("Welcome! This app compares model performance on predicting energy consumption per capita.")

# Load the CSV data
data = pd.read_csv("sample_data.csv")

# Load trained models and scaler
rf = joblib.load("rf_model.pkl")
gb = joblib.load("gb_model.pkl")
lr = joblib.load("lr_model.pkl")
scaler = joblib.load("scaler.pkl")

# Define input features
features = [
    'Access_to_electricity_of_population',
    'GDP per capita',
    'Financial_flows_to_developing_countries_US',
    'Renewable electricity Generating Capacity per capita',
    'Electricity_from_fossil_fuels_TWh'
]

# Get inputs and scale
X = data[features]
X_scaled = scaler.transform(X)
y_true = data['Primary_energy_consumption_per_capita_kWh_person']

# Make predictions
pred_rf = rf.predict(X_scaled)
pred_gb = gb.predict(X_scaled)
pred_lr = lr.predict(X_scaled)

# Evaluate each model
def evaluate_model(name, y_true, y_pred):
    return {
        "Model": name,
        "MSE": mean_squared_error(y_true, y_pred),
        "R2 Score": r2_score(y_true, y_pred)
    }

results = [
    evaluate_model("Random Forest", y_true, pred_rf),
    evaluate_model("Gradient Boosting", y_true, pred_gb),
    evaluate_model("Linear Regression", y_true, pred_lr)
]

# Display results
st.subheader("📊 Model Evaluation Results")
st.dataframe(pd.DataFrame(results))
