# Energy Prediction Model

A web application for predicting global energy consumption per capita using machine learning. The app provides both manual and batch (CSV) prediction modes, a secure login system, and a user-friendly interface built with Streamlit.

---

## Features
- **Manual Prediction**: Enter feature values manually to get instant predictions.
- **Batch Prediction**: Upload a CSV file for batch predictions and download results.
- **Login System**: Simple authentication to restrict access.
- **Modern UI**: Custom background and styled interface.

---

## File Descriptions

| File/Folder                  | Description                                                                 |
|------------------------------|-----------------------------------------------------------------------------|
| `app.py`                     | Main Streamlit app for prediction and UI.                                   |
| `login.py`                   | Handles user authentication.                                                |
| `credentials.py`             | Stores login credentials (default: admin/password123).                      |
| `rf_model.pkl`               | Pre-trained Random Forest model for energy prediction.                      |
| `scaler.pkl`                 | Scaler object for feature normalization.                                    |
| `imputer.pkl`                | (If used) Imputer for handling missing values.                              |
| `requirements.txt`           | Python dependencies.                                                        |
| `render.yaml`                | Deployment configuration for Render.com.                                    |
| `sample_energy_mixed_case.csv`| Example CSV for batch prediction.                                           |
| `wallpaper.png`              | Background image for the app.                                               |
| `README.md`                  | Project documentation (this file).                                          |

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd <project-directory>
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

1. **Run the app:**
   ```bash
   streamlit run app.py
   ```
2. **Login:**
   - Username: `admin`
   - Password: `password123`

3. **Prediction Modes:**
   - **Manual Input:** Enter values for:
     - Access to Electricity (% population)
     - GDP per Capita
     - Financial Flows to Developing Countries (USD)
     - Renewable Electricity Capacity per Capita
     - Electricity from Fossil Fuels (TWh)
   - **CSV Upload:** Upload a CSV file with the following columns:
     - `Access_to_electricity_of_population`
     - `gdp_per_capita`
     - `Financial_flows_to_developing_countries_US`
     - `Renewable_electricity_generating_capacity_per_capita`
     - `Electricity_from_fossil_fuels_TWh`
   - Download the results as a CSV after prediction.

---

## Sample Data

See `sample_energy_mixed_case.csv` for the required format:

```
Access_to_electricity_of_population,gdp_per_capita,Financial_flows_to_developing_countries_US,Renewable_electricity_generating_capacity_per_capita,Electricity_from_fossil_fuels_TWh
95.2,12000,5000000,1.8,1000
...
```

---

## Deployment

- The app can be deployed on [Render.com](https://render.com/) using the provided `render.yaml`.
- The default port is set to `10000`.

---

## Notes
- **Credentials**: For demo purposes, credentials are stored in plain text. Change them for production use.
- **Model/Scaler Files**: Ensure `rf_model.pkl` and `scaler.pkl` are present in the root directory.
- **Imputer**: If used, ensure `imputer.pkl` is also present.
- **Background Image**: `wallpaper.png` is used for UI styling.
