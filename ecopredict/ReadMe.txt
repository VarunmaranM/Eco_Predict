# ⚡️ EcoPredict: EV Charging Demand Forecasting

This project uses the Prophet time-series model to forecast the demand for EV charging sessions. The interactive Streamlit dashboard allows for simulating different future scenarios.

---

## 🛠️ Setup Instructions

**1. Create a Virtual Environment (Recommended)**
Use a virtual environment to manage dependencies for this specific project.

```bash
# Create a virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

**2. Install Dependencies**
Install the Prophet library and other dependencies from the `requirements.txt` file.
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

Follow these steps in order from within the `eco_predict` directory.

**1. Generate Sample Data (Run once)**
This script creates the `charging_data.csv` file containing historical demand data.
```bash
python generate_data.py
```

**2. Train the Model (Run once)**
This script trains the Prophet forecasting model and saves it as `forecast_model.json`.
```bash
python forecast.py
```

**3. Launch the Application**
Start the interactive Streamlit dashboard.
```bash
python -m streamlit run app.py
```