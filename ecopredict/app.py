import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.serialize import model_from_json
from prophet.plot import plot_plotly
import numpy as np
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="EcoPredict Pro Demand Forecast",
    page_icon="⚡️",
    layout="wide"
)

# --- Model & Data Loading ---
@st.cache_resource
def load_model_and_data():
    """Load the Prophet model and historical data."""
    try:
        with open('forecast_model.json', 'r') as fin:
            model = model_from_json(fin.read())
        data = pd.read_csv('charging_data.csv')
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        return model, data
    except FileNotFoundError:
        return None, None

model, historical_df = load_model_and_data()

if model is None:
    st.error("🚨 Model or data files not found! Please run `generate_data.py` and `forecast.py` first.")
    st.stop()

# --- UI Sidebar ---
st.sidebar.title("Forecast Configuration")
st.sidebar.markdown("Adjust the parameters to simulate different future scenarios for charging demand.")

forecast_hours = st.sidebar.slider('Hours to Forecast (1-7 days)', 12, 168, 48)

st.sidebar.subheader("Scenario Modifiers")

# Scenario-based temperature
weather_scenario = st.sidebar.selectbox(
    "Weather Scenario",
    ('Normal ☀️', 'Heatwave 🔥', 'Cold Snap ❄️')
)
if 'Heatwave' in weather_scenario:
    temp_avg = 35.0
elif 'Cold Snap' in weather_scenario:
    temp_avg = 5.0
else:
    temp_avg = 22.0
avg_temp = st.sidebar.slider('Override Average Temperature (°C)', -5.0, 45.0, temp_avg)

# Event-based demand multiplier
is_event = st.sidebar.checkbox("Simulate Public Holiday / Special Event?")
event_multiplier = 1.35 if is_event else 1.0

# --- Main Dashboard ---
st.title("⚡️ EcoPredict Pro: Charging Demand Dashboard")
st.markdown(f"Forecasting the next **{forecast_hours} hours** with a **'{weather_scenario}'** scenario.")

# --- Forecast Generation ---
with st.spinner('Generating forecast based on your parameters...'):
    time.sleep(0.5) # Simulate a heavier computation
    # 1. Create future dataframe
    last_timestamp = historical_df['timestamp'].max()
    future_dates = pd.date_range(start=last_timestamp, periods=forecast_hours + 1, freq='H')[1:]
    future_df = pd.DataFrame({'ds': future_dates})

    # 2. Add regressors
    future_df['day_of_week'] = future_df['ds'].dt.dayofweek
    temp_variation = 7 * np.sin(2 * np.pi * future_df['ds'].dt.hour / 24)
    future_df['temperature'] = avg_temp + temp_variation

    # 3. Make prediction
    forecast = model.predict(future_df)
    
    # 4. Apply "cheat" multiplier for events
    forecast['yhat'] = forecast['yhat'] * event_multiplier
    forecast['yhat_lower'] = forecast['yhat_lower'] * event_multiplier
    forecast['yhat_upper'] = forecast['yhat_upper'] * event_multiplier
    forecast['yhat'] = forecast['yhat'].clip(lower=0) # Don't allow negative predictions

# --- Key Metrics Display ---
st.subheader("Forecast Summary")

# Find peak demand and total sessions
peak_demand = forecast['yhat'].max()
total_sessions = forecast['yhat'].sum()
peak_time = forecast.loc[forecast['yhat'].idxmax()]['ds']

# Rule-based demand level classification
if peak_demand > 70:
    demand_level = "Critical 🔴"
elif peak_demand > 55:
    demand_level = "High 🟠"
else:
    demand_level = "Normal 🟢"

# Display metrics in columns
col1, col2, col3 = st.columns(3)
col1.metric("Predicted Peak Demand", f"{int(peak_demand)} sessions/hr", help="The highest number of concurrent charging sessions predicted in the forecast period.")
col2.metric("Total Predicted Sessions", f"{int(total_sessions):,} sessions", help="The cumulative number of charging sessions predicted over the entire forecast period.")
col3.metric("Grid Demand Level", demand_level, help="A classification of the peak demand on the grid. 'High' or 'Critical' levels may require load balancing.")
st.caption(f"Peak demand is expected around **{peak_time.strftime('%Y-%m-%d, %H:%M')}**.")

# --- Main Forecast Chart ---
st.markdown("---")
fig = plot_plotly(model, forecast)
fig.update_layout(
    title=f'Demand Forecast for the Next {forecast_hours} Hours',
    xaxis_title='Time',
    yaxis_title='Number of Charging Sessions'
)
# Overlay historical data for context
fig.add_scatter(
    x=historical_df['timestamp'].tail(24*7), # Show last week of history
    y=historical_df['number_of_charging_sessions'].tail(24*7),
    mode='lines', name='Historical Data', line=dict(color='royalblue', dash='dot')
)
st.plotly_chart(fig, use_container_width=True)


# --- Expander for Details ---
with st.expander("See Forecast Data & Methodology"):
    st.write("### Raw Forecast Data")
    display_df = forecast[['ds', 'yhat']].rename(columns={
        'ds': 'Timestamp', 'yhat': 'Predicted Sessions'
    })
    display_df['Predicted Sessions'] = display_df['Predicted Sessions'].round().astype(int)
    st.dataframe(display_df, use_container_width=True)
    
    st.write("### How It Works")
    st.markdown("""
    This dashboard uses Facebook's **Prophet** time-series model.
    - **Core Model**: It analyzes historical trends, weekly and daily seasonality to make a baseline forecast.
    - **Regressors**: The model is enhanced by including external factors like **temperature** and **day of the week** to improve accuracy.
    - **Scenario Simulation**: The "cheats" you control, like weather scenarios and special events, apply logical adjustments on top of the AI's prediction to simulate real-world conditions.
    """)