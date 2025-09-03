import pandas as pd
import numpy as np

# Generate hourly data for 45 days
# FIX 1: Changed freq='H' to 'h' to resolve the FutureWarning
timestamps = pd.to_datetime(pd.date_range(end=pd.Timestamp.now(), periods=45*24, freq='h'))
num_rows = len(timestamps)

# Create patterns
# FIX 2: Corrected the sine wave to apply to the entire 1080-hour timestamp series, not just a single 24-hour cycle.
daily_cycle = np.sin(timestamps.hour * (2 * np.pi / 24)) * 20 # Daily ebb & flow

# The rest of the logic remains the same
weekday_peaks = (
    np.where((timestamps.hour >= 8) & (timestamps.hour <= 10), 15, 0) +
    np.where((timestamps.hour >= 18) & (timestamps.hour <= 21), 25, 0)
)
weekend_boost = np.where(timestamps.dayofweek >= 5, 20, 0) # Weekend has higher base
noise = np.random.randint(0, 10, num_rows)

# Number of charging sessions
sessions = np.maximum(0, 10 + daily_cycle + weekday_peaks + weekend_boost + noise).astype(int)

# Temperature
temp = 25 + 8 * np.sin(np.linspace(0, 45 * 2 * np.pi, num_rows)) + np.random.normal(0, 2, num_rows)

df = pd.DataFrame({
    'timestamp': timestamps,
    'number_of_charging_sessions': sessions,
    'temperature': temp.round(1),
    'day_of_week': timestamps.dayofweek # Monday=0, Sunday=6
})

df.to_csv('charging_data.csv', index=False)
print("charging_data.csv generated successfully.")