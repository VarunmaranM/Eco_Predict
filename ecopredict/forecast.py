import pandas as pd
from prophet import Prophet
from prophet.serialize import model_to_json

# Load and prepare data
df = pd.read_csv('charging_data.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Prophet requires columns 'ds' and 'y'
df_prophet = df.rename(columns={
    'timestamp': 'ds',
    'number_of_charging_sessions': 'y'
})

# Initialize and train the model
# We include temperature and day_of_week as extra regressors
model = Prophet()
model.add_regressor('temperature')
model.add_regressor('day_of_week')
model.fit(df_prophet)

# Save the model
with open('forecast_model.json', 'w') as fout:
    fout.write(model_to_json(model))

print("Prophet model saved successfully.")