# ===============================
# Part 1: Machine Learning Model
# ===============================
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Load dataset
data = pd.read_csv("weather data 70 years.csv")

# Drop duplicates and fix Date
data.drop_duplicates(inplace=True)
data['Date'] = pd.to_datetime(data['Date'])

# Remove rows with Rain == 0
data = data[data['Rain'] != 0].copy()

# Handle missing values
imputer = KNNImputer(n_neighbors=5)
data[['Rain', 'Temp Max', 'Temp Min']] = imputer.fit_transform(
    data[['Rain', 'Temp Max', 'Temp Min']]
)

# Outlier handling
def handle_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[column] = np.where(df[column] < lower, lower, df[column])
    df[column] = np.where(df[column] > upper, upper, df[column])
    return df

for col in ['Rain', 'Temp Max', 'Temp Min']:
    data = handle_outliers(data, col)

# Feature engineering
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day
data['DayOfYear'] = data['Date'].dt.dayofyear
data['Season'] = data['Month'] % 12 // 3 + 1

# Lag features
data['Rain_Lag1'] = data['Rain'].shift(1)
data['Temp_Max_Lag1'] = data['Temp Max'].shift(1)
data['Temp_Min_Lag1'] = data['Temp Min'].shift(1)

# Rolling stats
data['Rain_Rolling_Mean'] = data['Rain'].rolling(window=7).mean()
data['Temp_Max_Rolling_Mean'] = data['Temp Max'].rolling(window=7).mean()
data['Temp_Min_Rolling_Mean'] = data['Temp Min'].rolling(window=7).mean()

data.dropna(inplace=True)

# Scaling
scaler = MinMaxScaler()
features = [
    'Temp Max', 'Temp Min', 'Year', 'Month', 'Day', 'DayOfYear', 'Season',
    'Rain_Lag1', 'Temp_Max_Lag1', 'Temp_Min_Lag1',
    'Rain_Rolling_Mean', 'Temp_Max_Rolling_Mean', 'Temp_Min_Rolling_Mean'
]
data[features] = scaler.fit_transform(data[features])

# Train/Test Split
X = data[features]
y = data['Rain']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RandomForest (fast version)
rf_model = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# Save model and scaler for ESP32 usage
joblib.dump(rf_model, "best_random_forest_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# Evaluation
y_pred = rf_model.predict(X_test)
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))


# ===============================
# Part 2: ESP32 Sensor Code
# ===============================
"""
Note:
This part is written for MicroPython and will run only on the ESP32 board.
It will NOT run on PC or Google Colab.

Required libraries on ESP32:
- machine
- dht
- time

⚠️ Important:
- The trained RandomForest model (`best_random_forest_model.pkl`) and scaler (`scaler.pkl`) cannot be used directly on ESP32 
  because scikit-learn is not supported on microcontrollers.
- The ESP32 should only collect sensor data and send it to a PC or server 
  (via Serial or MQTT), where the trained ML model will make predictions.
"""

import dht
from machine import Pin
import time

# Sensor setup
dht_sensor = dht.DHT22(Pin(4))  # DHT22 connected to GPIO 4
soil_sensor = Pin(34)           # ADC pin for soil moisture sensor

while True:
    try:
        dht_sensor.measure()
        temp = dht_sensor.temperature()
        humidity = dht_sensor.humidity()
        soil_moisture = soil_sensor.value()

        print("Temp:", temp, "Humidity:", humidity, "Soil:", soil_moisture)

        # Send this data to PC (via Serial or MQTT) for rainfall prediction
        time.sleep(2)

    except Exception as e:
        print("Error:", e)
