
# 🌦 Rainfall Prediction & IoT Integration

This project uses **Machine Learning** to predict daily rainfall based on historical weather data and sends the prediction—along with real-time temperature and humidity—to the **ThingSpeak IoT platform**.

---

## 📁 Directory Structure

Update file paths in the scripts with your actual locations:

```

/data/
├── filtered\_daily\_weather\_data.csv      ← Historical weather dataset (Windows path example used in code)

/scripts/
├── weather\_data\_processing.py           ← Data cleaning, feature engineering, and model training
├── rainfall\_prediction.py               ← Prediction + ThingSpeak API integration

````

> **Note:** In Windows, prefer raw strings or forward slashes for file paths in Python (e.g., `r"C:\Users\hp\Desktop\filtered_daily_weather_data.csv"` or `"C:/Users/hp/Desktop/filtered_daily_weather_data.csv"`).

---

## 🧰 Dependencies

Install the required Python packages:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn shap joblib requests
````

---

## 🚀 How It Works

1. **Load & Clean Data**

   * Reads the historical CSV file.
   * Drops duplicates and filters out zero-rainfall rows.
   * Imputes missing values (mean or KNN).
   * Converts the `Date` column to datetime.

2. **Feature Engineering**

   * Extracts `Year`, `Month`, `Day`, `DayOfYear`, and `Season`.
   * Adds lag features: `Rain_Lag1`, `Temp_Max_Lag1`, `Temp_Min_Lag1`.
   * (Optional) Adds rolling means.
   * Scales features using `MinMaxScaler`.

3. **Train Model**

   * Trains a **Random Forest Regressor**.
   * Tunes hyperparameters with **GridSearchCV**.
   * Evaluates with MSE/MAE/R² and saves:

     * Processed dataset → `refined_weather_data.csv`
     * Best model → `best_random_forest_model.pkl`

4. **Predict Rainfall**

   * Accepts real-time `Temp Max` and `Temp Min`.
   * Uses the latest historical row for contextual features.
   * (Includes a simple rule: if very hot/humid thresholds are met, predicted rain can be clamped to 0 in the provided example.)

5. **Send Data to ThingSpeak**

   * Posts **Temperature**, **Humidity**, and **Predicted Rainfall** to your ThingSpeak channel via HTTP API.

---

## 🖥 Quick Commands

### 1) Train the Model

```bash
python weather_data_processing.py
```

### 2) Predict & Send to ThingSpeak

```bash
python rainfall_prediction.py
```

---

## 📈 Sample Console Output

```
Enter the current temperature: 28
Enter the current humidity: 75
Enter the maximum temperature for prediction: 30
Enter the minimum temperature for prediction: 20

Predicted rainfall considering temperature and recent weather data: 12.45 mm
Data sent to ThingSpeak successfully! Response: 123456
```

---

## 📡 ThingSpeak Setup

1. Create a ThingSpeak channel with **3 fields**:

   * Field 1 → Temperature
   * Field 2 → Humidity
   * Field 3 → Predicted Rainfall

2. Get your **Write API Key** from ThingSpeak.

3. Update it in `rainfall_prediction.py`:

```python
THINGSPEAK_WRITE_API_KEY = 'YOUR_API_KEY'
```

> (Optional) Keep your keys in environment variables or a `.env` file for safety.

---

## 🔧 Sensor Snippet (Optional)

If you’re using **ESP32/ESP8266** with **DHT22** and an **analog moisture sensor**, you can read temperature/humidity/moisture and forward them to the Python script or directly to ThingSpeak. The provided MicroPython example in your code (`dht`, `ADC`) prints values to serial; adapt as needed for your setup.

---

## 📌 Notes

* Ensure your CSV(s) only contain valid weather rows and the correct columns: `Date`, `Rain`, `Temp Max`, `Temp Min`.
* For Windows paths, escape backslashes or use raw strings.
* Retrain periodically with fresh data for better accuracy.
* For more advanced modeling (probabilistic forecasts, seasonality), consider additional models or feature sets.

---

## 📄 License

This project is open-source under the **MIT License**.

```
