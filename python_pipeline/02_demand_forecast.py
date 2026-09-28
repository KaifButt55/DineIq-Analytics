import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(BASE_DIR, 'raw_data')
MODELS = os.path.join(BASE_DIR, 'models', 'python')
os.makedirs(MODELS, exist_ok=True)
OUT = os.path.join(BASE_DIR, 'python_pipeline')

print("⏳ Loading...")
orders = pd.read_csv(f'{RAW}/orders.csv', parse_dates=['order_date'])
items = pd.read_csv(f'{RAW}/order_items.csv')
df = items.merge(orders[['order_id','order_date','status']], on='order_id')
df = df[df['status']=='Completed']

# Daily demand
daily = df.groupby(df['order_date'].dt.date)['quantity'].sum().reset_index()
daily.columns = ['date','demand']
daily['date'] = pd.to_datetime(daily['date'])
daily = daily.sort_values('date').reset_index(drop=True)

# Features
daily['dayofweek'] = daily['date'].dt.dayofweek
daily['month'] = daily['date'].dt.month
daily['day'] = daily['date'].dt.day
daily['lag_1'] = daily['demand'].shift(1)
daily['lag_7'] = daily['demand'].shift(7)
daily = daily.dropna()

# Chronological split (NO leakage)
split = int(len(daily) * 0.8)
train, test = daily[:split], daily[split:]

features = ['dayofweek','month','day','lag_1','lag_7']
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(train[features], train['demand'])
preds = model.predict(test[features])

mae = mean_absolute_error(test['demand'], preds)
rmse = np.sqrt(mean_squared_error(test['demand'], preds))
r2 = r2_score(test['demand'], preds)

print(f"\n📊 Metrics: MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.4f}")

joblib.dump(model, f'{MODELS}/demand_forecast.pkl')

# Save predictions
result = test[['date','demand']].copy()
result['predicted'] = preds
result.to_csv(f'{OUT}/demand_predictions_python.csv', index=False)
print(f"✅ Model saved + predictions saved")