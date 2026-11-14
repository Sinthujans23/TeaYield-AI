
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_FILE = BASE_DIR / "data" / "processed" / "tea_features.csv"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_FILE, parse_dates=["period"])
df = df.sort_values("period").reset_index(drop=True)

# Features
features = [
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_12",
    "month",
    "year",
    "rolling_mean_3"
]

X = df[features]
y = df["production_mn_kg"]

# Chronological split (80% training, 20% testing)
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# Train Random Forest
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\nRandom Forest Training Completed!")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nModel Performance:")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2 Score: {r2:.4f}")

# Save trained model and feature order
joblib.dump(
    {"model": model, "features": features},
    MODEL_DIR / "random_forest.pkl"
)

# Save test predictions
results = pd.DataFrame({
    "period": df["period"].iloc[split_index:].values,
    "actual": y_test.values,
    "predicted": predictions
})

results.to_csv(
    RESULTS_DIR / "rf_predictions.csv",
    index=False
)

print("\nModel saved to models/random_forest.pkl")
print("Predictions saved to results/rf_predictions.csv")
