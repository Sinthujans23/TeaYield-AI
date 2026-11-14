
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_FILE = BASE_DIR / "data" / "processed" / "tea_features.csv"
RESULTS_DIR = BASE_DIR / "results"

df = pd.read_csv(DATA_FILE)
df = df.sort_values("period").reset_index(drop=True)

# Same test period as ML models
split_index = int(len(df) * 0.8)
test = df.iloc[split_index:].copy()

# Last year's same month production
predictions = test["lag_12"].values
actual = test["production_mn_kg"].values

mae = mean_absolute_error(actual, predictions)
rmse = np.sqrt(mean_squared_error(actual, predictions))
r2 = r2_score(actual, predictions)

print("\nSEASONAL NAIVE BASELINE")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2: {r2:.4f}")

results = pd.DataFrame({
    "period": test["period"],
    "actual": actual,
    "seasonal_naive": predictions
})

results.to_csv(
    RESULTS_DIR / "seasonal_naive_predictions.csv",
    index=False
)

print("\nBaseline evaluation completed!")
