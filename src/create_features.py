import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = BASE_DIR / "data" / "raw" / "tea_production.csv"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "tea_features.csv"

df = pd.read_csv(INPUT_FILE, comment="#")

df = df[["period", "value"]].copy()
df["period"] = pd.to_datetime(df["period"])

df = df.rename(columns={
    "value": "production_mn_kg"
})

df = df.sort_values("period").reset_index(drop=True)

# Previous months production
df["lag_1"] = df["production_mn_kg"].shift(1)
df["lag_2"] = df["production_mn_kg"].shift(2)
df["lag_3"] = df["production_mn_kg"].shift(3)
df["lag_12"] = df["production_mn_kg"].shift(12)

# Month and year
df["month"] = df["period"].dt.month
df["year"] = df["period"].dt.year

# Previous 3 months average
df["rolling_mean_3"] = (
    df["production_mn_kg"]
    .shift(1)
    .rolling(window=3)
    .mean()
)

# Remove rows without enough history
df = df.dropna().reset_index(drop=True)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(OUTPUT_FILE, index=False)

print("Feature engineering completed!")
print("Total usable records:", len(df))
print(df.head())
print("Saved to:", OUTPUT_FILE)