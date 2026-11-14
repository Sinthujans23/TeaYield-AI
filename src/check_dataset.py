import pandas as pd

df = pd.read_csv(
    "data/raw/tea_production.csv",
    comment="#"
)

df = df[["period", "value"]]

df["period"] = pd.to_datetime(df["period"])

df = df.rename(columns={
    "value": "production_mn_kg"
})

df = df.sort_values("period").reset_index(drop=True)

print(df.head())

print("Total records:", len(df))

print("Missing values:")
print(df.isnull().sum())