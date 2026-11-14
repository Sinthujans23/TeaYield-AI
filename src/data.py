import pandas as pd
import numpy as np

LAGS = [1, 2, 3, 6, 12]

def load_data(path):
    df = pd.read_csv(path)
    required = {"date", "production_mn_kg"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain {sorted(required)}; found {list(df.columns)}")
    df = df[["date", "production_mn_kg"]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    df["production_mn_kg"] = pd.to_numeric(df["production_mn_kg"], errors="raise")
    df = df.sort_values("date")
    if df.date.dt.to_period('M').duplicated().any():
        raise ValueError("Duplicate months detected")
    if df.production_mn_kg.isna().any() or (df.production_mn_kg < 0).any():
        raise ValueError("Missing or negative production values")
    df["date"] = df.date.dt.to_period("M").dt.to_timestamp()
    expected = pd.date_range(df.date.min(), df.date.max(), freq="MS")
    if len(expected) != len(df):
        raise ValueError("Missing calendar months: fill only with verified source observations")
    if len(df) < 36:
        raise ValueError("At least 36 consecutive months required; more is strongly recommended")
    return df.reset_index(drop=True)

def features(df):
    d = df.copy()
    d['month'] = d.date.dt.month
    d['year'] = d.date.dt.year
    for lag in LAGS:
        d[f'lag_{lag}'] = d.production_mn_kg.shift(lag)
    d['rolling_3'] = d.production_mn_kg.shift(1).rolling(3).mean()
    d['rolling_6'] = d.production_mn_kg.shift(1).rolling(6).mean()
    d = d.dropna().reset_index(drop=True)
    cols = ['month', 'year'] + [f'lag_{i}' for i in LAGS] + ['rolling_3','rolling_6']
    return d, cols
