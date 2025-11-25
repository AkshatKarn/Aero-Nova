# preprocessing/clean_weather.py
"""
Clean historical weather data and save to data/processed/weather_master.csv

 - loads data/historical_weather.csv
 - detect timestamp column -> 'timestamp'
 - coerce temperature/humidity/wind/precip to numeric
 - convert units if needed (not automatic; extend if you know units)
 - handle missing values with median or forward-fill small gaps
 - resample to hourly mean
"""

import os
import pandas as pd
import numpy as np

RAW_PATH = "data/historical_weather.csv"
OUT_DIR = "data/processed"
OUT_PATH = os.path.join(OUT_DIR, "weather_master.csv")
os.makedirs(OUT_DIR, exist_ok=True)

def find_datetime_col(df):
    for c in df.columns:
        if "date" in c.lower() or "time" in c.lower() or "timestamp" in c.lower():
            return c
    return None

def coerce_numeric_series(s):
    if s.dtype == object:
        s2 = s.astype(str).str.replace(',','').str.strip().replace({'': None, 'NA': None, 'NULL': None})
        return pd.to_numeric(s2, errors='coerce')
    return pd.to_numeric(s, errors='coerce')

print("Loading:", RAW_PATH)
df = pd.read_csv(RAW_PATH, low_memory=False)
print("Shape:", df.shape)

dt_col = find_datetime_col(df)
if dt_col:
    print("Detected datetime column:", dt_col)
    df['timestamp'] = pd.to_datetime(df[dt_col], errors='coerce')
    df = df.drop(columns=[dt_col], errors='ignore')
else:
    print("No datetime detected in weather file.")

# common weather column normalization (extend mapping as needed)
col_map = {}
for c in df.columns:
    lc = c.lower()
    if "temp" in lc and "dew" not in lc:
        col_map[c] = "temperature"
    if "humidity" in lc:
        col_map[c] = "humidity"
    if "wind" in lc and ("speed" in lc or "spd" in lc):
        col_map[c] = "windspeed"
    if "rain" in lc or "precip" in lc:
        col_map[c] = "precipitation"

if col_map:
    df = df.rename(columns=col_map)
    print("Renamed:", col_map)

# coerce numeric
for c in df.columns:
    if c == 'timestamp':
        continue
    df[c] = coerce_numeric_series(df[c])

# drop rows without timestamp
if 'timestamp' in df.columns:
    n0 = len(df)
    df = df.dropna(subset=['timestamp']).reset_index(drop=True)
    print("Dropped rows with invalid timestamp:", n0 - len(df))

# drop duplicates
df = df.drop_duplicates().reset_index(drop=True)

# Impute: small gaps -> forward-fill then median
num_cols = df.select_dtypes(include=['number']).columns.tolist()
if len(num_cols):
    df[num_cols] = df[num_cols].fillna(method='ffill').fillna(df[num_cols].median())

# resample hourly mean
if 'timestamp' in df.columns:
    df_hour = df.set_index('timestamp').resample('1h').mean().reset_index()
    df = df_hour
    print("Resampled hourly shape:", df.shape)

df.to_csv(OUT_PATH, index=False)
print("Saved cleaned weather to:", OUT_PATH)
