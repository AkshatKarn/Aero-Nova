# preprocessing/clean_aqi.py
"""
Clean AQI historical data and save to data/processed/aqi_master.csv

What it does:
 - loads data/aqi_historical_data.csv
 - detects datetime column and standardizes to 'timestamp' (pd.Timestamp)
 - standardizes common AQI/pollutant column names (basic mapping)
 - converts numeric columns, handles separators like commas
 - removes duplicates, removes impossible values, clips outliers using percentiles
 - imputes small missing values per-column with median
 - resamples to hourly mean (if timestamp present)
 - saves cleaned file to data/processed/aqi_master.csv
"""

import os
import pandas as pd
import numpy as np

RAW_PATH = "data/aqi_historical_data.csv"
OUT_DIR = "data/processed"
OUT_PATH = os.path.join(OUT_DIR, "aqi_master.csv")
os.makedirs(OUT_DIR, exist_ok=True)

def find_datetime_col(df):
    for c in df.columns:
        if "date" in c.lower() or "time" in c.lower() or "timestamp" in c.lower():
            return c
    for c in df.columns:
        if c.lower() in ("ts","datetime","time"):
            return c
    return None

def coerce_numeric_series(s):
    if s.dtype == object:
        # remove commas, strip spaces, convert to numeric
        s2 = s.astype(str).str.replace(',','').str.strip().replace({'': None, 'nan': None, 'NA': None, 'NULL': None})
        return pd.to_numeric(s2, errors='coerce')
    return pd.to_numeric(s, errors='coerce')

print("Loading:", RAW_PATH)
df = pd.read_csv(RAW_PATH, low_memory=False)
print("Shape:", df.shape)

# detect datetime
dt_col = find_datetime_col(df)
if dt_col:
    print("Detected datetime column:", dt_col)
    df['timestamp'] = pd.to_datetime(df[dt_col], errors='coerce')
    df = df.drop(columns=[dt_col], errors='ignore')
else:
    print("No datetime column detected. If timestamps are missing, the file will be left index-based.")
    if 'timestamp' not in df.columns:
        # try to infer from common column names
        pass

# Basic column name normalization (you can extend mapping if needed)
col_map = {}
# lowercase versions -> keep original if exist
for c in df.columns:
    lc = c.lower()
    if lc in ("pm2.5","pm2_5","pm25","pm_2_5"):
        col_map[c] = "PM2_5"
    if lc in ("pm10","pm_10"):
        col_map[c] = "PM10"
    if "no2" in lc and "no2" not in col_map:
        col_map[c] = "NO2"
    if "so2" in lc:
        col_map[c] = "SO2"
    if "co" == lc or "co2" in lc:
        col_map[c] = "CO"
    if "aqi" in lc and lc != "aqi_category":
        col_map[c] = "AQI"
# rename if any mapping
if col_map:
    df = df.rename(columns=col_map)
    print("Renamed columns:", col_map)

# Coerce numeric columns
for c in df.columns:
    if c == 'timestamp':
        continue
    df[c] = coerce_numeric_series(df[c])

# drop rows with no timestamp (if timestamp exists)
if 'timestamp' in df.columns:
    n_before = len(df)
    df = df.dropna(subset=['timestamp']).reset_index(drop=True)
    print(f"Dropped {n_before - len(df)} rows with invalid timestamp")

# drop duplicate rows
n_before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
print("Dropped duplicates:", n_before - len(df))

# Clip unrealistic values (per-domain knowledge)
# e.g., AQI negative? clip 0 to 1000; pollutant negative -> set NaN
for pollutant in ['AQI','PM2_5','PM10','NO2','SO2','CO']:
    if pollutant in df.columns:
        # replace negative/zero anomalies accordingly
        df.loc[df[pollutant] < 0, pollutant] = np.nan
        # clip AQI upper bound reasonably
        if pollutant == 'AQI':
            df[pollutant] = df[pollutant].clip(lower=0, upper=1000)

# Impute numeric columns with median (temporary)
num_cols = df.select_dtypes(include=['number']).columns.tolist()
if num_cols:
    medians = df[num_cols].median()
    df[num_cols] = df[num_cols].fillna(medians)

# If we have timestamp, resample to hourly mean to align with other datasets
if 'timestamp' in df.columns:
    df_hour = df.set_index('timestamp').resample('1h').mean().reset_index()
    print("Resampled hourly shape:", df_hour.shape)
    df = df_hour

# Save
df.to_csv(OUT_PATH, index=False)
print("Saved cleaned AQI to:", OUT_PATH)
