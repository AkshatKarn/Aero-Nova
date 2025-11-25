# preprocessing/clean_traffic.py
"""
Clean traffic dataset and save to data/processed/traffic_master.csv

 - loads the traffic CSV with 'traffic' substring in name under data/
 - detect timestamp column, convert to 'timestamp'
 - coerce numeric columns (vehicle_count, congestion index etc)
 - fill small gaps, drop huge missing blocks
 - resample to hourly mean
"""

import os
import pandas as pd
import numpy as np
from glob import glob

RAW_GLOB = "data/*traffic*.csv"
OUT_DIR = "data/processed"
OUT_PATH = os.path.join(OUT_DIR, "traffic_master.csv")
os.makedirs(OUT_DIR, exist_ok=True)

candidates = glob(RAW_GLOB)
if not candidates:
    raise FileNotFoundError("No traffic file found matching data/*traffic*.csv")
RAW_PATH = candidates[0]
print("Using traffic file:", RAW_PATH)

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

df = pd.read_csv(RAW_PATH, low_memory=False)
print("Loaded traffic shape:", df.shape)

dt_col = find_datetime_col(df)
if dt_col:
    print("Detected datetime column:", dt_col)
    df['timestamp'] = pd.to_datetime(df[dt_col], errors='coerce')
    df = df.drop(columns=[dt_col], errors='ignore')
else:
    print("No datetime column detected in traffic file.")

# normalize common columns
# Example mapping; adjust to your column names as needed
col_map = {}
for c in df.columns:
    lc = c.lower()
    if "vehicle" in lc or "count" in lc:
        col_map[c] = "vehicle_count"
    if "congest" in lc or "congestion" in lc:
        col_map[c] = "congestion_index"
    if "speed" in lc:
        col_map[c] = "avg_speed"

if col_map:
    df = df.rename(columns=col_map)
    print("Renamed:", col_map)

# coerce numeric
for c in df.columns:
    if c == 'timestamp':
        continue
    df[c] = coerce_numeric_series(df[c])

# drop rows w/o timestamp
if 'timestamp' in df.columns:
    df = df.dropna(subset=['timestamp']).reset_index(drop=True)

# drop duplicates
df = df.drop_duplicates().reset_index(drop=True)

# fill small gaps then median
num_cols = df.select_dtypes(include=['number']).columns.tolist()
if num_cols:
    df[num_cols] = df[num_cols].fillna(method='ffill').fillna(df[num_cols].median())

# resample hourly
if 'timestamp' in df.columns:
    df_hour = df.set_index('timestamp').resample('1h').mean().reset_index()
    df = df_hour
    print("Resampled hourly shape:", df.shape)

df.to_csv(OUT_PATH, index=False)
print("Saved cleaned traffic to:", OUT_PATH)
