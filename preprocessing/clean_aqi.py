# preprocessing/clean_aqi.py
"""
Clean LIVE AQI data and save to data/live_processed/aqi_live_master.csv

INPUT  : aeronova/data/aqi_live_data.csv
OUTPUT : aeronova/data/live_processed/aqi_live_master.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np

# --------------------------------------------------
# PATH SETUP (FINAL, FIXED)
# --------------------------------------------------
BASE = Path(__file__).resolve().parent.parent   # aeronova/
DATA_DIR = BASE / "data"
LIVE_DIR = DATA_DIR / "live_processed"

LIVE_DIR.mkdir(parents=True, exist_ok=True)

RAW_PATH = DATA_DIR / "aqi_live_data.csv"
OUT_PATH = LIVE_DIR / "aqi_live_master.csv"

# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def find_datetime_col(df):
    for c in df.columns:
        lc = c.lower()
        if "date" in lc or "time" in lc or "timestamp" in lc:
            return c
    return None


def coerce_numeric_series(s):
    if s.dtype == object:
        s = (
            s.astype(str)
             .str.replace(",", "", regex=False)
             .str.strip()
             .replace({"": None, "nan": None, "NA": None, "NULL": None})
        )
    return pd.to_numeric(s, errors="coerce")


# --------------------------------------------------
# LOAD
# --------------------------------------------------
if not RAW_PATH.exists():
    raise FileNotFoundError(f"❌ AQI live input file not found: {RAW_PATH}")

print(f"[INFO] Loading AQI live data from: {RAW_PATH}")
df = pd.read_csv(RAW_PATH, low_memory=False)
print("[INFO] Raw shape:", df.shape)

# --------------------------------------------------
# DATETIME HANDLING
# --------------------------------------------------
dt_col = find_datetime_col(df)
if dt_col:
    print("[INFO] Detected datetime column:", dt_col)
    df["timestamp"] = pd.to_datetime(df[dt_col], errors="coerce")
    df.drop(columns=[dt_col], inplace=True, errors="ignore")
else:
    raise ValueError("❌ No datetime column found in AQI live data")

df = df.dropna(subset=["timestamp"]).reset_index(drop=True)

# --------------------------------------------------
# COLUMN NORMALIZATION
# --------------------------------------------------
col_map = {}
for c in df.columns:
    lc = c.lower()
    if lc in ("pm2.5", "pm2_5", "pm25", "pm_2_5"):
        col_map[c] = "PM2_5"
    elif lc in ("pm10", "pm_10"):
        col_map[c] = "PM10"
    elif "no2" in lc:
        col_map[c] = "NO2"
    elif "so2" in lc:
        col_map[c] = "SO2"
    elif lc == "co":
        col_map[c] = "CO"
    elif "aqi" in lc and lc != "aqi_category":
        col_map[c] = "AQI"

if col_map:
    df.rename(columns=col_map, inplace=True)
    print("[INFO] Renamed columns:", col_map)

# --------------------------------------------------
# NUMERIC CLEANING
# --------------------------------------------------
for c in df.columns:
    if c != "timestamp":
        df[c] = coerce_numeric_series(df[c])

# remove negatives
for col in ["AQI", "PM2_5", "PM10", "NO2", "SO2", "CO"]:
    if col in df.columns:
        df.loc[df[col] < 0, col] = np.nan
        if col == "AQI":
            df[col] = df[col].clip(0, 1000)

# --------------------------------------------------
# IMPUTATION
# --------------------------------------------------
num_cols = df.select_dtypes(include="number").columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# --------------------------------------------------
# RESAMPLE (HOURLY)
# --------------------------------------------------
df = (
    df.set_index("timestamp")
      .resample("1h")
      .mean()
      .reset_index()
)

print("[INFO] Hourly resampled shape:", df.shape)

# --------------------------------------------------
# SAVE
# --------------------------------------------------
df.to_csv(OUT_PATH, index=False)
print(f"[OK] Clean AQI live data saved → {OUT_PATH}")
