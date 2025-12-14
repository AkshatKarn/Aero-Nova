# preprocessing/clean_traffic.py
"""
Clean LIVE traffic data and save to data/live_processed/traffic_live_master.csv

INPUT  : aeronova/data/traffic_timeseries.csv
OUTPUT : aeronova/data/live_processed/traffic_live_master.csv
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

RAW_PATH = DATA_DIR / "traffic_timeseries.csv"
OUT_PATH = LIVE_DIR / "traffic_live_master.csv"

# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def find_datetime_col(df):
    for c in df.columns:
        lc = c.lower()
        if "timestamp" in lc or "date" in lc or "time" in lc:
            return c
    return None


def coerce_numeric_series(s):
    if s.dtype == object:
        s = (
            s.astype(str)
             .str.replace(",", "", regex=False)
             .str.strip()
             .replace({"": None, "NA": None, "NULL": None})
        )
    return pd.to_numeric(s, errors="coerce")


# --------------------------------------------------
# LOAD
# --------------------------------------------------
if not RAW_PATH.exists():
    raise FileNotFoundError(f"❌ Traffic live input file not found: {RAW_PATH}")

print(f"[INFO] Loading traffic live data from: {RAW_PATH}")
df = pd.read_csv(RAW_PATH, low_memory=False)
print("[INFO] Raw shape:", df.shape)

# --------------------------------------------------
# DATETIME HANDLING
# --------------------------------------------------
dt_col = find_datetime_col(df)
if not dt_col:
    raise ValueError("❌ No timestamp/datetime column found in traffic live data")

print("[INFO] Detected datetime column:", dt_col)
df["timestamp"] = pd.to_datetime(df[dt_col], errors="coerce")
df.drop(columns=[dt_col], inplace=True, errors="ignore")
df = df.dropna(subset=["timestamp"]).reset_index(drop=True)

# --------------------------------------------------
# COLUMN NORMALIZATION
# --------------------------------------------------
col_map = {}
for c in df.columns:
    lc = c.lower()

    if "vehicle" in lc or "count" in lc:
        col_map[c] = "vehicle_count"
    elif "congest" in lc:
        col_map[c] = "congestion_index"
    elif "speed" in lc:
        col_map[c] = "avg_speed"
    elif "confidence" in lc:
        col_map[c] = "confidence"

if col_map:
    df.rename(columns=col_map, inplace=True)
    print("[INFO] Renamed columns:", col_map)

# --------------------------------------------------
# NUMERIC CLEANING
# --------------------------------------------------
for c in df.columns:
    if c != "timestamp":
        df[c] = coerce_numeric_series(df[c])

# remove negative anomalies
num_cols = df.select_dtypes(include="number").columns
for c in num_cols:
    df.loc[df[c] < 0, c] = np.nan

# --------------------------------------------------
# IMPUTATION
# --------------------------------------------------
df[num_cols] = (
    df[num_cols]
    .fillna(method="ffill")
    .fillna(df[num_cols].median())
)

# --------------------------------------------------
# REMOVE DUPLICATES
# --------------------------------------------------
df = df.drop_duplicates().reset_index(drop=True)

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
print(f"[OK] Clean traffic live data saved → {OUT_PATH}")
