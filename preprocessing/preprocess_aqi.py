# preprocess_aqi.py
"""
Preprocess cleaned LIVE AQI data and save final master file.

INPUT  : aeronova/data/live_processed/aqi_live_master.csv
OUTPUT : aeronova/data/processed/aqi_master.csv
"""

from pathlib import Path
import pandas as pd

# --------------------------------------------------
# PATH SETUP (FINAL)
# --------------------------------------------------
BASE = Path(__file__).resolve().parent.parent   # aeronova/
DATA_DIR = BASE / "data"
LIVE_DIR = DATA_DIR / "live_processed"
PROCESSED_DIR = DATA_DIR / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

INFILE = LIVE_DIR / "aqi_live_master.csv"
OUTFILE = PROCESSED_DIR / "aqi_master.csv"

# --------------------------------------------------
# LOAD
# --------------------------------------------------
if not INFILE.exists():
    raise FileNotFoundError(f"❌ AQI live master file not found: {INFILE}")

print(f"[INFO] Loading cleaned AQI from: {INFILE}")
df = pd.read_csv(INFILE)
print("[INFO] Input shape:", df.shape)

# --------------------------------------------------
# BASIC VALIDATION (OPTIONAL BUT SAFE)
# --------------------------------------------------
required_cols = {"timestamp"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"❌ Missing required columns in AQI data: {missing}")

# --------------------------------------------------
# (OPTIONAL) FINAL SORT / DEDUP
# --------------------------------------------------
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df = df.dropna(subset=["timestamp"])
df = df.drop_duplicates().sort_values("timestamp").reset_index(drop=True)

# --------------------------------------------------
# SAVE
# --------------------------------------------------
df.to_csv(OUTFILE, index=False)
print(f"[OK] Final AQI master saved → {OUTFILE}")
