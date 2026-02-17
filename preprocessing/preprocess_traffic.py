# preprocess_traffic.py
"""
Preprocess cleaned LIVE traffic data and save final master file.

INPUT  : aeronova/data/live_processed/traffic_live_master.csv
OUTPUT : aeronova/data/processed/traffic_master.csv
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

INFILE = LIVE_DIR / "traffic_live_master.csv"
OUTFILE = PROCESSED_DIR / "traffic_master.csv"

# --------------------------------------------------
# LOAD
# --------------------------------------------------
if not INFILE.exists():
    raise FileNotFoundError(f"❌ Traffic live master file not found: {INFILE}")

print(f"[INFO] Loading cleaned traffic from: {INFILE}")
df = pd.read_csv(INFILE, low_memory=False)
print("[INFO] Input shape:", df.shape)

# --------------------------------------------------
# BASIC VALIDATION
# --------------------------------------------------
required_cols = {"timestamp"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"❌ Missing required columns in traffic data: {missing}")

# --------------------------------------------------
# TIMESTAMP NORMALIZATION (Power BI SAFE)
# --------------------------------------------------
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df = df.dropna(subset=["timestamp"])

# remove timezone if any
try:
    df["timestamp"] = df["timestamp"].dt.tz_localize(None)
except Exception:
    pass

# --------------------------------------------------
# FINAL CLEANUP
# --------------------------------------------------
df = (
    df.drop_duplicates()
      .sort_values("timestamp")
      .reset_index(drop=True)
)

# --------------------------------------------------
# OPTIONAL: COLUMN SELECTION (UNCOMMENT IF NEEDED)
# --------------------------------------------------
# keep_cols = [
#     "timestamp",
#     "city",
#     "location",
#     "vehicle_count",
#     "avg_speed",
#     "congestion_index",
#     "confidence"
# ]
# df = df[[c for c in keep_cols if c in df.columns]]

# --------------------------------------------------
# SAVE
# --------------------------------------------------
df.to_csv(OUTFILE, index=False)
print(f"[OK] Final traffic master saved → {OUTFILE}")
