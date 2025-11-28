# check_aqi.py
import os
import pandas as pd

p = "data/processed/aqi_master.csv"
if not os.path.exists(p):
    print("File not found:", p)
    raise SystemExit(0)

df = pd.read_csv(p)
print("Loaded:", p, "shape:", df.shape)
# print first 30 columns (if many)
print("Columns sample:", df.columns.tolist()[:30])

# find AQI-like columns and show stats
aqi_cols = [c for c in df.columns if 'aqi' in c.lower()]
if not aqi_cols:
    print("No AQI-like column found in aqi_master.csv")
else:
    for c in aqi_cols:
        s = pd.to_numeric(df[c], errors='coerce').dropna()
        print("\nFound AQI column:", c)
        print(s.describe())
