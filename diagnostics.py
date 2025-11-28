# diagnostics.py
import pandas as pd, numpy as np, os
print("PWD:", os.getcwd())

# load processed files
paths = ["data/processed/aqi_master.csv","data/processed/weather_master.csv","data/processed/traffic_master.csv"]
for p in paths:
    print("\nChecking:", p)
    if os.path.exists(p):
        df = pd.read_csv(p)
        print(" -> shape:", df.shape)
        print(" -> columns sample:", df.columns.tolist()[:20])
        # show AQI-like stats
        for c in df.columns:
            if 'aqi' in c.lower():
                s = pd.to_numeric(df[c], errors='coerce').dropna()
                print("   AQI col:", c)
                print(s.describe())
                break
    else:
        print(" -> MISSING")

# quick merged check (same logic as train script)
def load_and_resample(p):
    d = pd.read_csv(p)
    for col in d.columns:
        if any(k in col.lower() for k in ['timestamp','time','date']):
            d['timestamp'] = pd.to_datetime(d[col], errors='coerce')
            break
    try:
        d = d.set_index('timestamp').resample('1h').mean().reset_index()
    except Exception:
        pass
    return d

try:
    aq = load_and_resample(paths[0])
    we = load_and_resample(paths[1])
    tr = load_and_resample(paths[2])
    merged = pd.merge_asof(aq.sort_values('timestamp'), we.sort_values('timestamp'), on='timestamp', direction='nearest', tolerance=pd.Timedelta("1h"))
    merged = pd.merge_asof(merged.sort_values('timestamp'), tr.sort_values('timestamp'), on='timestamp', direction='nearest', tolerance=pd.Timedelta("1h"))
    print("\nMerged shape:", merged.shape)
    print("Merged columns sample:", merged.columns.tolist()[:40])
    if 'AQI_real' in merged.columns:
        y = pd.to_numeric(merged['AQI_real'], errors='coerce').dropna()
    else:
        ycols=[c for c in merged.columns if 'aqi' in c.lower()]
        y = pd.to_numeric(merged[ycols[0]], errors='coerce').dropna() if ycols else pd.Series([])
    print("y describe:\n", y.describe())
    print("y unique count:", y.nunique())
    print("Top y value counts:\n", y.value_counts().head(10).to_string())
except Exception as e:
    print("Merged check failed:", e)
