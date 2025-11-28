#!/usr/bin/env python3
"""
process_live_data_noedit.py

No-edit script for these exact files (place this file in your project root):
  - data/aqi_live_data.csv
  - data/live_weather.csv
  - data/traffic_timeseries.csv

Outputs are saved into:
  - data/live_processed/aqi_live_master.csv
  - data/live_processed/weather_live_master.csv
  - data/live_processed/traffic_live_master.csv

Run:
    python process_live_data_noedit.py
"""
import os
import sys
import traceback
import pandas as pd
import numpy as np

ROOT = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT, "data")
OUT_DIR = os.path.join(DATA_DIR, "live_processed")
os.makedirs(OUT_DIR, exist_ok=True)

# Fixed input filenames (no-edit)
INPUT_AQI = os.path.join(DATA_DIR, "aqi_live_data.csv")
INPUT_WEATHER = os.path.join(DATA_DIR, "live_weather.csv")
INPUT_TRAFFIC = os.path.join(DATA_DIR, "traffic_timeseries.csv")

def read_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    try:
        return pd.read_csv(path)
    except Exception:
        # fallback tries
        return pd.read_csv(path, low_memory=False, engine="python")

def tidy_columns(df):
    df = df.copy()
    df.columns = [
        str(c).strip().lower().replace(" ", "_").replace("-", "_")
        for c in df.columns
    ]
    return df

def find_datetime_cols(df):
    cols = []
    for c in df.columns:
        if any(k in c for k in ("date", "time", "timestamp", "ts")):
            cols.append(c)
    return cols

def coerce_numeric_by_keywords(df, keywords):
    for col in df.columns:
        for k in keywords:
            if k in col:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                break
    return df

def parse_datetimes(df, dt_cols):
    for c in dt_cols:
        try:
            df[c] = pd.to_datetime(df[c], errors="coerce", infer_datetime_format=True)
        except Exception:
            # leave column as-is if parsing fails
            pass
    return df

def fill_numerics(df, strategy="median"):
    num_cols = df.select_dtypes(include=[np.number]).columns
    if strategy == "median":
        for c in num_cols:
            med = df[c].median()
            if pd.isna(med):
                med = 0
            df[c] = df[c].fillna(med)
    elif strategy == "ffill":
        df[num_cols] = df[num_cols].fillna(method="ffill").fillna(method="bfill").fillna(0)
    else:
        df[num_cols] = df[num_cols].fillna(0)
    return df

def fill_objects(df):
    obj_cols = df.select_dtypes(include=["object"]).columns
    for c in obj_cols:
        df[c] = df[c].fillna("Unknown")
    return df

def drop_dupes(df):
    return df.drop_duplicates()

def save(df, out_name):
    out_path = os.path.join(OUT_DIR, out_name)
    df.to_csv(out_path, index=False)
    print(f"[✓] Saved: {out_path}")

def process_aqi():
    print("Processing AQI...")
    df = read_csv(INPUT_AQI)
    df = tidy_columns(df)
    df = drop_dupes(df)

    # Coerce common pollutant columns
    df = coerce_numeric_by_keywords(df, ["pm2", "pm25", "pm10", "no2", "so2", "o3", "co", "aqi", "index"])

    dt_cols = find_datetime_cols(df)
    df = parse_datetimes(df, dt_cols)

    df = fill_numerics(df, strategy="median")
    df = fill_objects(df)

    if not dt_cols:
        df["ingestion_ts"] = pd.Timestamp.now()

    # Add small meta columns if missing
    if "source" not in df.columns:
        df["source"] = "aqi_live"
    save(df, "aqi_live_master.csv")
    print(f"AQI rows: {len(df)}")

def process_weather():
    print("Processing WEATHER...")
    df = read_csv(INPUT_WEATHER)
    df = tidy_columns(df)
    df = drop_dupes(df)

    # numeric weather-like columns
    df = coerce_numeric_by_keywords(df, ["temp", "temperature", "humidity", "pressure", "wind", "rain", "precip", "visibility"])

    dt_cols = find_datetime_cols(df)
    df = parse_datetimes(df, dt_cols)
    if dt_cols:
        df = df.sort_values(by=dt_cols[0], na_position="last")

    # forward/backfill numeric then median fallback
    df = fill_numerics(df, strategy="ffill")
    df = fill_objects(df)

    if not dt_cols:
        df["ingestion_ts"] = pd.Timestamp.now()

    if "source" not in df.columns:
        df["source"] = "weather_live"
    save(df, "weather_live_master.csv")
    print(f"WEATHER rows: {len(df)}")

def process_traffic():
    print("Processing TRAFFIC...")
    df = read_csv(INPUT_TRAFFIC)
    df = tidy_columns(df)
    df = drop_dupes(df)

    # coerce likely numeric traffic cols
    df = coerce_numeric_by_keywords(df, ["count", "vehicle", "speed", "flow", "occupancy", "volume"])

    dt_cols = find_datetime_cols(df)
    df = parse_datetimes(df, dt_cols)
    if dt_cols:
        df = df.sort_values(by=dt_cols[0], na_position="last")

    df = fill_numerics(df, strategy="ffill")
    df = fill_objects(df)

    if not dt_cols:
        df["ingestion_ts"] = pd.Timestamp.now()

    if "source" not in df.columns:
        df["source"] = "traffic_live"
    save(df, "traffic_live_master.csv")
    print(f"TRAFFIC rows: {len(df)}")

def main():
    try:
        missing = []
        for p in (INPUT_AQI, INPUT_WEATHER, INPUT_TRAFFIC):
            if not os.path.exists(p):
                missing.append(p)
        if missing:
            print("[ERROR] Required input files missing:")
            for m in missing:
                print("   -", m)
            sys.exit(1)

        process_aqi()
        process_weather()
        process_traffic()
        print("\nAll done. Master CSVs are in:", OUT_DIR)

    except Exception as e:
        print("\n[ERROR] processing failed:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
