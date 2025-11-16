# ===========================
#  FULL DATA FETCH SCHEDULER
# ===========================

import sys
import os
import time
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine

# allow importing scripts/*
sys.path.append(".")

# your scripts
from scripts.fetch_AQI import fetch_live as fetch_aqi_live
from scripts.fetch_weather import run_live_for_duration
from scripts.fetch_traffic import simulate_time_series

# --------------------------------
# DATABASE CONFIG
# --------------------------------
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    print("[WARN] DATABASE_URL not found — using local sqlite test DB.")
    DATABASE_URL = "sqlite:///./test_local_db.sqlite"

def get_engine():
    return create_engine(DATABASE_URL, pool_pre_ping=True)

# --------------------------------
# SAFE SQL PUSH
# --------------------------------
def push_to_sql(df: pd.DataFrame, table_name: str, expected_cols: list):
    if df is None or df.empty:
        print(f"[SKIP] No rows to push for {table_name}")
        return 0

    for c in expected_cols:
        if c not in df.columns: df[c] = pd.NA

    df = df[expected_cols]

    try:
        engine = get_engine()
        df.to_sql(table_name, engine, if_exists="append", index=False)
        print(f"[OK] → Inserted {len(df)} rows into {table_name}")
        return len(df)
    except Exception as e:
        print(f"[ERROR] SQL insert failed for {table_name}: {e}")
        return 0

# --------------------------------
# NORMALIZATIONS
# --------------------------------

def normalize_aqi_csv(csv="aqi_live_data.csv"):
    if not Path(csv).exists():
        return pd.DataFrame()

    df = pd.read_csv(csv)

    df = df.rename(columns={
        "Station": "City",
        "Lat": "Latitude",
        "Lon": "Longitude",
        "PM2_5": "PM25",
        "O3": "Ozone",
        "Type": "Source"
    })

    expected = ["City","Latitude","Longitude","Timestamp","AQI","PM25","PM10",
                "CO","NO2","SO2","Ozone","Source"]

    for c in expected:
        if c not in df.columns: df[c] = pd.NA

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="ignore")
    return df[expected]


def normalize_weather_csv(csv="live_weather.csv"):
    if not Path(csv).exists():
        return pd.DataFrame()

    df = pd.read_csv(csv)

    df = df.rename(columns={
        "Temperature (°C)": "Temperature",
        "Humidity (%)": "Humidity",
        "Pressure (hPa)": "Pressure",
        "Wind Speed (m/s)": "Wind_Speed",
        "Wind Direction (°)": "Wind_Direction",
        "Cloudiness (%)": "Cloudiness",
        "Visibility (m)": "Visibility"
    })

    expected = ["City","Latitude","Longitude","Timestamp","Temperature","Humidity",
                "Pressure","Wind_Speed","Wind_Direction","Cloudiness","Visibility","Source"]

    for c in expected:
        if c not in df.columns: df[c] = pd.NA

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="ignore")
    return df[expected]


def normalize_traffic_csv(csv="traffic_timeseries.csv"):
    if not Path(csv).exists():
        return pd.DataFrame()

    df = pd.read_csv(csv)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="ignore")

    expected = [
        "city","latitude","longitude","timestamp","currentSpeed",
        "freeFlowSpeed","currentTravelTime","freeFlowTravelTime",
        "confidence","roadClosure"
    ]

    for c in expected:
        if c not in df.columns: df[c] = pd.NA

    return df[expected]

# --------------------------------
# MAIN SCHEDULER
# --------------------------------

def run_full_fetch():
    print("\n===============================")
    print("🚀 FULL SCHEDULER STARTED")
    print("===============================\n")

    # ---------------------------------------------------
    # 1. AQI FULL FETCH
    # ---------------------------------------------------
    print("📌 Fetching FULL AQI live dataset...")
    fetch_aqi_live()    # this collects ALL stations
    print("✔ AQI data fetched.\n")

    # ---------------------------------------------------
    # 2. WEATHER — fetch for 10 minutes (10× more data)
    # ---------------------------------------------------
    print("📌 Fetching WEATHER live data (10 min, 60 sec interval)...")
    run_live_for_duration(run_minutes=10, interval_seconds=60)
    print("✔ Weather data fetched.\n")

    # ---------------------------------------------------
    # 3. TRAFFIC — fetch for 10 minutes (multiple runs)
    # ---------------------------------------------------
    print("📌 Fetching TRAFFIC live data (10 min, 60 sec interval)...")
    simulate_time_series(duration_minutes=10, interval_seconds=60)
    print("✔ Traffic data fetched.\n")

    # Allow CSV buffering
    time.sleep(3)

    # ---------------------------------------------------
    # 4. PUSH INTO SQL (clean-normalized)
    # ---------------------------------------------------
    print("\n📤 Pushing normalized data into SQL...\n")

    # AQI
    df_aqi = normalize_aqi_csv()
    push_to_sql(df_aqi, "aqi_live_data", [
        "City","Latitude","Longitude","Timestamp","AQI","PM25","PM10",
        "CO","NO2","SO2","Ozone","Source"
    ])

    # WEATHER
    df_weather = normalize_weather_csv()
    push_to_sql(df_weather, "live_weather", [
        "City","Latitude","Longitude","Timestamp","Temperature","Humidity",
        "Pressure","Wind_Speed","Wind_Direction","Cloudiness","Visibility","Source"
    ])

    # TRAFFIC
    df_traffic = normalize_traffic_csv()
    push_to_sql(df_traffic, "traffic_timeseries", [
        "city","latitude","longitude","timestamp","currentSpeed",
        "freeFlowSpeed","currentTravelTime","freeFlowTravelTime",
        "confidence","roadClosure"
    ])

    print("\n🎉 ALL DONE! Full scheduler cycle completed.\n")


if __name__ == "__main__":
    run_full_fetch()
 