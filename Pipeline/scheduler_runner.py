# Full data fetch scheduler — writes fetched CSVs into project-root data/ and pushes normalized rows into SQL

import sys
import os
import time
import shutil
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine

# allow importing scripts/*
sys.path.append(".")

# ---------------------------
# your fetcher functions (should exist in scripts/)
# ---------------------------
from scripts.fetch_AQI import fetch_live as fetch_aqi_live
from scripts.fetch_weather import run_live_for_duration
from scripts.fetch_traffic import simulate_time_series

# ===========================
# CONFIG: discover PROJECT ROOT, DATA DIR + DB URL
# ===========================
# find project root heuristically: first ancestor that contains .git or common project folders
_this_file = Path(__file__).resolve()
project_root = None
for p in _this_file.parents:
    if (p / ".git").exists() or (p / "Pipeline").exists() or (p / "scripts").exists() or (p / "preprocessing").exists():
        project_root = p
        break
# fallback to parents[1] (two levels up) if nothing found
if project_root is None:
    project_root = _this_file.parents[1]

ROOT = project_root            # project root (Aero-Nova-2)
DATA_DIR = ROOT / "data"       # <-- THIS IS THE TARGET: project-root/data/
DATA_DIR.mkdir(parents=True, exist_ok=True)

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    print("[WARN] DATABASE_URL not found — using local sqlite test DB.")
    DATABASE_URL = f"sqlite:///{(ROOT / 'test_local_db.sqlite').as_posix()}"

def get_engine():
    return create_engine(DATABASE_URL, pool_pre_ping=True)


# --------------------------------
# SAFE SQL PUSH
# --------------------------------
def push_to_sql(df: pd.DataFrame, table_name: str, expected_cols: list):
    if df is None or df.empty:
        print(f"[SKIP] No rows to push for {table_name}")
        return 0

    # ensure expected columns present
    for c in expected_cols:
        if c not in df.columns:
            df[c] = pd.NA

    # select only expected (order preserved)
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
# NORMALIZATIONS (read from DATA_DIR)
# --------------------------------
def normalize_aqi_csv(csv_name="aqi_live_data.csv"):
    csv_path = DATA_DIR / csv_name
    if not csv_path.exists():
        print(f"[WARN] AQI csv missing: {csv_path}")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    df = df.rename(columns={
        "Station": "City",
        "Lat": "Latitude",
        "Lon": "Longitude",
        "PM2_5": "PM25",
        "O3": "Ozone",
        "Type": "Source",
        "timestamp": "Timestamp",
        "Timestamp": "Timestamp"
    })

    expected = ["City","Latitude","Longitude","Timestamp","AQI","PM25","PM10",
                "CO","NO2","SO2","Ozone","Source"]

    for c in expected:
        if c not in df.columns:
            df[c] = pd.NA

    # try robust datetime parsing
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    return df[expected]


def normalize_weather_csv(csv_name="live_weather.csv"):
    csv_path = DATA_DIR / csv_name
    if not csv_path.exists():
        print(f"[WARN] Weather csv missing: {csv_path}")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    df = df.rename(columns={
        "Temperature (°C)": "Temperature",
        "Temperature": "Temperature",
        "Humidity (%)": "Humidity",
        "Humidity": "Humidity",
        "Pressure (hPa)": "Pressure",
        "Wind Speed (m/s)": "Wind_Speed",
        "Wind Direction (°)": "Wind_Direction",
        "Cloudiness (%)": "Cloudiness",
        "Visibility (m)": "Visibility",
        "timestamp": "Timestamp",
        "Timestamp": "Timestamp"
    })

    expected = ["City","Latitude","Longitude","Timestamp","Temperature","Humidity",
                "Pressure","Wind_Speed","Wind_Direction","Cloudiness","Visibility","Source"]

    for c in expected:
        if c not in df.columns:
            df[c] = pd.NA

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    return df[expected]


def normalize_traffic_csv(csv_name="traffic_timeseries.csv"):
    csv_path = DATA_DIR / csv_name
    if not csv_path.exists():
        print(f"[WARN] Traffic csv missing: {csv_path}")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)

    # try to normalize common column names
    # ensure timestamp col exists
    if "timestamp" not in df.columns:
        for c in df.columns:
            if "time" in c.lower() or "date" in c.lower():
                df = df.rename(columns={c: "timestamp"})
                break

    # coerce timestamp
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    expected = [
        "city","latitude","longitude","timestamp","currentSpeed",
        "freeFlowSpeed","currentTravelTime","freeFlowTravelTime",
        "confidence","roadClosure"
    ]

    for c in expected:
        if c not in df.columns:
            df[c] = pd.NA

    return df[expected]


# --------------------------------
# UTIL: move generated files into DATA_DIR
# --------------------------------
def move_to_data_if_exists(filenames):
    """
    Move files that may have been written either to:
      - project root (ROOT / fname)
      - scheduler folder (this script's folder)
      - current working directory (Path.cwd() / fname)
    into DATA_DIR (project_root/data/).
    Returns list of moved filenames.
    """
    moved = []
    candidates = []
    for fname in filenames:
        candidates.append(ROOT / fname)            # project_root/fname
        candidates.append(_this_file.parent / fname)  # scheduler folder / fname
        candidates.append(Path.cwd() / fname)      # current working dir / fname

    # dedupe while preserving order
    seen = set()
    uniq = []
    for p in candidates:
        if p not in seen:
            uniq.append(p)
            seen.add(p)

    for src in uniq:
        if src.exists():
            dst = DATA_DIR / src.name
            try:
                # create parent if needed
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))
                moved.append(src.name)
            except Exception as e:
                print(f"[WARN] Could not move {src} -> {dst}: {e}")

    if moved:
        print("[INFO] Moved files into data/:", moved)
    return moved


# --------------------------------
# MAIN SCHEDULER
# --------------------------------
def run_full_fetch():
    print("\n===============================")
    print("🚀 FULL SCHEDULER STARTED")
    print("===============================\n")
    print(f"Project root: {ROOT}")
    print(f"Data dir: {DATA_DIR}\n")

    # ---------------------------------------------------
    # 1. AQI FULL FETCH
    # ---------------------------------------------------
    print("📌 Fetching FULL AQI live dataset...")
    try:
        fetch_aqi_live()
        print("✔ AQI data fetched.")
    except Exception as e:
        print(f"[ERROR] fetch_aqi_live failed: {e}")

    # ---------------------------------------------------
    # 2. WEATHER — fetch for 10 minutes (10× more data)
    # ---------------------------------------------------
    print("\n📌 Fetching WEATHER live data (10 min, 60 sec interval)...")
    try:
        run_live_for_duration(run_minutes=10, interval_seconds=60)
        print("✔ Weather data fetched.")
    except Exception as e:
        print(f"[ERROR] run_live_for_duration failed: {e}")

    # ---------------------------------------------------
    # 3. TRAFFIC — fetch for 10 minutes (multiple runs)
    # ---------------------------------------------------
    print("\n📌 Fetching TRAFFIC live data (10 min, 60 sec interval)...")
    try:
        simulate_time_series(duration_minutes=10, interval_seconds=60)
        print("✔ Traffic data fetched.")
    except Exception as e:
        print(f"[ERROR] simulate_time_series failed: {e}")

    # Allow CSV buffering
    time.sleep(2)

    # ---------------------------------------------------
    # Move produced CSVs into data/ (if fetchers wrote to repo root or cwd)
    # ---------------------------------------------------
    produced_files = ["aqi_live_data.csv", "live_weather.csv", "traffic_timeseries.csv"]
    move_to_data_if_exists(produced_files)

    # list data dir contents for debug
    print("\n[data folder contents]")
    for p in sorted(DATA_DIR.glob("*")):
        print(" -", p.name)

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
