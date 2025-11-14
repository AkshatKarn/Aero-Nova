import requests
import pandas as pd
import time
import os
from datetime import datetime, timedelta

# ---------------- CONFIG ----------------
OWM_API_KEY = "ab1c6a5cd00c250e7f9621ba1ef2ed67"   # <-- change if you want
CSV_HIST = "historical_weather.csv"
CSV_LIVE = "live_weather.csv"

# how long to run live collection (minutes) and interval (seconds)
RUN_MINUTES = 30
LIVE_INTERVAL_SECONDS = 60  # fetch every 60 seconds (1 minute)

CITIES = {
    "Indore": (22.7196, 75.8577),
    "Bhopal": (23.2599, 77.4126),
    "Ujjain": (23.1765, 75.7885),
    "Dewas": (22.9676, 76.0534)
}

# ---------------- HELPERS ----------------
def ensure_cols(df):
    """Ensure consistent column order and parse Timestamp."""
    if df.empty:
        return df
    expected = [
        "City", "Latitude", "Longitude", "Timestamp",
        "Temperature (°C)", "Humidity (%)", "Pressure (hPa)",
        "Wind Speed (m/s)", "Wind Direction (°)", "Cloudiness (%)",
        "Visibility (m)", "Source"
    ]
    for c in expected:
        if c not in df.columns:
            df[c] = pd.NA
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    return df[expected]

def append_csv_no_dup(df_new, csv_path, key_cols=["City","Timestamp"]):
    """Append df_new to csv_path, avoid duplicates by key_cols, return inserted count."""
    df_new = ensure_cols(df_new)
    if df_new.empty:
        return 0

    if os.path.exists(csv_path):
        old = pd.read_csv(csv_path, parse_dates=["Timestamp"])
        old = ensure_cols(old)
        combined = pd.concat([old, df_new], ignore_index=True)
        before = len(old)
        # drop duplicates keeping last (new)
        combined = combined.drop_duplicates(subset=key_cols, keep="last")
        combined.to_csv(csv_path, index=False)
        inserted = len(combined) - before
        return max(inserted, 0)
    else:
        df_new.to_csv(csv_path, index=False)
        return len(df_new)

# ---------------- FETCH HISTORICAL (Open-Meteo) ----------------
def fetch_historical_openmeteo(city, lat, lon, days=5):
    """
    Fetch last `days` days of hourly data from Open-Meteo archive.
    Returns a dataframe (may be empty if failed).
    """
    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=days)
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&hourly=temperature_2m,relative_humidity_2m,pressure_msl,"
        f"windspeed_10m,winddirection_10m,cloudcover,visibility"
        f"&timezone=UTC"
    )
    try:
        print(f"📡 Fetching historical ({days}d) for {city} ...")
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        data = r.json().get("hourly", {})
        if not data:
            print(f"⚠ No hourly data in Open-Meteo response for {city}")
            return pd.DataFrame()
        df = pd.DataFrame({
            "City": city,
            "Latitude": lat,
            "Longitude": lon,
            "Timestamp": data.get("time", []),
            "Temperature (°C)": data.get("temperature_2m", []),
            "Humidity (%)": data.get("relative_humidity_2m", []),
            "Pressure (hPa)": data.get("pressure_msl", []),
            "Wind Speed (m/s)": data.get("windspeed_10m", []),
            "Wind Direction (°)": data.get("winddirection_10m", []),
            "Cloudiness (%)": data.get("cloudcover", []),
            "Visibility (m)": data.get("visibility", []),
            "Source": "Open-Meteo"
        })
        return df
    except requests.RequestException as e:
        print(f"❌ Historical fetch error for {city}: {e}")
        return pd.DataFrame()

# ---------------- FETCH LIVE (OpenWeather) ----------------
def fetch_live_openweather(city, lat, lon, retries=3):
    """
    Fetch current weather from OpenWeatherMap.
    Retries `retries` times on network errors.
    Returns single-row DataFrame or empty DataFrame on failure.
    """
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OWM_API_KEY}&units=metric"
    attempt = 0
    while attempt < retries:
        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            data = r.json()
            ts = datetime.utcfromtimestamp(data.get("dt", int(datetime.utcnow().timestamp())))
            row = {
                "City": city,
                "Latitude": lat,
                "Longitude": lon,
                "Timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "Temperature (°C)": data.get("main", {}).get("temp"),
                "Humidity (%)": data.get("main", {}).get("humidity"),
                "Pressure (hPa)": data.get("main", {}).get("pressure"),
                "Wind Speed (m/s)": data.get("wind", {}).get("speed"),
                "Wind Direction (°)": data.get("wind", {}).get("deg"),
                "Cloudiness (%)": data.get("clouds", {}).get("all"),
                "Visibility (m)": data.get("visibility"),
                "Source": "OpenWeather"
            }
            return pd.DataFrame([row])
        except requests.RequestException as e:
            attempt += 1
            wait = 2 ** attempt
            print(f"⚠ Live fetch attempt {attempt} failed for {city}: {e} — retrying in {wait}s")
            time.sleep(wait)
    print(f"❌ Live fetch failed for {city} after {retries} attempts.")
    return pd.DataFrame()

# ---------------- MAIN FLOW ----------------
def collect_one_time_historical(days=5):
    """Fetch historical for all cities once and save to CSV_HIST (no duplicates)."""
    frames = []
    for city, (lat, lon) in CITIES.items():
        df = fetch_historical_openmeteo(city, lat, lon, days=days)
        if not df.empty:
            frames.append(df)
    if frames:
        all_hist = pd.concat(frames, ignore_index=True)
        inserted = append_csv_no_dup(all_hist, CSV_HIST)
        print(f"✅ Historical saved: {CSV_HIST} (new rows: {inserted})")
    else:
        print("⚠ No historical data collected.")

def run_live_for_duration(run_minutes=RUN_MINUTES, interval_seconds=LIVE_INTERVAL_SECONDS):
    """
    Run live fetch loop for `run_minutes` minutes; fetch every `interval_seconds`.
    Each iteration fetches live data for all cities and appends to CSV_LIVE.
    """
    end_time = datetime.now() + timedelta(minutes=run_minutes)
    iteration = 0
    print(f"\n⏳ Starting live collection for {run_minutes} minutes — interval {interval_seconds}s\n")
    while datetime.now() < end_time:
        iteration += 1
        start_iter = datetime.now()
        print(f"[{start_iter.strftime('%Y-%m-%d %H:%M:%S')}] Iteration {iteration}: fetching live data for {len(CITIES)} cities...")
        live_frames = []
        for city, (lat, lon) in CITIES.items():
            df_live = fetch_live_openweather(city, lat, lon)
            if not df_live.empty:
                live_frames.append(df_live)
        if live_frames:
            all_live = pd.concat(live_frames, ignore_index=True)
            inserted = append_csv_no_dup(all_live, CSV_LIVE)
            print(f" → Inserted {inserted} new live rows into {CSV_LIVE}")
        else:
            print(" → No live rows fetched this iteration.")
        # compute how long to sleep to maintain roughly the interval (simple)
        elapsed = (datetime.now() - start_iter).total_seconds()
        sleep_for = interval_seconds - elapsed
        if sleep_for > 0:
            print(f"Sleeping for {int(sleep_for)}s...\n")
            time.sleep(sleep_for)
        else:
            print("Next iteration starting immediately (fetch took longer than interval).\n")
    print("✅ Completed live collection run.\n")

if __name__ == "__main__":
    # 1) Fetch & save historical once (5 days). Change days param if needed.
    collect_one_time_historical(days=5)

    # 2) Run live collection for RUN_MINUTES (default 30) with given interval
    run_live_for_duration(run_minutes=RUN_MINUTES, interval_seconds=LIVE_INTERVAL_SECONDS)

    print("All done. Files:")
    print(" - Historical CSV:", os.path.abspath(CSV_HIST))
    print(" - Live CSV:", os.path.abspath(CSV_LIVE))
