import requests
import pandas as pd
from datetime import datetime, timedelta
import os

# ---------------- CONFIG ----------------
OWM_API_KEY = "ab1c6a5cd00c250e7f9621ba1ef2ed67"

CSV_HIST = "historical_weather.csv"
CSV_LIVE = "live_weather.csv"

CITIES = {
    "Indore": (22.7196, 75.8577),
    "Bhopal": (23.2599, 77.4126),
    "Ujjain": (23.1765, 75.7885),
    "Dewas": (22.9676, 76.0534)
}

# ---------------- CLEAN HELPER ----------------
def clean_df(df):
    """Ensure consistent column format"""
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    return df

def save_csv(df_new, csv_path):
    """Append to CSV with no duplicates"""
    if df_new.empty:
        return 0

    df_new = clean_df(df_new)

    if os.path.exists(csv_path):
        old = pd.read_csv(csv_path, parse_dates=["Timestamp"])
        old = clean_df(old)

        combined = pd.concat([old, df_new], ignore_index=True)

        # remove duplicate rows (City + Timestamp)
        combined = combined.drop_duplicates(subset=["City", "Timestamp"], keep="last")

        combined.to_csv(csv_path, index=False)
        return len(combined) - len(old)
    else:
        df_new.to_csv(csv_path, index=False)
        return len(df_new)

# ---------------- OPEN-METEO (Historical) ----------------
def fetch_historical_weather(city, lat, lon, days=5):
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

    print(f"📡 Fetching historical for {city}...")
    r = requests.get(url)
    if r.status_code != 200:
        print(f"❌ Error: {r.text}")
        return pd.DataFrame()

    data = r.json().get("hourly", {})

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame({
        "City": city,
        "Latitude": lat,
        "Longitude": lon,
        "Timestamp": data["time"],
        "Temperature (°C)": data["temperature_2m"],
        "Humidity (%)": data["relative_humidity_2m"],
        "Pressure (hPa)": data["pressure_msl"],
        "Wind Speed (m/s)": data["windspeed_10m"],
        "Wind Direction (°)": data["winddirection_10m"],
        "Cloudiness (%)": data["cloudcover"],
        "Visibility (m)": data["visibility"],
        "Source": "Open-Meteo"
    })

    return df

# ---------------- OPENWEATHER (Live) ----------------
def fetch_live_weather(city, lat, lon):
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OWM_API_KEY}&units=metric"
    r = requests.get(url)

    if r.status_code != 200:
        print(f"❌ Live fetch failed for {city}")
        return pd.DataFrame()

    data = r.json()
    ts = datetime.utcfromtimestamp(data["dt"]).strftime("%Y-%m-%d %H:%M:%S")

    df = pd.DataFrame([{
        "City": city,
        "Latitude": lat,
        "Longitude": lon,
        "Timestamp": ts,
        "Temperature (°C)": data["main"]["temp"],
        "Humidity (%)": data["main"]["humidity"],
        "Pressure (hPa)": data["main"]["pressure"],
        "Wind Speed (m/s)": data["wind"]["speed"],
        "Wind Direction (°)": data["wind"].get("deg"),
        "Cloudiness (%)": data["clouds"]["all"],
        "Visibility (m)": data.get("visibility"),
        "Source": "OpenWeather"
    }])

    return df

# ---------------- MAIN ----------------
def main():
    hist_frames = []
    live_frames = []

    # Fetch historical for all cities
    for city, (lat, lon) in CITIES.items():
        df = fetch_historical_weather(city, lat, lon, days=5)
        if not df.empty:
            hist_frames.append(df)

    # Fetch live for all cities
    for city, (lat, lon) in CITIES.items():
        print(f"🌤 Fetching live for {city}...")
        df = fetch_live_weather(city, lat, lon)
        if not df.empty:
            live_frames.append(df)

    # Combine dataframes
    hist_all = pd.concat(hist_frames, ignore_index=True) if hist_frames else pd.DataFrame()
    live_all = pd.concat(live_frames, ignore_index=True) if live_frames else pd.DataFrame()

    # Save them separately
    hist_count = save_csv(hist_all, CSV_HIST)
    live_count = save_csv(live_all, CSV_LIVE)

    print(f"✅ Historical updated → {CSV_HIST} (new rows: {hist_count})")
    print(f"🌤 Live updated → {CSV_LIVE} (new rows: {live_count})")

# ---------------- RUN ----------------
if __name__ == "__main__":
    main()
