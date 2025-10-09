
import requests
import pandas as pd
from datetime import datetime, timedelta
import os

# ----------------- CONFIG -----------------
OWM_API_KEY = "ab1c6a5cd00c250e7f9621ba1ef2ed67"  # OpenWeatherMap API key

# ----------------- STATIONS (50 Points) -----------------
stations = {
    # Indore (13)
    "Indore - MG Road": (22.718, 75.847),
    "Indore - Vijay Nagar": (22.735, 75.88),
    "Indore - AB Road": (22.726, 75.889),
    "Indore - Rau": (22.74, 75.83),
    "Indore - Palasia": (22.72, 75.88),
    "Indore - Rajwada": (22.7196, 75.855),
    "Indore - Airport Road": (22.72, 75.80),
    "Indore - Bhawarkua": (22.72, 75.84),
    "Indore - Annapurna": (22.70, 75.86),
    "Indore - Sanwer Road Industrial": (22.79, 75.84),
    "Indore - Ring Road": (22.74, 75.88),
    "Indore - CAT Road": (22.78, 75.86),
    "Indore - Mhow Naka": (22.69, 75.87),

    # Bhopal (13)
    "Bhopal - MP Nagar": (23.238, 77.4125),
    "Bhopal - Habibganj": (23.26, 77.398),
    "Bhopal - Arera Hills": (23.251, 77.412),
    "Bhopal - Ayodhya Bypass": (23.267, 77.435),
    "Bhopal - Lalghati": (23.285, 77.383),
    "Bhopal - Kolar Road": (23.20, 77.43),
    "Bhopal - Bairagarh": (23.29, 77.36),
    "Bhopal - Govindpura Industrial": (23.26, 77.47),
    "Bhopal - Misrod": (23.20, 77.49),
    "Bhopal - BHEL Area": (23.27, 77.49),
    "Bhopal - Shahpura": (23.21, 77.43),
    "Bhopal - Piplani": (23.25, 77.48),
    "Bhopal - Ashoka Garden": (23.27, 77.42),

    # Ujjain (12)
    "Ujjain - Freeganj": (23.18, 75.78),
    "Ujjain - Nanakheda": (23.16, 75.79),
    "Ujjain - Mahakal": (23.18, 75.77),
    "Ujjain - Dewas Gate": (23.19, 75.79),
    "Ujjain - Vikram Nagar": (23.20, 75.82),
    "Ujjain - Engineering College Road": (23.21, 75.82),
    "Ujjain - Agar Road": (23.21, 75.78),
    "Ujjain - Indore Road": (23.18, 75.81),
    "Ujjain - Industrial Area": (23.22, 75.80),
    "Ujjain - Railway Station": (23.19, 75.79),
    "Ujjain - Jaisinghpura": (23.20, 75.79),
    "Ujjain - Gopal Mandir": (23.18, 75.79),

    # Dewas (12)
    "Dewas - City Center": (22.97, 76.05),
    "Dewas - Industrial Area": (22.95, 76.07),
    "Dewas - Uday Nagar": (23.01, 76.09),
    "Dewas - Station Road": (22.97, 76.06),
    "Dewas - A.B. Road": (22.96, 76.07),
    "Dewas - Rajendra Nagar": (22.98, 76.05),
    "Dewas - Bavlia": (22.95, 76.10),
    "Dewas - Bhopal Road": (22.99, 76.08),
    "Dewas - Meera Nagar": (22.96, 76.04),
    "Dewas - Kailash Nagar": (22.95, 76.05),
    "Dewas - Hatpipliya Road": (22.93, 76.07),
    "Dewas - Kalani Nagar": (22.97, 76.09),
}


# ----------------- FUNCTIONS -----------------
def fetch_owm_historical(stations, days=5):
    """Fetch OWM air pollution historical data for N days (time series)."""
    base_url = "http://api.openweathermap.org/data/2.5/air_pollution/history"
    all_data = []

    end_time = int(datetime.utcnow().timestamp())
    start_time = int((datetime.utcnow() - timedelta(days=days)).timestamp())

    for station, (lat, lon) in stations.items():
        params = {"lat": lat, "lon": lon, "start": start_time, "end": end_time, "appid": OWM_API_KEY}
        print(f"📡 Fetching history for {station} ...")
        resp = requests.get(base_url, params=params)

        if resp.status_code != 200:
            print(f"❌ Error {resp.status_code} for {station}: {resp.text}")
            continue

        for item in resp.json().get("list", []):
            ts = datetime.utcfromtimestamp(item["dt"]).strftime("%Y-%m-%d %H:%M:%S")
            components = item.get("components", {})
            all_data.append({
                "Station": station, "Latitude": lat, "Longitude": lon,
                "Timestamp": ts,
                "CO": components.get("co"), "NO": components.get("no"), "NO2": components.get("no2"),
                "O3": components.get("o3"), "SO2": components.get("so2"),
                "PM2_5": components.get("pm2_5"), "PM10": components.get("pm10"),
                "NH3": components.get("nh3"),
                "AQI": item.get("main", {}).get("aqi"),
                "Source": "OpenWeatherMap-History"
            })

    df = pd.DataFrame(all_data)
    if not df.empty:
        df.to_csv("owm_timeseries.csv", index=False, encoding="utf-8")
        print(f"✅ Historical data saved with {len(df)} rows → owm_timeseries.csv")
    else:
        print("⚠ No historical data found!")
    return df


def fetch_owm_hourly(stations):
    """Fetch last 1 hour AQI data and append to CSV"""
    base_url = "http://api.openweathermap.org/data/2.5/air_pollution/history"
    end_time = int(datetime.utcnow().timestamp())
    start_time = end_time - 3600
    all_data = []

    for station, (lat, lon) in stations.items():
        params = {"lat": lat, "lon": lon, "start": start_time, "end": end_time, "appid": OWM_API_KEY}
        print(f"📡 Fetching hourly data for {station} ...")
        resp = requests.get(base_url, params=params)

        if resp.status_code != 200:
            print(f"❌ Error {resp.status_code} for {station}: {resp.text}")
            continue

        for item in resp.json().get("list", []):
            ts = datetime.utcfromtimestamp(item["dt"]).strftime("%Y-%m-%d %H:%M:%S")
            components = item.get("components", {})
            all_data.append({
                "Station": station, "Latitude": lat, "Longitude": lon,
                "Timestamp": ts,
                "CO": components.get("co"), "NO": components.get("no"), "NO2": components.get("no2"),
                "O3": components.get("o3"), "SO2": components.get("so2"),
                "PM2_5": components.get("pm2_5"), "PM10": components.get("pm10"),
                "NH3": components.get("nh3"),
                "AQI": item.get("main", {}).get("aqi"),
                "Source": "OpenWeatherMap-Hourly"
            })

    if all_data:
        df = pd.DataFrame(all_data)
        file_exists = os.path.isfile("owm_timeseries.csv")

        df.to_csv("owm_timeseries.csv", mode="a", header=not file_exists, index=False, encoding="utf-8")
        print(f"✅ Hourly data appended with {len(df)} rows")
        return df
    else:
        print("⚠ No hourly data found!")
        return pd.DataFrame()


# ----------------- MAIN -----------------
if __name__ == "_main_":
    df_hist = fetch_owm_historical(stations, days=5)
    print(df_hist.head())
    df_hour = fetch_owm_hourly(stations)
    print(df_hour.head())
