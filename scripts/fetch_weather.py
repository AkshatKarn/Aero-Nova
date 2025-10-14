import requests
import pandas as pd
from datetime import datetime, timedelta

# ---------------- CONFIG ----------------
OWM_API_KEY = "ab1c6a5cd00c250e7f9621ba1ef2ed67"   # OpenWeather API Key
VC_API_KEY = "QBTAZGQ6EEZZHXR6P9Q2F9PPR"           # VisualCrossing API Key

CITIES = {
    "Indore": (22.7196, 75.8577),
    "Bhopal": (23.2599, 77.4126),
    "Ujjain": (23.1793, 75.7849),
    "Dewas": (22.9659, 76.0553)
}

CSV_FILE = "weather_data_timeseries.csv"

# ---------------- VISUAL CROSSING (Historical) ----------------
def fetch_historical_weather(city, lat, lon, days=5):
    start_date = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    end_date = datetime.utcnow().strftime("%Y-%m-%d")
    
    url = (
        f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/"
        f"{lat},{lon}/{start_date}/{end_date}"
        f"?unitGroup=metric&key={VC_API_KEY}&include=hours&contentType=json"
    )
    
    print(f"📡 Fetching historical weather for {city} ({start_date} → {end_date})")
    r = requests.get(url)
    if r.status_code != 200:
        print(f"❌ Error {r.status_code} for {city}: {r.text}")
        return pd.DataFrame()
    
    data = r.json()
    rows = []
    for day in data.get("days", []):
        for hour in day.get("hours", []):
            rows.append({
                "City": city,
                "Latitude": lat,
                "Longitude": lon,
                "Timestamp": f"{day.get('datetime')} {hour.get('datetime')}",
                "Temperature (°C)": hour.get("temp"),
                "Feels Like (°C)": hour.get("feelslike"),
                "Humidity (%)": hour.get("humidity"),
                "Pressure (hPa)": hour.get("pressure"),
                "Wind Speed (m/s)": hour.get("windspeed"),
                "Wind Direction (°)": hour.get("winddir"),
                "Cloudiness (%)": hour.get("cloudcover"),
                "Visibility (km)": hour.get("visibility"),
                "Conditions": hour.get("conditions"),
                "Source": "VisualCrossing"
            })
    return pd.DataFrame(rows)

# ---------------- OPENWEATHER (Live) ----------------
def fetch_live_weather(city, lat, lon):
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OWM_API_KEY}&units=metric"
    r = requests.get(url)
    if r.status_code != 200:
        print(f"❌ Live fetch failed for {city}: {r.text}")
        return pd.DataFrame()
    
    data = r.json()
    ts = datetime.utcfromtimestamp(data["dt"]).strftime("%Y-%m-%d %H:%M:%S")
    
    row = {
        "City": city,
        "Latitude": lat,
        "Longitude": lon,
        "Timestamp": ts,
        "Temperature (°C)": data["main"]["temp"],
        "Feels Like (°C)": data["main"]["feels_like"],
        "Humidity (%)": data["main"]["humidity"],
        "Pressure (hPa)": data["main"]["pressure"],
        "Wind Speed (m/s)": data["wind"]["speed"],
        "Wind Direction (°)": data["wind"]["deg"],
        "Cloudiness (%)": data["clouds"]["all"],
        "Visibility (m)": data.get("visibility", None),
        "Conditions": data["weather"][0]["main"],
        "Source": "OpenWeather"
    }
    return pd.DataFrame([row])

# ---------------- MAIN ----------------
def main():
    all_data = []

    # Step 1: Fetch Historical (Visual Crossing)
    for city, (lat, lon) in CITIES.items():
        hist_df = fetch_historical_weather(city, lat, lon, days=5)
        if not hist_df.empty:
            all_data.append(hist_df)

    # Step 2: Fetch Live (OpenWeather)
    for city, (lat, lon) in CITIES.items():
        print(f"🌤 Fetching live weather for {city} ...")
        live_df = fetch_live_weather(city, lat, lon)
        if not live_df.empty:
            all_data.append(live_df)

    # Step 3: Save to CSV
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        try:
            old_df = pd.read_csv(CSV_FILE)
            final_df = pd.concat([old_df, final_df], ignore_index=True)
        except FileNotFoundError:
            pass

        final_df.to_csv(CSV_FILE, index=False)
        print(f"✅ Weather data (History + Live) saved → {CSV_FILE}")
    else:
        print("⚠ No data fetched.")

if __name__ == "__main__":
    main()
