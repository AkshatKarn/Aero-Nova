import requests
import pandas as pd
import time
from datetime import datetime, timedelta

API_KEY = "Wrm8quHpGYlr2O68g13DQmWJYp6r5Ysj"

# City coordinates
CITIES = {
    "Indore": (22.7196, 75.8577),
    "Dewas": (22.9676, 76.0534),
    "Ujjain": (23.1765, 75.7885),
    "Bhopal": (23.2599, 77.4126)
}

def fetch_data(city, lat, lon):
    url = f"https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
    params = {"point": f"{lat},{lon}", "key": API_KEY}
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()["flowSegmentData"]
            return {
                "city": city,
                "latitude": lat,
                "longitude": lon,
                "timestamp": datetime.now(),
                "currentSpeed": data.get("currentSpeed"),
                "freeFlowSpeed": data.get("freeFlowSpeed"),
                "currentTravelTime": data.get("currentTravelTime"),
                "freeFlowTravelTime": data.get("freeFlowTravelTime"),
                "confidence": data.get("confidence"),
                "roadClosure": data.get("roadClosure"),
            }
    except Exception as e:
        print(f"Error for {city}: {e}")
    return None

def simulate_time_series(duration_minutes=60, interval_seconds=60):
    end_time = datetime.now() + timedelta(minutes=duration_minutes)
    all_data = []

    print("Collecting data...")
    while datetime.now() < end_time:
        for city, (lat, lon) in CITIES.items():
            record = fetch_data(city, lat, lon)
            if record:
                all_data.append(record)
                print(f"{datetime.now().strftime('%H:%M:%S')} | {city} data added")
        time.sleep(interval_seconds)  # Wait before next fetch

    df = pd.DataFrame(all_data)
    df.to_csv("traffic_timeseries.csv", index=False)
    print("✅ Bulk historical-like traffic dataset saved as traffic_timeseries.csv")

if __name__ == "__main__":
    simulate_time_series(duration_minutes=30, interval_seconds=30)
