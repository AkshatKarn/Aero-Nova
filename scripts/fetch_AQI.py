import requests
import pandas as pd
from datetime import datetime, timedelta
import os

# ----------------- CONFIG -----------------
OWM_API_KEY = "ab1c6a5cd00c250e7f9621ba1ef2ed67"

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

# ----------------- SAVE FUNCTION -----------------
def save_csv(df, filename):
    exists = os.path.exists(filename)
    df.to_csv(filename, mode="a", header=not exists, index=False)

# ----------------- FETCH HISTORICAL -----------------
def fetch_historical(days=5):
    url = "http://api.openweathermap.org/data/2.5/air_pollution/history"
    end = int(datetime.utcnow().timestamp())
    start = end - days * 86400

    rows = []
    
    for station, (lat, lon) in stations.items():
        print(f"Historical → {station}")
        params = {"lat": lat, "lon": lon, "start": start, "end": end, "appid": OWM_API_KEY}
        r = requests.get(url, params=params)
        if r.status_code != 200:
            print(f"Error {r.status_code} @ {station}")
            continue

        for item in r.json().get("list", []):
            t = datetime.utcfromtimestamp(item["dt"]).strftime("%Y-%m-%d %H:%M:%S")
            c = item["components"]
            rows.append([station, lat, lon, t, *c.values(), item["main"]["aqi"], "Historical"])

    cols = ["Station","Lat","Lon","Timestamp","CO","NO","NO2","O3","SO2","PM2_5","PM10","NH3","AQI","Type"]
    df = pd.DataFrame(rows, columns=cols)
    save_csv(df, "aqi_historical_data.csv")
    print(f"✅ Historical saved ({len(df)} rows)")

# ----------------- FETCH LIVE -----------------
def fetch_live():
    url = "http://api.openweathermap.org/data/2.5/air_pollution"
    
    rows = []
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    for station, (lat, lon) in stations.items():
        print(f"Live → {station}")
        params = {"lat": lat, "lon": lon, "appid": OWM_API_KEY}
        r = requests.get(url, params=params)
        if r.status_code != 200:
            print(f"Error {r.status_code} @ {station}")
            continue

        item = r.json()["list"][0]
        c = item["components"]

        rows.append([station, lat, lon, now, *c.values(), item["main"]["aqi"], "Live"])
    
    cols = ["Station","Lat","Lon","Timestamp","CO","NO","NO2","O3","SO2","PM2_5","PM10","NH3","AQI","Type"]
    df = pd.DataFrame(rows, columns=cols)
    save_csv(df, "aqi_live_data.csv")
    print(f"✅ Live saved ({len(df)} rows)")

# ----------------- RUN -----------------
if __name__ == "__main__":
    fetch_historical(days=5)
    fetch_live()
