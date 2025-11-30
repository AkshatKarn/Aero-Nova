#!/usr/bin/env python3
"""
fetch_openweather_and_save.py (robust)

Fetch hourly forecast from OpenWeather OneCall and save mapped columns to:
  data/live_processed/weather_live_master.csv

If LOCATION_NAME is not exact, script will try case-insensitive and substring matches.
"""
import os
import requests
import pandas as pd
from datetime import datetime, timezone

# ------------- CONFIG -------------
OPENWEATHER_KEY = "ab1c6a5cd00c250e7f9621ba1ef2ed67"

# Put whatever string you like here; script will try to match intelligently.
LOCATION_NAME = "Indore -Ujjain"

OUT_DIR = os.path.join("data", "live_processed")
OUT_PATH = os.path.join(OUT_DIR, "weather_live_master.csv")
os.makedirs(OUT_DIR, exist_ok=True)

LOCATIONS = {
    "Indore": (22.7196, 75.8577),
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

# ------------- helpers -------------
def match_location(name, locations):
    # exact match
    if name in locations:
        return name
    # case-insensitive exact
    for k in locations:
        if k.lower() == name.lower():
            return k
    # split words and try substring match (more flexible)
    parts = [p.strip().lower() for p in name.replace('-', ' ').split() if p.strip()]
    if parts:
        # prefer matches containing most parts
        scores = []
        for k in locations:
            kl = k.lower()
            score = sum(1 for p in parts if p in kl)
            if score > 0:
                scores.append((score, k))
        if scores:
            scores.sort(reverse=True)
            return scores[0][1]
    return None

def fetch_onecall_hourly(lat, lon, key, max_hours=48):
    if not key:
        raise RuntimeError("OPENWEATHER_KEY not provided.")
    url = (
        f"https://api.openweathermap.org/data/2.5/onecall"
        f"?lat={lat}&lon={lon}&exclude=minutely,daily,alerts,current&units=metric&appid={key}"
    )
    r = requests.get(url, timeout=12)
    r.raise_for_status()
    js = r.json()
    hours = js.get("hourly", [])
    rows = []
    for h in hours[:max_hours]:
        dt = datetime.fromtimestamp(h.get("dt"), tz=timezone.utc)
        rows.append({
            "timestamp": dt.isoformat(),
            "Temperature (°C)": h.get("temp"),
            "Humidity (%)": h.get("humidity"),
            "Pressure (hPa)": h.get("pressure"),
            "Wind Speed (m/s)": h.get("wind_speed"),
            "Wind Direction (°)": h.get("wind_deg"),
            "Cloudiness (%)": h.get("clouds")
        })
    return pd.DataFrame(rows)

def main():
    matched = match_location(LOCATION_NAME, LOCATIONS)
    if not matched:
        print("ERROR: LOCATION_NAME did not match any available keys.")
        print("You set:", LOCATION_NAME)
        print("Pick one of these exact keys or a substring that matches:")
        for k in sorted(LOCATIONS.keys()):
            print("  -", k)
        raise SystemExit(1)

    lat, lon = LOCATIONS[matched]
    print(f"Using location key: '{matched}' -> coords ({lat}, {lon})")
    print("Fetching OpenWeather hourly forecast ...")
    try:
        df = fetch_onecall_hourly(lat, lon, OPENWEATHER_KEY, max_hours=48)
    except Exception as e:
        raise SystemExit("OpenWeather fetch failed: " + str(e))

    if df.empty:
        raise SystemExit("No hourly data returned from OpenWeather.")
    df.to_csv(OUT_PATH, index=False)
    print("Saved hourly weather to:", OUT_PATH)
    print(df.head(6).to_string(index=False))

if __name__ == "__main__":
    main()
