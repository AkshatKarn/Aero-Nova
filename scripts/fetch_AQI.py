import requests
import pandas as pd
from datetime import datetime
import os

# ================= CONFIG =================
OWM_API_KEY = os.getenv("5759ac0332f5d274da97a9a072f16664") or "5759ac0332f5d274da97a9a072f16664"

# ================= STATIONS =================
stations = {
    "Indore - MG Road": (22.718, 75.847),
    "Indore - Vijay Nagar": (22.735, 75.88),
    "Indore - Rajwada": (22.7196, 75.855),
    "Bhopal - MP Nagar": (23.238, 77.4125),
    "Bhopal - Habibganj": (23.26, 77.398),
    "Ujjain - Mahakal": (23.18, 75.77),
    "Dewas - City Center": (22.97, 76.05),
}

# ================= AQI FUNCTIONS =================
def calc_aqi(cp, breakpoints):
    for c_low, c_high, i_low, i_high in breakpoints:
        if c_low <= cp <= c_high:
            return round(((i_high - i_low) / (c_high - c_low)) * (cp - c_low) + i_low)
    return None

PM25_BP = [(0,12,0,50),(12.1,35.4,51,100),(35.5,55.4,101,150),
           (55.5,150.4,151,200),(150.5,250.4,201,300),
           (250.5,350.4,301,400),(350.5,500.4,401,500)]

PM10_BP = [(0,54,0,50),(55,154,51,100),(155,254,101,150),
           (255,354,151,200),(355,424,201,300),
           (425,504,301,400),(505,604,401,500)]

NO2_BP = [(0,53,0,50),(54,100,51,100),(101,360,101,150),
          (361,649,151,200),(650,1249,201,300)]

SO2_BP = [(0,35,0,50),(36,75,51,100),(76,185,101,150),
          (186,304,151,200),(305,604,201,300)]

O3_BP = [(0,54,0,50),(55,70,51,100),(71,85,101,150),
         (86,105,151,200),(106,200,201,300)]

# ================= SAVE =================
def save_csv(df, file):
    df.to_csv(file, mode="a", header=not os.path.exists(file), index=False)

# ================= FETCH =================
def fetch_data(live=True, days=5):
    if live:
        url = "http://api.openweathermap.org/data/2.5/air_pollution"
    else:
        url = "http://api.openweathermap.org/data/2.5/air_pollution/history"

    rows = []
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    for station, (lat, lon) in stations.items():
        params = {"lat": lat, "lon": lon, "appid": OWM_API_KEY}
        if not live:
            end = int(datetime.utcnow().timestamp())
            start = end - days * 86400
            params.update({"start": start, "end": end})

        r = requests.get(url, params=params)
        if r.status_code != 200:
            continue

        data_list = r.json().get("list", [])
        for item in data_list:
            c = item["components"]

            pm25 = c.get("pm2_5")
            pm10 = c.get("pm10")
            no2 = c.get("no2")
            so2 = c.get("so2")
            o3 = c.get("o3")

            aqi_pm25 = calc_aqi(pm25, PM25_BP) if pm25 else None
            aqi_pm10 = calc_aqi(pm10, PM10_BP) if pm10 else None
            aqi_no2 = calc_aqi(no2, NO2_BP) if no2 else None
            aqi_so2 = calc_aqi(so2, SO2_BP) if so2 else None
            aqi_o3 = calc_aqi(o3, O3_BP) if o3 else None

            final_aqi = max(filter(None, [
                aqi_pm25, aqi_pm10, aqi_no2, aqi_so2, aqi_o3
            ]))

            rows.append([
                station, lat, lon, now,
                c.get("co"), c.get("no"), no2, o3, so2,
                pm25, pm10, c.get("nh3"),
                aqi_pm25, aqi_pm10, aqi_no2, aqi_so2, aqi_o3,
                final_aqi,
                "Live" if live else "Historical"
            ])

    cols = [
        "Station","Lat","Lon","Timestamp",
        "CO","NO","NO2","O3","SO2","PM2_5","PM10","NH3",
        "PM25_AQI","PM10_AQI","NO2_AQI","SO2_AQI","O3_AQI",
        "FINAL_AQI","Type"
    ]

    df = pd.DataFrame(rows, columns=cols)
    save_csv(df, "aqi_live.csv" if live else "aqi_historical.csv")
    print("✅ Data saved:", len(df))

# ================= RUN =================
if __name__ == "__main__":
    fetch_data(live=False, days=5)
    fetch_data(live=True)
