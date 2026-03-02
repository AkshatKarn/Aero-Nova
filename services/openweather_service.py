import requests
import os
import time
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("OPENWEATHER_API_KEY")

def get_raw_pollution_data(lat, lon):

    url = f"https://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"

    print("Loaded API Key:", API_KEY)
    print("Final URL:", url)
    
    response = requests.get(url)
    print("Status Code:", response.status_code)
    print("Response Text:", response.text)

    if response.status_code != 200:
        return None

    data = response.json()

    if "list" not in data or not data["list"]:
        return None

    components = data["list"][0]["components"]

    return {
        "pm25": components.get("pm2_5"),
        "pm10": components.get("pm10"),
        "no2": components.get("no2"),
        "o3": components.get("o3"),
        "so2": components.get("so2"),
        "co": components.get("co"),
    }



def get_historical_pollution_data(lat, lon):

    end = int(time.time())  # current time
    start = end - (24 * 60 * 60)  # last 24 hours

    url = f"https://api.openweathermap.org/data/2.5/air_pollution/history?lat={lat}&lon={lon}&start={start}&end={end}&appid={API_KEY}"

    response = requests.get(url)

    if response.status_code != 200:
        return None

    data = response.json()

    if "list" not in data:
        return None

    return data["list"]