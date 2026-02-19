import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

AQI_TOKEN = os.getenv("AQI_TOKEN")


def get_live_aqi(lat: float, lon: float):
    """
    Fetch live AQI data from AQICN API using latitude & longitude.
    Returns cleaned dictionary or None if failed.
    """

    if not AQI_TOKEN:
        raise ValueError("AQI_TOKEN not found in environment variables.")

    url = f"https://api.waqi.info/feed/geo:{lat};{lon}/?token={AQI_TOKEN}"

    try:
        response = requests.get(url, timeout=10)
        data = response.json()

        if data.get("status") != "ok":
            return None

        aqi_data = data["data"]

        return {
            "aqi": aqi_data.get("aqi"),
            "city": aqi_data.get("city", {}).get("name"),
            "pm25": aqi_data.get("iaqi", {}).get("pm25", {}).get("v"),
            "pm10": aqi_data.get("iaqi", {}).get("pm10", {}).get("v"),
            "no2": aqi_data.get("iaqi", {}).get("no2", {}).get("v"),
            "o3": aqi_data.get("iaqi", {}).get("o3", {}).get("v"),
            "so2": aqi_data.get("iaqi", {}).get("so2", {}).get("v"),
            "co": aqi_data.get("iaqi", {}).get("co", {}).get("v"),
        }

    except requests.RequestException:
        return None

def get_user_location_by_ip():
    """
    Detect user location using IP address.
    Returns latitude, longitude and city name.
    """

    try:
        response = requests.get("http://ip-api.com/json/", timeout=5)
        data = response.json()

        if data.get("status") == "success":
            return {
                "latitude": data.get("lat"),
                "longitude": data.get("lon"),
                "city": data.get("city")
            }
        else:
            return None

    except requests.RequestException:
        return None
