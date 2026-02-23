import mysql.connector
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

def get_connection():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASS"),
        database=os.getenv("DB_NAME")
    )

def save_aqi_data(data, lat, lon):
    conn = get_connection()
    cursor = conn.cursor()

    timestamp_hour = datetime.now().replace(minute=0, second=0, microsecond=0)

    query = """
    INSERT INTO aqi_station_hourly
    (station_name, city, latitude, longitude, timestamp_hour,
     aqi, pm25, pm10, no2, o3, so2, co)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE
        aqi=VALUES(aqi),
        pm25=VALUES(pm25),
        pm10=VALUES(pm10),
        no2=VALUES(no2),
        o3=VALUES(o3),
        so2=VALUES(so2),
        co=VALUES(co);
    """

    values = (
        data.get("city"),
        data.get("city"),
        lat,
        lon,
        timestamp_hour,
        data.get("aqi"),
        data.get("pm25"),
        data.get("pm10"),
        data.get("no2"),
        data.get("o3"),
        data.get("so2"),
        data.get("co")
    )

    cursor.execute(query, values)
    conn.commit()

    cursor.close()
    conn.close()