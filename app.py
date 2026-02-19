import streamlit as st
import folium
from streamlit_folium import st_folium
from services.aqi_service import get_live_aqi, get_user_location_by_ip
from utils.aqi_category import get_category_with_color

st.set_page_config(page_title="Aero-Nova | Live AQI", layout="wide")

st.title("🌍 Aero-Nova Live Air Quality Monitor")

# ---------------------------------------------------
# STEP 1: Detect IP Location
# ---------------------------------------------------

with st.spinner("Detecting your location..."):
    location = get_user_location_by_ip()

if not location:
    st.error("Unable to detect location.")
    st.stop()

default_lat = location["latitude"]
default_lon = location["longitude"]
city = location["city"]

st.success(f"📍 Approx Location Detected: {city}")

# ---------------------------------------------------
# STEP 2: Render Map Centered on IP
# ---------------------------------------------------

st.subheader("🗺 Adjust Your Location (Click on Map)")

m = folium.Map(location=[default_lat, default_lon], zoom_start=12)

folium.Marker(
    [default_lat, default_lon],
    tooltip="Detected Location",
    icon=folium.Icon(color="blue")
).add_to(m)

map_data = st_folium(m, width=900, height=500)

# ---------------------------------------------------
# STEP 3: Capture Clicked Location
# ---------------------------------------------------

if map_data and map_data.get("last_clicked"):

    latitude = map_data["last_clicked"]["lat"]
    longitude = map_data["last_clicked"]["lng"]

    st.info(f"📍 Selected Location: {latitude:.4f}, {longitude:.4f}")

else:
    latitude = default_lat
    longitude = default_lon

# ---------------------------------------------------
# STEP 4: Fetch AQI
# ---------------------------------------------------

if st.button("Get AQI for Selected Location"):

    with st.spinner("Fetching live AQI..."):
        data = get_live_aqi(latitude, longitude)

    if not data:
        st.error("Failed to fetch AQI.")
        st.stop()

    aqi_value = data["aqi"]
    category, color = get_category_with_color(aqi_value)

    st.markdown(
        f"<h1 style='color:{color}; font-size:80px; text-align:center;'>{aqi_value}</h1>",
        unsafe_allow_html=True
    )

    st.markdown(
        f"<h3 style='color:{color}; text-align:center;'>{category}</h3>",
        unsafe_allow_html=True
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    col1.metric("PM2.5", data.get("pm25"))
    col2.metric("PM10", data.get("pm10"))
    col3.metric("NO2", data.get("no2"))