import streamlit as st
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim

from services.aqi_service import get_live_aqi, get_user_location_by_ip
from utils.aqi_category import get_category_with_color

st.set_page_config(page_title="Aero-Nova | Live AQI", layout="wide")

st.title("🌍 Aero-Nova Live Air Quality Monitor")

# ---------------------------------------------------
# STEP 1: Detect IP Location
# ---------------------------------------------------

with st.spinner("Detecting your location..."):
    ip_location = get_user_location_by_ip()

if not ip_location:
    st.error("Unable to detect location.")
    st.stop()

default_lat = ip_location["latitude"]
default_lon = ip_location["longitude"]
city = ip_location["city"]

st.success(f"📍 Approx Location Detected: {city}")

# ---------------------------------------------------
# STEP 2: Address Search
# ---------------------------------------------------

st.subheader("🔎 Search Location")

address_input = st.text_input("Enter Address, Area or City")

search_lat = None
search_lon = None

if address_input:
    geolocator = Nominatim(user_agent="aero_nova_app")
    location = geolocator.geocode(address_input)

    if location:
        search_lat = location.latitude
        search_lon = location.longitude
        st.success(f"📍 Found: {location.address}")
    else:
        st.warning("Address not found. Try a more specific query.")

# ---------------------------------------------------
# STEP 3: Determine Map Center
# ---------------------------------------------------

if search_lat and search_lon:
    map_center = [search_lat, search_lon]
else:
    map_center = [default_lat, default_lon]

# ---------------------------------------------------
# STEP 4: Render Interactive Map
# ---------------------------------------------------

st.subheader("🗺 Adjust Location (Drag Marker or Click Map)")

m = folium.Map(location=map_center, zoom_start=14, control_scale=True)

# Add multiple tile layers
folium.TileLayer("cartodbpositron").add_to(m)
folium.TileLayer("Stamen Terrain").add_to(m)
folium.LayerControl().add_to(m)

# Draggable marker
marker = folium.Marker(
    location=map_center,
    draggable=True
)
marker.add_to(m)

map_data = st_folium(m, width=1000, height=500)

# ---------------------------------------------------
# STEP 5: Capture Final Location
# ---------------------------------------------------

if search_lat and search_lon:
    final_lat = search_lat
    final_lon = search_lon

elif map_data and map_data.get("last_clicked"):
    final_lat = map_data["last_clicked"]["lat"]
    final_lon = map_data["last_clicked"]["lng"]

elif map_data and map_data.get("last_object_clicked"):
    final_lat = map_data["last_object_clicked"]["lat"]
    final_lon = map_data["last_object_clicked"]["lng"]

else:
    final_lat = default_lat
    final_lon = default_lon

st.info(f"📍 Selected Coordinates: {final_lat:.5f}, {final_lon:.5f}")

# ---------------------------------------------------
# STEP 6: Fetch AQI
# ---------------------------------------------------

if st.button("Get AQI for Selected Location"):

    with st.spinner("Fetching live AQI..."):
        data = get_live_aqi(final_lat, final_lon)

    if not data:
        st.error("Failed to fetch AQI.")
        st.stop()

    aqi_value = data["aqi"]
    category, color = get_category_with_color(aqi_value)

    st.markdown(
        f"<h1 style='color:{color}; font-size:90px; text-align:center;'>{aqi_value}</h1>",
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
