import streamlit as st
import pydeck as pdk
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
    final_lat = search_lat
    final_lon = search_lon
else:
    final_lat = default_lat
    final_lon = default_lon

# ---------------------------------------------------
# STEP 4: Render Pydeck Map
# ---------------------------------------------------

st.subheader("🗺 Location Preview")

view_state = pdk.ViewState(
    latitude=final_lat,
    longitude=final_lon,
    zoom=16,
    pitch=0,
)

layer = pdk.Layer(
    "ScatterplotLayer",
    data=[{"lat": final_lat, "lon": final_lon}],
    get_position="[lon, lat]",
    get_radius=150,
    get_fill_color=[255, 0, 0],
)

deck = pdk.Deck(
    map_style="road",
    initial_view_state=view_state,
    layers=[layer],
)

st.pydeck_chart(deck)

st.info(f"📍 Selected Coordinates: {final_lat:.5f}, {final_lon:.5f}")

# ---------------------------------------------------
# STEP 5: Fetch AQI
# ---------------------------------------------------

if st.button("Get AQI for This Location"):

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
