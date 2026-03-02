import streamlit as st
import pydeck as pdk
from geopy.geocoders import Nominatim
import sys
import os
import plotly.graph_objects as go
import datetime

# Ensure project root in path
sys.path.append(os.path.abspath("."))

from services.openweather_service import (
    get_raw_pollution_data,
    get_historical_pollution_data
)
from services.aqi_service import get_user_location_by_ip
from services.db_service import save_aqi_data
from utils.aqi_category import get_category_with_color
from utils.indian_aqi_calculator import calculate_indian_aqi

st.set_page_config(page_title="Aero-Nova | Live AQI", layout="wide")

st.title("🌍 Aero-Nova Live Air Quality Monitor")

# ---------------------------------------------------
# STEP 1: Detect IP Location
# ---------------------------------------------------

if "user_lat" not in st.session_state:

    with st.spinner("Detecting your location..."):
        try:
            ip_location = get_user_location_by_ip()
        except:
            ip_location = None

    if ip_location and ip_location.get("latitude") and ip_location.get("longitude"):
        st.session_state.user_lat = ip_location.get("latitude")
        st.session_state.user_lon = ip_location.get("longitude")
        st.success("📍 Location detected successfully.")
    else:
        st.error("❌ Can't fetch location. Please enter manually.")
        st.stop()

# ---------------------------------------------------
# STEP 2: Manual Address Search
# ---------------------------------------------------

st.subheader("🔎 Search Location")

from geopy.exc import GeocoderTimedOut, GeocoderUnavailable

address_input = st.text_input("Enter Address, Area or City", key="address_input")

if address_input and st.session_state.get("last_searched") != address_input:
    try:
        geolocator = Nominatim(user_agent="aero_nova_app", timeout=10)
        location = geolocator.geocode(address_input)

        if location:
            st.session_state.user_lat = location.latitude
            st.session_state.user_lon = location.longitude
            st.session_state.last_searched = address_input
            st.success(f"📍 Found: {location.address}")
        else:
            st.warning("Address not found.")

    except (GeocoderTimedOut, GeocoderUnavailable):
        st.warning("⚠ Location service temporarily unavailable.")

        
# ---------------------------------------------------
# STEP 3: Final Coordinates
# ---------------------------------------------------

final_lat = st.session_state.user_lat
final_lon = st.session_state.user_lon

# ---------------------------------------------------
# STEP 4: Map View
# ---------------------------------------------------

with st.expander("🗺 Adjust Location Manually (Optional)"):

    view_state = pdk.ViewState(
        latitude=final_lat,
        longitude=final_lon,
        zoom=14,
        pitch=0,
    )

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=[{"lat": final_lat, "lon": final_lon}],
        get_position="[lon, lat]",
        get_radius=200,
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
# STEP 5: Fetch AQI + Show Trend
# ---------------------------------------------------

if st.button("Get AQI for Selected Location"):

    with st.spinner("Fetching pollution data..."):
        try:
            raw_data = get_raw_pollution_data(final_lat, final_lon)
        except:
            raw_data = None

    if not raw_data:
        st.error("❌ Failed to fetch pollution data.")
        st.stop()

    # 🔥 FULL INDIAN AQI ENGINE
    aqi_value, dominant, breakdown = calculate_indian_aqi({
        "pm25": raw_data.get("pm25"),
        "pm10": raw_data.get("pm10"),
        "no2": raw_data.get("no2"),
        "o3": raw_data.get("o3"),
        "so2": raw_data.get("so2"),
        "co": raw_data.get("co"),
    })

    if aqi_value is None:
        st.error("Unable to calculate AQI.")
        st.stop()

    category, color = get_category_with_color(aqi_value)

    # Save to DB
    save_aqi_data(
        {
            "aqi": aqi_value,
            "pm25": raw_data.get("pm25"),
            "pm10": raw_data.get("pm10"),
            "no2": raw_data.get("no2"),
            "o3": raw_data.get("o3"),
            "so2": raw_data.get("so2"),
            "co": raw_data.get("co"),
        },
        final_lat,
        final_lon,
        "Unknown"
    )

    # -------- Display AQI --------
    st.markdown(
        f"<h1 style='color:{color}; font-size:90px; text-align:center;'>{aqi_value}</h1>",
        unsafe_allow_html=True
    )

    st.markdown(
        f"<h3 style='color:{color}; text-align:center;'>Indian AQI (CPCB Standard) - {category}</h3>",
        unsafe_allow_html=True
    )

    st.write(f"### Dominant Pollutant: {dominant.upper()}")

    st.subheader("📊 Pollutant Sub-Index Breakdown")

    cols = st.columns(len(breakdown))

    for col, (pollutant, value) in zip(cols, breakdown.items()):
        col.metric(pollutant.upper(), value)

    st.divider()

    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)

    col1.metric("PM2.5", raw_data.get("pm25") or "N/A")
    col2.metric("PM10", raw_data.get("pm10") or "N/A")
    col3.metric("NO2", raw_data.get("no2") or "N/A")

    col4.metric("O3", raw_data.get("o3") or "N/A")
    col5.metric("SO2", raw_data.get("so2") or "N/A")
    col6.metric("CO", raw_data.get("co") or "N/A")

    # -------- Historical Trend --------
    st.subheader("📈 Last 24 Hours PM2.5 Trend")

    history = get_historical_pollution_data(final_lat, final_lon)

    if history:
        timestamps = []
        pm25_values = []

        for entry in history:
            dt = datetime.datetime.fromtimestamp(entry["dt"])
            pm25 = entry["components"]["pm2_5"]
            timestamps.append(dt)
            pm25_values.append(pm25)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=timestamps,
            y=pm25_values,
            mode='lines+markers',
            name='PM2.5'
        ))

        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="PM2.5 (µg/m³)",
            template="plotly_dark"
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Unable to fetch historical data.")