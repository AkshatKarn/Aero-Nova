import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
from PIL import Image
import base64
from io import BytesIO

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Aero-Nova | AI-Driven AQI",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ======================================================
# PATH SETUP (ROBUST)
# ======================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

AQI_FILE = DATA_DIR / "aqi_live_data.csv"
WEATHER_FILE = DATA_DIR / "live_weather.csv"
TRAFFIC_FILE = DATA_DIR / "traffic_timeseries.csv"

# ======================================================
# LOAD LOGO
# ======================================================
def load_logo(path="logo.png"):
    img = Image.open(path).convert("RGBA")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

LOGO = load_logo(BASE_DIR / "logo.png")

# ======================================================
# LOAD DATA
# ======================================================
aqi_df = pd.read_csv(AQI_FILE, parse_dates=["Timestamp"])
weather_df = pd.read_csv(WEATHER_FILE, parse_dates=["Timestamp"])
traffic_df = pd.read_csv(TRAFFIC_FILE, parse_dates=["timestamp"])

# sort & latest rows
aqi_df = aqi_df.sort_values("Timestamp")
weather_df = weather_df.sort_values("Timestamp")
traffic_df = traffic_df.sort_values("timestamp")

latest_aqi = aqi_df.iloc[-1]
latest_weather = weather_df.iloc[-1]
latest_traffic = traffic_df.iloc[-1]

# ======================================================
# HERO + CSS
# ======================================================
HERO_BG = "https://images.unsplash.com/photo-1505842465776-3d0c1f6b2a4b?auto=format&fit=crop&w=2000&q=80"

st.markdown(f"""
<style>
* {{ font-family: 'Segoe UI', sans-serif; }}

.hero {{
    height: 90vh;
    background-image: url('{HERO_BG}');
    background-size: cover;
    background-position: center;
    display: flex;
    justify-content: center;
    align-items: center;
}}

.glass {{
    background: rgba(255,255,255,0.18);
    border-radius: 20px;
    padding: 40px 60px;
    border: 1px solid rgba(255,255,255,0.35);
    backdrop-filter: blur(14px);
    box-shadow: 0 8px 30px rgba(0,0,0,0.3);
    text-align: center;
}}

.section-title {{
    font-size: 36px;
    font-weight: 800;
    margin: 50px 0 20px 0;
}}

.card {{
    background: rgba(255,255,255,0.25);
    padding: 25px;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.4);
    backdrop-filter: blur(10px);
    box-shadow: 0 6px 20px rgba(0,0,0,0.25);
    text-align: center;
}}
</style>
""", unsafe_allow_html=True)

# ======================================================
# HERO SECTION
# ======================================================
st.markdown(f"""
<div class="hero">
  <div class="glass">
    <img src="data:image/png;base64,{LOGO}" width="200"><br><br>
    <h1 style="color:white;font-size:48px;font-weight:800;">
      AI-Driven Nowcasting of Air Quality & Mobility
    </h1>
    <p style="color:white;font-size:18px;">
      Real-time AQI • What-If Scenarios • Forecasting
    </p>
  </div>
</div>
""", unsafe_allow_html=True)

# ======================================================
# SECTION 1 — LIVE AQI
# ======================================================
st.markdown("<div class='section-title'>Live Air Quality</div>", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="card">
      <h3>Current AQI</h3>
      <div style="font-size:50px;font-weight:900;">{int(latest_aqi["AQI"])}</div>
      <p>{latest_aqi["Station"]}</p>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.metric("PM2.5 (µg/m³)", round(latest_aqi["PM2_5"], 1))

with c3:
    st.metric("PM10 (µg/m³)", round(latest_aqi["PM10"], 1))

with c4:
    st.metric("Traffic Speed (km/h)", round(latest_traffic["currentSpeed"], 1))

# ======================================================
# SECTION 2 — WEATHER
# ======================================================
st.markdown("<div class='section-title'>Current Weather</div>", unsafe_allow_html=True)

w1, w2, w3, w4 = st.columns(4)

w1.metric("Temperature (°C)", round(latest_weather["Temperature (°C)"], 1))
w2.metric("Humidity (%)", round(latest_weather["Humidity (%)"], 1))
w3.metric("Pressure (hPa)", round(latest_weather["Pressure (hPa)"], 1))
w4.metric("Wind Speed (m/s)", round(latest_weather["Wind Speed (m/s)"], 1))

# ======================================================
# SECTION 3 — WHAT-IF SCENARIO ENGINE
# ======================================================
st.markdown("<div class='section-title'>What-If Scenario Engine</div>", unsafe_allow_html=True)

colA, colB = st.columns(2)

with colA:
    traffic_change = st.slider("Traffic Change (%)", -50, 200, 0)
    industry_change = st.slider("Industrial Emissions (%)", -50, 200, 0)
    vehicle_change = st.slider("Vehicular Emissions (%)", -50, 200, 0)

with colB:
    humidity_impact = st.slider("Humidity Impact (%)", -50, 50, 0)
    wind_impact = st.slider("Wind Speed Impact (%)", -50, 50, 0)
    pm25_sensitivity = st.slider("PM2.5 Sensitivity", 1, 10, 5)
    pm10_sensitivity = st.slider("PM10 Sensitivity", 1, 10, 5)

def calculate_new_aqi(base):
    delta = (
        traffic_change * 0.3 +
        industry_change * 0.25 +
        vehicle_change * 0.2 -
        humidity_impact * 0.15 -
        wind_impact * 0.2 +
        pm25_sensitivity * 2 +
        pm10_sensitivity * 1.5
    ) / 40
    return max(0, int(base + delta))

if st.button("Calculate Scenario"):
    new_aqi = calculate_new_aqi(latest_aqi["AQI"])
    diff = new_aqi - latest_aqi["AQI"]

    st.subheader("Scenario Output")

    o1, o2 = st.columns(2)

    with o1:
        st.markdown(f"""
        <div class="card">
          <h3>New AQI</h3>
          <div style="font-size:46px;font-weight:900;">{new_aqi}</div>
          <p>Change: {diff:+}</p>
        </div>
        """, unsafe_allow_html=True)

    with o2:
        if new_aqi < 100:
            st.success("Air quality acceptable.")
        elif new_aqi < 200:
            st.warning("Sensitive groups should be cautious.")
        else:
            st.error("Poor air quality. Avoid outdoor activity.")

# ======================================================
# SECTION 4 — AQI TREND (CSV)
# ======================================================
st.markdown("<div class='section-title'>AQI Trend (Dataset)</div>", unsafe_allow_html=True)

trend_df = aqi_df.tail(200)

fig = px.line(
    trend_df,
    x="Timestamp",
    y="AQI",
    title="AQI Trend Over Time"
)
st.plotly_chart(fig, use_container_width=True)

# ======================================================
# FOOTER
# ======================================================
st.markdown("""
<hr>
<center>
<h4>Aero-Nova © 2025</h4>
<p>AI-Driven Nowcasting of Air Quality & Mobility</p>
</center>
""", unsafe_allow_html=True)
