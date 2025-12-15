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
# GLOBAL BACKGROUND CSS
# ======================================================
st.markdown("""
<style>

/* Full App Background */
[data-testid="stAppViewContainer"] {
    background-image: url("https://images.unsplash.com/photo-1579003593419-98f949b9398f?q=80&w=1600&auto=format&fit=crop");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

[data-testid="stApp"] {
    background: transparent;
}

.main {
    background-color: rgba(0,0,0,0);
}

</style>
""", unsafe_allow_html=True)

# ======================================================
# PATH SETUP
# ======================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

AQI_FILE = DATA_DIR / "aqi_live_data.csv"
WEATHER_FILE = DATA_DIR / "live_weather.csv"
TRAFFIC_FILE = DATA_DIR / "traffic_timeseries.csv"
AQI_TREND_FILE = DATA_DIR / "processed" / "aqi_master.csv"


# ======================================================
# LOAD LOGO
# ======================================================
def load_logo(path):
    img = Image.open(path).convert("RGBA")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

LOGO = load_logo(BASE_DIR / "logo.png")

# ======================================================
# LOAD DATA
# ======================================================
aqi_df = pd.read_csv(AQI_FILE)

# 🔥 normalize column names (THIS FIXES EVERYTHING)
aqi_df.columns = aqi_df.columns.str.strip().str.lower()

# now safely convert timestamp
aqi_df["timestamp"] = pd.to_datetime(aqi_df["timestamp"], errors="coerce")

# drop bad rows
aqi_df = aqi_df.dropna(subset=["timestamp"])

# sort by time
aqi_df = aqi_df.sort_values("timestamp")



# ======================================================
# WEATHER
# ======================================================

weather_df = pd.read_csv(WEATHER_FILE)

# normalize column names
weather_df.columns = weather_df.columns.str.strip().str.lower()

# ✅ parse timestamp SAFELY (by column name, not iloc)
if "timestamp" in weather_df.columns:
    weather_df["timestamp"] = pd.to_datetime(weather_df["timestamp"], errors="coerce")
else:
    weather_df["timestamp"] = pd.to_datetime(weather_df.iloc[:, 0], errors="coerce")

# drop bad rows & sort
weather_df = weather_df.dropna(subset=["timestamp"]).sort_values("timestamp")


# TRAFFIC
traffic_df = pd.read_csv(TRAFFIC_FILE)
traffic_df.columns = traffic_df.columns.str.strip().str.lower()
traffic_df["timestamp"] = pd.to_datetime(traffic_df.iloc[:, 0], errors="coerce")
traffic_df = traffic_df.dropna(subset=["timestamp"]).sort_values("timestamp")

latest_aqi = aqi_df.iloc[-1]
if weather_df.empty:
    latest_weather = None
else:
    latest_weather = weather_df.iloc[-1]

if traffic_df.empty:
    latest_traffic = None
else:
    latest_traffic = traffic_df.iloc[-1]


def aqi_category(aqi):
    try:
        aqi = int(aqi)
    except:
        return "Unknown"

    if aqi == 1:
        return "Good 🟢"
    elif aqi == 2:
        return "Satisfactory 🟡"
    elif aqi == 3:
        return "Moderate 🟠"
    elif aqi == 4:
        return "Poor 🔴"
    elif aqi == 5:
        return "Very Poor 🟣"
    elif aqi == 6:
        return "Severe ⚫"
    elif aqi == 7:
        return "Extreme ☠"
    else:
        return "Out of Range"


# ======================================================
# HERO + UI CSS
# ======================================================
st.markdown("""
<style>

* { font-family: 'Segoe UI', sans-serif; }

.hero {
    height: 90vh;
    background-image: url("https://images.unsplash.com/photo-1505842465776-3d0c1f6b2a4b?auto=format&fit=crop&w=2000&q=80");
    background-size: cover;
    background-position: center;
    display: flex;
    justify-content: center;
    align-items: center;
}

.glass {
    background: rgba(255,255,255,0.18);
    border-radius: 22px;
    padding: 45px 70px;
    border: 1px solid rgba(255,255,255,0.35);
    backdrop-filter: blur(14px);
    box-shadow: 0 10px 35px rgba(0,0,0,0.35);
    text-align: center;
}

.section-title {
    font-size: 42px;
    font-weight: 800;
    margin: 60px 0 30px 0;
    color: white;
}

.card {
background: rgba(255,255,255,0.25);
    padding: 25px;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.4);
    backdrop-filter: blur(12px);
    box-shadow: 0 6px 20px rgba(0,0,0,0.25);
    text-align: center;

    /* 🔥 FIXED SIZE */
    height: 200px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}

.metric-title {
    font-size: 18px;
    opacity: 0.9;
}

.metric-value {
    font-size: 36px;
    font-weight: 800;
}

/* White Sliders */
div[data-baseweb="slider"] > div > div {
    background-color: white !important;
}
div[data-baseweb="slider"] div[role="slider"] {
    background-color: white !important;
    border: 2px solid white;
}

</style>
""", unsafe_allow_html=True)

# ======================================================
# HERO SECTION
# ======================================================
st.markdown(f"""
<div class="hero">
  <div class="glass">
    <img src="data:image/png;base64,{LOGO}" width="460"><br><br>
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
# LOCATION SELECTOR (FIXED CITIES)
# ======================================================
st.markdown("<div class='section-title'>Select Location</div>", unsafe_allow_html=True)

CITY_MAP = {
    "Indore": "Indore",
    "Dewas": "Dewas",
    "Bhopal": "Bhopal",
    "Ujjain": "Ujjain"
}

selected_city = st.selectbox(
    "Choose City",
    list(CITY_MAP.keys()),
    index=0
)

# Filter AQI data by city
city_mask = aqi_df["station"].str.contains(CITY_MAP[selected_city], case=False, na=False)
aqi_city_df = aqi_df[city_mask]

if aqi_city_df.empty:
    st.warning("No AQI data available for selected city.")
else:
    latest_aqi = aqi_city_df.iloc[-1]

# AQI trend data filtered by city (from live AQI data)
aqi_trend_city = aqi_df[
    aqi_df["station"].str.contains(
        CITY_MAP[selected_city], case=False, na=False
    )
]



# ======================================================
# LIVE AQI
# ======================================================
st.markdown("<div class='section-title'>Live Air Quality</div>", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)

aqi_val = int(latest_aqi["aqi"])
aqi_cat = aqi_category(aqi_val)

with c1:
    st.markdown(
        f"""
        <div class="card">
            <div style="font-size:18px;opacity:0.85;">Current AQI</div>
            <div style="font-size:56px;font-weight:900;margin-top:6px;">
                {aqi_val}
            </div>
            <div style="margin-top:10px;font-size:16px;font-weight:600;">
                {aqi_cat}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with c2:
    st.markdown(f"""
    <div class="card">
        <div class="metric-title">PM2.5 (µg/m³)</div>
        <div class="metric-value">{latest_aqi["pm2_5"]:.1f}</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="card">
        <div class="metric-title">PM10 (µg/m³)</div>
        <div class="metric-value">{latest_aqi["pm10"]:.1f}</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    if latest_traffic is None:
        st.markdown(f"""
        <div class="card">
            <div class="metric-title">Traffic Speed (km/h)</div>
            <div class="metric-value">N/A</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="card">
            <div class="metric-title">Traffic Speed (km/h)</div>
            <div class="metric-value">{latest_traffic["currentspeed"]:.1f}</div>
        </div>
        """, unsafe_allow_html=True)


st.markdown("<div class='section-title'>Current Weather</div>", unsafe_allow_html=True)

if weather_df.empty:
    st.warning("Weather data not available.")
else:
    latest_weather = weather_df.iloc[-1]

    w1, w2, w3, w4 = st.columns(4)

    with w1:
        st.markdown(f"""
        <div class="card">
            <div class="metric-title">Temperature (°C)</div>
            <div class="metric-value">{latest_weather.get("temperature (°c)", np.nan):.1f}</div>
        </div>
        """, unsafe_allow_html=True)

    with w2:
        st.markdown(f"""
        <div class="card">
            <div class="metric-title">Humidity (%)</div>
            <div class="metric-value">{latest_weather.get("humidity (%)", np.nan):.1f}</div>
        </div>
        """, unsafe_allow_html=True)

    with w3:
        st.markdown(f"""
        <div class="card">
            <div class="metric-title">Pressure (hPa)</div>
            <div class="metric-value">{latest_weather.get("pressure (hpa)", np.nan):.1f}</div>
        </div>
        """, unsafe_allow_html=True)

    with w4:
        st.markdown(f"""
        <div class="card">
            <div class="metric-title">Wind Speed (m/s)</div>
            <div class="metric-value">{latest_weather.get("wind speed (m/s)", np.nan):.1f}</div>
        </div>
        """, unsafe_allow_html=True)

# ======================================================
# WHAT-IF ENGINE
# ======================================================
st.markdown("<div class='section-title'>What-If Scenario Engine</div>", unsafe_allow_html=True)

left, right = st.columns(2)

with left:
    traffic_change = st.slider("Traffic Change (%)", -50, 200, 0)
    industry_change = st.slider("Industrial Emissions (%)", -50, 200, 0)
    vehicle_change = st.slider("Vehicular Emissions (%)", -50, 200, 0)

with right:
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
    base_aqi = int(latest_aqi["aqi"])

    new_aqi = calculate_new_aqi(base_aqi)
    diff = new_aqi - base_aqi


    st.markdown(f"""
    <div class="card">
        <h3>Scenario AQI</h3>
        <div style="font-size:48px;font-weight:900;">{new_aqi}</div>
        <p>Change: {diff:+}</p>
    </div>
    """, unsafe_allow_html=True)

# ===============================
# AQI TREND (FROM LIVE AQI DATA)
# ===============================
st.markdown("<div class='section-title'>AQI Trend</div>", unsafe_allow_html=True)

trend_df = aqi_df[
    aqi_df["station"].str.contains(
        CITY_MAP[selected_city],
        case=False,
        na=False
    )
]

if trend_df.empty:
    st.info("Not enough AQI data available to display trend.")
else:
    fig = px.scatter(
        trend_df,
        x="timestamp",
        y="aqi",
        title=f"AQI Trend – {selected_city}"
    )

    fig.update_layout(
        template="plotly_dark",
        height=450,
        xaxis_title="Time",
        yaxis_title="AQI"
    )

    st.plotly_chart(fig, use_container_width=True)