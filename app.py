#!/usr/bin/env python3
"""
Aero-Nova — AI-Driven Nowcasting of Air Quality & Mobility
FULL BEAUTIFUL GLASS UI + FULL FUNCTIONALITY (Option A)

NOTE:
- This is PART 1 of 4.
- Paste all 4 parts into one single app.py
- Run using: streamlit run app.py
"""

import os
import json
import base64
import importlib
from datetime import datetime, timedelta
import requests
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import plotly.graph_objects as go
from io import BytesIO
from PIL import Image

# --------------------------------------------------------------------
# 🔵 LOAD & CONVERT USER LOGO TO TRANSPARENT PNG BASE64
# --------------------------------------------------------------------
def load_and_convert_logo():
    logo_path = r"C:\Users\aksha\OneDrive\Desktop\Aero-Nova-2\logo.png"   # ← store logo.png in same folder as app.py

    img = Image.open(logo_path).convert("RGBA")

    datas = img.getdata()
    new_data = []
    for item in datas:
        # remove white background (optional)
        if item[0] > 240 and item[1] > 240 and item[2] > 240:
            new_data.append((255, 255, 255, 0))
        else:
            new_data.append(item)

    img.putdata(new_data)

    buffer = BytesIO()
    img.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return encoded

LOGO_BASE64 = load_and_convert_logo()


# --------------------------------------------------------------------
# 🌟 STREAMLIT PAGE CONFIG + BEAUTIFUL GLASSMORPHISM UI THEME
# --------------------------------------------------------------------

st.set_page_config(page_title="Aero-Nova AQI Dashboard", layout="wide")

BACKGROUND_URL = (
    "https://images.unsplash.com/photo-1505842465776-3d0c1f6b2a4b?"
    "auto=format&fit=crop&w=2000&q=80"
)

st.markdown(
    f"""
<style>

* {{
    font-family: 'Segoe UI', sans-serif;
}}

[data-testid="stAppViewContainer"] {{
    background: url('{BACKGROUND_URL}') no-repeat center center fixed;
    background-size: cover;
}}

/* GLASS CARD */
.glass-card {{
    background: rgba(255, 255, 255, 0.18);
    padding: 28px;
    border-radius: 18px;
    border: 1px solid rgba(255, 255, 255, 0.40);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    box-shadow: 0px 8px 24px rgba(0,0,0,0.25);
    margin-bottom: 20px;
}}

/* AQI Metric Card */
.metric-card {{
    padding: 18px;
    border-radius: 15px;
    color: white;
    text-align: center;
    font-weight: 600;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.25);
    width: 260px;
}}

/* FIXED TOP-LEFT LOGO */
#fixed-logo {{
    position: fixed;
    top: 18px;
    left: 18px;
    width: 140px;
    z-index: 99999;
}}

</style>

<!-- Transparent PNG Logo -->
<img id="fixed-logo" src="data:image/png;base64,{LOGO_BASE64}">
""",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------
# 🌟 BEAUTIFUL DASHBOARD TITLE
# --------------------------------------------------------------------

st.markdown(
    """
<h1 style="color:white; font-size:48px; font-weight:800;">
    Aero-Nova — AQI Live Monitoring & Forecast
</h1>
<p style="color:#e8e8e8; font-size:18px; margin-top:-10px;">
    Real-Time Air Quality • Weather Influence • 72-Hour ML Forecast • AI-Powered Insights
</p>
<br>
""",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------
# 🌍 CORE AQI BREAKPOINTS & CALCULATIONS
# --------------------------------------------------------------------

# AQI breakpoints for PM2.5 and PM10
PM25_BP = [(0,30,0,50),(31,60,51,100),(61,90,101,200),(91,120,201,300),(121,250,301,400),(251,500,401,500)]
PM10_BP = [(0,50,0,50),(51,100,51,100),(101,250,101,200),(251,350,201,300),(351,430,301,400)]

BP_MAP = {
    "pm2_5": PM25_BP, "pm25": PM25_BP, "pm2.5": PM25_BP,
    "pm10": PM10_BP, "no2": PM10_BP, "so2": PM10_BP,
    "o3": PM25_BP, "nh3": PM25_BP, "co": PM10_BP
}

def calc_subindex(C, bps):
    try:
        C = float(C)
    except:
        return None
    for (Cl, Ch, Il, Ih) in bps:
        if Cl <= C <= Ch:
            return ((Ih - Il) / (Ch - Cl)) * (C - Cl) + Il
    return None

def compute_aqi_from_pollutants(meas):
    sub = {}
    for k, v in meas.items():
        if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
            continue
        key = k.lower().replace(".", "_")
        bps = BP_MAP.get(key)
        if bps:
            si = calc_subindex(v, bps)
            if si is not None:
                sub[key.upper()] = round(si, 2)

    if not sub:
        return None, None, {}

    dominant, dom_value = max(sub.items(), key=lambda x: x[1])
    final_aqi = int(round(dom_value))
    return final_aqi, dominant, sub


def aqi_category(aqi):
    try:
        aqi = float(aqi)
    except:
        return "Unknown"

    if aqi <= 50: return "Good"
    if aqi <= 100: return "Satisfactory"
    if aqi <= 200: return "Moderate"
    if aqi <= 300: return "Poor"
    if aqi <= 400: return "Very Poor"
    return "Severe"


# --------------------------------------------------------------------
# 🌬 PARSE REASONS & SUGGESTIONS CLEANLY
# --------------------------------------------------------------------

def parse_list_field(x):
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(map(str, x))
    if isinstance(x, str):
        x = x.strip()
        try:
            js = json.loads(x)
            if isinstance(js, (list, tuple)):
                return list(map(str, js))
        except:
            pass
        try:
            val = eval(x, {"__builtins__": None}, {})
            if isinstance(val, (list, tuple)):
                return list(map(str, val))
        except:
            pass
        return [x]
    return [str(x)]


# --------------------------------------------------------------------
# 🔗 API ENDPOINTS
# --------------------------------------------------------------------

OWM_WEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"
OWM_FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"
OWM_AIR_URL = "https://api.openweathermap.org/data/2.5/air_pollution"
OWM_AIR_FORECAST_URL = "https://api.openweathermap.org/data/2.5/air_pollution/forecast"
OPENAQ_URL = "https://api.openaq.org/v3/latest"

# Default locations
LOCATIONS = {
    "Indore": (22.7180, 75.8470),
    "Dewas": (22.9667, 75.3333),
    "Ujjain": (23.1765, 75.7885),
    "Bhopal - MP Nagar": (23.2380, 77.4125)
}


# --------------------------------------------------------------------
# 🌦 FETCH FROM OPENWEATHER LIVE AQI
# --------------------------------------------------------------------

def fetch_owm_air(lat, lon, api_key):
    if not api_key:
        return None

    params = {"lat": lat, "lon": lon, "appid": api_key}
    r = requests.get(OWM_AIR_URL, params=params, timeout=12)
    r.raise_for_status()

    js = r.json()
    lst = js.get("list", [])
    if not lst:
        return None

    rec = lst[0]
    comps = rec.get("components", {})

    return {
        "components": comps,
        "timestamp": rec.get("dt"),
        "raw": rec
    }


# --------------------------------------------------------------------
# 🌦 FETCH WEATHER FORECAST FROM OPENWEATHER
# --------------------------------------------------------------------

def fetch_owm_forecast_weather(lat, lon, api_key, hours=48):
    if not api_key:
        return pd.DataFrame()

    params = {"lat": lat, "lon": lon, "appid": api_key, "units": "metric"}
    r = requests.get(OWM_FORECAST_URL, params=params, timeout=12)
    r.raise_for_status()

    js = r.json()
    rows = []

    for entry in js.get("list", [])[: (hours // 3) + 1]:
        ts = datetime.utcfromtimestamp(entry["dt"])
        rows.append({
            "timestamp": ts,
            "temperature": entry["main"].get("temp"),
            "humidity": entry["main"].get("humidity"),
            "pressure": entry["main"].get("pressure"),
            "wind_speed": entry["wind"].get("speed"),
            "wind_deg": entry["wind"].get("deg"),
        })

    return pd.DataFrame(rows)


# --------------------------------------------------------------------
# 🌫 FETCH AIR QUALITY FORECAST FROM OPENWEATHER
# --------------------------------------------------------------------

def fetch_owm_air_forecast(lat, lon, api_key, hours=48):
    if not api_key:
        return None

    params = {"lat": lat, "lon": lon, "appid": api_key}
    r = requests.get(OWM_AIR_FORECAST_URL, params=params, timeout=12)
    r.raise_for_status()

    js = r.json()
    lst = js.get("list", [])[:hours]

    rows = []
    for rec in lst:
        ts = rec.get("dt")
        comps = rec.get("components", {})
        rows.append({"timestamp": datetime.utcfromtimestamp(ts), **comps})

    return pd.DataFrame(rows) if rows else None


# --------------------------------------------------------------------
# 🧠 TRY LOCAL current_aqi.py OR CACHE
# --------------------------------------------------------------------

def try_run_local_current_aqi(ignore_if_stale=True, stale_repeat_threshold=3):
    report = None

    try:
        if os.path.exists("current_aqi.py"):
            mod = importlib.import_module("current_aqi")
            importlib.reload(mod)

            if hasattr(mod, "calculate_current_aqi"):
                r = mod.calculate_current_aqi()
                if isinstance(r, dict):
                    report = r
    except:
        pass

    # CSV fallback
    csv_path = "results/current_report.csv"
    json_path = "results/current_report.json"

    if report is None and os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            if not df.empty:
                report = df.iloc[-1].to_dict()
        except:
            pass

    if report is None and os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                report = json.load(f)
        except:
            pass

    # Stale detection
    if ignore_if_stale and os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            aqi_cols = [c for c in df.columns if "aqi" in c.lower()]

            if aqi_cols:
                col = df[aqi_cols[0]].dropna().astype(float)
                if len(col) >= stale_repeat_threshold:
                    last_vals = col.tail(stale_repeat_threshold).values
                    if np.allclose(last_vals, last_vals[0]):
                        return None
        except:
            pass

    return report


# --------------------------------------------------------------------
# 🎛 SIDEBAR SETTINGS
# --------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Settings")

    owm_key_input = st.text_input(
        "OpenWeather API Key",
        value=os.getenv("OWM_KEY", "")
    )

    external_forecast_url = st.text_input("External Forecast URL (optional)")

    source = st.selectbox(
        "Choose AQI Source Priority",
        ["local (current_aqi.py/results)", "OpenWeather (recommended)", "OpenAQ (fallback)"]
    )

    refresh = st.button("Refresh Data")
    forecast_hours = st.slider("Forecast duration (hours)", 6, 72, 24)
    smoothing = st.slider("Forecast Smoothing", 0, 6, 1)
    show_ci = st.checkbox("Show Confidence Interval", True)
    anchor_observed = st.checkbox("Anchor Observation to Forecast", True)


# --------------------------------------------------------------------
# 🔐 SAFE SECRETS HANDLING
# --------------------------------------------------------------------

OWM_KEY = owm_key_input or os.getenv("OWM_KEY", "")

try:
    sec = getattr(st, "secrets", {})
    if isinstance(sec, dict) and "OWM_KEY" in sec and not OWM_KEY:
        OWM_KEY = sec["OWM_KEY"]
except:
    pass


# --------------------------------------------------------------------
# 📍 LOCATION PICKER
# --------------------------------------------------------------------

left, right = st.columns([2, 1])

with left:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("📍 Select Location")

    loc = st.selectbox("Choose city", list(LOCATIONS.keys()))
    lat, lon = LOCATIONS[loc]

    st.write(f"**Coordinates:** {lat:.4f}, {lon:.4f}")
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("### ℹ️ Info")
    st.write("Priority: Local → OpenWeather → OpenAQ")
    st.markdown("</div>", unsafe_allow_html=True)




# --------------------------------------------------------------------
# 🌟 GLASS AQI CARD (Modern Tile)
# --------------------------------------------------------------------

def aqi_glass_card(aqi_value, title="Current AQI"):
    cat = aqi_category(aqi_value)
    colors = {
        "Good": "#2ecc71",
        "Satisfactory": "#9acd32",
        "Moderate": "#f1c40f",
        "Poor": "#e67e22",
        "Very Poor": "#e74c3c",
        "Severe": "#7f1d1d",
        "Unknown": "#7f8c8d"
    }
    c = colors.get(cat, "#7f8c8d")

    html = f"""
    <div class="metric-card" style="background:{c};">
        <div style="font-size:15px; opacity:0.95;">{title}</div>
        <div style="font-size:42px; font-weight:800; margin-top:4px;">
            {int(aqi_value)}
        </div>
        <div style="font-size:14px; margin-top:6px;">
            {cat}
        </div>
    </div>
    """
    return html


# --------------------------------------------------------------------
# 🌍 FETCH LIVE AQI
# --------------------------------------------------------------------

local_report = None
remote_pollutants = {}
owm_air_snapshot = None
remote_station = None

# 1) Local AQI first
if source.startswith("local"):
    local_report = try_run_local_current_aqi(ignore_if_stale=True)
    if local_report is None:
        st.info("Local AQI missing or stale → switching to remote source.")

# 2) OpenWeather AQI
if (source == "OpenWeather (recommended)") or (local_report is None and source.startswith("local")):
    if OWM_KEY:
        try:
            owm_air_snapshot = fetch_owm_air(lat, lon, OWM_KEY)
            if owm_air_snapshot:
                remote_pollutants = owm_air_snapshot.get("components", {})
        except Exception as e:
            st.warning("OpenWeather AQI fetch failed: " + str(e))
    else:
        st.info("OpenWeather key missing → skipping OWM live AQI.")

# 3) OpenAQ fallback
if not remote_pollutants and (
    source == "OpenAQ (fallback)" or (local_report is None and not owm_air_snapshot)
):
    try:
        r = requests.get(
            OPENAQ_URL,
            params={"coordinates": f"{lat},{lon}", "radius": 5000, "limit": 3},
            timeout=12,
        )
        r.raise_for_status()
        js = r.json()

        results = js.get("results", [])
        if results:
            best = None
            for r0 in results:
                if r0.get("latest"):
                    best = r0
                    break
            if best is None:
                best = results[0]

            meas = best.get("latest") or best.get("measurements", []) or best.get("parameters", [])
            for m in meas:
                param = m.get("parameter") or m.get("name")
                val = m.get("value")
                if param:
                    remote_pollutants[param.lower()] = val

            remote_station = best

    except Exception as e:
        st.warning("OpenAQ fetch failed: " + str(e))


# --------------------------------------------------------------------
# 🌟 DISPLAY LIVE AQI (Glass Card + Reasons + Suggestions)
# --------------------------------------------------------------------

st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
st.subheader("🌫 Live AQI Summary")

displayed_aqi = None
dominant = None
subindices = None

# Case 1 — Local AQI available
if local_report:
    aqi_val = None
    for k in ("AQI", "aqi", "predicted_AQI", "aqi_value"):
        if k in local_report:
            aqi_val = local_report[k]
            break

    if aqi_val is not None:
        displayed_aqi = float(aqi_val)
        st.markdown(aqi_glass_card(displayed_aqi, "Local AQI"), unsafe_allow_html=True)

        # Reasons
        reasons = parse_list_field(local_report.get("reasons"))
        suggestions = parse_list_field(local_report.get("suggestions"))

        if reasons:
            st.write("**🟦 Reasons**")
            for r in reasons:
                st.write("•", r)

        if suggestions:
            st.write("**🟩 Suggestions**")
            for s in suggestions:
                st.write("•", s)

# Case 2 — Remote pollutants used to compute AQI
if displayed_aqi is None and remote_pollutants:

    meas = {}
    for k, v in remote_pollutants.items():
        k0 = k.lower()
        if any(x in k0 for x in ("pm2", "pm10", "no2", "so2", "o3", "nh3", "co")):
            meas[k0] = v

    aqi_val, dom, sub = compute_aqi_from_pollutants(meas)

    if aqi_val is not None:
        displayed_aqi = float(aqi_val)
        dominant = dom
        subindices = sub

        st.markdown(aqi_glass_card(displayed_aqi, "Remote AQI"), unsafe_allow_html=True)

        if dom:
            st.write("**Dominant pollutant:**", dom)

        if sub:
            st.write("**Subindices:**")
            for k, v in sub.items():
                st.write(f"• {k}: {v}")

if displayed_aqi is None:
    st.info("No live AQI available.")

st.markdown("</div>", unsafe_allow_html=True)


# --------------------------------------------------------------------
# 🌦 WEATHER (EXOGENOUS INPUT FOR MODEL)
# --------------------------------------------------------------------

st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
st.subheader("🌦 Weather Conditions & Forecast")

weather_df = pd.DataFrame()

if OWM_KEY:
    try:
        weather_df = fetch_owm_forecast_weather(lat, lon, OWM_KEY, hours=max(24, forecast_hours))
        if not weather_df.empty:
            st.dataframe(weather_df.head(24))
        else:
            st.write("Weather forecast not available.")
    except Exception as e:
        st.warning("Weather forecast failed: " + str(e))
else:
    st.info("OpenWeather key missing → Weather forecast not fetched.")

st.markdown("</div>", unsafe_allow_html=True)


# --------------------------------------------------------------------
# 🔮 FORECAST (EXTERNAL → OPENWEATHER → LOCAL MODEL)
# --------------------------------------------------------------------

st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
st.subheader("🔮 AQI Forecast (AI + Weather)")

forecast_df = None

# 1) External Forecast API
if external_forecast_url:
    try:
        req = requests.get(external_forecast_url, timeout=10)
        req.raise_for_status()
        js = req.json()

        if isinstance(js, dict):
            if "timestamps" in js and "aqi_forecast" in js:
                df = pd.DataFrame({
                    "timestamp": pd.to_datetime(js["timestamps"]),
                    "aqi_forecast": js["aqi_forecast"],
                })
                forecast_df = df
    except Exception as e:
        st.warning("External forecast fetch failed: " + str(e))


# 2) OpenWeather Air Forecast
if forecast_df is None and OWM_KEY:
    try:
        temp = fetch_owm_air_forecast(lat, lon, OWM_KEY, hours=forecast_hours)
        if temp is not None and not temp.empty:
            temp["timestamp"] = pd.to_datetime(temp["timestamp"])

            aqi_vals = []
            for _, row in temp.iterrows():
                comps = {c: row[c] for c in ["pm2_5", "pm10", "no2", "so2", "o3", "co"] if c in row}
                aqi, dom, sub = compute_aqi_from_pollutants(comps)
                aqi_vals.append(aqi if aqi is not None else np.nan)

            temp["aqi_forecast"] = aqi_vals
            temp = temp[["timestamp", "aqi_forecast"]].dropna().reset_index(drop=True)

            if not temp.empty:
                forecast_df = temp.head(forecast_hours)
    except:
        pass


# --------------------------------------------------------------------
# 3) FALLBACK: Local ARIMA Model
# --------------------------------------------------------------------

MODEL_PATH = "saved_models/arima_aqi_weather_only.joblib"

if forecast_df is None and os.path.exists(MODEL_PATH):
    try:
        mdl = joblib.load(MODEL_PATH)
        if isinstance(mdl, dict):
            sm_res = mdl.get("model")
            meta = mdl.get("meta", {})
        else:
            sm_res = mdl
            meta = {}

        exog_cols = meta.get("exog_cols", [])

        # Build exogenous matrix if weather is available:
        X_future = None
        if exog_cols and not weather_df.empty:
            rows = []
            for i in range(forecast_hours):
                row = []
                for c in exog_cols:
                    match = [col for col in weather_df.columns if col.lower() == c.lower()]
                    val = weather_df.iloc[i].get(match[0]) if match else np.nan
                    try:
                        row.append(float(val))
                    except:
                        row.append(np.nan)
                rows.append(row)

            X_future = np.array(rows, dtype=float)
            for j in range(X_future.shape[1]):
                col = X_future[:, j]
                med = np.nanmedian(col)
                col[np.isnan(col)] = med
                X_future[:, j] = col

        pred = sm_res.get_forecast(steps=forecast_hours, exog=X_future)
        mean = pred.predicted_mean.astype(float)

        # Clean
        s = pd.Series(mean)
        s = s.interpolate().fillna(method="bfill").fillna(0)
        mean = s.values

        # Build DF
        start_ts = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
        times = [start_ts + timedelta(hours=i) for i in range(1, forecast_hours + 1)]
        df = pd.DataFrame({"timestamp": times, "aqi_forecast": mean})

        # CI
        if hasattr(pred, "conf_int"):
            ci = pred.conf_int()
            df["lower95"] = ci.iloc[:, 0].values
            df["upper95"] = ci.iloc[:, 1].values

        # Smoothing
        if smoothing > 0:
            df["aqi_forecast"] = df["aqi_forecast"].rolling(smoothing, min_periods=1).mean()

        forecast_df = df

    except Exception as e:
        st.warning("Local model forecast failed: " + str(e))


if forecast_df is None:
    st.info("No forecast data available.")

else:
    st.dataframe(forecast_df.head(10))

st.markdown("</div>", unsafe_allow_html=True)



# --------------------------------------------------------------------
# 📈 OBSERVED + FORECAST PLOT
# --------------------------------------------------------------------

st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
st.subheader("📈 Live AQI Trend + Forecast Curve")

fig = go.Figure()

# ---------------- Observed
obs_x, obs_y = [], []

if local_report:
    ts = local_report.get("timestamp") or local_report.get("time")
    try:
        ts = pd.to_datetime(ts)
    except:
        ts = datetime.utcnow()

    val = None
    for k in ("AQI", "aqi", "predicted_AQI", "aqi_value"):
        if k in local_report:
            val = local_report[k]
            break

    if val is not None:
        obs_x.append(ts)
        obs_y.append(float(val))

elif displayed_aqi is not None:
    obs_x.append(datetime.utcnow())
    obs_y.append(float(displayed_aqi))

# Anchor observed to forecast start
if anchor_observed and obs_x and forecast_df is not None and not forecast_df.empty:
    start_ts = forecast_df["timestamp"].iloc[0]
    if obs_x[0] < start_ts:
        obs_x.append(start_ts)
        obs_y.append(obs_y[0])

# Plot observed
if obs_x and obs_y:
    fig.add_trace(go.Scatter(
        x=obs_x, y=obs_y,
        mode="markers+lines" if len(obs_x) > 1 else "markers",
        name="Observed AQI",
        marker=dict(size=10),
        line=dict(width=3)
    ))


# ---------------- Forecast Plot
if forecast_df is not None and not forecast_df.empty:
    fig.add_trace(go.Scatter(
        x=forecast_df["timestamp"],
        y=forecast_df["aqi_forecast"],
        mode="lines+markers",
        name="Forecast AQI",
        marker=dict(size=7),
        line=dict(shape="spline", width=2)
    ))

    # CI
    if show_ci and "lower95" in forecast_df and "upper95" in forecast_df:
        fig.add_trace(go.Scatter(
            x=list(forecast_df["timestamp"]) + list(forecast_df["timestamp"])[::-1],
            y=list(forecast_df["upper95"]) + list(forecast_df["lower95"])[::-1],
            fill="toself",
            fillcolor="rgba(200,200,200,0.2)",
            line=dict(color="rgba(0,0,0,0)"),
            name="Confidence Interval",
            hoverinfo="skip"
        ))


# ---------------- Dynamic Y-range
vals = list(obs_y)
if forecast_df is not None and not forecast_df.empty:
    vals += list(forecast_df["aqi_forecast"].astype(float))

if vals:
    ymin, ymax = min(vals), max(vals)
    pad = max(5, (ymax - ymin) * 0.15)
    fig.update_yaxes(range=[max(0, ymin - pad), ymax + pad])
else:
    fig.update_yaxes(autorange=True)


fig.update_layout(
    xaxis_title="Time (UTC)",
    yaxis_title="AQI",
    height=520,
    template="plotly_white",
    legend=dict(orientation="v", x=1, y=1)
)

st.plotly_chart(fig, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)



# --------------------------------------------------------------------
# 📄 DATA PREVIEW + DEBUG INFORMATION
# --------------------------------------------------------------------

st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
st.subheader("📄 Data Preview & Debug Info")

if local_report:
    st.write("### Local AQI Report")
    try:
        preview = {}
        for k in ["timestamp", "AQI", "aqi", "predicted_AQI", "reasons", "suggestions"]:
            if k in local_report:
                preview[k] = local_report[k]
        st.json(preview)
    except:
        st.json(local_report)

if owm_air_snapshot:
    st.write("### OpenWeather Components")
    st.json(owm_air_snapshot.get("components", {}))

if remote_station:
    st.write("### OpenAQ Station Metadata")
    st.json(remote_station)

if forecast_df is not None and not forecast_df.empty:
    st.write("### Forecast Preview")
    st.dataframe(forecast_df.head(min(24, len(forecast_df))))
else:
    st.info("No forecast data available.")

st.markdown("</div>", unsafe_allow_html=True)


# --------------------------------------------------------------------
# 🏁 FOOTER
# --------------------------------------------------------------------

st.markdown(
    """
<br>
<div class='glass-card'>
    <h3 style='margin-bottom:8px;'>Aero-Nova — AI-Driven Nowcasting</h3>
    <p style='margin-top:-8px;'>
        This dashboard integrates <b>OpenWeather</b>, <b>OpenAQ</b>, and <b>Local ML Models</b>
        to deliver high-resolution AQI insights & next-hours forecasting.
        <br><br>
        © 2025 Aero-Nova | Designed with ❤️ using Streamlit + AI
    </p>
</div>
""",
    unsafe_allow_html=True
)

