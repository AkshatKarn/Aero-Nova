#!/usr/bin/env python3
"""
app_streamlit_remote.py

Streamlit app that:
 - fetches pollutant data from OpenAQ (no key)
 - fetches weather from Open-Meteo (no key)
 - computes current AQI from pollutant concentrations
 - loads a saved ARIMA model (if present) and forecasts 24h using weather exog
 - accepts one local upload (CSV) which can override live AQI or provide traffic

Run:
    pip install streamlit pandas requests joblib plotly numpy
    streamlit run app_streamlit_remote.py
"""
import os, math
import streamlit as st
import requests
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import plotly.graph_objects as go

st.set_page_config(page_title="AQI Live (remote) + Forecast", layout="wide")

# ---------------- USER CONFIG ----------------
MODEL_PATH = "saved_models/arima_aqi_weather_only.joblib"   # model saved by your pipeline
DEFAULT_LOCATION = "Indore - MG Road"
LOCATIONS = {
    "Indore - MG Road": (22.718, 75.847),
    "Indore - Vijay Nagar": (22.735, 75.88),
    "Indore - AB Road": (22.726, 75.889),
    "Indore - Rajwada": (22.7196, 75.855),
    "Bhopal - MP Nagar": (23.238, 77.4125)
}
OPENAQ_URL = "https://api.openaq.org/v2/latest"
OPENMETEO_URL = "https://api.open-meteo.com/v1/forecast"

# ---------------- AQI breakpoints helpers ----------------
PM25_BP = [(0,30,0,50),(31,60,51,100),(61,90,101,200),(91,120,201,300),(121,250,301,400),(251,500,401,500)]
PM10_BP = [(0,50,0,50),(51,100,51,100),(101,250,101,200),(251,350,201,300),(351,430,301,400)]
GAS_BP  = PM10_BP
O3_BP   = PM25_BP

BP_MAP = {
    "pm2_5": PM25_BP, "pm2.5": PM25_BP, "pm25": PM25_BP,
    "pm10": PM10_BP, "no2": GAS_BP, "so2": GAS_BP,
    "o3": O3_BP, "nh3": PM25_BP
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
    """
    meas: dict with pollutant concentrations in µg/m3 keys like 'pm25','pm10','no2','so2','o3','nh3'
    returns: (aqi, dominant_pollutant, subindices dict)
    """
    sub = {}
    for k,v in meas.items():
        if v is None or (isinstance(v,float) and np.isnan(v)): continue
        key = k.lower().replace(".", "_")
        bps = BP_MAP.get(key)
        if bps:
            si = calc_subindex(v, bps)
            if si is not None:
                sub[k.upper()] = round(si, 2)
    if not sub:
        return None, None, {}
    # final AQI is max subindex
    dominant, dom_si = max(sub.items(), key=lambda x: x[1])
    final_aqi = int(round(dom_si))
    return final_aqi, dominant, sub

def aqi_category(aqi):
    try: aqi=float(aqi)
    except: return "Unknown"
    if aqi<=50: return "Good"
    if aqi<=100: return "Satisfactory"
    if aqi<=200: return "Moderate"
    if aqi<=300: return "Poor"
    if aqi<=400: return "Very Poor"
    return "Severe"

def reason_for_dominant(dominant):
    if dominant is None: return "No pollutant info."
    dom = dominant.lower()
    if "pm2" in dom: return "PM2.5 high — vehicles, burning, stagnant air."
    if "pm10" in dom: return "PM10 high — dust/construction."
    if "no2" in dom: return "NO2 high — traffic/combustion."
    if "so2" in dom: return "SO2 high — industrial/thermal."
    if "o3" in dom: return "O3 high — photochemical in sunlight."
    return "Mixed sources."

# ---------------- Remote fetch helpers ----------------
def fetch_openaq_nearby(lat, lon, radius_m=5000):
    params = {"coordinates": f"{lat},{lon}", "radius": radius_m, "limit": 100}
    r = requests.get(OPENAQ_URL, params=params, timeout=10)
    r.raise_for_status()
    js = r.json()
    # Aggregate latest measurements at nearest location (take first result)
    results = js.get("results", [])
    if not results:
        return None
    # pick nearest by distance if provided
    best = results[0]
    measurements = best.get("measurements", [])
    vals = {}
    # map OpenAQ parameter names to our keys
    for m in measurements:
        param = m.get("parameter", "").lower()
        val = m.get("value", None)
        # OpenAQ uses pm25, pm10, no2, so2, o3, nh3 typically
        vals[param] = val
    return vals, best

def fetch_open_meteo(lat, lon, hours=48):
    params = {
        "latitude": lat, "longitude": lon,
        "hourly": ",".join(["temperature_2m","relativehumidity_2m","pressure_msl","windspeed_10m","winddirection_10m","cloudcover"]),
        "forecast_days": max(1, (hours//24)+1),
        "timezone": "UTC"
    }
    r = requests.get(OPENMETEO_URL, params=params, timeout=10)
    r.raise_for_status()
    js = r.json()
    h = js.get("hourly", {})
    times = h.get("time", [])
    rows = []
    for i,t in enumerate(times[:hours]):
        rows.append({
            "timestamp": t,
            "Temperature (°C)": h.get("temperature_2m", [None])[i] if i < len(h.get("temperature_2m", [])) else None,
            "Humidity (%)": h.get("relativehumidity_2m", [None])[i] if i < len(h.get("relativehumidity_2m", [])) else None,
            "Pressure (hPa)": h.get("pressure_msl", [None])[i] if i < len(h.get("pressure_msl", [])) else None,
            "Wind Speed (m/s)": h.get("windspeed_10m", [None])[i] if i < len(h.get("windspeed_10m", [])) else None,
            "Wind Direction (°)": h.get("winddirection_10m", [None])[i] if i < len(h.get("winddirection_10m", [])) else None,
            "Cloudiness (%)": h.get("cloudcover", [None])[i] if i < len(h.get("cloudcover", [])) else None,
        })
    return pd.DataFrame(rows)

# ---------------- UI ----------------
st.title("AQI Live (remote) + ARIMA Forecast (auto-fetch)")

col1, col2 = st.columns([2,1])
with col2:
    st.markdown("**Settings**")
    loc = st.selectbox("Choose location (remote fetch)", list(LOCATIONS.keys()), index=list(LOCATIONS.keys()).index(DEFAULT_LOCATION))
    hours = st.slider("Forecast hours", 6, 72, 24)
    st.markdown("You can upload **one** local CSV (optional). Use it to supply a local live AQI override or traffic CSV.")
    uploaded = st.file_uploader("Upload local CSV (optional, single file)", type=["csv"])

with col1:
    st.header("Live remote data")
    lat, lon = LOCATIONS[loc]
    st.write(f"Using coordinates: {lat:.4f}, {lon:.4f}")

    # fetch pollutants
    try:
        pq_res = fetch_openaq_nearby(lat, lon)
        if pq_res is None:
            st.warning("OpenAQ returned no nearby station. Pollutant data unavailable.")
            pollutants = {}
            station_info = None
        else:
            pollutants, station_info = pq_res
            st.write("Source: OpenAQ (nearest station)")
            st.json({"station": station_info.get("location"), "city": station_info.get("city")})
    except Exception as e:
        st.error("OpenAQ fetch failed: " + str(e))
        pollutants = {}
        station_info = None

    # optionally override using uploaded local CSV if user indicates it's a live AQI file
    override_live = False
    if uploaded:
        df_local = pd.read_csv(uploaded)
        st.write("Uploaded file head:")
        st.dataframe(df_local.head())
        # Let user choose how to use this file
        use_choice = st.radio("How to use uploaded file?", ("Ignore", "Use as 'live AQI' override (take last row)","Use as traffic data (not used automatically)"))
        if use_choice.startswith("Use as 'live AQI'"):
            # try to detect pollutant columns in uploaded file
            aqi_col = None
            for c in df_local.columns:
                if "aqi" in c.lower():
                    aqi_col = c; break
            # if pollutant columns available, use them; else, if aqi present, override final aqi
            pollutant_cols = [c for c in df_local.columns if any(x in c.lower() for x in ("pm2","pm10","no2","so2","o3","nh3"))]
            if pollutant_cols:
                last = df_local.iloc[-1]
                for pc in pollutant_cols:
                    pollutants[pc.lower()] = float(last.get(pc, np.nan))
                st.success("Local upload used to override pollutant concentrations.")
                override_live = True
            elif aqi_col:
                last = df_local[aqi_col].iloc[-1]
                # we cannot convert AQI back to pollutant concentrations; pass as special flag
                st.info(f"Local AQI override found: {last}. We'll display it as current AQI (does not affect model exog).")
                pollutants["_local_aqi_override"] = float(last)
                override_live = True
            else:
                st.warning("Uploaded file doesn't have recognizable pollutant/AQI columns. Ignoring as live override.")

    # compute current AQI from pollutants (unless local AQI override exists)
    local_override_aqi = pollutants.get("_local_aqi_override", None)
    if local_override_aqi is not None:
        current_aqi = int(local_override_aqi)
        dominant = None
        subindices = {}
    else:
        # map OpenAQ names to our keys
        # openaq uses pm25, pm10, no2, so2, o3, nh3 typically
        meas = {}
        for k in ["pm25","pm2_5","pm2.5","pm10","no2","so2","o3","nh3"]:
            if k in pollutants:
                meas[k] = pollutants[k]
        # also try keys like 'pm25' etc present in dict
        for k in list(pollutants.keys()):
            if k not in meas and any(x in k for x in ["pm2","pm10","no2","so2","o3","nh3"]):
                meas[k] = pollutants[k]
        current_aqi, dominant, subindices = compute_aqi_from_pollutants(meas)

    # show current AQI
    st.subheader("Current AQI")
    if current_aqi is None:
        st.warning("Could not compute AQI from remote data.")
    else:
        st.metric("AQI", f"{current_aqi}", delta=None)
        st.write("Category:", aqi_category(current_aqi))
        if dominant:
            st.write("Dominant pollutant:", dominant)
            st.write("Reason:", reason_for_dominant(dominant))
        if subindices:
            st.write("Subindices:", subindices)

# ---------------- Weather fetch ----------------
with st.expander("Weather (used as exogenous for model)"):
    try:
        weather_df = fetch_open_meteo(lat, lon, hours=72)
        st.write("Weather source: Open-Meteo (hourly, UTC). Showing next 24 rows:")
        st.dataframe(weather_df.head(24))
    except Exception as e:
        st.error("Open-Meteo fetch failed: " + str(e))
        weather_df = pd.DataFrame()

# ---------------- Forecast using saved ARIMA (if exists) ----------------
with st.spinner("Forecasting..."):
    model_exists = os.path.exists(MODEL_PATH)
    if not model_exists:
        st.info("No saved ARIMA model found at saved_models/. The app still shows current AQI only.")
    else:
        try:
            saved = joblib.load(MODEL_PATH)
            if isinstance(saved, dict):
                sm_res = saved.get("model")
                meta = saved.get("meta", {})
            else:
                sm_res = saved
                meta = {}
            exog_cols = meta.get("exog_cols", []) if isinstance(meta, dict) else []
        except Exception as e:
            st.error("Failed to load model: " + str(e))
            sm_res = None
            exog_cols = []

        if sm_res is None:
            st.info("No usable model loaded.")
        else:
            needs_exog = bool(exog_cols)
            X_future = None
            if needs_exog:
                # prepare exog array for forecast_hours: try live weather last row duplicated
                if weather_df is not None and not weather_df.empty:
                    last = weather_df.iloc[0]  # first rows correspond to UTC times; we used head above
                    row = []
                    for c in exog_cols:
                        # case-insensitive match to weather_df columns
                        match = None
                        for k in weather_df.columns:
                            if k.lower() == c.lower(): match=k; break
                        if match is None:
                            for k in weather_df.columns:
                                if c.lower() in k.lower(): match=k; break
                        if match is not None:
                            val = weather_df.iloc[0].get(match, np.nan)
                        else:
                            val = np.nan
                        row.append(float(val) if (pd.notnull(val)) else 0.0)
                    X_future = np.tile(np.array(row).reshape(1,-1), (hours,1))
                else:
                    # fallback zeros
                    X_future = np.zeros((hours, len(exog_cols)))
            # perform forecast
            try:
                if X_future is not None:
                    pred = sm_res.get_forecast(steps=hours, exog=X_future)
                else:
                    pred = sm_res.get_forecast(steps=hours)
                mean = np.array(pred.predicted_mean).astype(float)
                conf = pred.conf_int()
            except Exception as e:
                st.error("Forecast failed: " + str(e))
                mean = np.array([np.nan]*hours)
                conf = None

            # build forecast df with timestamps
            start_ts = datetime.utcnow().replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
            times = [start_ts + timedelta(hours=i) for i in range(hours)]
            forecast_df = pd.DataFrame({"timestamp": times, "aqi_forecast": np.round(mean,2)})
            if conf is not None and not conf.empty:
                forecast_df["lower95"] = conf.iloc[:,0].values
                forecast_df["upper95"] = conf.iloc[:,1].values

            st.success("Forecast ready")
            st.dataframe(forecast_df.head(24))

# ---------------- Chart: observed recent + forecast ----------------
with st.container():
    st.header("Observed (remote / optional local) + Forecast")
    # build observed series: use uploaded local if user chose override, else none
    observed_ts = []
    observed_vals = []
    # remote observed (OpenAQ doesn't give time series in this endpoint) — we can show only point for now
    if current_aqi is not None:
        observed_ts.append(datetime.utcnow())
        observed_vals.append(current_aqi)
    # if user uploaded full timeseries and chose it as live override, try to plot last N
    if uploaded and 'df_local' in locals():
        try:
            tcol = None
            for c in df_local.columns:
                if 'time' in c.lower() or 'date' in c.lower(): tcol = c; break
            aqi_col = None
            for c in df_local.columns:
                if 'aqi' in c.lower(): aqi_col = c; break
            if tcol and aqi_col:
                df_local[tcol] = pd.to_datetime(df_local[tcol], errors='coerce')
                local_plot = df_local[[tcol, aqi_col]].dropna().sort_values(tcol).tail(72)
                # extend observed series
                observed_ts += list(local_plot[tcol].to_list())
                observed_vals += list(pd.to_numeric(local_plot[aqi_col], errors='coerce').to_list())
        except Exception:
            pass

    fig = go.Figure()
    if observed_ts:
        fig.add_trace(go.Scatter(x=observed_ts, y=observed_vals, mode='lines+markers', name='Observed (remote/local)'))

    if os.path.exists(MODEL_PATH) and 'forecast_df' in locals():
        fig.add_trace(go.Scatter(x=forecast_df['timestamp'], y=forecast_df['aqi_forecast'], mode='lines+markers', name='Forecast'))
        if 'lower95' in forecast_df and 'upper95' in forecast_df:
            fig.add_trace(go.Scatter(
                x=list(forecast_df['timestamp']) + list(forecast_df['timestamp'])[::-1],
                y=list(forecast_df['upper95']) + list(forecast_df['lower95'])[::-1],
                fill='toself', fillcolor='rgba(200,200,200,0.2)', line=dict(color='rgba(0,0,0,0)'),
                name='95% CI', showlegend=True
            ))
    fig.update_layout(xaxis_title='Time (UTC)', yaxis_title='AQI', title='AQI Observed + Forecast')
    st.plotly_chart(fig, use_container_width=True)

st.caption("Notes: OpenAQ gives latest pollutant readings (not full timeseries). For more granular historical observed series, upload a local AQI timeseries CSV. The app tries to auto-build exogenous weather using Open-Meteo for forecasting.")
