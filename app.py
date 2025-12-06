#!/usr/bin/env python3
"""
app.py - Streamlit AQI app (OpenWeather primary, local fallback, safe secrets handling)

Features:
 - Uses OpenWeather (air pollution + forecast) when API key provided.
 - Falls back to local current_aqi.py or results/current_report.* (but ignores stale repeated values).
 - Optional external forecast URL.
 - Saved ARIMA model fallback for forecasts (if present).
 - Minimal CSS: background image + subtle overlay for readability.
 - Fixed parsing/display of reasons & suggestions.
 - Connected observed point to forecast (optional) so plot shows line connectivity.
 - Safe handling of st.secrets (won't crash if secrets file missing).
 - Copy this file to project root and run: `streamlit run app.py`.

Before running:
 - (recommended) create .streamlit/secrets.toml with OWM_KEY = "your_key" OR
 - set environment variable OWM_KEY OR paste key into sidebar input at runtime.
"""
import os
import json
import importlib
from datetime import datetime, timedelta
import requests
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import plotly.graph_objects as go

# ---------------- Page config & minimal CSS (background image + subtle overlay) ----------------
st.set_page_config(page_title="AQI Live + Forecast", layout="wide")

BG_IMAGE_URL = "https://images.unsplash.com/photo-1505842465776-3d0c1f6b2a4b?auto=format&fit=crop&w=2000&q=80"
st.markdown(
    f"""
    <style>
    [data-testid="stAppViewContainer"] {{
      background-image: url("{BG_IMAGE_URL}");
      background-size: cover;
      background-position: center;
      background-repeat: no-repeat;
      background-attachment: fixed;
    }}
    .app-overlay {{
      position: fixed;
      inset: 0;
      background: rgba(255,255,255,0.90); /* subtle wash for readability */
      pointer-events: none;
      z-index: -1;
    }}
    /* Try to lightly increase panel background opacity for readability */
    .css-1d391kg, .css-18e3th9, .css-1offfwp {{
      background-color: rgba(255,255,255,0.96) !important;
    }}
    </style>
    <div class="app-overlay"></div>
    """,
    unsafe_allow_html=True,
)

# ---------------- Config ----------------
MODEL_PATH = "saved_models/arima_aqi_weather_only.joblib"  # optional
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# OpenWeather endpoints
OWM_WEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"
OWM_FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"
OWM_AIR_URL = "https://api.openweathermap.org/data/2.5/air_pollution"
OWM_AIR_FORECAST_URL = "https://api.openweathermap.org/data/2.5/air_pollution/forecast"

# Optional OpenAQ fallback (v3)
OPENAQ_URL = "https://api.openaq.org/v3/latest"

LOCATIONS = {
    "Indore - MG Road": (22.718, 75.847),
    "Indore - Vijay Nagar": (22.735, 75.88),
    "Indore - AB Road": (22.726, 75.889),
    "Indore - Rajwada": (22.7196, 75.855),
    "Bhopal - MP Nagar": (23.238, 77.4125)
}

# ---------------- AQI breakpoints ----------------
PM25_BP = [(0,30,0,50),(31,60,51,100),(61,90,101,200),(91,120,201,300),(121,250,301,400),(251,500,401,500)]
PM10_BP = [(0,50,0,50),(51,100,51,100),(101,250,101,200),(251,350,201,300),(351,430,301,400)]
BP_MAP = {
    "pm2_5": PM25_BP, "pm2.5": PM25_BP, "pm25": PM25_BP,
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
    for k,v in meas.items():
        if v is None or (isinstance(v,(float,int)) and (np.isnan(v) or np.isinf(v))):
            continue
        key = k.lower().replace(".", "_")
        bps = BP_MAP.get(key)
        if bps:
            si = calc_subindex(v, bps)
            if si is not None:
                sub[key.upper()] = round(si, 2)
    if not sub:
        return None, None, {}
    dominant, dom_si = max(sub.items(), key=lambda x: x[1])
    final_aqi = int(round(dom_si))
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

# ---------------- parse list-like fields (reasons/suggestions) ----------------
def parse_list_field(x):
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return [str(i) for i in x]
    if isinstance(x, str):
        s = x.strip()
        # try JSON
        try:
            loaded = json.loads(s)
            if isinstance(loaded, (list, tuple)):
                return [str(i) for i in loaded]
        except Exception:
            pass
        # try python literal (safe-ish)
        try:
            val = eval(s, {"__builtins__": None}, {})
            if isinstance(val, (list, tuple)):
                return [str(i) for i in val]
        except Exception:
            pass
        # fallback: treat full string as one item
        return [s]
    return [str(x)]

# ---------------- OpenWeather fetch helpers ----------------
def fetch_owm_air(lat, lon, api_key):
    if not api_key:
        return None
    params = {"lat": lat, "lon": lon, "appid": api_key}
    r = requests.get(OWM_AIR_URL, params=params, timeout=10)
    r.raise_for_status()
    js = r.json()
    lst = js.get("list", [])
    if not lst:
        return None
    rec = lst[0]
    comps = rec.get("components", {})
    return {"components": comps, "timestamp": rec.get("dt"), "raw": rec}

def fetch_owm_forecast_weather(lat, lon, api_key, hours=48):
    if not api_key:
        return pd.DataFrame()
    params = {"lat": lat, "lon": lon, "appid": api_key, "units": "metric"}
    r = requests.get(OWM_FORECAST_URL, params=params, timeout=12)
    r.raise_for_status()
    js = r.json()
    rows = []
    for item in js.get("list", [])[: max(1, (hours//3) + 1)]:
        ts = datetime.utcfromtimestamp(item.get("dt"))
        rows.append({
            "timestamp": ts,
            "temperature": item.get("main", {}).get("temp"),
            "humidity": item.get("main", {}).get("humidity"),
            "pressure": item.get("main", {}).get("pressure"),
            "wind_speed": item.get("wind", {}).get("speed"),
            "wind_deg": item.get("wind", {}).get("deg"),
        })
    df = pd.DataFrame(rows)
    return df

def fetch_owm_air_forecast(lat, lon, api_key, hours=48):
    if not api_key:
        return None
    params = {"lat": lat, "lon": lon, "appid": api_key}
    r = requests.get(OWM_AIR_FORECAST_URL, params=params, timeout=12)
    r.raise_for_status()
    js = r.json()
    lst = js.get("list", [])[: max(1, hours)]
    rows = []
    for rec in lst:
        ts = rec.get("dt")
        comps = rec.get("components", {})
        rows.append({"timestamp": datetime.utcfromtimestamp(ts) if ts else None, **comps})
    if rows:
        return pd.DataFrame(rows)
    return None

# ---------------- local current_aqi loader (with stale-detection) ----------------
def try_run_local_current_aqi(ignore_if_stale=True, stale_repeat_threshold=3):
    report = None
    try:
        if os.path.exists("current_aqi.py"):
            ca = importlib.import_module("current_aqi")
            importlib.reload(ca)
            if hasattr(ca, "calculate_current_aqi"):
                try:
                    out = ca.calculate_current_aqi()
                    if isinstance(out, dict):
                        report = out
                except Exception:
                    pass
    except Exception:
        pass

    csv_path = os.path.join(RESULTS_DIR, "current_report.csv")
    json_path = os.path.join(RESULTS_DIR, "current_report.json")
    if report is None and os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            if not df.empty:
                report = df.iloc[-1].to_dict()
        except Exception:
            report = None
    if report is None and os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as fh:
                report = json.load(fh)
        except Exception:
            report = None

    # stale-check: if CSV has last N identical AQI values -> return None so app uses remote
    if ignore_if_stale and os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            aqi_cols = [c for c in df.columns if 'aqi' in c.lower()]
            if aqi_cols:
                col = aqi_cols[0]
                tail = df[col].dropna().astype(float) if not df[col].dropna().empty else pd.Series([])
                if len(tail) >= stale_repeat_threshold:
                    last_vals = tail.tail(stale_repeat_threshold).values
                    if np.allclose(last_vals, last_vals[0]):
                        return None
        except Exception:
            pass

    return report

# ---------------- UI: sidebar controls & safe secrets retrieval ----------------
st.title("AQI Live + Forecast — OpenWeather powered")

with st.sidebar:
    st.header("Settings")
    owm_key_input = st.text_input("OpenWeather API key (paste here or set env/.streamlit/secrets)", value=os.getenv("OWM_KEY", ""))
    external_forecast_url = st.text_input("External forecast URL (optional)")
    source = st.selectbox("Live AQI source (priority)", ["local (current_aqi.py/results)", "OpenWeather (recommended)", "OpenAQ (fallback)"])
    refresh = st.button("Refresh now")
    forecast_hours = st.slider("Forecast hours", 6, 72, 24)
    smoothing = st.slider("Forecast smoothing (rolling window)", 0, 6, 1)
    show_ci = st.checkbox("Show 95% CI (if available)", True)
    anchor_observed = st.checkbox("Anchor observed to forecast start", True)

# safe read from st.secrets if present (do not crash when secrets missing)
OWM_KEY = owm_key_input or os.getenv("OWM_KEY", "")
try:
    # st.secrets may throw on some setups if secrets file absent; guard it
    sec = {}
    try:
        sec = getattr(st, "secrets", {}) or {}
    except Exception:
        sec = {}
    sec_key = sec.get("OWM_KEY") if isinstance(sec, dict) else None
    if not OWM_KEY and sec_key:
        OWM_KEY = sec_key
except Exception:
    # ignore secrets errors
    pass

if not OWM_KEY:
    st.sidebar.info("OpenWeather key not set — paste in sidebar or create .streamlit/secrets.toml or set OWM_KEY env var.")

left, right = st.columns([2,1])
with right:
    st.markdown("**Quick info**")
    st.write("Priority: local -> OpenWeather -> OpenAQ. External forecast URL optional.")
with left:
    st.markdown("### Live remote data & current AQI")
    loc = st.selectbox("Choose location", list(LOCATIONS.keys()), index=0)
    lat, lon = LOCATIONS[loc]
    st.write(f"Coordinates: {lat:.4f}, {lon:.4f}")

# ---------------- Acquire live AQI (local first, else remote) ----------------
local_report = None
remote_pollutants = {}
remote_station = None
owm_air_snapshot = None

# Try local (with stale-detection)
if source.startswith("local"):
    local_report = try_run_local_current_aqi(ignore_if_stale=True)
    if local_report is None:
        st.info("Local report missing or detected stale — will try remote sources.")

# If OpenWeather chosen or local not present -> try OpenWeather air snapshot
if (source == "OpenWeather (recommended)") or (local_report is None and source.startswith("local")):
    if OWM_KEY:
        try:
            owm_air_snapshot = fetch_owm_air(lat, lon, OWM_KEY)
            if owm_air_snapshot and "components" in owm_air_snapshot:
                remote_pollutants = owm_air_snapshot["components"]
        except Exception as e:
            st.warning("OpenWeather air fetch failed: " + str(e))
    else:
        st.info("OpenWeather key not available; skipping OWM fetch.")

# If still no pollutants and OpenAQ chosen/fallback -> try OpenAQ
if not remote_pollutants and (source == "OpenAQ (fallback)" or (local_report is None and not owm_air_snapshot)):
    try:
        r = requests.get(OPENAQ_URL, params={"coordinates": f"{lat},{lon}", "radius": 5000, "limit": 3}, timeout=12)
        r.raise_for_status()
        js = r.json()
        results = js.get("results", []) or []
        if results:
            best = None
            for res in results:
                if res.get("latest"):
                    best = res; break
            if best is None:
                best = results[0]
            measurements = best.get("latest", []) if isinstance(best.get("latest"), list) else best.get("measurements", []) or best.get("parameters", [])
            for m in measurements:
                param = m.get("parameter") or m.get("param") or m.get("name")
                val = m.get("value", None)
                if param:
                    remote_pollutants[param.lower()] = val
            remote_station = best
    except Exception as e:
        st.warning("OpenAQ fetch failed: " + str(e))

# ---------------- Display live AQI & reasons nicely ----------------
def aqi_card_html(aqi_value, title="Live AQI"):
    cat = aqi_category(aqi_value)
    colors = {"Good":"#2ECC71","Satisfactory":"#99CC33","Moderate":"#F1C40F","Poor":"#E67E22","Very Poor":"#E74C3C","Severe":"#7F1D1D","Unknown":"#888"}
    bg = colors.get(cat, "#888")
    html = f"""
    <div style="padding:12px;border-radius:8px;background:{bg};color:white;max-width:360px;">
      <div style="font-size:13px;font-weight:600">{title}</div>
      <div style="font-size:36px;margin-top:6px;font-weight:700">{int(aqi_value)}</div>
      <div style="font-size:12px;margin-top:6px">{cat}</div>
    </div>
    """
    return html

displayed_aqi = None
displayed_subindices = None
displayed_dominant = None

# If local_report exists and not stale:
if local_report:
    aqi_val = None
    for k in ("AQI","aqi","predicted_AQI","aqi_value"):
        if k in local_report:
            aqi_val = local_report.get(k); break
    if aqi_val is not None:
        displayed_aqi = float(aqi_val)
        st.markdown(aqi_card_html(displayed_aqi, "Live AQI (local)"), unsafe_allow_html=True)
        st.write("**Reasons:**")
        for r in parse_list_field(local_report.get("reasons")):
            st.write("•", r)
        st.write("**Suggestions:**")
        for s in parse_list_field(local_report.get("suggestions")):
            st.write("•", s)
    else:
        st.warning("Local report present but no AQI field — falling back to remote snapshot.")

# If no local AQI displayed, try compute from remote pollutants
if displayed_aqi is None and remote_pollutants:
    meas = {}
    for k,v in remote_pollutants.items():
        k0 = k.lower()
        if any(x in k0 for x in ("pm25","pm2","pm10","no2","so2","o3","nh3","co")):
            meas[k0] = v
    curr_aqi, dom, sub = compute_aqi_from_pollutants(meas)
    if curr_aqi is not None:
        displayed_aqi = float(curr_aqi)
        displayed_subindices = sub
        displayed_dominant = dom
        st.markdown(aqi_card_html(displayed_aqi, "Live AQI (snapshot)"), unsafe_allow_html=True)
        if dom:
            st.write("Dominant pollutant:", dom)
        if sub:
            st.write("Subindices:")
            for k,v in sub.items():
                st.write("•", f"{k}: {v}")
    else:
        st.warning("Could not compute AQI from snapshot pollutants.")

if displayed_aqi is None:
    st.info("No live AQI available (local or remote).")

# ---------------- Weather (exogenous) and forecast fetching ----------------
with st.expander("Weather (used as exogenous for model)"):
    weather_df = pd.DataFrame()
    if OWM_KEY:
        try:
            weather_df = fetch_owm_forecast_weather(lat, lon, OWM_KEY, hours=max(24, forecast_hours))
            if not weather_df.empty:
                st.dataframe(weather_df.head(24))
            else:
                st.write("OpenWeather forecast returned no data.")
        except Exception as e:
            st.warning("OpenWeather forecast failed: " + str(e))
            # optional fallback to Open-Meteo (not required)
            try:
                params = {
                    "latitude": lat, "longitude": lon,
                    "hourly": ",".join(["temperature_2m","relativehumidity_2m","pressure_msl","windspeed_10m"]),
                    "forecast_days": max(1, (forecast_hours//24)+1),
                    "timezone": "UTC"
                }
                r = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=12)
                r.raise_for_status()
                js = r.json().get("hourly", {})
                times = js.get("time", [])[: max(24, forecast_hours)]
                rows = []
                for i,t in enumerate(times):
                    rows.append({
                        "timestamp": t,
                        "temperature": js.get("temperature_2m", [None])[i] if i < len(js.get("temperature_2m", [])) else None,
                        "humidity": js.get("relativehumidity_2m", [None])[i] if i < len(js.get("relativehumidity_2m", [])) else None,
                    })
                weather_df = pd.DataFrame(rows)
                st.dataframe(weather_df.head(24))
            except Exception as e2:
                st.write("Weather not available:", e2)
    else:
        st.write("OpenWeather key not provided; weather forecast not fetched here.")

# ---------------- External forecast helper ----------------
def fetch_external_forecast(url, lat, lon, hours=24):
    if not url:
        return None
    try:
        r = requests.get(url, params={"lat": lat, "lon": lon, "hours": hours}, timeout=12)
        r.raise_for_status()
        js = r.json()
    except Exception:
        r = requests.get(url, timeout=12)
        r.raise_for_status()
        js = r.json()
    if isinstance(js, dict):
        if "timestamps" in js and "aqi_forecast" in js:
            ts = js["timestamps"]; vals = js["aqi_forecast"]
        elif "forecast" in js and isinstance(js["forecast"], list):
            lst = js["forecast"]
            ts = [x.get("timestamp") for x in lst]; vals = [x.get("aqi") or x.get("aqi_forecast") for x in lst]
        else:
            ts = js.get("timestamps") or js.get("time") or []
            vals = js.get("aqi_forecast") or js.get("aqi") or []
    else:
        return None
    if not ts or not vals:
        return None
    df = pd.DataFrame({"timestamp": pd.to_datetime(ts), "aqi_forecast": np.array(vals, dtype=float)})
    return df

# ---------------- Forecast generation (external -> OWM air forecast -> local model fallback) ----------------
forecast_df = None

# 1) try external forecast URL if given
if external_forecast_url:
    try:
        ext_df = fetch_external_forecast(external_forecast_url, lat, lon, hours=forecast_hours)
        if ext_df is not None and not ext_df.empty:
            forecast_df = ext_df.sort_values("timestamp").reset_index(drop=True)
            st.info("Using external forecast endpoint.")
        else:
            st.warning("External forecast endpoint returned no usable data.")
    except Exception as e:
        st.warning("External forecast fetch failed: " + str(e))

# 2) try OpenWeather air forecast (convert components -> AQI)
if forecast_df is None and OWM_KEY:
    try:
        owm_air_fore = fetch_owm_air_forecast(lat, lon, OWM_KEY, hours=forecast_hours)
        if owm_air_fore is not None and not owm_air_fore.empty:
            temp = owm_air_fore.copy()
            temp['timestamp'] = pd.to_datetime(temp['timestamp'])
            aqi_vals = []
            for _, row in temp.iterrows():
                comps = {k: row.get(k) for k in temp.columns if k in ['pm2_5','pm10','no2','so2','o3','co']}
                aqi, dom, sub = compute_aqi_from_pollutants(comps)
                aqi_vals.append(aqi if aqi is not None else np.nan)
            temp['aqi_forecast'] = pd.Series(aqi_vals)
            temp = temp[['timestamp','aqi_forecast']].dropna().reset_index(drop=True)
            if not temp.empty:
                forecast_df = temp.head(forecast_hours)
    except Exception:
        pass

# 3) fallback to saved ARIMA model (if present)
if forecast_df is None and os.path.exists(MODEL_PATH):
    try:
        saved = joblib.load(MODEL_PATH)
        if isinstance(saved, dict):
            sm_res = saved.get("model"); meta = saved.get("meta", {})
        else:
            sm_res = saved; meta = {}
        exog_cols = meta.get("exog_cols", []) if isinstance(meta, dict) else []
    except Exception as e:
        st.warning("Failed to load saved model: " + str(e))
        sm_res = None; exog_cols = []
    if sm_res is not None:
        try:
            hours = forecast_hours
            X_future = None
            if exog_cols and (not weather_df.empty):
                col_map = {}
                for c in exog_cols:
                    match = None
                    for k in weather_df.columns:
                        if k.lower() == c.lower() or c.lower() in k.lower():
                            match = k; break
                    col_map[c] = match
                rows = []
                for i in range(hours):
                    row = []
                    for c in exog_cols:
                        match = col_map.get(c)
                        val = weather_df.iloc[i].get(match, np.nan) if match in weather_df.columns else np.nan
                        try: row.append(float(val))
                        except: row.append(np.nan)
                    rows.append(row)
                X_future = np.array(rows, dtype=float)
                X_future = np.where(np.isfinite(X_future), X_future, np.nan)
                for j in range(X_future.shape[1]):
                    col = X_future[:, j]
                    med = np.nanmedian(col)
                    if np.isnan(med): med = 0.0
                    col[np.isnan(col)] = med
                    X_future[:, j] = col
            if X_future is not None:
                pred = sm_res.get_forecast(steps=hours, exog=X_future)
            else:
                pred = sm_res.get_forecast(steps=hours)
            mean = np.array(pred.predicted_mean).astype(float)
            s = pd.Series(mean)
            s = s.interpolate(limit_direction="both").fillna(method="ffill").fillna(method="bfill").fillna(0.0)
            mean = s.values.astype(float)
            start_ts = datetime.utcnow().replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
            times = [start_ts + timedelta(hours=i) for i in range(forecast_hours)]
            forecast_df = pd.DataFrame({"timestamp": times, "aqi_forecast": np.round(mean, 3)})
            if hasattr(pred, "conf_int"):
                try:
                    conf = pred.conf_int()
                    forecast_df["lower95"] = conf.iloc[:,0].values
                    forecast_df["upper95"] = conf.iloc[:,1].values
                except Exception:
                    pass
            if smoothing > 0 and len(forecast_df) >= smoothing:
                forecast_df["aqi_forecast"] = forecast_df["aqi_forecast"].rolling(smoothing, min_periods=1).mean().round(3)
        except Exception as e:
            st.warning("Local model forecasting failed: " + str(e))

if forecast_df is None:
    st.info("No forecast available (external / OpenWeather forecast / local model not available).")

# ---------------- Observed + Forecast plotting ----------------
st.header("Observed (live) + Forecast")
fig = go.Figure()

# observed data (local_report preferred)
obs_x = []; obs_y = []
if local_report:
    ts = local_report.get("timestamp") or local_report.get("time")
    try:
        ots = pd.to_datetime(ts) if ts else datetime.utcnow()
    except:
        ots = datetime.utcnow()
    aqi_val = None
    for k in ("AQI","aqi","predicted_AQI","aqi_value"):
        if k in local_report:
            aqi_val = local_report.get(k); break
    if aqi_val is not None:
        obs_x.append(ots); obs_y.append(float(aqi_val))
elif displayed_aqi is not None:
    obs_x.append(datetime.utcnow()); obs_y.append(float(displayed_aqi))

# anchor observed to forecast start to show connectivity
if anchor_observed and obs_x and (forecast_df is not None and not forecast_df.empty):
    try:
        start_ts = pd.to_datetime(forecast_df['timestamp'].iloc[0])
        if obs_x[0] < start_ts:
            obs_x.append(start_ts); obs_y.append(obs_y[0])
    except Exception:
        pass

# add observed trace
if obs_x and obs_y:
    fig.add_trace(go.Scatter(
        x=obs_x, y=obs_y, mode='markers+lines' if len(obs_x) > 1 else 'markers',
        name='Observed', marker=dict(size=9), line=dict(width=3)
    ))

# add forecast trace
if forecast_df is not None and not forecast_df.empty:
    fig.add_trace(go.Scatter(
        x=forecast_df['timestamp'], y=forecast_df['aqi_forecast'],
        mode='lines+markers', name='Forecast', marker=dict(size=7),
        line=dict(shape='spline', width=2)
    ))
    if show_ci and 'lower95' in forecast_df and 'upper95' in forecast_df:
        fig.add_trace(go.Scatter(
            x=list(forecast_df['timestamp']) + list(forecast_df['timestamp'])[::-1],
            y=list(forecast_df['upper95']) + list(forecast_df['lower95'])[::-1],
            fill='toself', fillcolor='rgba(200,200,200,0.2)', line=dict(color='rgba(0,0,0,0)'),
            name='95% CI', hoverinfo='skip'
        ))

# dynamic y-axis range
vals = []
if obs_y: vals += obs_y
if forecast_df is not None and not forecast_df.empty:
    vals += list(forecast_df['aqi_forecast'].astype(float))
if vals:
    ymin = float(np.min(vals)); ymax = float(np.max(vals))
    pad = max(5, (ymax - ymin) * 0.12)
    fig.update_yaxes(range=[max(0, ymin - pad), ymax + pad])
else:
    fig.update_yaxes(autorange=True)

fig.update_layout(xaxis_title="Time (UTC)", yaxis_title="AQI", height=520, legend=dict(orientation="v", x=0.92, y=0.95))
st.plotly_chart(fig, use_container_width=True)

# ---------------- Footer: data previews & help expander ----------------
with st.expander("Data preview & help"):
    st.write("Priority for live AQI: local current_aqi.py/results (unless stale) → OpenWeather snapshot → OpenAQ fallback.")
    if local_report:
        st.write("Local report preview:")
        try:
            preview = {}
            for k in ["timestamp","AQI","aqi","category","predicted_AQI","reasons","suggestions"]:
                if k in local_report:
                    preview[k] = local_report[k]
            st.json(preview)
        except Exception:
            st.write(local_report)
    if owm_air_snapshot:
        st.write("OpenWeather air snapshot (components):")
        st.json(owm_air_snapshot.get("components", {}))
    if remote_station:
        st.write("OpenAQ station (raw):")
        st.json(remote_station)
    if forecast_df is not None and not forecast_df.empty:
        st.write("Forecast preview (first rows):")
        st.dataframe(forecast_df.head(min(24, len(forecast_df))))
    else:
        st.write("No forecast available to preview.")

st.caption("If the app shows the same AQI everywhere, paste 'Local report preview' and/or 'Forecast preview' outputs here and I'll help debug the source (likely local file or stale feed).")
