#!/usr/bin/env python3
"""
current_with_reason_openweather_fixed.py

- Tailored for OpenWeather Air Pollution CSV exports (values in µg/m3).
- Computes 8-hour O3 mean when enough history/timestamps available.
- Uses CPCB-like breakpoints (O3 8-hour style).
- Robust to missing columns, avoids pandas Series truthiness errors.
- Prints concise summary and saves results/current_report.json & .csv.
"""
import os
import json
from datetime import datetime
import pandas as pd
import math

# ---------------------------
# Paths (adjust if needed)
# ---------------------------
AQI_CSV = os.path.join("data", "live_processed", "aqi_live_master.csv")
WEATHER_CSV = os.path.join("data", "live_processed", "weather_live_master.csv")
TRAFFIC_CSV = os.path.join("data", "live_processed", "traffic_timeseries.csv")
OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------
# Breakpoints (coarse CPCB-like)
# ---------------------------
PM25_BP = [(0,30,0,50),(31,60,51,100),(61,90,101,200),(91,120,201,300),(121,250,301,400),(251,500,401,500)]
PM10_BP = [(0,50,0,50),(51,100,51,100),(101,250,101,200),(251,350,201,300),(351,430,301,400),(431,600,401,500)]
NO2_BP  = PM10_BP
SO2_BP  = PM10_BP
# O3 8-hour breakpoints (coarse)
O3_8BP  = [(0,50,0,50),(51,100,51,100),(101,168,101,200),(169,208,201,300),(209,748,301,500)]

BP_DICT = {
    "PM2.5": PM25_BP,
    "PM10": PM10_BP,
    "NO2": NO2_BP,
    "SO2": SO2_BP,
    "O3_8HR": O3_8BP,
    "NH3": PM25_BP
}

# ---------------------------
# Helpers: safe CSV read
# ---------------------------
def safe_read_csv(path):
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        try:
            return pd.read_csv(path, low_memory=False, engine="python")
        except Exception:
            return None

# ---------------------------
# Subindex calculation
# ---------------------------
def subindex(value, bps):
    try:
        C = float(value)
    except Exception:
        return None
    for Cl, Ch, Il, Ih in bps:
        if Cl <= C <= Ch:
            if Ch - Cl == 0:
                return float(Ih)
            return round(((Ih - Il) / (Ch - Cl)) * (C - Cl) + Il, 2)
    # if above final range return final Ih
    last = bps[-1]
    if C > last[1]:
        return float(last[3])
    return None

# ---------------------------
# O3 detection & 8-hour computation (OpenWeather: values in µg/m3)
# ---------------------------
def find_o3_column(df):
    if df is None:
        return None
    cols = list(df.columns)
    low = [c.lower() for c in cols]
    # prefer explicit candidates
    for cand in ("o3_8hr","o3_8","o3_1hr","o3","o3_ugm3","o3_ug","o3_µg"):
        if cand in low:
            return cols[low.index(cand)]
    # fallback: first column starting with 'o3'
    for i,c in enumerate(low):
        if c.startswith("o3"):
            return cols[i]
    return None

def compute_o3_8hr_value(df, o3_col):
    """
    Returns tuple: (raw_last_value, used_for_aqi_value, method_string)
    - Uses 8-hour resample/rolling when possible, else falls back to last instant value.
    - Assumes input O3 values are in µg/m3 (OpenWeather).
    """
    if o3_col is None or df is None:
        return None, None, "no_o3_column"
    # safe numeric series
    ser = pd.to_numeric(df[o3_col], errors="coerce").dropna()
    raw_last = float(ser.iloc[-1]) if not ser.empty else None

    # detect timestamp column
    ts_col = None
    cols_low = [c.lower() for c in df.columns]
    for candidate in ("timestamp","time","date","datetime"):
        if candidate in cols_low:
            ts_col = df.columns[cols_low.index(candidate)]
            break

    # if timestamp exists -> attempt hourly resample and compute 8-hr rolling mean
    if ts_col is not None:
        try:
            tmp = df[[ts_col, o3_col]].copy()
            tmp[ts_col] = pd.to_datetime(tmp[ts_col], errors="coerce")
            tmp = tmp.dropna(subset=[ts_col, o3_col])
            tmp = tmp.sort_values(ts_col)
            if tmp.shape[0] >= 1:
                tmp = tmp.set_index(ts_col)
                # resample hourly (use lowercase 'h' to avoid FutureWarning)
                hourly = tmp[o3_col].resample("h").mean().dropna()
                if hourly.shape[0] >= 8:
                    o3_8 = hourly.rolling(window=8, min_periods=6).mean().dropna()
                    if not o3_8.empty:
                        used = float(o3_8.iloc[-1])
                        return raw_last, round(used,2), "8hr_resample"
        except Exception:
            # fallback silently to other methods
            pass

    # fallback: if raw history has >=8 samples use rolling mean
    try:
        if ser.shape[0] >= 8:
            r8 = ser.rolling(window=8, min_periods=6).mean().dropna()
            if not r8.empty:
                used = float(r8.iloc[-1])
                return raw_last, round(used,2), "8hr_rolling"
    except Exception:
        pass

    # final fallback: instant last value
    return raw_last, raw_last if raw_last is not None else None, "instant"

# ---------------------------
# Generic helpers for finding columns & numeric extraction
# ---------------------------
def find_col_by_keywords(series_or_df, keywords):
    if series_or_df is None:
        return None
    if isinstance(series_or_df, pd.Series):
        keys = list(series_or_df.index)
    else:
        keys = list(series_or_df.columns)
    low = [k.lower() for k in keys]
    for kw in keywords:
        for i,k in enumerate(low):
            if kw in k:
                return keys[i]
    return None

def safe_extract_numeric(row_or_df, colname):
    if row_or_df is None or colname is None:
        return None
    try:
        v = row_or_df[colname]
        if pd.isna(v):
            return None
        return float(v)
    except Exception:
        return None

# ---------------------------
# Avoid Series truthiness: choose readable source for weather/traffic
# ---------------------------
def choose_row_source(row, df):
    # prefer explicit last-row Series if valid
    if row is not None:
        if not getattr(row, "empty", False):
            return row
    # else use DataFrame last row
    if df is not None and not df.empty:
        return df.iloc[-1]
    return None

# ---------------------------
# Main pipeline
# ---------------------------
def main():
    # load AQI/master CSV
    aqi_df = safe_read_csv(AQI_CSV)
    if aqi_df is None or aqi_df.empty:
        print("[ERROR] AQI CSV missing or empty:", AQI_CSV)
        raise SystemExit(1)

    # O3 detection and 8-hr logic (OpenWeather values in µg/m3)
    o3_col = find_o3_column(aqi_df)
    o3_raw, o3_used, o3_method = compute_o3_8hr_value(aqi_df, o3_col)

    # detect pollutant columns
    pm25_col = find_col_by_keywords(aqi_df, ["pm2.5","pm2_5","pm25","pm2"])
    pm10_col = find_col_by_keywords(aqi_df, ["pm10","pm_10"])
    no2_col  = find_col_by_keywords(aqi_df, ["no2","no_2"])
    so2_col  = find_col_by_keywords(aqi_df, ["so2","so_2"])
    nh3_col  = find_col_by_keywords(aqi_df, ["nh3","nh_3"])

    # helper to get last valid numeric
    def last_valid_value(df, col):
        if col is None:
            return None
        try:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            return float(s.iloc[-1]) if not s.empty else None
        except Exception:
            return None

    pm25_val = last_valid_value(aqi_df, pm25_col)
    pm10_val = last_valid_value(aqi_df, pm10_col)
    no2_val  = last_valid_value(aqi_df, no2_col)
    so2_val  = last_valid_value(aqi_df, so2_col)
    nh3_val  = last_valid_value(aqi_df, nh3_col)

    # prepare pollutant map (O3 uses o3_used)
    pollutant_map = {
        "PM2.5": pm25_val,
        "PM10": pm10_val,
        "NO2": no2_val,
        "SO2": so2_val,
        "O3": o3_used,
        "NH3": nh3_val
    }

    # compute subindices
    subindices = {}
    for pname, pval in pollutant_map.items():
        if pval is None:
            subindices[pname] = None
            continue
        if pname == "O3":
            si = subindex(pval, BP_DICT["O3_8HR"])
        else:
            si = subindex(pval, BP_DICT.get(pname))
        subindices[pname] = None if si is None else round(si,2)

    valid_subs = {k:v for k,v in subindices.items() if v is not None}
    if not valid_subs:
        print("[ERROR] No valid pollutant subindices could be computed.")
        raise SystemExit(1)

    final_aqi = int(round(max(valid_subs.values())))
    dominant = max(valid_subs, key=valid_subs.get)
    aqi_category = ("Good" if final_aqi <=50 else "Satisfactory" if final_aqi<=100 else
                    "Moderate" if final_aqi<=200 else "Poor" if final_aqi<=300 else
                    "Very Poor" if final_aqi<=400 else "Severe")

    # load weather & traffic dataframes and choose safe row sources
    weather_df = safe_read_csv(WEATHER_CSV)
    traffic_df = safe_read_csv(TRAFFIC_CSV)
    # raw last-row candidates (may be Series or None)
    weather_last_candidate = weather_df.iloc[-1] if (weather_df is not None and not weather_df.empty) else None
    traffic_last_candidate = traffic_df.iloc[-1] if (traffic_df is not None and not traffic_df.empty) else None
    weather_source = choose_row_source(weather_last_candidate, weather_df)
    traffic_source = choose_row_source(traffic_last_candidate, traffic_df)

    # detect keys from chosen sources (no boolean ambiguity)
    wind_key = find_col_by_keywords(weather_source, ["wind_speed","windspeed","wind"])
    hum_key  = find_col_by_keywords(weather_source, ["humidity","hum"])
    prec_key = find_col_by_keywords(weather_source, ["rain","precip","precipitation"])
    vis_key  = find_col_by_keywords(weather_source, ["vis","visibility"])

    veh_key  = find_col_by_keywords(traffic_source, ["count","vehicle","volume","veh_count","vehicles"])
    speed_key= find_col_by_keywords(traffic_source, ["speed","avg_speed","avg_speed_km","avg_speed_kph"])

    # safe numeric extraction
    wind_speed = safe_extract_numeric(weather_source, wind_key)
    humidity   = safe_extract_numeric(weather_source, hum_key)
    precip     = safe_extract_numeric(weather_source, prec_key)
    visibility = safe_extract_numeric(weather_source, vis_key)
    veh_count  = safe_extract_numeric(traffic_source, veh_key)
    avg_speed  = safe_extract_numeric(traffic_source, speed_key)

    # build reasons
    reasons = []
    reasons.append(f"Dominant pollutant: {dominant} (subindex {valid_subs[dominant]}).")
    reasons.append(f"O3 diagnostic: raw_last={o3_raw} µg/m3  used_for_aqi={o3_used} µg/m3  method={o3_method}  col={o3_col}")

    if wind_speed is not None:
        if wind_speed < 2:
            reasons.append(f"Low wind ({wind_speed} m/s) — poor dispersion.")
        elif wind_speed < 4:
            reasons.append(f"Light wind ({wind_speed} m/s) — limited dispersion.")
        else:
            reasons.append(f"Moderate/high wind ({wind_speed} m/s) — helps dispersion.")
    if humidity is not None:
        if humidity > 75:
            reasons.append(f"High humidity ({humidity}%) — hygroscopic growth can increase PM.")
    if precip is not None and precip > 0:
        reasons.append(f"Recent precipitation ({precip}) — washout likely reduced PM.")
    if visibility is not None:
        if visibility < 2000:
            reasons.append(f"Low visibility ({visibility}) — heavy particulate loading.")

    if veh_count is not None:
        if veh_count > 400:
            reasons.append(f"High vehicle count ({veh_count}) — likely higher vehicular PM/NO2.")
        elif veh_count > 200:
            reasons.append(f"Moderate vehicle count ({veh_count}).")
    if avg_speed is not None and avg_speed < 20:
        reasons.append(f"Low avg traffic speed ({avg_speed} km/h) — congestion contributes to emissions.")

    if weather_source is None:
        reasons.append("Weather data not available in live feed.")
    if traffic_source is None:
        reasons.append("Traffic data not available in live feed.")

    # suggestions
    suggestions = []
    if "PM2.5" in dominant:
        suggestions.append("Avoid outdoor exercise; use masks; reduce vehicle travel if possible.")
    elif "PM10" in dominant:
        suggestions.append("Limit dusty outdoor activities; control construction dust.")
    else:
        suggestions.append("Monitor updates; reduce exposure if sensitive.")
    if wind_speed is not None and wind_speed < 2:
        suggestions.append("Reduce local emissions (avoid burning, limit non-essential traffic) until dispersion improves.")
    if precip is not None and precip > 0:
        suggestions.append("Pollution likely to improve after rain - check updated AQI later.")

    # report build
    report = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "aqi": int(final_aqi),
        "category": aqi_category,
        "pollutant_values_last": {
            "PM2.5": pm25_val, "PM10": pm10_val, "NO2": no2_val, "SO2": so2_val, "O3_used_µg/m3": o3_used, "NH3": nh3_val
        },
        "subindices": subindices,
        "dominant": dominant,
        "o3_diag": {"raw": o3_raw, "used": o3_used, "method": o3_method, "column": o3_col},
        "weather_snapshot": {"wind": wind_speed, "humidity": humidity, "precip": precip, "visibility": visibility},
        "traffic_snapshot": {"vehicle_count": veh_count, "avg_speed": avg_speed},
        "reasons": reasons,
        "suggestions": suggestions
    }

    # save outputs
    with open(os.path.join(OUT_DIR, "current_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    pd.DataFrame([report]).to_csv(os.path.join(OUT_DIR, "current_report.csv"), index=False)

    # concise print
    print("\n===== CURRENT AQI (OpenWeather-safe & fixed) =====")
    print(f"AQI: {report['aqi']}  ({report['category']})")
    print(f"Dominant pollutant: {report['dominant']}")
    print("O3 diagnostic:", report["o3_diag"])
    print("\nTop reasons:")
    for r in reasons[:6]:
        print(" -", r)
    print("\nSaved:", os.path.join(OUT_DIR, "current_report.json"))

if __name__ == "__main__":
    main()
