#!/usr/bin/env python3
"""
train_arima_clean.py

- Trains a SARIMAX (ARIMA) model for AQI forecasting using:
    - Historical AQI (required) from data/live_processed/aqi_live_master.csv
    - Historical weather (optional) from data/live_processed/weather_live_master.csv as exogenous variables
- Traffic is intentionally NOT used (you said traffic dataset is from another city).
- Saves model to saved_models/arima_aqi_weather_only.joblib
- Also saves an in-sample diagnostics CSV in results/
"""

import os
import joblib
import warnings
from datetime import datetime
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Paths
AQI_CSV = os.path.join("data", "live_processed", "aqi_live_master.csv")
WEATHER_CSV = os.path.join("data", "live_processed", "weather_live_master.csv")
OUT_MODEL = os.path.join("saved_models", "arima_aqi_weather_only.joblib")
os.makedirs("saved_models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# ---------------- helpers ----------------
def safe_read(path):
    if not os.path.exists(path): return None
    try:
        return pd.read_csv(path)
    except:
        try:
            return pd.read_csv(path, engine="python", low_memory=False)
        except:
            return None

def find_time_col(df):
    if df is None: return None
    low = [c.lower() for c in df.columns]
    for cand in ("timestamp","time","date","datetime","dt"):
        if cand in low:
            return df.columns[low.index(cand)]
    # fallback: any column with 'time' or 'date'
    for i,c in enumerate(df.columns):
        if "time" in c.lower() or "date" in c.lower():
            return c
    return None

def find_aqi_col(df):
    if df is None: return None
    low = [c.lower() for c in df.columns]
    for cand in ("aqi","aqi_value","predicted_aqi","aqi_obs","value"):
        if cand in low:
            return df.columns[low.index(cand)]
    # fallback: any column that contains 'aqi'
    for i,c in enumerate(df.columns):
        if "aqi" in c.lower():
            return c
    return None

def select_weather_exog_cols(df):
    if df is None: return []
    keywords = ["wind","humidity","temp","temperature","rain","precip","visibility","pressure","cloud"]
    cols = []
    for c in df.columns:
        lc = c.lower()
        for kw in keywords:
            if kw in lc and c not in cols:
                cols.append(c)
    return cols

# ---------------- load AQI ----------------
aqi_df = safe_read(AQI_CSV)
if aqi_df is None or aqi_df.empty:
    raise SystemExit(f"[ERROR] AQI CSV missing or empty: {AQI_CSV}")

time_col = find_time_col(aqi_df)
if time_col:
    try:
        aqi_df[time_col] = pd.to_datetime(aqi_df[time_col], errors="coerce")
        aqi_df = aqi_df.sort_values(time_col).dropna(subset=[time_col]).set_index(time_col)
    except Exception:
        pass

aqi_col = find_aqi_col(aqi_df)
if aqi_col is None:
    raise SystemExit("[ERROR] Could not detect AQI column in aqi_live_master.csv. Ensure a column named 'aqi' or similar exists.")

y = pd.to_numeric(aqi_df[aqi_col], errors="coerce").dropna()
if y.empty:
    raise SystemExit("[ERROR] No numeric AQI values found.")

# ---------------- load weather (exog) ----------------
weather_df = safe_read(WEATHER_CSV)
exog_cols = []
X = None
if weather_df is not None and not weather_df.empty:
    tcol_w = find_time_col(weather_df)
    if tcol_w:
        try:
            weather_df[tcol_w] = pd.to_datetime(weather_df[tcol_w], errors="coerce")
            weather_df = weather_df.sort_values(tcol_w).dropna(subset=[tcol_w]).set_index(tcol_w)
        except Exception:
            pass
    # align to y index using nearest 1-hour tolerance
    try:
        merged = weather_df.reindex(weather_df.index.union(y.index)).sort_index()
        # reindex weather to y index using nearest with tolerance 1H
        exog_aligned = weather_df.reindex(y.index, method="nearest", tolerance=pd.Timedelta("1H"))
    except Exception:
        exog_aligned = weather_df.reindex(y.index, method="ffill")
    exog_cols = select_weather_exog_cols(exog_aligned)
    if exog_cols:
        X = exog_aligned[exog_cols].fillna(method="ffill").fillna(method="bfill")
    else:
        X = None

print(f"[INFO] Detected AQI column: {aqi_col} (N={len(y)}). Weather exog cols: {exog_cols}")

# ---------------- model selection & train ----------------
# ---------------- MODEL SELECTION & TRAINING (improved) ----------------
use_pmd = False
try:
    import pmdarima as pm
    use_pmd = True
except Exception:
    use_pmd = False

from statsmodels.tsa.statespace.sarimax import SARIMAX

# ensure index frequency if possible
try:
    if hasattr(y.index, "inferred_freq") and y.index.inferred_freq is None:
        y = y.asfreq('H')
except Exception:
    pass

order = (1,0,0)
seasonal_order = (0,0,0,0)

if use_pmd:
    print("[INFO] Using pmdarima.auto_arima with seasonal=True, m=24 ...")
    try:
        am = pm.auto_arima(
            y,
            exogenous=(X.values if X is not None else None),
            seasonal=True,
            m=24,
            stepwise=True,
            max_p=4, max_q=4, max_P=2, max_Q=2,
            max_d=2, max_D=1,
            start_p=0, start_q=0,
            information_criterion='aic',
            suppress_warnings=True,
            error_action='ignore',
            n_jobs=1
        )
        order = am.order
        seasonal_order = am.seasonal_order
        print(f"[INFO] auto_arima selected order={order} seasonal_order={seasonal_order}")
    except Exception as e:
        print("[WARN] auto_arima failed, falling back to grid search:", e)
        use_pmd = False

if not use_pmd:
    print("[INFO] pmdarima not available — running small grid search …")
    best_aic = np.inf
    best_order = (1,0,0)
    best_seasonal = (0,0,0,0)

    for p in range(0,3):
        for d in range(0,2):
            for q in range(0,3):
                # non seasonal
                try:
                    mod = SARIMAX(y, exog=X, order=(p,d,q),
                                  enforce_stationarity=False,
                                  enforce_invertibility=False)
                    res = mod.fit(disp=False, maxiter=100)
                    if res.aic < best_aic:
                        best_aic = res.aic
                        best_order = (p,d,q)
                        best_seasonal = (0,0,0,0)
                except Exception:
                    pass

                # simple seasonal
                try:
                    mod2 = SARIMAX(y, exog=X, order=(p,d,q),
                                   seasonal_order=(1,0,1,24),
                                   enforce_stationarity=False,
                                   enforce_invertibility=False)
                    res2 = mod2.fit(disp=False, maxiter=100)
                    if res2.aic < best_aic:
                        best_aic = res2.aic
                        best_order = (p,d,q)
                        best_seasonal = (1,0,1,24)
                except:
                    pass

    order = best_order
    seasonal_order = best_seasonal
    print(f"[INFO] selected order={order}, seasonal_order={seasonal_order}, AIC={best_aic:.2f}")

print(f"[INFO] Training final SARIMAX(order={order}, seasonal_order={seasonal_order})")
mod = SARIMAX(y, exog=X, order=order, seasonal_order=seasonal_order,
              enforce_stationarity=False, enforce_invertibility=False)
res = mod.fit(disp=False, maxiter=500)
print("[INFO] Model fit complete. AIC:", res.aic)

# Save model
model_meta = {
    "order": order,
    "seasonal_order": seasonal_order,
    "exog_cols": exog_cols,
    "training_end": str(y.index[-1]),
    "aic": float(res.aic)
}
joblib.dump({"model": res, "meta": model_meta}, OUT_MODEL)
print(f"[INFO] Saved model to {OUT_MODEL}")

# Save in-sample diagnostics
try:
    pred = res.get_prediction(start=0, end=len(y)-1, exog=X)
    pred_mean = pred.predicted_mean
    diag = pd.DataFrame({"y": y, "y_pred_in_sample": pred_mean}).reset_index()
    diag.to_csv(os.path.join("results","arima_insample.csv"), index=False)
    print("[INFO] saved results/arima_insample.csv")
except Exception:
    pass
