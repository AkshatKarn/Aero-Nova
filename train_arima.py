#!/usr/bin/env python3
"""
retrain_arima_with_scale_fix.py

1) Auto-detects historical + live AQI CSVs in project root / data/.
2) Merges into an hourly AQI series, fills small gaps.
3) Compares last training value to latest live AQI and if there is a
   clear scale mismatch (training values tiny), automatically rescales training y.
4) Aligns weather exogenous (optional) and trains SARIMAX (pmdarima auto_arima if available).
5) Saves model -> saved_models/arima_aqi_weather_only.joblib and diagnostics -> results/arima_insample.csv

Run:
    python retrain_arima_with_scale_fix.py
"""
import os, sys, warnings, joblib
from datetime import datetime
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------- Paths and candidates ----------
AQI_CANDIDATES = [
    "data/live_processed/aqi_live_master.csv",
    "aqi_live_master.csv",
    "aqi_live_data.csv",
    "aqi_historical_data.csv",
    "aqi_master.csv",
    "data/processed/aqi_master.csv",
    "data/aqi_master.csv",
    "aqi_historical.csv"
]
WEATHER_CANDIDATES = [
    "data/live_processed/weather_live_master.csv",
    "live_weather.csv",
    "historical_weather.csv",
    "data/historical_weather.csv",
    "data/live_processed/weather_live.csv"
]
OUT_MODEL = os.path.join("saved_models", "arima_aqi_weather_only.joblib")
os.makedirs("saved_models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# ---------- helpers ----------
def safe_read(path):
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        try:
            return pd.read_csv(path, engine="python", low_memory=False)
        except Exception:
            return None

def find_time_col(df):
    if df is None: return None
    low = [c.lower() for c in df.columns]
    for cand in ("timestamp","time","date","datetime","dt"):
        if cand in low:
            return df.columns[low.index(cand)]
    for c in df.columns:
        if "time" in c.lower() or "date" in c.lower():
            return c
    return None

def find_aqi_col(df):
    if df is None: return None
    low = [c.lower() for c in df.columns]
    for cand in ("aqi","aqi_value","predicted_aqi","aqi_obs","value"):
        if cand in low:
            return df.columns[low.index(cand)]
    for c in df.columns:
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

# ---------- collect and normalize AQI files ----------
found = []
for p in AQI_CANDIDATES:
    df = safe_read(p)
    if df is not None and not df.empty:
        found.append((p, df))

if not found:
    raise SystemExit("[ERROR] No AQI candidate files found. Put historical/live CSVs in project root or data/ folders.")

normed = []
for path, df in found:
    tcol = find_time_col(df)
    aqi_col = find_aqi_col(df)
    if tcol is None or aqi_col is None:
        print(f"[WARN] Skipping {path} — missing timestamp or AQI column.")
        continue
    tmp = df.copy()
    tmp[tcol] = pd.to_datetime(tmp[tcol], errors="coerce")
    tmp = tmp.dropna(subset=[tcol])
    tmp2 = tmp[[tcol, aqi_col]].rename(columns={tcol: "timestamp", aqi_col: "aqi"})
    normed.append(tmp2)
    print(f"[INFO] Loaded {path} -> time:{tcol} aqi:{aqi_col} rows:{len(tmp2)}")

if not normed:
    raise SystemExit("[ERROR] Found AQI files but none had usable timestamp+AQI columns.")

big = pd.concat(normed, axis=0, ignore_index=True)
big = big.drop_duplicates(subset=["timestamp"], keep="last")
big = big.sort_values("timestamp").reset_index(drop=True)
print(f"[INFO] Combined AQI rows: {len(big)} ({big['timestamp'].min()} -> {big['timestamp'].max()})")

# set hourly index
big["timestamp"] = pd.to_datetime(big["timestamp"], errors="coerce")
big = big.dropna(subset=["timestamp"]).set_index("timestamp")
full_idx = pd.date_range(start=big.index.min(), end=big.index.max(), freq="H")
big = big.reindex(full_idx)
# convert aqi to numeric
big["aqi"] = pd.to_numeric(big["aqi"], errors="coerce")
# fill small gaps with ffill then bfill
big["aqi"] = big["aqi"].fillna(method="ffill").fillna(method="bfill")

y = big["aqi"].dropna()
if y.empty:
    raise SystemExit("[ERROR] After merge no numeric AQI series available.")

print(f"[INFO] Final hourly series length: {len(y)} from {y.index.min()} to {y.index.max()}")

# ---------- read latest live AQI if available (to compute scale) ----------
latest_live = None
# try best candidate live files (prefer explicit live_master)
for lf in ["data/live_processed/aqi_live_master.csv", "aqi_live_data.csv", "aqi_live_master.csv"]:
    if os.path.exists(lf):
        try:
            dfl = pd.read_csv(lf)
            tcol = find_time_col(dfl)
            if tcol:
                dfl[tcol] = pd.to_datetime(dfl[tcol], errors="coerce")
                dfl = dfl.sort_values(tcol).dropna(subset=[tcol])
            if not dfl.empty:
                aqi_col = find_aqi_col(dfl)
                if aqi_col:
                    latest_live = pd.to_numeric(dfl[aqi_col].iloc[-1], errors="coerce")
                    print(f"[INFO] Found latest live AQI from {lf}: {latest_live}")
                    break
        except Exception:
            pass

# ---------- detect scale mismatch and auto-rescale if needed ----------
train_last = float(y.iloc[-1]) if len(y)>0 else None
applied_scale = 1.0
scaled = False
if latest_live is not None and train_last is not None and not np.isnan(latest_live) and not np.isnan(train_last):
    # if training last value tiny (<10) and live >> training_last, assume scaling issue
    if abs(train_last) < 10 and latest_live > (train_last * 3 + 10):
        # compute candidate scale, cap it to avoid insane factors
        scale = latest_live / (train_last if train_last != 0 else 1.0)
        # clamp scale to reasonable range [1, 1000]
        scale = max(1.0, min(scale, 1000.0))
        applied_scale = float(scale)
        print(f"[WARN] Scale mismatch detected: train_last={train_last} live_latest={latest_live}. Applying scale={applied_scale:.3f} to training series.")
        y = y * applied_scale
        scaled = True
    else:
        print("[INFO] No automatic rescaling required (train_last vs latest_live consistent).")
else:
    print("[INFO] Could not compute train_last or latest_live — skipping auto-rescale.")

# also defensive clamp values to [0, 500]
y = y.clip(lower=0.0, upper=500.0)

# ---------- prepare weather exogenous if available ----------
weather_df = None
for w in WEATHER_CANDIDATES:
    dfw = safe_read(w)
    if dfw is not None and not dfw.empty:
        weather_df = dfw
        weather_path = w
        break

X = None
exog_cols = []
if weather_df is not None:
    tcolw = find_time_col(weather_df)
    try:
        if tcolw:
            weather_df[tcolw] = pd.to_datetime(weather_df[tcolw], errors="coerce")
            weather_df = weather_df.sort_values(tcolw).dropna(subset=[tcolw]).set_index(tcolw)
    except Exception:
        pass
    # dedupe duplicate timestamps
    try:
        if weather_df.index.duplicated().any():
            weather_df = weather_df[~weather_df.index.duplicated(keep='last')]
    except Exception:
        pass
    # align to y index using nearest tolerance 1h
    try:
        exog_aligned = weather_df.reindex(y.index, method="nearest", tolerance=pd.Timedelta("1H"))
    except Exception:
        exog_aligned = weather_df.reindex(y.index, method="ffill")
    exog_cols = select_weather_exog_cols(exog_aligned)
    if exog_cols:
        X = exog_aligned[exog_cols].fillna(method="ffill").fillna(method="bfill")
        # drop exog columns that are all-NaN
        drop_cols = [c for c in X.columns if X[c].isna().all()]
        if drop_cols:
            print(f"[WARN] Dropping exog cols with all-NaN: {drop_cols}")
            X = X.drop(columns=drop_cols)
            exog_cols = [c for c in exog_cols if c not in drop_cols]
        print(f"[INFO] Using weather exog cols: {exog_cols} (from {weather_path})")
    else:
        print("[INFO] Weather found but no exog columns matched; training without exog.")
        X = None
        exog_cols = []
else:
    print("[INFO] No weather file found — training without exog.")

# ---------- model selection & training (pmdarima auto_arima if available) ----------
use_pmd = False
try:
    import pmdarima as pm
    use_pmd = True
except Exception:
    use_pmd = False

from statsmodels.tsa.statespace.sarimax import SARIMAX

# ensure hourly freq if possible
try:
    if y.index.inferred_freq is None:
        y = y.asfreq("H")
except Exception:
    pass

order = (1,0,0)
seasonal_order = (0,0,0,0)
if use_pmd:
    print("[INFO] pmdarima found, running auto_arima (seasonal m=24)...")
    try:
        am = pm.auto_arima(y, exogenous=(X.values if X is not None else None),
                           seasonal=True, m=24, stepwise=True,
                           max_p=4, max_q=4, max_P=2, max_Q=2,
                           max_d=2, max_D=1, suppress_warnings=True, error_action="ignore")
        order = am.order
        seasonal_order = getattr(am, "seasonal_order", (0,0,0,0))
        print(f"[INFO] auto_arima chose order={order} seasonal_order={seasonal_order}")
    except Exception as e:
        print("[WARN] auto_arima failed, falling back to grid search:", e)
        use_pmd = False

if not use_pmd:
    print("[INFO] Running small grid search for ARIMA order (this may take some minutes)...")
    best_aic = np.inf
    best_order = (1,0,0)
    best_seasonal = (0,0,0,0)
    for p in range(0,3):
        for d in range(0,2):
            for q in range(0,3):
                try:
                    mod = SARIMAX(y, exog=X, order=(p,d,q),
                                  enforce_stationarity=False, enforce_invertibility=False)
                    res = mod.fit(disp=False, maxiter=100)
                    if res.aic < best_aic:
                        best_aic = res.aic
                        best_order = (p,d,q)
                        best_seasonal = (0,0,0,0)
                except Exception:
                    pass
                # try simple seasonal candidate
                try:
                    mod2 = SARIMAX(y, exog=X, order=(p,d,q), seasonal_order=(1,0,1,24),
                                   enforce_stationarity=False, enforce_invertibility=False)
                    res2 = mod2.fit(disp=False, maxiter=100)
                    if res2.aic < best_aic:
                        best_aic = res2.aic
                        best_order = (p,d,q)
                        best_seasonal = (1,0,1,24)
                except Exception:
                    pass
    order = best_order
    seasonal_order = best_seasonal
    print(f"[INFO] selected order={order}, seasonal_order={seasonal_order} (AIC={best_aic:.2f})")

print(f"[INFO] Training SARIMAX(order={order}, seasonal_order={seasonal_order}) ...")
mod = SARIMAX(y, exog=X, order=order, seasonal_order=seasonal_order,
              enforce_stationarity=False, enforce_invertibility=False)
res = mod.fit(disp=False, maxiter=600)
print("[INFO] Model fit complete. AIC:", res.aic)

# ---------- save model and metadata ----------
meta = {"order": order, "seasonal_order": seasonal_order, "exog_cols": exog_cols, "training_end": str(y.index[-1]), "aic": float(res.aic), "applied_scale": applied_scale, "scaled": scaled}
joblib.dump({"model": res, "meta": meta}, OUT_MODEL)
print(f"[INFO] Saved model to {OUT_MODEL}")

# ---------- save in-sample diagnostics ----------
try:
    pred = res.get_prediction(start=0, end=len(y)-1, exog=X)
    df_diag = pd.DataFrame({"y": y, "y_pred_in_sample": pred.predicted_mean}).reset_index()
    df_diag.to_csv(os.path.join("results","arima_insample.csv"), index=False)
    print("[INFO] Saved in-sample diagnostics to results/arima_insample.csv")
except Exception:
    pass

print("[DONE] retrain_arima_with_scale_fix.py finished.")
print("Model meta:", meta)
