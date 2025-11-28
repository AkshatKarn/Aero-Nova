#!/usr/bin/env python3
"""
train_model.py

Safe, importable training & utility module.

- Helper functions are defined at module level (safe to import).
- Training & argparse logic is inside main() and guarded by if __name__ == "__main__".
- Exposes load_pipeline_and_predict(...) for runtime inference imports.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# -----------------------
# Defaults
# -----------------------
DEFAULT_DATA_DIR = os.path.join(os.getcwd(), "processed")
DEFAULT_AQI_FILE = "aqi_master.csv"
DEFAULT_TRAFFIC_FILE = "traffic_master.csv"
DEFAULT_WEATHER_FILE = "weather_master.csv"
DEFAULT_OUT_DIR = "saved_models"
DEFAULT_MODEL_FILENAME = "aqi_pipeline_with_metadata_no_leak.joblib"

# -----------------------
# CPCB-like breakpoints (replace with official tables if desired)
# -----------------------
POLLUTANT_BREAKPOINTS = {
    "PM2.5": [(0.0,30.0,0,50),(30.1,60.0,51,100),(60.1,90.0,101,200),(90.1,120.0,201,300),(120.1,250.0,301,400),(250.1,500.0,401,500)],
    "PM10":  [(0.0,50.0,0,50),(50.1,100.0,51,100),(100.1,250.0,101,200),(250.1,350.0,201,300),(350.1,430.0,301,400),(430.1,600.0,401,500)],
    "NO2":   [(0.0,40.0,0,50),(40.1,80.0,51,100),(80.1,180.0,101,200),(180.1,280.0,201,300),(280.1,400.0,301,400),(400.1,1000.0,401,500)],
    "SO2":   [(0.0,40.0,0,50),(40.1,80.0,51,100),(80.1,380.0,101,200),(380.1,800.0,201,300),(800.1,1600.0,301,400),(1600.1,2000.0,401,500)],
    "O3":    [(0.0,50.0,0,50),(50.1,100.0,51,100),(100.1,168.0,101,200),(168.1,208.0,201,300),(208.1,748.0,301,400),(748.1,1000.0,401,500)],
    "CO":    [(0.0,1.0,0,50),(1.1,2.0,51,100),(2.1,10.0,101,200),(10.1,17.0,201,300),(17.1,34.0,301,400),(34.1,50.0,401,500)],
    "NH3":   [(0.0,200.0,0,50),(200.1,400.0,51,100),(400.1,800.0,101,200),(800.1,1200.0,201,300),(1200.1,1800.0,301,400),(1800.1,4000.0,401,500)],
}

# -----------------------
# Compatibility helper for OneHotEncoder param names
# -----------------------
def make_onehot(**kwargs):
    """
    Return a OneHotEncoder instance with compatible argument for both sklearn<=1.3 and sklearn>=1.4.
    Tries 'sparse_output' first, falls back to 'sparse' if necessary.
    """
    try:
        return OneHotEncoder(**kwargs, sparse_output=False)
    except TypeError:
        # older sklearn expects 'sparse'
        return OneHotEncoder(**{k if k != "sparse_output" else "sparse": v for k, v in kwargs.items()}, sparse=False)

# -----------------------
# Utilities
# -----------------------
def interpolate_aqi_array(concs, breakpoints):
    concs = np.array(concs, dtype=float)
    out = np.full(concs.shape, np.nan, dtype=float)
    for (c_lo, c_hi, a_lo, a_hi) in breakpoints:
        mask = (concs >= c_lo) & (concs <= c_hi)
        if c_hi == c_lo:
            out[mask] = a_lo
        else:
            out[mask] = ((a_hi - a_lo) / (c_hi - c_lo)) * (concs[mask] - c_lo) + a_lo
    return out

def compute_aqi_columns(df, pollutant_breakpoints):
    cols_lower_map = {c.lower().replace(".", "").replace(" ", ""): c for c in df.columns}
    aqi_data = {}
    for pol, bps in pollutant_breakpoints.items():
        key = pol.lower().replace(".", "").replace(" ", "")
        found = cols_lower_map.get(key)
        if found:
            aqi_vals = interpolate_aqi_array(df[found].fillna(np.nan).values, bps)
            aqi_data[f"AQI_{pol}"] = aqi_vals
        else:
            aqi_data[f"AQI_{pol}"] = np.full(len(df), np.nan)
    aqi_df = pd.DataFrame(aqi_data, index=df.index)
    aqi_df["AQI_real"] = aqi_df.max(axis=1, skipna=True)
    return aqi_df

def load_and_preview(aqi_path, traffic_path, weather_path):
    if not os.path.exists(aqi_path) or not os.path.exists(traffic_path) or not os.path.exists(weather_path):
        missing = [p for p in (aqi_path, traffic_path, weather_path) if not os.path.exists(p)]
        raise FileNotFoundError("Missing files:\n" + "\n".join(missing))
    aqi = pd.read_csv(aqi_path)
    traffic = pd.read_csv(traffic_path)
    weather = pd.read_csv(weather_path)
    return aqi, traffic, weather

def ensure_timestamp(df):
    for col in df.columns:
        if col.lower() in ("timestamp","time","datetime","date") or "time" in col.lower() or "date" in col.lower():
            try:
                df[col] = pd.to_datetime(df[col])
                return df.set_index(col)
            except Exception:
                continue
    first = df.columns[0]
    df[first] = pd.to_datetime(df[first], errors='coerce')
    if df[first].isna().all():
        raise ValueError("No parseable datetime column found.")
    return df.set_index(first)

def merge_datasets(aqi_df, traffic_df, weather_df, resample_rule="H"):
    aqi_dt = ensure_timestamp(aqi_df)
    traffic_dt = ensure_timestamp(traffic_df)
    weather_dt = ensure_timestamp(weather_df)
    try:
        aqi_rs = aqi_dt.resample(resample_rule).mean()
    except Exception:
        aqi_rs = aqi_dt
    try:
        traffic_rs = traffic_dt.resample(resample_rule).mean()
    except Exception:
        traffic_rs = traffic_dt
    try:
        weather_rs = weather_dt.resample(resample_rule).mean()
    except Exception:
        weather_rs = weather_dt
    merged = aqi_rs.join([traffic_rs, weather_rs], how="outer")
    merged = merged.sort_index()
    return merged

def make_columns_unique(df, sep="_dup"):
    cols = list(df.columns)
    seen = {}
    new_cols = []
    for c in cols:
        if c not in seen:
            seen[c] = 1
            new_cols.append(c)
        else:
            new_name = f"{c}{sep}{seen[c]}"
            seen[c] += 1
            while new_name in seen:
                new_name = f"{c}{sep}{seen[c]}"
                seen[c] += 1
            seen[new_name] = 1
            new_cols.append(new_name)
    df.columns = new_cols
    return df

def drop_allnull_columns(df):
    allnull = [c for c in df.columns if df[c].isna().all()]
    if allnull:
        return df.drop(columns=allnull), allnull
    return df, []

# -----------------------
# Prepare ML dataset (removes AQI_* features to avoid leakage)
# -----------------------
def prepare_ml_dataset(merged_df, pollutant_breakpoints, drop_exact_duplicate_columns=False):
    aqi_cols_df = compute_aqi_columns(merged_df, pollutant_breakpoints)
    merged_with_aqi = pd.concat([merged_df.reset_index(drop=True), aqi_cols_df.reset_index(drop=True)], axis=1)

    # handle duplicate column labels
    if merged_with_aqi.columns.duplicated().any():
        if drop_exact_duplicate_columns:
            merged_with_aqi = merged_with_aqi.loc[:, ~merged_with_aqi.columns.duplicated()]
        else:
            merged_with_aqi = make_columns_unique(merged_with_aqi, sep="_dup")

    if not merged_with_aqi.index.is_unique:
        merged_with_aqi = merged_with_aqi.reset_index(drop=True)

    merged_with_aqi, _ = drop_allnull_columns(merged_with_aqi)

    merged_with_aqi["AQI_real"] = pd.to_numeric(merged_with_aqi["AQI_real"], errors="coerce")
    df_ok = merged_with_aqi[~merged_with_aqi["AQI_real"].isna()].copy().reset_index(drop=True)

    numeric_cols = df_ok.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if not (str(c).startswith("AQI_") or c == "AQI_real")]
    categorical_cols = df_ok.select_dtypes(include=["object","category","bool"]).columns.tolist()

    final_feature_cols = feature_cols + categorical_cols
    return df_ok, final_feature_cols, "AQI_real"

# -----------------------
# Training pipeline
# -----------------------
def build_and_train(df, feature_cols, target_col, save_path, time_split=False, tolerance=10.0):
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    if X.shape[0] < 20:
        raise ValueError(f"Too few rows to train ({X.shape[0]}). Need more data.")

    if time_split:
        timestamps = pd.to_datetime(df.index.to_series(), errors="coerce")
        if timestamps.isna().all():
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        else:
            cutoff = timestamps.quantile(0.8)
            train_mask = timestamps <= cutoff
            if train_mask.sum() < 10 or (~train_mask).sum() < 10:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            else:
                X_train, X_test = X.loc[train_mask], X.loc[~train_mask]
                y_train, y_test = y.loc[train_mask], y.loc[~train_mask]
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    numeric_feats = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_feats = X_train.select_dtypes(include=["object","category","bool"]).columns.tolist()

    numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    if categorical_feats:
        categorical_pipeline = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("ohe", make_onehot(handle_unknown="ignore"))])
        preprocessor = ColumnTransformer([("num", numeric_pipeline, numeric_feats), ("cat", categorical_pipeline, categorical_feats)], remainder="drop")
    else:
        preprocessor = ColumnTransformer([("num", numeric_pipeline, numeric_feats)], remainder="drop")

    models = {
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "HGB": HistGradientBoostingRegressor(random_state=42)
    }

    trained_models = {}
    for name, model in models.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        r2 = r2_score(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        rmse = mse ** 0.5
        mae = mean_absolute_error(y_test, preds)
        within_tol = np.mean(np.abs(preds - y_test) <= tolerance) * 100.0
        trained_models[name] = {"pipeline": pipe, "metrics": {"r2": r2, "rmse": rmse, "mae": mae, "within_tol_pct": within_tol}}

    best_name = max(trained_models.keys(), key=lambda k: (trained_models[k]["metrics"]["within_tol_pct"], trained_models[k]["metrics"]["r2"]))
    best_pipeline = trained_models[best_name]["pipeline"]

    saved_obj = {
        "pipeline": best_pipeline,
        "feature_names": feature_cols,
        "numeric_features": numeric_feats,
        "categorical_features": categorical_feats,
        "pollutant_breakpoints": POLLUTANT_BREAKPOINTS,
        "training_metrics": trained_models[best_name]["metrics"],
        "model_name": best_name
    }
    joblib.dump(saved_obj, save_path)
    return save_path, trained_models

# -----------------------
# Exported helper for inference
# -----------------------
def load_pipeline_and_predict(model_file, df_new):
    """
    Safe helper for inference. Loads saved joblib object (the structure saved by build_and_train)
    and returns predictions for df_new (DataFrame).
    """
    saved = joblib.load(model_file)
    pipeline = saved["pipeline"]
    feature_names = saved["feature_names"]
    pollutant_breakpoints = saved.get("pollutant_breakpoints", POLLUTANT_BREAKPOINTS)

    # compute AQI columns if pollutant data present (not used in training features but kept for convenience)
    aqi_cols = compute_aqi_columns(df_new, pollutant_breakpoints)
    df_combined = pd.concat([df_new.reset_index(drop=True), aqi_cols.reset_index(drop=True)], axis=1)

    for col in feature_names:
        if col not in df_combined.columns:
            df_combined[col] = np.nan

    X_new = df_combined[feature_names].copy()
    for c in X_new.columns:
        X_new[c] = pd.to_numeric(X_new[c], errors="ignore")

    preds = pipeline.predict(X_new)
    return preds

# -----------------------
# Main entrypoint (guarded)
# -----------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train AQI model and save pipeline")
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR, help="Folder containing aqi_master.csv, traffic_master.csv, weather_master.csv")
    parser.add_argument("--aqi", default=DEFAULT_AQI_FILE)
    parser.add_argument("--traffic", default=DEFAULT_TRAFFIC_FILE)
    parser.add_argument("--weather", default=DEFAULT_WEATHER_FILE)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--model_name", default=DEFAULT_MODEL_FILENAME)
    parser.add_argument("--drop_exact_duplicate_columns", action="store_true")
    parser.add_argument("--time_split", action="store_true")
    parser.add_argument("--tolerance", type=float, default=10.0)
    args = parser.parse_args()

    DATA_DIR = args.data_dir
    AQI_CSV = os.path.join(DATA_DIR, args.aqi)
    TRAFFIC_CSV = os.path.join(DATA_DIR, args.traffic)
    WEATHER_CSV = os.path.join(DATA_DIR, args.weather)
    MODEL_SAVE_DIR = args.out_dir
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    SAVE_PATH = os.path.join(MODEL_SAVE_DIR, args.model_name)

    print("Loading files from:", DATA_DIR)
    aqi, traffic, weather = load_and_preview(AQI_CSV, TRAFFIC_CSV, WEATHER_CSV)

    print("Merging datasets (hourly)...")
    merged = merge_datasets(aqi, traffic, weather, resample_rule="H")

    print("Preparing ML dataset (computing AQI_real and removing leakage)...")
    df_ok, feature_cols, target_col = prepare_ml_dataset(merged, POLLUTANT_BREAKPOINTS, drop_exact_duplicate_columns=args.drop_exact_duplicate_columns)

    print(f"Training rows: {df_ok.shape[0]}, Features: {len(feature_cols)}")
    if df_ok.shape[0] < 30:
        print("Warning: very few training rows; model may not generalize well.")

    print("Training models (this may take some time)...")
    model_file, trained_models = build_and_train(df_ok, feature_cols, target_col, SAVE_PATH, time_split=args.time_split, tolerance=args.tolerance)

    print("Saved model to:", model_file)
    print("Training metrics (best):", trained_models[max(trained_models.keys(), key=lambda k: trained_models[k]["metrics"]["within_tol_pct"])]["metrics"])

if __name__ == "__main__":
    main()
