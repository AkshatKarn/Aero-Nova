# /mnt/data/train_model.py
import os
import sys
import json
import joblib
from datetime import datetime
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

# ---------------- CONFIG ----------------
AQI_PATH = "data/processed/aqi_master.csv"
WEATHER_PATH = "data/processed/weather_master.csv"
TRAFFIC_PATH = "data/processed/traffic_master.csv"
MODEL_DIR = "results/models"
METRIC_DIR = "results/metrics"
RANDOM_STATE = 42
TEST_SIZE = 0.2
MERGE_TOLERANCE = pd.Timedelta("1h")   # use lower-case 'h'
RESAMPLE_RULE = "1h"
# ---------------------------------------

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(METRIC_DIR, exist_ok=True)

def load_csv(path):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return None
    try:
        df = pd.read_csv(path, low_memory=False)
        print(f"Loaded {path} -> shape {df.shape}")
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to read {path}: {e}")

def find_datetime_col(df):
    if df is None:
        return None
    for c in df.columns:
        lc = c.lower()
        if "date" in lc or "time" in lc or "timestamp" in lc:
            return c
    for c in df.columns:
        if c.lower() in ("ts","time","timestamp","datetime"):
            return c
    return None

def ensure_timestamp(df, dt_col):
    if dt_col is None:
        return df
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df[dt_col], errors='coerce')
    # drop original datetime column if different
    if dt_col != 'timestamp':
        df = df.drop(columns=[dt_col], errors='ignore')
    # drop rows that couldn't parse timestamp (safe because processed files should be clean)
    df = df.dropna(subset=['timestamp']).reset_index(drop=True)
    return df

def resample_hourly_mean(df):
    df = df.copy().set_index('timestamp').sort_index()
    df_hour = df.resample(RESAMPLE_RULE).mean()
    df_hour = df_hour.reset_index()
    return df_hour

# ---------------- Load processed files ----------------
print("Using processed files (data/processed/*.csv)")
df_aqi = load_csv(AQI_PATH)
df_weather = load_csv(WEATHER_PATH)
df_traffic = load_csv(TRAFFIC_PATH)

if df_aqi is None:
    raise FileNotFoundError(f"Missing AQI file: {AQI_PATH}")
if df_weather is None:
    raise FileNotFoundError(f"Missing weather file: {WEATHER_PATH}")
if df_traffic is None:
    raise FileNotFoundError(f"Missing traffic file: {TRAFFIC_PATH}")

# ---------------- Convert / align timestamps if present ----------------
dfs = []
names = []
for df, name in [(df_aqi, "aqi"), (df_weather, "weather"), (df_traffic, "traffic")]:
    dt_col = find_datetime_col(df)
    if dt_col:
        print(f"Detected datetime column '{dt_col}' in {name}; converting to 'timestamp' and resampling hourly.")
        df_ts = ensure_timestamp(df, dt_col)
        try:
            df_hour = resample_hourly_mean(df_ts)
            dfs.append(df_hour)
            names.append(name)
            print(f"{name} resampled hourly -> shape {df_hour.shape}")
        except Exception as e:
            print(f"Resample failed for {name}, using raw timestamped data: {e}")
            dfs.append(df_ts)
            names.append(name)
    else:
        print(f"No datetime detected in {name}; including raw (index-based).")
        dfs.append(df)
        names.append(name)

# ---------------- Merge strategy ----------------
all_have_ts = all(('timestamp' in d.columns) for d in dfs)
if all_have_ts and len(dfs) >= 2:
    print("Merging on timestamp using merge_asof (nearest within 1 hour).")
    for i in range(len(dfs)):
        dfs[i] = dfs[i].sort_values('timestamp').reset_index(drop=True)
    df_merged = dfs[0]
    for other in dfs[1:]:
        df_merged = pd.merge_asof(df_merged.sort_values('timestamp'),
                                  other.sort_values('timestamp'),
                                  on='timestamp',
                                  direction='nearest',
                                  tolerance=MERGE_TOLERANCE)
    print("Merged shape (time-aware):", df_merged.shape)
else:
    # fallback: index-wise concat (truncate to minimum length)
    print("Not all files have timestamp; performing index-wise concat (truncate to min length).")
    min_len = min(len(d) for d in dfs)
    dfs_trunc = [d.iloc[:min_len].reset_index(drop=True) for d in dfs]
    df_merged = pd.concat(dfs_trunc, axis=1)
    print("Merged shape (index-concat):", df_merged.shape)

# ---------------- Identify target column (AQI) ----------------
TARGET_CANDIDATES = ["AQI", "aqi", "aqi_value", "AirQualityIndex", "AQI_Value", "aqi_val"]
target_col = None
for t in TARGET_CANDIDATES:
    if t in df_merged.columns:
        target_col = t
        break
if target_col is None:
    for c in df_merged.columns:
        if "aqi" in c.lower():
            target_col = c
            break
if target_col is None:
    print("Merged columns:", df_merged.columns.tolist())
    raise KeyError("AQI target column not found in merged data. Ensure processed AQI file contains an AQI column.")

print("Using target column:", target_col)

# drop rows with missing target
missing_target = int(df_merged[target_col].isna().sum())
if missing_target:
    print(f"Dropping {missing_target} rows with missing target ({target_col}).")
    df_merged = df_merged.dropna(subset=[target_col]).reset_index(drop=True)

# ---------------- Prepare features ----------------
X = df_merged.drop(columns=[target_col]).copy()
y = df_merged[target_col].copy()

# Coerce object columns to numeric where possible (handle commas)
for c in X.columns:
    if X[c].dtype == object:
        try:
            X[c] = pd.to_numeric(X[c].astype(str).str.replace(',',''), errors='coerce')
        except Exception:
            X[c] = pd.to_numeric(X[c], errors='coerce')

# Convert timestamp -> epoch seconds for numeric model input (if present)
if 'timestamp' in X.columns:
    try:
        X['timestamp'] = pd.to_datetime(X['timestamp'], errors='coerce').astype('int64') // 10**9
    except Exception:
        pass

# ---------------- DROP ALL-NaN & NON-NUMERIC COLUMNS (important) ----------------
# Drop columns where all values are NaN (imputer would skip them and cause shape mismatch)
all_nan_cols = [c for c in X.columns if X[c].isna().all()]
if all_nan_cols:
    print("Dropping columns that are ALL NaN:", all_nan_cols)
    X = X.drop(columns=all_nan_cols)

# Drop remaining non-numeric columns (booleans are allowed)
non_numeric = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c]) and not pd.api.types.is_bool_dtype(X[c])]
if non_numeric:
    print("Dropping non-numeric columns (could not coerce):", non_numeric)
    X = X.drop(columns=non_numeric)

# Show NaNs before imputation
nan_counts = X.isna().sum()
nan_counts = nan_counts[nan_counts > 0].sort_values(ascending=False)
if len(nan_counts):
    print("NaN counts in features before imputation:")
    print(nan_counts.to_string())
else:
    print("No NaNs in features before imputation.")

# ---------------- Impute numeric NaNs with median (SimpleImputer) ----------------
imp = SimpleImputer(strategy="median")
X_values = imp.fit_transform(X)    # numpy array shape = (n_rows, n_final_columns)
# Rebuild DataFrame using the current X.columns (these columns were not dropped above)
X = pd.DataFrame(X_values, columns=list(X.columns), index=X.index)

# Final check
if X.isna().any().any():
    print("Warning: NaNs remain after imputation; dropping rows with NaNs.")
    df_comb = pd.concat([X, y.reset_index(drop=True)], axis=1)
    df_comb = df_comb.dropna(axis=0)
    # Re-assign X and y after dropping
    y = df_comb[target_col]
    X = df_comb.drop(columns=[target_col])

print("Final feature matrix shape:", X.shape)

if X.shape[1] == 0:
    raise ValueError("No numeric features available after preprocessing. Check processed CSVs.")

# ---------------- Train/test split ----------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

# ---------------- Linear Regression ----------------
print("\nTraining LinearRegression (baseline)...")
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
lin_pred = lin_reg.predict(X_test)
lin_rmse = np.sqrt(mean_squared_error(y_test, lin_pred))
lin_r2 = r2_score(y_test, lin_pred)
print(f"LinearRegression -> RMSE: {lin_rmse:.4f}, R2: {lin_r2:.4f}")

# ---------------- XGBoost (optional) ----------------
xgb_model = None
xgb_rmse = None
xgb_r2 = None
try:
    from xgboost import XGBRegressor
    print("\nTraining XGBoost...")
    xgb_model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_STATE,
        verbosity=0
    )
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
    xgb_r2 = r2_score(y_test, xgb_pred)
    print(f"XGBoost -> RMSE: {xgb_rmse:.4f}, R2: {xgb_r2:.4f}")
except Exception as e:
    print("XGBoost not trained (missing or failed):", e)
    xgb_model = None

# ---------------- SHAP or fallback importance ----------------
feature_importance_df = None
explainer = None
if xgb_model is not None:
    try:
        import shap
        print("\nComputing SHAP values for feature importances...")
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X_test)
        mean_abs = np.mean(np.abs(shap_values), axis=0)
        feature_importance_df = pd.DataFrame({"feature": X.columns, "importance": mean_abs}).sort_values("importance", ascending=False).reset_index(drop=True)
        print("Top features (SHAP):\n", feature_importance_df.head(10).to_string(index=False))
    except Exception as e:
        print("SHAP not available or failed:", e)

if feature_importance_df is None:
    try:
        coeffs = np.abs(lin_reg.coef_)
        feature_importance_df = pd.DataFrame({"feature": X.columns, "importance": coeffs}).sort_values("importance", ascending=False).reset_index(drop=True)
        print("Top features (Linear coef fallback):\n", feature_importance_df.head(10).to_string(index=False))
    except Exception as e:
        print("Could not compute fallback feature importances:", e)
        feature_importance_df = None

# ---------------- Save artifacts ----------------
ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

def save_obj(obj, path):
    try:
        joblib.dump(obj, path)
        print("Saved:", path)
    except Exception as e:
        print("Failed saving", path, ":", e)

# save linear model
save_obj(lin_reg, os.path.join(MODEL_DIR, f"linear_model_{ts}.joblib"))
save_obj(lin_reg, os.path.join(MODEL_DIR, "linear_model_latest.joblib"))

# save xgb model if trained
if xgb_model is not None:
    save_obj(xgb_model, os.path.join(MODEL_DIR, f"xgb_model_{ts}.joblib"))
    save_obj(xgb_model, os.path.join(MODEL_DIR, "xgb_model_latest.joblib"))

# save explainer if created
if explainer is not None:
    try:
        save_obj(explainer, os.path.join(MODEL_DIR, f"explainer_{ts}.joblib"))
        save_obj(explainer, os.path.join(MODEL_DIR, "explainer_latest.joblib"))
    except Exception as e:
        print("Warning: saving explainer failed (pickling):", e)

# save feature importance CSV
if feature_importance_df is not None:
    fi_path = os.path.join(MODEL_DIR, f"feature_importance_{ts}.csv")
    feature_importance_df.to_csv(fi_path, index=False)
    feature_importance_df.to_csv(os.path.join(MODEL_DIR, "feature_importance_latest.csv"), index=False)
    print("Saved feature importance:", fi_path)

# --- SAVE TRUE FEATURE LIST USED IN TRAINING (NEW) ---
feature_list_path = os.path.join(MODEL_DIR, "features_latest.json")
try:
    with open(feature_list_path, "w") as f:
        json.dump(list(X.columns), f)
    print("Saved training feature list:", feature_list_path)
except Exception as e:
    print("Failed saving feature list:", e)

# save metrics
metrics = {
    "data_used": {
        "aqi": AQI_PATH,
        "weather": WEATHER_PATH,
        "traffic": TRAFFIC_PATH
    },
    "n_rows_merged": int(df_merged.shape[0]) if 'df_merged' in globals() else int(X.shape[0]),
    "n_features": int(X.shape[1]),
    "linear": {"rmse": float(lin_rmse), "r2": float(lin_r2)},
    "xgboost": {"rmse": float(xgb_rmse) if xgb_rmse is not None else None,
                "r2": float(xgb_r2) if xgb_r2 is not None else None},
    "timestamp": ts
}
metrics_path = os.path.join(METRIC_DIR, f"metrics_{ts}.json")
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)
with open(os.path.join(METRIC_DIR, "metrics_latest.json"), "w") as f:
    json.dump(metrics, f, indent=2)

print("\nTraining complete.")
print("Models saved in:", MODEL_DIR)
print("Metrics saved in:", METRIC_DIR)
if feature_importance_df is not None:
    print("Top 5 features:\n", feature_importance_df.head(5).to_string(index=False))
