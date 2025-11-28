import pandas as pd
import joblib
import os

# ---------------------------------------------------------
# PATH SETUP
# ---------------------------------------------------------
ROOT = os.path.abspath(os.path.dirname(__file__))

INPUT_DIR = os.path.join(ROOT, "data", "live_processed")
OUTPUT_DIR = os.path.join(ROOT, "saved_models")

# Create output folder if not exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Master CSV paths
AQI_CSV = os.path.join(INPUT_DIR, "aqi_live_master.csv")
WEATHER_CSV = os.path.join(INPUT_DIR, "weather_live_master.csv")
TRAFFIC_CSV = os.path.join(INPUT_DIR, "traffic_live_master.csv")

# Output joblib paths
AQI_JOBLIB = os.path.join(OUTPUT_DIR, "aqi_live_master.joblib")
WEATHER_JOBLIB = os.path.join(OUTPUT_DIR, "weather_live_master.joblib")
TRAFFIC_JOBLIB = os.path.join(OUTPUT_DIR, "traffic_live_master.joblib")

COMBINED_JOBLIB = os.path.join(OUTPUT_DIR, "combined_master.joblib")

# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
def load_df(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ERROR] CSV not found: {path}")
    return pd.read_csv(path)

def save_joblib(df, path):
    joblib.dump(df, path)
    print(f"[✓] Saved: {path}")

def find_datetime_col(df):
    for col in df.columns:
        if any(k in col for k in ("date", "time", "timestamp", "ts")):
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce")
                return col
            except:
                continue
    return None

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    print("\nLoading master CSV files...")
    df_aqi = load_df(AQI_CSV)
    df_weather = load_df(WEATHER_CSV)
    df_traffic = load_df(TRAFFIC_CSV)

    # ---------------------------------------------------------
    # SAVE INDIVIDUAL JOBLIBS
    # ---------------------------------------------------------
    print("\nSaving individual joblibs...")
    save_joblib(df_aqi, AQI_JOBLIB)
    save_joblib(df_weather, WEATHER_JOBLIB)
    save_joblib(df_traffic, TRAFFIC_JOBLIB)

    # ---------------------------------------------------------
    # COMBINE ALL DATASETS
    # ---------------------------------------------------------
    print("\nCombining datasets...")

    aqi_ts = find_datetime_col(df_aqi)
    weather_ts = find_datetime_col(df_weather)
    traffic_ts = find_datetime_col(df_traffic)

    if aqi_ts: df_aqi.rename(columns={aqi_ts: "timestamp"}, inplace=True)
    if weather_ts: df_weather.rename(columns={weather_ts: "timestamp"}, inplace=True)
    if traffic_ts: df_traffic.rename(columns={traffic_ts: "timestamp"}, inplace=True)

    # sort
    for df in (df_aqi, df_weather, df_traffic):
        if "timestamp" in df.columns:
            df.sort_values("timestamp", inplace=True)

    # merge
    df_combined = pd.merge(df_aqi, df_weather, on="timestamp", how="outer", suffixes=("_aqi", "_weather"))
    df_combined = pd.merge(df_combined, df_traffic, on="timestamp", how="outer")

    if "timestamp" in df_combined.columns:
        df_combined.sort_values("timestamp", inplace=True)

    # save combined joblib
    save_joblib(df_combined, COMBINED_JOBLIB)

    print(f"\n[✓] All joblibs saved successfully in {OUTPUT_DIR}")
    print(f"Rows in combined dataset: {len(df_combined)}")

if __name__ == "__main__":
    main()
