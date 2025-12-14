# preprocess_traffic.py
import os
from pathlib import Path
import pandas as pd

# candidate data dirs relative to project root (parent of this script)
BASE = Path(__file__).resolve().parent.parent
CANDIDATE_DIRS = [
    BASE / "data",
    BASE / "data" / "processed",
    BASE / "data" / "live_processed",
]

# expected filenames (adjust if your filenames differ)
KAGGLE = "Banglore_traffic_Dataset.csv"
API = "traffic_timeseries.csv"
OUTFILE = "traffic_master.csv"
OUT_DIR = BASE / "data" / "processed"


def find_file_in_candidates(fname):
    for d in CANDIDATE_DIRS:
        p = d / fname
        if p.exists():
            return p
    return None


def safe_read_csv(path):
    try:
        # low_memory=False helps prevent dtype inference issues for large csvs
        return pd.read_csv(path, low_memory=False)
    except Exception as e:
        print(f"[ERROR] Failed reading {path}: {e}")
        return None


def normalize_cols(df):
    df = df.copy()
    df.columns = (df.columns
                    .str.strip()
                    .str.replace(" ", "", regex=False)
                    .str.replace("/", "", regex=False)
                    .str.lower())
    return df


def kaggle_to_common(df):
    rename = {
        "date": "timestamp",
        "area_name": "city",
        "road_intersection_name": "location",
        "traffic_volume": "traffic_volume",
        "average_speed": "avg_speed",
        "travel_time_index": "travel_time_index",
        "congestion_level": "congestion_level",
        "road_capacity_utilization": "road_capacity_utilization",
        "incident_reports": "incident_reports",
        "environmental_impact": "environmental_impact",
        "public_transport_usage": "public_transport_usage",
        "traffic_signal_compliance": "traffic_signal_compliance",
        "parking_usage": "parking_usage",
        "pedestrian_and_cyclist_count": "pedestrian_cyclist_count",
        "weather_conditions": "weather_conditions",
        "roadwork_and_construction_activity": "roadwork_activity"
    }
    df = df.rename(columns=rename)
    return df


def api_to_common(df):
    rename = {
        "city": "city",
        "latitude": "latitude",
        "longitude": "longitude",
        "timestamp": "timestamp",
        "currentspeed": "current_speed",
        "currentSpeed": "current_speed",
        "current_speed": "current_speed",
        "freeflowspeed": "free_flow_speed",
        "freeflowtraveltime": "free_flow_travel_time",
        "currenttraveltime": "current_travel_time",
        "freeFlowSpeed": "free_flow_speed",
        "currentTravelTime": "current_travel_time",
        "freeFlowTravelTime": "free_flow_travel_time",
        "confidence": "confidence",
        "roadclosure": "road_closure",
        "roadClosure": "road_closure"
    }
    df.columns = [c.strip() for c in df.columns]
    col_map = {}
    for c in df.columns:
        k = c.lower().replace(" ", "").replace("_", "")
        for key in rename:
            if key.lower().replace(" ", "").replace("_", "") == k:
                col_map[c] = rename[key]
                break
    if col_map:
        df = df.rename(columns=col_map)
    return df


def parse_timestamp(df, col="timestamp"):
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def fill_missing(df):
    # numeric fill
    num = df.select_dtypes(include=["number"]).columns
    if len(num):
        df[num] = df[num].fillna(df[num].median())

    # categorical fill
    cat = df.select_dtypes(include=["object", "category"]).columns
    for c in cat:
        if not df[c].mode().empty:
            df[c] = df[c].fillna(df[c].mode()[0])
        else:
            df[c] = df[c].fillna("unknown")
    return df


def normalize_and_format_timestamp_column(df, ts_col="timestamp"):
    """
    Coerce to datetime, remove timezone/microseconds and format as YYYY-MM-DD HH:MM:SS strings.
    Returns (df, n_null) where n_null is number of null timestamps after coercion.
    """
    if ts_col not in df.columns:
        return df, 0

    # coerce to datetime (bad strings -> NaT)
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce", utc=False)

    # remove timezone (if any) and microseconds; keep naive datetime
    try:
        df[ts_col] = df[ts_col].dt.tz_localize(None)
    except Exception:
        # ignore if tz_localize not applicable
        pass

    # format to string without microseconds for safe CSV -> Power BI friendly
    df[ts_col] = df[ts_col].dt.strftime("%Y-%m-%d %H:%M:%S")

    # the previous step turns NaT into NaN strings; convert those back to pd.NA
    df.loc[df[ts_col].isna(), ts_col] = pd.NA

    n_null = int(df[ts_col].isna().sum())
    return df, n_null


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    kag_path = find_file_in_candidates(KAGGLE)
    api_path = find_file_in_candidates(API)

    if not kag_path:
        print(f"[ERROR] Kaggle file not found in candidate dirs: {KAGGLE}")
    else:
        print(f"[INFO] Found Kaggle -> {kag_path}")

    if not api_path:
        print(f"[ERROR] API file not found in candidate dirs: {API}")
    else:
        print(f"[INFO] Found API -> {api_path}")

    if not kag_path and not api_path:
        print("[ERROR] No input files found. Exiting.")
        return

    dfs = []
    if kag_path:
        kag = safe_read_csv(kag_path)
        if kag is not None:
            kag = normalize_cols(kag)
            kag = kaggle_to_common(kag)
            kag = parse_timestamp(kag, "timestamp")
            dfs.append(kag)

    if api_path:
        apid = safe_read_csv(api_path)
        if apid is not None:
            apid = normalize_cols(apid)
            apid = api_to_common(apid)
            apid = parse_timestamp(apid, "timestamp")
            if "current_speed" in apid.columns and "avg_speed" not in apid.columns:
                apid = apid.rename(columns={"current_speed": "avg_speed"})
            dfs.append(apid)

    if not dfs:
        print("[ERROR] No valid dataframes to combine. Exiting.")
        return

    combined = pd.concat(dfs, ignore_index=True, sort=False).drop_duplicates().reset_index(drop=True)
    combined = fill_missing(combined)

    # normalize and format timestamp column for Power BI
    combined, null_ts = normalize_and_format_timestamp_column(combined, ts_col="timestamp")
    if null_ts > 0:
        print(f"[WARN] {null_ts} rows have invalid or missing timestamps after coercion; they are set to blank.")

    outpath = OUT_DIR / OUTFILE
    try:
        combined.to_csv(outpath, index=False)
        print(f"[OK] Traffic master saved -> {outpath} (rows: {len(combined)})")
    except Exception as e:
        print(f"[ERROR] Failed to write {outpath}: {e}")

    print("If you want fewer columns, we can select a subset (timestamp, city, location, traffic_volume, avg_speed, congestion_level).")


if __name__ == "__main__":
    main()
