# preprocess_traffic.py
import os
import pandas as pd

RAW_DIR = "../data"
OUT_DIR = "../processed"
KAGGLE = "Banglore_traffic_Dataset.csv"
API = "traffic_timeseries.csv"
OUTFILE = "traffic_master.csv"

def normalize_cols(df):
    df = df.copy()
    df.columns = df.columns.str.strip().str.replace(" ", "_").str.replace("/", "_").str.lower()
    return df

def kaggle_to_common(df):
    # original kaggle columns (seen from your print)
    # map to common names
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
    # api columns already close to common names; rename for consistency
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
        "freeflowtraveltime": "free_flow_travel_time",
        "freeFlowSpeed": "free_flow_speed",
        "currentTravelTime": "current_travel_time",
        "freeFlowTravelTime": "free_flow_travel_time",
        "confidence": "confidence",
        "roadclosure": "road_closure",
        "roadClosure": "road_closure"
    }
    # lower-case keys and columns to catch variants
    df.columns = [c.strip() for c in df.columns]
    # no guarantee of uniform case so we'll coerce column names
    col_map = {}
    for c in df.columns:
        k = c.lower().replace(" ", "").replace("_", "")
        # match keys in rename by simplified form
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
    num = df.select_dtypes(include=["number"]).columns
    df[num] = df[num].fillna(df[num].median())
    cat = df.select_dtypes(include=["object"]).columns
    for c in cat:
        if not df[c].mode().empty:
            df[c] = df[c].fillna(df[c].mode()[0])
        else:
            df[c] = df[c].fillna("unknown")
    return df

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    kp = os.path.join(RAW_DIR, KAGGLE)
    ap = os.path.join(RAW_DIR, API)

    if not os.path.exists(kp):
        print(f"[ERROR] Kaggle file not found: {kp}")
        return
    if not os.path.exists(ap):
        print(f"[ERROR] API file not found: {ap}")
        return

    kag = pd.read_csv(kp)
    api = pd.read_csv(ap)

    kag = normalize_cols(kag)
    api = normalize_cols(api)

    print("Kaggle columns:", list(kag.columns))
    print("API columns   :", list(api.columns))

    kag = kaggle_to_common(kag)
    api = api_to_common(api)

    # parse timestamps
    kag = parse_timestamp(kag, "timestamp")
    api = parse_timestamp(api, "timestamp")

    # If API has current_speed, move to avg_speed if avg not present
    if "current_speed" in api.columns and "avg_speed" not in api.columns:
        api = api.rename(columns={"current_speed": "avg_speed"})

    # unify columns of interest (keep wide set)
    combined = pd.concat([kag, api], ignore_index=True, sort=False)
    combined = combined.drop_duplicates().reset_index(drop=True)
    combined = fill_missing(combined)

    outpath = os.path.join(OUT_DIR, OUTFILE)
    combined.to_csv(outpath, index=False)
    print(f"[OK] Traffic master saved -> {outpath} (rows: {len(combined)})")
    print("If you want fewer columns, we can select a subset (timestamp, city, location, traffic_volume, avg_speed, congestion_level).")

if __name__ == "__main__":
    main()
