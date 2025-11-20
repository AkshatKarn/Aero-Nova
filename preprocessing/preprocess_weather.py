# preprocess_weather.py
import os
import pandas as pd

RAW_DIR = "../data"
OUT_DIR = "../processed"
FILES = ["historical_weather.csv", "live_weather.csv"]
OUTFILE = "weather_master.csv"

def normalize_cols(df):
    df = df.copy()
    df.columns = df.columns.str.strip().str.replace(" ", "_").str.replace("(", "").str.replace(")", "").str.lower()
    return df

def parse_timestamp(df):
    for c in df.columns:
        if "timestamp" in c or "time" in c:
            df[c] = pd.to_datetime(df[c], errors="coerce")
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
    dfs = []
    for f in FILES:
        p = os.path.join(RAW_DIR, f)
        if not os.path.exists(p):
            print(f"[WARN] Missing {p} — skipping.")
            continue
        df = pd.read_csv(p)
        df = normalize_cols(df)
        df = parse_timestamp(df)
        dfs.append(df)
    if not dfs:
        print("No Weather files found. Exiting.")
        return
    combined = pd.concat(dfs, ignore_index=True, sort=False)
    combined = combined.drop_duplicates().reset_index(drop=True)
    combined = fill_missing(combined)
    outpath = os.path.join(OUT_DIR, OUTFILE)
    combined.to_csv(outpath, index=False)
    print(f"[OK] Weather master saved -> {outpath} (rows: {len(combined)})")

if __name__ == "__main__":
    main()
