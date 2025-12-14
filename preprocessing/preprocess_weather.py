# preprocess_weather.py
import pandas as pd
from pathlib import Path

# project base (parent of this script)
BASE = Path(__file__).resolve().parent.parent

# candidate dirs to search for source files
CANDIDATE_DIRS = [
    BASE / "data",
    BASE / "data" / "processed",
    BASE / "data" / "live_processed",
]

# expected filenames (adjust if your filenames differ)
FILES = ["historical_weather.csv", "live_weather.csv"]
OUTFILE = "weather_master.csv"

# output directory (kept under data/processed as requested)
OUT_DIR = BASE / "data" / "processed"


def find_file(fname):
    """Return Path if found in any candidate dir, else None."""
    for d in CANDIDATE_DIRS:
        p = d / fname
        if p.exists():
            return p
    return None


def safe_read_csv(path):
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[ERROR] Failed to read {path}: {e}")
        return None


def normalize_cols(df):
    df = df.copy()
    # use regex=False for literal replacements to avoid warnings
    cols = (df.columns
              .str.strip()
              .str.replace(" ", "_", regex=False)
              .str.replace("(", "", regex=False)
              .str.replace(")", "", regex=False)
              .str.lower())
    df.columns = cols
    return df


def parse_timestamp(df):
    # convert any column that looks like timestamp/time to datetime
    for c in df.columns:
        if "timestamp" in c or ("time" in c and "timezone" not in c):
            try:
                df[c] = pd.to_datetime(df[c], errors="coerce")
            except Exception:
                # leave as-is if conversion fails
                pass
    return df


def fill_missing(df):
    num_cols = df.select_dtypes(include=["number"]).columns
    if len(num_cols):
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for c in cat_cols:
        if not df[c].mode().empty:
            df[c] = df[c].fillna(df[c].mode()[0])
        else:
            df[c] = df[c].fillna("unknown")
    return df


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    found_any = False
    dfs = []

    for fname in FILES:
        p = find_file(fname)
        if not p:
            print(f"[WARN] Missing {fname} — not found in candidate dirs.")
            continue

        print(f"[INFO] Found {fname} at {p}")
        df = safe_read_csv(p)
        if df is None:
            print(f"[WARN] Skipping {p} due to read error.")
            continue

        df = normalize_cols(df)
        df = parse_timestamp(df)
        # optional: tag source file column to trace provenance
        df["source_file"] = p.name
        dfs.append(df)
        found_any = True

    if not found_any or not dfs:
        print("[ERROR] No valid Weather files found. Exiting.")
        return

    combined = pd.concat(dfs, ignore_index=True, sort=False)
    combined = combined.drop_duplicates().reset_index(drop=True)
    combined = fill_missing(combined)

    outpath = OUT_DIR / OUTFILE
    try:
        combined.to_csv(outpath, index=False)
        print(f"[OK] Weather master saved -> {outpath} (rows: {len(combined)})")
    except Exception as e:
        print(f"[ERROR] Failed to write output {outpath}: {e}")


if __name__ == "__main__":
    main()
