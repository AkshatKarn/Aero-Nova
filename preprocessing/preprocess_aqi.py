# preprocess_aqi.py
import pandas as pd
from pathlib import Path

# BASE directory = project root (parent of /preprocessing folder)
BASE = Path(__file__).resolve().parent.parent

# Candidate dirs under BASE/data
CANDIDATE_DIRS = [
    BASE / "data",
    BASE / "data" / "processed",
    BASE / "data" / "live_processed",
]

INFILES = ["aqi_historical_data.csv", "aqi_live_data.csv"]
OUTFILE = "aqi_master.csv"

# <-- Change output to data/processed (so it will appear under processed)
OUT_DIR = BASE / "data" / "processed"


def find_file(fname):
    """Return Path if found in any candidate dir, else None."""
    for d in CANDIDATE_DIRS:
        p = d / fname
        if p.exists():
            return p
    return None


def load_csv_if_exists(path):
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[ERROR] Failed reading {path}: {e}")
        return None


def main():
    found = {}
    for fname in INFILES:
        p = find_file(fname)
        if p:
            print(f"[INFO] Found {fname} at: {p}")
            found[fname] = p
        else:
            print(f"[WARN] Missing {fname} — not found in candidate dirs.")

    if not found:
        print("[ERROR] No AQI files found. Exiting.")
        return

    dfs = []
    for fname in INFILES:
        p = found.get(fname)
        if p:
            df = load_csv_if_exists(p)
            if df is not None:
                df["source_file"] = p.name  # better: actual source filename
                dfs.append(df)

    if not dfs:
        print("[ERROR] No valid dataframes loaded. Exiting.")
        return

    master = pd.concat(dfs, ignore_index=True, sort=False)

    # Ensure output directory exists
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    out_path = OUT_DIR / OUTFILE
    master.to_csv(out_path, index=False)

    print(f"[OK] Wrote merged AQI master to: {out_path}")


if __name__ == "__main__":
    main()
