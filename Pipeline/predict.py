# /mnt/data/predict.py
import os
import argparse
import json as _json
import joblib
import pandas as pd
import numpy as np

MODEL_DIR = "results/models"
PROCESSED_DIR = "data/processed"

# Use the features JSON saved from training (preferred). Keep CSV fallback for compatibility.
FEATURE_FILE_JSON = os.path.join(MODEL_DIR, "features_latest.json")
FEATURE_FILE_CSV = os.path.join(MODEL_DIR, "feature_importance_latest.csv")
XGB_FILE = os.path.join(MODEL_DIR, "xgb_model_latest.joblib")
LIN_FILE = os.path.join(MODEL_DIR, "linear_model_latest.joblib")
EXPLAINER_FILE = os.path.join(MODEL_DIR, "explainer_latest.joblib")


# ---------------- LOADERS ----------------
def load_model():
    if os.path.exists(XGB_FILE):
        return joblib.load(XGB_FILE), "xgb"
    if os.path.exists(LIN_FILE):
        return joblib.load(LIN_FILE), "linear"
    raise FileNotFoundError("Model not found. Train model first.")


def load_feature_list():
    # Preferred: load training feature list saved as JSON (guarantees order)
    if os.path.exists(FEATURE_FILE_JSON):
        try:
            with open(FEATURE_FILE_JSON, "r") as f:
                features = _json.load(f)
            print("Loaded feature list from JSON:", FEATURE_FILE_JSON)
            return features
        except Exception as e:
            print("Failed to load features JSON:", e)

    # Fallback: read feature_importance CSV (may be sorted by importance and thus wrong order)
    if os.path.exists(FEATURE_FILE_CSV):
        try:
            df = pd.read_csv(FEATURE_FILE_CSV)
            print("Loaded feature list from CSV (fallback):", FEATURE_FILE_CSV)
            return df["feature"].tolist()
        except Exception as e:
            print("Failed to load feature_importance CSV:", e)

    raise FileNotFoundError("No feature list found. Train model to generate 'features_latest.json'.")


def compute_medians():
    paths = [
        f"{PROCESSED_DIR}/aqi_master.csv",
        f"{PROCESSED_DIR}/weather_master.csv",
        f"{PROCESSED_DIR}/traffic_master.csv",
    ]
    dfs = []
    for p in paths:
        if os.path.exists(p):
            d = pd.read_csv(p)
            if "timestamp" in d.columns:
                d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce")
                d = d.dropna(subset=["timestamp"]).set_index("timestamp").resample("1h").mean().reset_index()
            dfs.append(d)

    if len(dfs) == 0:
        return {}

    merged = dfs[0]
    for other in dfs[1:]:
        merged = pd.merge_asof(
            merged.sort_values("timestamp"),
            other.sort_values("timestamp"),
            on="timestamp",
            direction="nearest",
            tolerance=pd.Timedelta("1h")
        )

    # numeric-only
    for c in merged.columns:
        merged[c] = pd.to_numeric(merged[c], errors="coerce")

    return merged.median(numeric_only=True).to_dict()


# ---------------- PREPARE INPUT ----------------
def prepare_input(df_in, feature_list, medians):
    df = df_in.copy()

    # Convert strings numbers to numeric
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Convert timestamp
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce").astype("int64") // 10**9

    # Final frame with EXACT columns in EXACT order
    final = pd.DataFrame(columns=feature_list)

    for col in feature_list:
        if col in df.columns:
            final[col] = df[col]
        else:
            final[col] = medians.get(col, 0)

    # If everything is NaN for a column, fill with median of final (numeric only)
    try:
        final = final.fillna(final.median(numeric_only=True))
    except Exception:
        final = final.fillna(0)

    return final


# ---------------- EXPLAINER ----------------
def explain(model_type, model, X_row):
    if hasattr(model, "feature_importances_"):
        fi = model.feature_importances_
        df = pd.DataFrame({"feature": X_row.columns, "importance": fi})
        df = df.sort_values("importance", ascending=False)
        return df
    return None


# ---------------- MAIN PREDICTION ----------------
def predict_dataframe(df_input):
    model, model_type = load_model()
    features = load_feature_list()
    medians = compute_medians()

    X_ready = prepare_input(df_input, features, medians)

    # debug: print mismatch info if any
    incoming = list(X_ready.columns)
    expected = list(features)
    missing = [c for c in expected if c not in incoming]
    extra = [c for c in incoming if c not in expected]
    if missing or extra:
        print(f"[DEBUG] expected {len(expected)} features, incoming {len(incoming)}")
        if missing:
            print("[DEBUG] MISSING features (should not happen):", missing)
        if extra:
            print("[DEBUG] EXTRA features (should not happen):", extra)

    preds = model.predict(X_ready)

    fi = None
    if len(X_ready) > 0:
        fi = explain(model_type, model, X_ready.iloc[[0]])
    else:
        print("[WARNING] No rows found in input → X_ready is empty.")


    return preds, fi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="CSV input")
    parser.add_argument("--json", help="JSON row input")
    args = parser.parse_args()

    # Load input dataframe from --file OR --json
    if args.file:
        # Allow .json input too — but handle single-object JSON correctly
        if args.file.lower().endswith(".json"):
            # read file as JSON then convert to DataFrame
            with open(args.file, "r", encoding="utf-8") as fh:
                obj = _json.load(fh)
            if isinstance(obj, list):
                df = pd.DataFrame(obj)
            elif isinstance(obj, dict):
                df = pd.DataFrame([obj])
            else:
                # fallback to pandas reader for other formats
                df = pd.read_json(args.file)
        else:
            df = pd.read_csv(args.file)

    elif args.json:
        # args.json is a JSON string for a single row
        df = pd.DataFrame([_json.loads(args.json)])
    else:
        print("Provide --file or --json")
        return

    preds, fi = predict_dataframe(df)
    df["predicted_AQI"] = preds

    print("\n----------- RESULT -----------")
    print(df.to_string(index=False))

    if fi is not None:
        print("\nTop Feature Importances:")
        print(fi.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
