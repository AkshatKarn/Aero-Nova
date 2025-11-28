# predict.py (updated - auto-mapping feature names)
import joblib
import pandas as pd
import numpy as np
import os
from typing import Tuple, List, Optional

MODEL_PATH = r"saved_models/combined_master.joblib"
SAVED_MODELS_DIR = os.path.dirname(MODEL_PATH)

def normalize_col(name: str) -> str:
    if pd.isna(name):
        return ""
    s = str(name).strip().lower()
    s = s.replace(".", "_").replace("-", "_").replace(" ", "_")
    s = s.replace("__", "_")
    return s

def find_model_file_in_folder(folder: str) -> Optional[str]:
    if not os.path.isdir(folder):
        return None
    candidates = []
    for fname in os.listdir(folder):
        path = os.path.join(folder, fname)
        if not os.path.isfile(path):
            continue
        ln = fname.lower()
        if not (ln.endswith(".joblib") or ln.endswith(".pkl")):
            continue
        # ignore master data joblibs
        if ln.endswith("_master.joblib") or ln.endswith("_master.pkl"):
            continue
        candidates.append(path)
    # prefer files with 'pipeline' or 'model' in name
    for p in candidates:
        ln = os.path.basename(p).lower()
        if "pipeline" in ln or "model" in ln or "final" in ln:
            return p
    return candidates[0] if candidates else None

def load_model() -> Tuple[object, Optional[List[str]]]:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"MODEL_PATH does not exist: {MODEL_PATH}")

    saved = joblib.load(MODEL_PATH)

    # if dictionary with pipeline/model inside
    if isinstance(saved, dict):
        for key in ("pipeline", "model", "estimator", "pipe"):
            if key in saved:
                pipeline = saved[key]
                feature_names = saved.get("feature_names", None)
                return pipeline, feature_names
        # maybe values contain estimator
        for v in saved.values():
            if hasattr(v, "predict"):
                pipeline = v
                feature_names = saved.get("feature_names", None)
                return pipeline, feature_names

    # If loaded is a DataFrame -> it's data, not model. Find actual model file in same folder
    if isinstance(saved, pd.DataFrame):
        print("[info] Loaded joblib is a DataFrame (data), not a model.")
        model_file = find_model_file_in_folder(SAVED_MODELS_DIR)
        if not model_file:
            raise RuntimeError(
                f"No model file found in '{SAVED_MODELS_DIR}'.\n"
                "Place your trained pipeline joblib in that folder and name it e.g. 'pipeline.joblib'."
            )
        print(f"[info] Found candidate model file: {model_file}. Loading it...")
        loaded = joblib.load(model_file)
        if isinstance(loaded, dict):
            for key in ("pipeline", "model", "estimator", "pipe"):
                if key in loaded:
                    pipeline = loaded[key]
                    feature_names = loaded.get("feature_names", None)
                    return pipeline, feature_names
            # fallback to any estimator inside dict
            for v in loaded.values():
                if hasattr(v, "predict"):
                    pipeline = v
                    feature_names = loaded.get("feature_names", None)
                    return pipeline, feature_names
            raise RuntimeError(f"Could not find estimator inside {model_file}.")
        else:
            if hasattr(loaded, "predict"):
                pipeline = loaded
                feat_names = None
                if hasattr(pipeline, "feature_names_in_"):
                    feat_names = list(getattr(pipeline, "feature_names_in_"))
                return pipeline, feat_names

    # If saved is direct estimator/pipeline
    if hasattr(saved, "predict"):
        pipeline = saved
        feat_names = None
        if hasattr(pipeline, "feature_names_in_"):
            feat_names = list(getattr(pipeline, "feature_names_in_"))
        return pipeline, feat_names

    raise RuntimeError("Unable to load a model/pipeline from provided joblib file(s).")

# -----------------------
# Helper to map common pollutant inputs to model feature names
# -----------------------
COMMON_KEYSETS = {
    "pm25": {"pm2_5", "pm25", "pm_2_5", "pm2"},
    "pm10": {"pm10", "pm_10"},
    "no2":  {"no2", "no_2"},
    "so2":  {"so2", "so_2"},
    "o3":   {"o3", "o_3"},
    "nh3":  {"nh3", "nh_3"},
}

def build_feature_map(expected_features: List[str]) -> dict:
    """
    Given the expected feature names (original model column names), return a mapping:
      normalized_expected -> original_expected
    """
    return { normalize_col(f): f for f in expected_features }

def map_inputs_to_features(feature_map: dict, pm25=None, pm10=None, no2=None):
    """
    Attempt to map the three provided pollutant values into whatever features the model expects.
    Returns a dict of original_feature_name -> value (or np.nan).
    """
    mapped = {}
    # start with NaN for all expected features
    for norm_f, orig_f in feature_map.items():
        mapped[orig_f] = np.nan

    # build reverse lookup: check normalized expected features for substrings
    # For each common pollutant, try to find first matching expected feature
    provided = {"pm25": pm25, "pm10": pm10, "no2": no2}
    for pollutant, val in provided.items():
        if val is None:
            continue
        keys_to_match = COMMON_KEYSETS[pollutant]
        found = False
        # exact substring match in normalized expected features
        for norm_f, orig_f in feature_map.items():
            for k in keys_to_match:
                if k in norm_f:
                    mapped[orig_f] = val
                    found = True
                    break
            if found:
                break
        # if not found, as fallback try matching by suffix/prefix like contains pollutant letters
        if not found:
            for norm_f, orig_f in feature_map.items():
                if pollutant in norm_f:
                    mapped[orig_f] = val
                    found = True
                    break
        # final fallback: do nothing here; other code may place values by position

    # If none of the provided values were mapped (very unlikely), place them in first columns
    all_mapped_values = [v for v in mapped.values() if not pd.isna(v)]
    if len(all_mapped_values) == 0:
        ofeatures = list(feature_map.values())
        if len(ofeatures) >= 1 and pm25 is not None: mapped[ofeatures[0]] = pm25
        if len(ofeatures) >= 2 and pm10 is not None: mapped[ofeatures[1]] = pm10
        if len(ofeatures) >= 3 and no2 is not None: mapped[ofeatures[2]] = no2

    return mapped

# -----------------------
# Predict single row
# -----------------------
def predict_single(pm25: float, pm10: float, no2: float):
    try:
        pipeline, feature_names = load_model()
    except Exception as e:
        print("[ERROR] loading model:", e)
        return

    # If pipeline exposes feature names
    if feature_names is None and hasattr(pipeline, "feature_names_in_"):
        feature_names = list(getattr(pipeline, "feature_names_in_"))

    # If still None, try to infer a minimal feature list
    if feature_names is None:
        # create default normalized names
        feature_names = ["pm2_5", "pm10", "no2"]

    # ensure list
    feature_names = list(feature_names)

    # Build normalized map and then mapped values
    feature_map = build_feature_map(feature_names)   # normalized -> original
    mapped_values = map_inputs_to_features(feature_map, pm25=pm25, pm10=pm10, no2=no2)

    # Create single-row DataFrame with original feature column names in correct order
    row = { orig: mapped_values.get(orig, np.nan) for orig in feature_names }

    # If any feature still missing in row, fill with 0 (or np.nan) - choose 0 for numerical stability
    for k in row:
        if pd.isna(row[k]):
            row[k] = 0.0

    X = pd.DataFrame([row], columns=feature_names)
    try:
        pred = pipeline.predict(X)[0]
        print("Predicted AQI:", pred)
        return pred
    except Exception as e:
        print("[ERROR] pipeline prediction failed:", e)
        return

# -----------------------
# Predict from CSV (keeps earlier behavior but improved mapping)
# -----------------------
def predict_from_csv(csv_path: str, out_path: Optional[str] = None):
    try:
        pipeline, feature_names = load_model()
    except Exception as e:
        print("[ERROR] loading model:", e)
        return

    df = pd.read_csv(csv_path)
    # keep original columns saved for output
    orig_cols = df.columns.tolist()
    # normalize df cols
    df.columns = [normalize_col(c) for c in orig_cols]

    # determine feature names
    if feature_names is None and hasattr(pipeline, "feature_names_in_"):
        feature_names = list(getattr(pipeline, "feature_names_in_"))
    if feature_names is None:
        # fallback: use all df columns as features
        feature_names = df.columns.tolist()

    # Build normalized feature map
    feature_map = build_feature_map(feature_names)  # normalized -> original

    # Ensure df contains normalized columns for all expected features
    # If not present, create with NaN
    for norm_f, orig_f in feature_map.items():
        if norm_f not in df.columns:
            df[norm_f] = np.nan

    # Reorder columns to pipeline expected (normalized form)
    norm_feature_order = [normalize_col(fn) for fn in feature_names]
    X = df[norm_feature_order]

    # Predict
    try:
        preds = pipeline.predict(X)
    except Exception as e:
        print("[ERROR] pipeline prediction failed:", e)
        return

    # attach predicted column and restore original column names for output
    result = df.copy()
    result["predicted_AQI"] = preds
    # rename cols back to original input names
    rename_map = {normalize_col(o): o for o in orig_cols}
    result = result.rename(columns=rename_map)

    if out_path:
        result.to_csv(out_path, index=False)
        print(f"Prediction saved to: {out_path}")
    else:
        print(result.head())
    return result

# -----------------------
# If run as script
# -----------------------
if __name__ == "__main__":
    # example direct predict
    try:
        predict_single(100, 150, 60)
    except Exception as e:
        print("[ERROR] predict_single failed:", e)
