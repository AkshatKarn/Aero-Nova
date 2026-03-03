import os
import json
import time
import pickle
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

MODEL_DIR = "models"
META_PATH = os.path.join(MODEL_DIR, "metadata.json")
RETRAIN_INTERVAL = 6 * 60 * 60  # 6 hours


def get_model_path(parameter):
    return os.path.join(MODEL_DIR, f"{parameter}.pkl")


def load_metadata():
    if not os.path.exists(META_PATH):
        return {}
    with open(META_PATH, "r") as f:
        return json.load(f)


def save_metadata(meta):
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(META_PATH, "w") as f:
        json.dump(meta, f)


def should_retrain(parameter):
    meta = load_metadata()
    last_trained = meta.get(parameter, 0)
    return (time.time() - last_trained) > RETRAIN_INTERVAL


def train_model(series, parameter):
    model = ARIMA(series, order=(2, 1, 2))
    model_fit = model.fit()

    os.makedirs(MODEL_DIR, exist_ok=True)

    with open(get_model_path(parameter), "wb") as f:
        pickle.dump(model_fit, f)

    meta = load_metadata()
    meta[parameter] = time.time()
    save_metadata(meta)

    return model_fit


def load_model(parameter):
    with open(get_model_path(parameter), "rb") as f:
        return pickle.load(f)


def forecast_next_24(series, parameter):

    series = pd.Series(series).dropna()

    if len(series) < 10:
        return None

    model_path = get_model_path(parameter)

    if not os.path.exists(model_path) or should_retrain(parameter):
        model = train_model(series, parameter)
    else:
        model = load_model(parameter)

    forecast = model.forecast(steps=24)
    return forecast.tolist()