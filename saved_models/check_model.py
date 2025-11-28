import joblib
s = joblib.load("saved_models/aqi_pipeline_with_metadata_no_leak.joblib")

print("Model:", s.get("model_name"))
print("Training metrics:", s.get("training_metrics"))
feat = s.get("feature_names", [])
print("Num features:", len(feat))
print("Any leaked AQI columns?:", [f for f in feat if str(f).startswith("AQI_")])
print("First 40 features:", feat[:40])
