# check_columns.py (put this in project root, not inside preprocessing/)
import os
import pandas as pd

print("Current working dir:", os.getcwd())
print("Files in current folder:", os.listdir("."))

data_folder = "data"
if os.path.isdir(data_folder):
    print("\nFiles in ./data/:", os.listdir(data_folder))
else:
    print("\nNo ./data/ folder found in current dir.")

files = [
    os.path.join(data_folder, "historical_weather.csv"),
    os.path.join(data_folder, "live_weather.csv"),
    os.path.join(data_folder, "aqi_historical_data.csv"),
    os.path.join(data_folder, "aqi_live_data.csv"),
    os.path.join(data_folder, "traffic_timeseries.csv"),
    os.path.join(data_folder, "Banglore_traffic_Dataset.csv")
]

for f in files:
    try:
        print("\n====", f, "====")
        df = pd.read_csv(f)
        print(df.columns.tolist())
    except Exception as e:
        print("Error reading", f, ":", e)
