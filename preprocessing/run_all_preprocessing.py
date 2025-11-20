# run_all_preprocessing.py
import subprocess
import sys
import os

SCRIPTS = [
    "preprocess_aqi.py",
    "preprocess_weather.py",
    "preprocess_traffic.py"
]

def run(script):
    print("\n>>> Running", script)
    rc = subprocess.call([sys.executable, script])
    if rc != 0:
        print(f"[ERROR] {script} returned {rc}")

if __name__ == "__main__":
    cwd = os.getcwd()
    print("Working dir:", cwd)
    for s in SCRIPTS:
        if os.path.exists(s):
            run(s)
        else:
            print("[SKIP] Not found:", s)
