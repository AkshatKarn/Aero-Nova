#!/usr/bin/env python3
"""
processing_scheduler.py
Runs ALL Python scripts inside /preprocessing exactly ONCE.
Intended to run IMMEDIATELY AFTER full_fetch_scheduler.py
"""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PRE_DIR = ROOT / "preprocessing"

def run_preprocessing_once():
    print("\n====================================")
    print("🔧 RUNNING PREPROCESSING SCRIPTS...")
    print("====================================\n")

    if not PRE_DIR.exists():
        print(f"[ERROR] preprocessing folder not found at: {PRE_DIR}")
        return

    scripts = sorted(PRE_DIR.glob("*.py"))

    if not scripts:
        print("[WARN] No .py files found inside preprocessing/")
        return

    for script in scripts:
        print(f"\n▶ Running {script.name} ...")
        try:
            subprocess.run(["python", str(script)], check=True)
            print(f"✔ Completed: {script.name}")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Script failed: {script.name} -> {e}")

    print("\n✅ All preprocessing scripts completed.\n")


if __name__ == "__main__":
    run_preprocessing_once()
