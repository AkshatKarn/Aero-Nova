from pathlib import Path

# Project paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

# Data files you’ll use later
RAW_AIR_QUALITY = DATA_DIR / "air_quality_raw.csv"
RAW_MOBILITY = DATA_DIR / "mobility_raw.csv"

# Model artifacts (future steps)
MODEL_DIR = RESULTS_DIR / "models"
FIGURES_DIR = RESULTS_DIR / "figures"
REPORTS_DIR = RESULTS_DIR / "reports"

# Create folders at runtime if needed (used by scripts)
def ensure_dirs():
    for d in [DATA_DIR, RESULTS_DIR, MODEL_DIR, FIGURES_DIR, REPORTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
