from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FIGURES_DIR = PROJECT_ROOT / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

PERIOD_RANGES = {
    "Pre-pandemic": (2017, 2019),
    "Pandemic": (2020, 2021),
    "Recovery": (2022, 2024),
}

PERIOD_ORDER = ["Pre-pandemic", "Pandemic", "Recovery"]

MODEL_FORMULA = "demand_mw ~ C(hour) * C(day_type) + C(month)"
DEFAULT_TEST_FRACTION = 0.20
DEFAULT_SIMULATION_RUNS = 1000
DEFAULT_RANDOM_SEED = 42

# Actual workbook layout for the official NGCP Hourly Demand per Grid file.
NGCP_LUZON_SHEET = "LUZON HOURLY LOAD 2013-2025"
NGCP_HEADER_ROW = 1
NGCP_DATE_COLUMN = "DATE"
