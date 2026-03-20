"""
Central configuration for the King County House Price project.

All paths, feature lists, model hyperparameters, and tuning constants
live here so notebooks and pipeline scripts stay in sync.
"""

from pathlib import Path

# ── Directories ───────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"

# ── Data paths ────────────────────────────────────────────────────────────────
RAW_DATA_PATH = DATA_DIR / "raw" / "king_ country_ houses_aa.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "cleaned_data.csv"

# ── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.2

# ── Target & columns to drop ──────────────────────────────────────────────────
TARGET = "price"
# Dropped before any modelling: id has no signal; date is not used for seasonality
DROP_FEATURES = ["id", "date"]

# ── Feature groups ────────────────────────────────────────────────────────────
# Right-skewed features that benefit from log1p transformation
LOG_TRANSFORM_FEATURES = [
    "sqft_living",
    "sqft_lot",
    "sqft_above",
    "sqft_basement",
    "sqft_living15",
    "sqft_lot15",
]

# Categorical features passed to CatBoost natively (after zipcode is replaced by its rank)
CATBOOST_CAT_FEATURES = ["waterfront", "view", "condition"]

# ── Model hyperparameters ─────────────────────────────────────────────────────
RF_PARAMS = {
    "n_estimators": 100,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

XGB_PARAMS = {
    "n_estimators": 100,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

GB_PARAMS = {
    "n_estimators": 100,
    "random_state": RANDOM_STATE,
}

CATBOOST_PARAMS = {
    "iterations": 100,
    "random_seed": RANDOM_STATE,
    "verbose": False,
}

STACKING_CV = 5
