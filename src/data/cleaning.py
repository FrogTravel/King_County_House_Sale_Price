"""
Data loading and cleaning.

Responsibilities:
- Load the raw CSV from data/raw/
- Remove confirmed input errors (e.g. bedrooms = 33)
- Fix column data types
- Persist the cleaned dataset to data/processed/

This module does NOT apply any feature transformations.
Transformations that must be fit on the training set only
(log scaling, zipcode encoding, etc.) belong in src/features/engineering.py.
"""

import pandas as pd

from src.config import RAW_DATA_PATH, PROCESSED_DATA_PATH


# ── Loading ───────────────────────────────────────────────────────────────────

def load_raw_data() -> pd.DataFrame:
    """Load the raw dataset from data/raw/."""
    return pd.read_csv(RAW_DATA_PATH)


def load_clean_data() -> pd.DataFrame:
    """Load the cleaned dataset from data/processed/."""
    return pd.read_csv(PROCESSED_DATA_PATH)


# ── Cleaning steps ────────────────────────────────────────────────────────────

def fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Parse the date column to datetime."""
    result = df.copy()
    result["date"] = pd.to_datetime(result["date"])
    return result


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove confirmed data entry errors.

    - bedrooms = 33: a single row with an impossibly high bedroom count.
      The property has a normal sqft_living (~1,620 sq ft) — clear typo.
    """
    return df[df["bedrooms"] < 33].reset_index(drop=True)


# ── Persistence ───────────────────────────────────────────────────────────────

def save_clean_data(df: pd.DataFrame) -> None:
    """Save the cleaned DataFrame to data/processed/."""
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"  Saved → {PROCESSED_DATA_PATH}  ({len(df):,} rows)")


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_cleaning_pipeline() -> pd.DataFrame:
    """
    End-to-end cleaning pipeline:
        load → fix dtypes → remove outliers → save to processed/

    Returns the cleaned DataFrame.
    """
    data = load_raw_data()
    print(f"  Loaded raw data: {data.shape[0]:,} rows × {data.shape[1]} columns")

    data = fix_dtypes(data)
    data = remove_outliers(data)
    print(f"  After cleaning:  {data.shape[0]:,} rows × {data.shape[1]} columns")

    save_clean_data(data)
    return data
