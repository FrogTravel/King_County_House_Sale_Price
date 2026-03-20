"""
Feature engineering.

Responsibilities:
- Log-transform right-skewed numeric features
- Build a zipcode → ordinal rank mapping (fit on training data only)
- Replace raw zipcode with its rank
- Decompose yr_renovated into interpretable binary + numeric features
- Expose a single preprocess_features() function for the full pipeline

All functions that learn from data (zipcode ranking) accept a pre-fitted
artifact so they can be applied to the test set without leakage.
"""

import numpy as np
import pandas as pd

from src.config import LOG_TRANSFORM_FEATURES, TARGET


# ── Log transformation ────────────────────────────────────────────────────────

def log_transform_features(
    df: pd.DataFrame,
    features: list = LOG_TRANSFORM_FEATURES,
) -> pd.DataFrame:
    """
    Apply log1p to right-skewed numeric features.

    log1p handles zeros safely (e.g. sqft_basement = 0 when no basement).
    Only transforms columns that actually exist in the DataFrame.
    """
    result = df.copy()
    for col in features:
        if col in result.columns:
            result[col] = np.log1p(result[col])
    return result


# ── Zipcode encoding ──────────────────────────────────────────────────────────

def create_zipcode_ranking_map(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> pd.DataFrame:
    """
    Build a zipcode → ordinal rank lookup from training data only.

    Strategy: rank zipcodes by their mean cost-per-sqft (price / sqft_living).
    Using cost-per-sqft rather than raw median price reduces the influence of
    property size variation within a zipcode.

    Returns a DataFrame with columns ['zipcode', 'zipcode_ordinal_rank'].
    """
    combined = X_train[["zipcode", "sqft_living"]].copy()
    combined[TARGET] = y_train.values
    combined["cost_per_sqft"] = combined[TARGET] / combined["sqft_living"]

    mean_cost = combined.groupby("zipcode")["cost_per_sqft"].mean()
    ranking = mean_cost.rank(method="dense", ascending=True).astype(int)
    return ranking.reset_index().rename(columns={"cost_per_sqft": "zipcode_ordinal_rank"})


def apply_zipcode_ranking(
    df: pd.DataFrame,
    ranking_map: pd.DataFrame,
) -> pd.DataFrame:
    """
    Replace the raw zipcode column with its pre-computed ordinal rank.

    Zipcodes unseen during training (can arise in test splits from other folds)
    receive the median rank as a safe fallback.
    """
    result = df.copy()
    result = result.merge(
        ranking_map[["zipcode", "zipcode_ordinal_rank"]],
        on="zipcode",
        how="left",
    )
    median_rank = ranking_map["zipcode_ordinal_rank"].median()
    result["zipcode_ordinal_rank"] = result["zipcode_ordinal_rank"].fillna(median_rank)
    result = result.drop(columns=["zipcode"])
    return result


# ── yr_renovated decomposition ────────────────────────────────────────────────

def engineer_renovation_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Decompose yr_renovated into two interpretable features:

    - was_renovated (int 0/1): whether the property was ever renovated.
    - years_since_renovation (int): years between last renovation and 2015
      (the last year in the dataset); 0 if never renovated.

    The original yr_renovated column is dropped.
    """
    result = df.copy()
    result["was_renovated"] = (result["yr_renovated"] > 0).astype(int)
    result["years_since_renovation"] = result["yr_renovated"].apply(
        lambda y: (2015 - int(y)) if y > 0 else 0
    )
    result = result.drop(columns=["yr_renovated"])
    return result


# ── Full preprocessing pipeline ───────────────────────────────────────────────

def preprocess_features(
    df: pd.DataFrame,
    zipcode_ranking_map: pd.DataFrame,
) -> pd.DataFrame:
    """
    Full feature preprocessing pipeline (applied identically to train and test):

    1. Drop uninformative columns (id, date)
    2. Log-transform right-skewed numeric features
    3. Engineer renovation features from yr_renovated
    4. Replace zipcode with its pre-computed ordinal rank

    The zipcode_ranking_map must be built from training data only
    (via create_zipcode_ranking_map) to avoid target leakage.
    """
    result = df.copy()

    # 1. Drop uninformative columns (ignore if already absent)
    result = result.drop(columns=["id", "date"], errors="ignore")

    # 2. Log transform skewed features
    result = log_transform_features(result)

    # 3. Renovation features
    result = engineer_renovation_features(result)

    # 4. Zipcode ordinal encoding
    result = apply_zipcode_ranking(result, zipcode_ranking_map)

    return result
