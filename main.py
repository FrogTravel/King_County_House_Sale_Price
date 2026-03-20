"""
King County House Sale Price Prediction — end-to-end pipeline.

Usage:
    python main.py
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    CATBOOST_CAT_FEATURES,
    PROCESSED_DATA_PATH,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
)
from src.data.cleaning import run_cleaning_pipeline
from src.features.engineering import create_zipcode_ranking_map, preprocess_features
from src.models.evaluate import compare_models, evaluate_model, print_metrics
from src.models.train import get_models, train_model


def main() -> None:
    print("=" * 60)
    print("  King County House Sale Price Prediction")
    print("=" * 60)

    # ── 1. Data cleaning ──────────────────────────────────────────────────────
    print("\n[1/4] Cleaning data...")
    run_cleaning_pipeline()

    # ── 2. Load & split ───────────────────────────────────────────────────────
    print("\n[2/4] Splitting data...")
    data = pd.read_csv(PROCESSED_DATA_PATH)
    X = data.drop(columns=[TARGET])
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=X["waterfront"],
        random_state=RANDOM_STATE,
    )
    print(f"  Train: {X_train.shape[0]:,} samples")
    print(f"  Test:  {X_test.shape[0]:,} samples")

    # Sanity check: all zipcodes in the dataset must appear in the training set
    # so the ranking map covers the full domain.
    assert set(X_train["zipcode"]) == set(data["zipcode"]), (
        "Training set is missing zipcodes present in the full dataset. "
        "Increase TEST_SIZE or re-seed."
    )

    # ── 3. Feature engineering ────────────────────────────────────────────────
    print("\n[3/4] Engineering features...")
    zipcode_ranking_map = create_zipcode_ranking_map(X_train, y_train)

    X_train_fe = preprocess_features(X_train, zipcode_ranking_map)
    X_test_fe  = preprocess_features(X_test,  zipcode_ranking_map)
    print(f"  Feature matrix: {X_train_fe.shape[1]} columns")

    # Resolve which categorical columns are still present after feature engineering
    cat_cols = [c for c in CATBOOST_CAT_FEATURES if c in X_train_fe.columns]

    # ── 4. Train & evaluate ───────────────────────────────────────────────────
    print("\n[4/4] Training and evaluating models...")
    models  = get_models()
    results = {}

    for name, model in models.items():
        print(f"\n  ── {name} {'─' * max(1, 40 - len(name))}")
        trained = train_model(model, X_train_fe, y_train, cat_features=cat_cols)
        metrics = evaluate_model(trained, X_train_fe, y_train, X_test_fe, y_test)
        results[name] = metrics
        print_metrics(name, metrics)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Model Comparison (sorted by Test R²)")
    print("=" * 60)
    summary = compare_models(results)
    print(summary.to_string())
    print()


if __name__ == "__main__":
    main()
