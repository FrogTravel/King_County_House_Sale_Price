# King County House Sale Price Prediction

**Course:** Data Science & Machine Learning
**Author:** Ekaterina Levchenko
**Date:** March 2026

---

## Project Overview

This project builds and evaluates regression models to predict residential house sale prices in King County, Washington (Seattle metro area). The dataset covers 21,613 sales recorded between May 2014 and May 2015. The full pipeline covers exploratory data analysis, data cleaning and feature engineering, and a comparative evaluation of five machine learning models ranging from a linear baseline to gradient boosting ensembles.

The best model — CatBoost — achieves a test R² of **0.8975** and a test RMSE of **$119,007**, compared to a linear regression baseline of R² = 0.735 and RMSE = $191,484.

---

## Project Structure

```
king_county_price_prediction/
├── data/
│   └── raw/                        # Original dataset (king_ country_ houses_aa.csv)
├── notebooks/
│   └── eda.ipynb                   # Exploratory Data Analysis notebook
├── src/
│   └── data/
│       ├── modelling.ipynb         # Model training and evaluation
│       └── cleaning.py             # Preprocessing utilities
├── reports/
│   ├── eda_report.md               # Structured EDA findings
│   └── REPORT_EDA.md               # Raw EDA notes
└── README.md
```

---

## Dataset

| Property | Value |
|---|---|
| Source | [Kaggle](https://www.kaggle.com/datasets/minasameh55/king-country-houses-aa/data)  |
| Period | May 2014 – May 2015 |
| Rows | 21,613 |
| Features | 21 (including target) |
| Target | `price` (continuous, USD) |
| Missing values | None |

### Feature Summary

| Feature | Type | Description |
|---|---|---|
| `price` | float | Target — sale price (USD) |
| `sqft_living` | int | Interior living area (sq ft) |
| `sqft_above` | int | Above-ground area (sq ft) |
| `sqft_basement` | int | Basement area (sq ft); 0 = no basement |
| `sqft_lot` | int | Lot size (sq ft) |
| `sqft_living15` | int | Living area as of 2015 (may reflect unreported renovations) |
| `sqft_lot15` | int | Lot area as of 2015 |
| `grade` | int 1–13 | King County construction & design quality grade |
| `bedrooms` | int | Number of bedrooms |
| `bathrooms` | float | Number of bathrooms (fractional values used) |
| `floors` | float | Number of floors; fractional values (1.5, 2.5, 3.5) |
| `waterfront` | int 0/1 | Binary — waterfront property (very imbalanced: 0.75% positive) |
| `view` | int 0–4 | View quality rating |
| `condition` | int 1–5 | Overall condition of the property |
| `yr_built` | int | Year built |
| `yr_renovated` | int | Year last renovated; 0 = never renovated |
| `zipcode` | int | Zip code — 70 unique values; treated as categorical |
| `lat` / `long` | float | Geographic coordinates |
| `id` | int | Unique identifier — dropped (no signal) |
| `date` | datetime | Sale date — dropped (no seasonality analysis performed) |

---

## Exploratory Data Analysis

### Distributions

Several features are heavily right-skewed and require log transformation for linear modeling:

- **Right-skewed:** `price`, `sqft_living`, `sqft_lot`, `sqft_above`, `sqft_living15`, `sqft_lot15`
- **Spike at zero:** `sqft_basement` (no basement) and `yr_renovated` (never renovated)
- **Categorical / low-cardinality:** `floors` (6 values), `waterfront`, `view`, `condition`, `grade`, `zipcode`
- **Near-uniform:** `yr_built` spans 1900–2015 with no dominant peak

Log transformation of `sqft_lot` brings it closer to normal, but it remains non-Gaussian due to a few extreme large-lot properties.

### Target Variable

| Statistic | Value |
|---|---|
| Minimum | $75,000 |
| Median | $450,000 |
| Mean | $540,088 |
| Maximum | $7,700,000 |

### Geographic Patterns

Properties are concentrated in the Seattle/Bellevue metro area (latitude 47.3–47.8, longitude −122.5 to −121.3). Key patterns:

- Prices cluster in the north and along waterfront corridors (Mercer Island, Bellevue).
- Zipcode boundaries are visible in `lat`/`long` scatter plots, confirming zipcode as a strong location proxy.
- Latitude shows a moderate positive correlation with price — properties further north tend to command higher prices.
- Waterfront properties are sparse (0.75% of the dataset) but geographically identifiable and command a clear premium.

### Correlations with Price (Pearson)

| Feature | Correlation |
|---|---|
| `sqft_living` | 0.70 |
| `grade` | 0.67 |
| `sqft_above` | 0.61 |
| `sqft_living15` | 0.59 |
| `bathrooms` | 0.53 |
| `view` | 0.40 |
| `sqft_basement` | 0.32 |
| `lat` | 0.31 |
| `bedrooms` | 0.31 |
| `waterfront` | 0.27 |
| `yr_built` | 0.05 (weak) |

Living space (sqft) and construction quality (grade) are the dominant predictors.

### Data Quality & Anomalies

- **No missing values** across all 21,613 rows.
- **1 input error:** `bedrooms = 33` — removed from the dataset.
- **12 rows with 0 bedrooms:** Possibly empty lots listed for sale (contradicted by `floors ≥ 1`); retained.
- **`sqft_living15` discrepancy:** Differs from `sqft_living` in rows where `yr_renovated = 0`, suggesting renovations were performed but not recorded.
- **3 extreme `sqft_lot` rows** (z-score > 20): These are real large-lot properties, not input errors — retained.

### Outlier Strategy

| z-score threshold | % of rows flagged |
|---|---|
| z > 2 | 35.8% — too aggressive |
| z > 3 | 13.2% — potentially pacceptable |

The 33-bedroom row was removed separately as a confirmed input error.

---

## Data Cleaning & Feature Engineering

The following transformations were applied:

**Log Transformations**
`price`, `sqft_living`, `sqft_lot`, `sqft_above`, `sqft_living15`, `sqft_lot15` were log-transformed to meet linear regression assumptions. Tree-based models are scale-invariant but still benefit from normalized distributions.

**Zipcode Encoding**
Zipcode was encoded as an ordinal rank based on median sale price within the training set. This avoids data leakage (ranks are computed only on training data) and captures the geographic pricing signal without imposing a misleading linear ordering.

**Categorical Features**
`waterfront`, `view`, and `condition` were treated as categorical. CatBoost handles these natively; other models received ordinal encoding.

**Dropped Features**
`id` (unique identifier, no predictive signal) and `date` were dropped.

**Train / Test Split**
80% / 20% split, stratified by `waterfront` to ensure both splits represent the rare waterfront class (0.75% of data). The zipcode ranking map was built exclusively on training data to prevent leakage.

---

## Modeling

Five models were trained and evaluated on the held-out test set (4,323 samples, 20%).

### Models

**1. Linear Regression (Baseline)**
Interpretable OLS model. Assumes linear relationships between log-transformed features and price. Establishes a lower-bound benchmark.

**2. Random Forest**
100 decision trees trained on bootstrapped samples with random feature subsets. Handles non-linearity naturally and is robust to outliers and irrelevant features.

**3. XGBoost**
Gradient boosting with L1/L2 regularization. Sequentially corrects residuals; delivers strong out-of-the-box performance on structured tabular data.

**4. Stacking Ensemble**
Layer 1 (base models): Random Forest, XGBoost, Gradient Boosting — each trained with 5-fold cross-validation to prevent target leakage.
Layer 2 (meta-model): Linear Regression learns the optimal combination of base model predictions.

**5. CatBoost**
Gradient boosting with native categorical feature support for `waterfront`, `view`, and `condition`. Uses ordered boosting with symmetric trees to reduce overfitting.

---

## Results

All models were evaluated on the same held-out 20% test set.

| Model | Train R² | Test R² | Test RMSE | Test MAE |
|---|---|---|---|---|
| Linear Regression | 0.729 | 0.735 | $191,484 | $112,554 |
| Random Forest | 0.983 | 0.882 | $127,574 | $69,339 |
| XGBoost | 0.978 | 0.886 | $125,562 | $68,042 |
| Stacking Ensemble | 0.961 | 0.896 | $119,958 | $66,747 |
| **CatBoost** | **0.951** | **0.898** | **$119,007** | $67,710 |

**Best model: CatBoost** — highest test R² (0.8975) and lowest RMSE ($119,007).

The Stacking Ensemble achieves the best MAE ($66,747), making it a strong alternative when average error magnitude is the primary concern. Random Forest shows the largest train/test gap (0.983 vs. 0.882), indicating the highest degree of overfitting among ensemble methods. Linear Regression underfits across the board, confirming that the underlying relationships are substantially non-linear.

All ensemble models reduced test RMSE by 34–38% relative to the linear baseline.

---

## Conclusions

1. **Living space and quality dominate.** `sqft_living`, `grade`, `sqft_above`, and `sqft_living15` are the strongest individual predictors of price. Lot size and year built contribute relatively little.

2. **Location encodes strongly.** Latitude and zipcode (via ordinal rank by median price) capture geographic price gradients that no single structural feature can replicate.

3. **Log transformations matter for linear models.** Right-skewed features must be log-transformed for linear regression; tree-based models are more tolerant but still benefit from cleaner distributions.

4. **CatBoost generalizes best.** Its native categorical handling and ordered boosting strategy produced the best balance of training fit and test generalization.

5. **Stacking is competitive.** Combining RF, XGBoost, and Gradient Boosting via a meta-learner achieves the lowest MAE and is only marginally behind CatBoost on R² and RMSE.

### Future Work

- **Hyperparameter tuning** (grid/random search or Optuna) for all tree-based models.
- **SHAP explainability** — quantify feature importance and per-prediction contributions.
- **Geospatial features** — proximity to schools, transit stops, green spaces.
- **Seasonality** — extract month/season from `date` to capture temporal price patterns.
- **Renovation discrepancy flag** — engineer a feature from the `sqft_living` vs. `sqft_living15` mismatch where `yr_renovated = 0`.
