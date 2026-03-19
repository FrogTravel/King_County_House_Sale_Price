# King County House Sales — EDA Report

## 1. Dataset Overview

- **Rows:** 21,613
- **Columns:** 21
- **Sales period:** May 2014 – May 2015
- **No missing values** in any column

## 2. Features

| Feature | Type | Description |
|---|---|---|
| `id` | int | Unique identifier — no predictive signal |
| `date` | datetime | Sale date |
| `price` | float | **Target variable** — sale price |
| `bedrooms` | int | Number of bedrooms |
| `bathrooms` | float | Number of bathrooms |
| `sqft_living` | int | Interior living area (sq ft) |
| `sqft_lot` | int | Lot size (sq ft) |
| `floors` | float | Number of floors |
| `waterfront` | int | Binary — waterfront view (0/1) |
| `view` | int | View quality (0–4) |
| `condition` | int | Overall condition (1–5) |
| `grade` | int | King County grade (1–13) |
| `sqft_above` | int | Above-ground area (sq ft) |
| `sqft_basement` | int | Basement area (sq ft) |
| `yr_built` | int | Year built |
| `yr_renovated` | int | Year last renovated (0 = never) |
| `zipcode` | int | Zip code — categorical |
| `lat` / `long` | float | Geographic coordinates |
| `sqft_living15` | int | Living area in 2015 (post-renovation) |
| `sqft_lot15` | int | Lot area in 2015 (post-renovation) |

**Candidates for removal:** `id`, `date`

## 3. Data Quality

- **No null values** across all 21 columns
- **Anomalies found:**
  - 12 entries with `bedrooms = 0` — possibly empty lots listed for sale (min `floors` is 1, which is contradictory)
  - 1 entry with `bedrooms = 33` — clear outlier, removed in cleaning
  - `sqft_living15` differs from `sqft_living` in many rows where `yr_renovated = 0` — suggests renovations are not always recorded

## 4. Feature Distributions

### Heavily right-skewed (log transform recommended)
- `price`, `sqft_living`, `sqft_lot`, `sqft_above`, `sqft_living15`, `sqft_lot15`
- `sqft_basement` — high spike at 0 (no basement); right-skewed among non-zero values

### Categorical / low-cardinality
| Feature | Unique values | Notes |
|---|---|---|
| `floors` | 6 | Fractional values (1.5, 2.5, 3.5) — treat as ordered categorical |
| `waterfront` | 2 | Very imbalanced — only 0.75% of properties have waterfront |
| `view` | 5 | Imbalanced — 90% have `view = 0` |
| `condition` | 5 | More items with condition = 3, 4 and 5 |
| `grade` | 12 | Near-normal distribution |
| `zipcode` | 70 | Categorical — encodes location |

### Near-uniform
- `yr_built` — spans 1900–2015, roughly uniform

### Spike-at-zero
- `yr_renovated` — vast majority are 0 (never renovated); treat as binary + value for renovated subset

## 5. Geographic Analysis

Properties are concentrated in the Seattle/Bellevue area (lat ~47.3–47.8, long ~-122.5 to -121.3).

Key geographic patterns:
- **Price** clusters in the north and along the waterfront (Mercer Island, Bellevue corridor)
- **Waterfront** properties are sparse but geographically identifiable
- **Zipcode** boundaries are clearly visible in scatter plots of `lat`/`long`

## 6. Correlations with Price

From the correlation heatmap (Pearson):

| Feature | Correlation with `price` |
|---|---|
| `sqft_living` | **highest** |
| `grade` | high |
| `sqft_above` | high |
| `sqft_living15` | high |
| `bathrooms` | moderate |
| `view` | moderate |
| `sqft_basement` | moderate |
| `bedrooms` | low–moderate |
| `waterfront` | low–moderate |
| `lat` | moderate (north = more expensive) |
| `yr_built` | low (weak) |
| `condition` | low |
| `sqft_lot` / `sqft_lot15` | low |
| `long` | low |
| `zipcode` | low (ordinal encoding misleads — treat as categorical) |

`id` and `date` show no meaningful signal.

**Note:** `sqft_living15` vs `sqft_living` discrepancy in rows with `yr_renovated = 0` is unexplained.

## 7. Outliers

### Obvious single outlier
- **Row 9714:** `bedrooms = 33` — input error, removed

### sqft_lot / sqft_lot15 extreme values
Three rows flagged (indices 9714, 20452, 13464):

| Index | sqft_lot z-score | sqft_lot15 z-score | Price z-score |
|---|---|---|---|
| 9714 | 20.7 | 31.4 | 1.1 |
| 20452 | 20.7 | 31.0 | 2.9 |
| 13464 | 10.1 | 20.1 | 0.7 |

These are real large-lot properties (not input errors)

### Global z-score analysis

| Threshold | % of rows flagged as outlier in at least one feature |
|---|---|
| z > 2 | **35.8%** |
| z > 3 | **13.2%** |

> With z > 2, over a third of the dataset is flagged — aggressive removal would lose too much data. A threshold of z > 3 (~13%) is more practical, or feature-specific capping.

## 8. Target Analysis — Price

- **Min:** $75,000 | **Max:** $7,700,000 | **Median:** $450,000 | **Mean:** $540,088
- Heavily **right-skewed** — log transform produces a near-normal distribution
- Strongest predictors: `sqft_living`, `grade`, `sqft_above`, `sqft_living15`

## 9. Key Findings & Recommendations

1. **Log-transform** `price`, `sqft_living`, `sqft_lot`, `sqft_above`, `sqft_living15`, `sqft_lot15` before modeling
2. **Drop** `id` and `date` (no signal); or extract month/season from `date` if seasonality is expected
3. **Treat as categorical:** `zipcode`, `view`, `condition`, `waterfront`, `floors`
4. **Encode** `yr_renovated` as binary `was_renovated` + a separate `years_since_renovation` for non-zero rows
5. **Investigate** `sqft_living15` / `sqft_living` mismatch with `yr_renovated = 0` — consider creating a `renovation_discrepancy` flag
6. **Remove** the 33-bedroom row; consider capping or removing the 3 extreme `sqft_lot` outliers
7. **Outlier strategy:** z-score threshold of 3 removes ~13% of data — acceptable; z > 2 removes 36% — too aggressive
