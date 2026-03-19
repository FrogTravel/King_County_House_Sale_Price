import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


categorical_features = ["zipcode", "floors", "waterfront", "view", "condition", "grade"]
numerical_features = ["bathrooms", "sqft_living", "sqft_lot", "sqft_above", "sqft_basement", "yr_built", "yr_renovated", "lat", "long", "sqft_living15", "sqft_lot15"]

log_n_features = ["sqft_basement", "sqft_living", "sqft_lot", "sqft_above", "sqft_living15", "sqft_lot15"] # TODO should we log the price?


def log_right_skewed_features(df, features):
  preprocess_data = df.copy()

  for feature in features:
    preprocess_data[feature] = np.log1p(preprocess_data[feature]) # There are 0 in some features

  return preprocess_data


def fix_dtypes(df):
  df["date"] = pd.to_datetime(df["date"])
  return df


# The original encode_zipcode_to_ordinal function is retained here for reference or future use.
# However, the user's request will be fulfilled by a new set of functions for X_train/y_train.
# Creating additional feature to encode the zipcode
# The idea is that zipcodes represent different areas.
# Different areas have different cost_per_sqft
# Then we can encode the zipcode as ordinal, where
# the biggest values is the most expensive area
# the smalles value is the cheapest area
def encode_zipcode_to_ordinal(df):
  result = df.copy()
  result["cost_per_sqft"] = df["price"] / df["sqft_living"]

  # Calculate the mean cost per sqft for each zipcode
  zipcode_mean_cost = result.groupby("zipcode")["cost_per_sqft"].mean()

  # Rank the zipcodes based on their mean cost per sqft
  # Using 'dense' method so that ranks are consecutive integers
  zipcode_rank_series = zipcode_mean_cost.rank(method='dense', ascending=True)

  # Convert the Series to a DataFrame and reset index to make 'zipcode' a column for merging
  zipcode_rank_df = zipcode_rank_series.reset_index(name='zipcode_ordinal_rank')

  # Merge this new ranking feature back into the original DataFrame copy
  result = result.merge(zipcode_rank_df[['zipcode', 'zipcode_ordinal_rank']], on="zipcode", how="left")

  # Drop the intermediate 'cost_per_sqft' column
  result = result.drop(columns=["cost_per_sqft"])

  return result


# Function to create the zipcode ranking map from training data
def create_zipcode_ranking_map(X_train_df, y_train_series):
    train_combined = X_train_df.copy()
    train_combined['price'] = y_train_series.values # Ensure alignment
    train_combined['cost_per_sqft'] = train_combined['price'] / train_combined['sqft_living']

    zipcode_mean_cost = train_combined.groupby('zipcode')['cost_per_sqft'].mean()
    zipcode_rank_series = zipcode_mean_cost.rank(method='dense', ascending=True)
    zipcode_rank_df = zipcode_rank_series.reset_index(name='zipcode_ordinal_rank')
    return zipcode_rank_df


# Function to apply the zipcode ranking map to a DataFrame
def apply_zipcode_ranking(df_to_transform, zipcode_ranking_map_df):
    df_transformed = df_to_transform.copy()
    df_transformed = df_transformed.merge(zipcode_ranking_map_df[['zipcode', 'zipcode_ordinal_rank']], on='zipcode', how='left')
    # Handle zipcodes in test set not seen in training set if necessary (e.g., fill with median/mode or a special value)
    # For simplicity, we'll assume all zipcodes in test are in train, or will be NaN if not.
    # Optionally, df_transformed['zipcode_ordinal_rank'] = df_transformed['zipcode_ordinal_rank'].fillna(some_default_value)
    df_transformed = df_transformed.drop(columns=['zipcode'])
    return df_transformed


def standardize_numerical_features(df, features):
  scaler = StandardScaler()
  df[features] = scaler.fit_transform(df[features])
  return df


def preprocess_features(df, ranking_map):
  result = log_right_skewed_features(df, log_n_features) # Linear Model perform better without this!
  result = fix_dtypes(result)
  result = apply_zipcode_ranking(result, ranking_map) # Has big impact on results
  result = standardize_numerical_features(result, numerical_features) # Has almost no impact on results
  result = result.drop(["date"], axis = 1)

  return result
