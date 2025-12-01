#!/usr/bin/env python3
"""
Train XGBoost model WITHOUT data leakage.
This script retrains the model using only clean features.
"""

import os
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from datetime import datetime
import json

print("="*60)
print("TRAINING CLEAN XGBOOST MODEL (NO DATA LEAKAGE)")
print("="*60)

# Configuration
class Config:
    TRAIN_PATH = '../data/train.parquet'
    TEST_PATH = '../data/test.parquet'
    LISTINGS_PATH = '../data/listings.parquet'
    MODEL_PATH = '../data/xgboost_model_no_leakage'
    RANDOM_STATE = 42

config = Config()

# Initialize Spark
print("\n1. Initializing Spark...")
spark = SparkSession.builder \
    .appName("AirbnbXGBoost_NoLeakage") \
    .master("local[*]") \
    .config("spark.driver.memory", "4g") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")
print(f"   Spark version: {spark.version}")

# Load data
print("\n2. Loading data...")
train_spark = spark.read.parquet(config.TRAIN_PATH)
test_spark = spark.read.parquet(config.TEST_PATH)
listings_spark = spark.read.parquet(config.LISTINGS_PATH)

train_spark.cache()
test_spark.cache()
listings_spark.cache()

print(f"   Train: {train_spark.count():,} rows")
print(f"   Test: {test_spark.count():,} rows")
print(f"   Listings: {listings_spark.count():,} rows")

# Feature Engineering - NO DATA LEAKAGE
print("\n3. Creating CLEAN features (NO DATA LEAKAGE)...")

# CLEAN User features - counts only (NO rating-based statistics)
print("   - Computing user review counts...")
user_stats = train_spark.groupBy("user_id").agg(
    count("item_id").alias("user_review_count")
    # REMOVED: avg("rating"), stddev("rating"), min("rating"), max("rating")
).withColumnRenamed("user_id", "user_id_stats")

# CLEAN Item features - counts only (NO rating-based statistics)
print("   - Computing item review counts...")
item_stats = train_spark.groupBy("item_id").agg(
    count("user_id").alias("item_review_count")
    # REMOVED: avg("rating"), stddev("rating"), min("rating"), max("rating")
).withColumnRenamed("item_id", "item_id_stats")

# Join features to train data
print("   - Joining features to train data...")
train_with_features = train_spark \
    .join(user_stats, train_spark.user_id == user_stats.user_id_stats, "left") \
    .join(item_stats, train_spark.item_id == item_stats.item_id_stats, "left") \
    .drop("user_id_stats", "item_id_stats") \
    .join(listings_spark, train_spark.listing_id == listings_spark.listing_id, "left")

# Apply same to test data
print("   - Joining features to test data...")
test_with_features = test_spark \
    .join(user_stats, test_spark.user_id == user_stats.user_id_stats, "left") \
    .join(item_stats, test_spark.item_id == item_stats.item_id_stats, "left") \
    .drop("user_id_stats", "item_id_stats") \
    .join(listings_spark, test_spark.listing_id == listings_spark.listing_id, "left")

print("   ✓ Clean feature engineering complete")

# Convert to Pandas
print("\n4. Converting to Pandas...")
train_df = train_with_features.toPandas()
test_df = test_with_features.toPandas()
print(f"   Train shape: {train_df.shape}")
print(f"   Test shape: {test_df.shape}")

# Define CLEAN feature columns - NO target-derived statistics
feature_cols = [
    'user_id', 'item_id',
    'user_review_count',  # Count only (not rating-based)
    'item_review_count',  # Count only (not rating-based)
    'price', 'accommodates', 'bedrooms', 'beds',
    'minimum_nights', 'number_of_reviews',
    'review_scores_rating', 'review_scores_location', 'review_scores_value',
    'latitude', 'longitude'
]

# Handle categorical columns
print("\n5. Encoding categorical features...")
categorical_cols = ['property_type', 'room_type', 'neighbourhood_cleansed']
label_encoders = {}

for col_name in categorical_cols:
    if col_name in train_df.columns:
        le = LabelEncoder()
        combined = pd.concat([train_df[col_name].fillna('unknown'), 
                              test_df[col_name].fillna('unknown')])
        le.fit(combined)
        train_df[f'{col_name}_encoded'] = le.transform(train_df[col_name].fillna('unknown'))
        test_df[f'{col_name}_encoded'] = le.transform(test_df[col_name].fillna('unknown'))
        label_encoders[col_name] = le
        feature_cols.append(f'{col_name}_encoded')
        print(f"   - Encoded {col_name}")

# Handle boolean columns
bool_cols = ['host_is_superhost', 'instant_bookable']
for col_name in bool_cols:
    if col_name in train_df.columns:
        train_df[col_name] = train_df[col_name].astype(float).fillna(0)
        test_df[col_name] = test_df[col_name].astype(float).fillna(0)
        feature_cols.append(col_name)

# Create derived features (CLEAN - no rating-based)
print("\n6. Creating derived features...")

# Price per person
train_df['price_per_person'] = train_df['price'] / (train_df['accommodates'] + 1e-6)
test_df['price_per_person'] = test_df['price'] / (test_df['accommodates'] + 1e-6)
feature_cols.append('price_per_person')

# Bedroom ratio
train_df['bedroom_ratio'] = train_df['bedrooms'] / (train_df['accommodates'] + 1e-6)
test_df['bedroom_ratio'] = test_df['bedrooms'] / (test_df['accommodates'] + 1e-6)
feature_cols.append('bedroom_ratio')

# Bed ratio
train_df['bed_ratio'] = train_df['beds'] / (train_df['accommodates'] + 1e-6)
test_df['bed_ratio'] = test_df['beds'] / (test_df['accommodates'] + 1e-6)
feature_cols.append('bed_ratio')

# Review score composite (from AIRBNB scores - not our constructed ratings!)
train_df['review_score_composite'] = (
    train_df['review_scores_rating'].fillna(0) * 0.5 +
    train_df['review_scores_location'].fillna(0) * 0.3 +
    train_df['review_scores_value'].fillna(0) * 0.2
)
test_df['review_score_composite'] = (
    test_df['review_scores_rating'].fillna(0) * 0.5 +
    test_df['review_scores_location'].fillna(0) * 0.3 +
    test_df['review_scores_value'].fillna(0) * 0.2
)
feature_cols.append('review_score_composite')

# Select only available features
available_features = [f for f in feature_cols if f in train_df.columns]
print(f"   Using {len(available_features)} CLEAN features")

# Handle missing values with median imputation
print("\n7. Handling missing values...")
numeric_features = [f for f in available_features if f not in ['user_id', 'item_id']]
for col in numeric_features:
    if col in train_df.columns:
        median_val = train_df[col].median()
        if pd.notna(median_val):
            train_df[col] = train_df[col].fillna(median_val)
            test_df[col] = test_df[col].fillna(median_val)
        else:
            train_df[col] = train_df[col].fillna(0)
            test_df[col] = test_df[col].fillna(0)

# Prepare X and y
X_train = train_df[available_features]
y_train = train_df['rating'].values
X_test = test_df[available_features]
y_test = test_df['rating'].values

print(f"   X_train shape: {X_train.shape}")
print(f"   X_test shape: {X_test.shape}")

# Hyperparameter tuning with cross-validation
print("\n8. Hyperparameter tuning with cross-validation...")
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [4, 5, 6],
    'learning_rate': [0.03, 0.05, 0.1],
    'min_child_weight': [1, 3],
    'subsample': [0.8, 0.9],
    'colsample_bytree': [0.8, 0.9],
}

base_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    random_state=config.RANDOM_STATE,
    n_jobs=-1
)

start_time = datetime.now()
print(f"   Starting at {start_time.strftime('%H:%M:%S')}...")

search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_grid,
    n_iter=20,
    scoring='neg_root_mean_squared_error',
    cv=5,
    verbose=1,
    random_state=config.RANDOM_STATE,
    n_jobs=-1
)

search.fit(X_train, y_train)

end_time = datetime.now()
duration = (end_time - start_time).total_seconds()
print(f"   Completed in {duration:.1f} seconds")

best_cv_rmse = -search.best_score_
print(f"\n   Best CV RMSE: {best_cv_rmse:.4f}")
print(f"   Best parameters: {search.best_params_}")

# Train final model
print("\n9. Training final model with optimal parameters...")
model = xgb.XGBRegressor(
    **search.best_params_,
    objective='reg:squarederror',
    random_state=config.RANDOM_STATE,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Generate predictions
y_pred = model.predict(X_test)
y_pred = np.clip(y_pred, 1.0, 5.0)

# Calculate metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = np.mean(np.abs(y_test - y_pred))

print("\n" + "="*60)
print("FINAL RESULTS (NO DATA LEAKAGE)")
print("="*60)
print(f"Test RMSE: {rmse:.4f}")
print(f"Test MAE:  {mae:.4f}")
print(f"CV RMSE:   {best_cv_rmse:.4f}")

# Feature importance
print("\nTop 15 Feature Importance:")
feature_importance = pd.DataFrame({
    'feature': available_features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

for idx, row in feature_importance.head(15).iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

# Save model and artifacts
print("\n10. Saving model and artifacts...")
os.makedirs(config.MODEL_PATH, exist_ok=True)

# Save model
model.save_model(f"{config.MODEL_PATH}/xgboost_model.json")

# Save feature importance
feature_importance.to_csv(f"{config.MODEL_PATH}/feature_importance.csv", index=False)

# Also save to the location the dashboard expects
feature_importance.to_csv('../data/feature_importance_no_leakage.csv', index=False)

# Save model info
model_info = {
    'feature_names': available_features,
    'categorical_columns': list(label_encoders.keys()),
    'test_rmse': float(rmse),
    'test_mae': float(mae),
    'cv_rmse': float(best_cv_rmse),
    'n_features': len(available_features),
    'optimal_hyperparameters': search.best_params_,
    'training_timestamp': datetime.now().isoformat(),
    'data_leakage': False,
    'note': 'This model uses only clean features without any target-derived statistics'
}

with open(f"{config.MODEL_PATH}/model_info.json", 'w') as f:
    json.dump(model_info, f, indent=2)

# Update the optimized model path too
optimized_path = '../data/xgboost_model_optimized'
os.makedirs(optimized_path, exist_ok=True)
model.save_model(f"{optimized_path}/xgboost_model.json")
feature_importance.to_csv(f"{optimized_path}/feature_importance.csv", index=False)
with open(f"{optimized_path}/model_info.json", 'w') as f:
    json.dump(model_info, f, indent=2)

print(f"\n✓ Model saved to {config.MODEL_PATH}/")
print(f"✓ Model also saved to {optimized_path}/")

# Stop Spark
spark.stop()

print("\n" + "="*60)
print("TRAINING COMPLETE - NO DATA LEAKAGE")
print("="*60)
print(f"\nVerification: Top feature should NOT be 'user_avg_rating'")
print(f"Top feature: {feature_importance.iloc[0]['feature']} ({feature_importance.iloc[0]['importance']:.4f})")

if 'user_avg_rating' in available_features or 'item_avg_rating' in available_features:
    print("\n⚠️ WARNING: Leaky features still present!")
else:
    print("\n✓ SUCCESS: No leaky features in the model!")

