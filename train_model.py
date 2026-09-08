# ============================================================
# CROP YIELD PREDICTION - MODEL TRAINING
# ============================================================

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ------------------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------------------

data = pd.read_csv("crop_yield.csv")

print("Dataset shape:", data.shape)

print("\nFirst 5 rows:")
print(data.head())


# ------------------------------------------------------------
# 2. CLEAN COLUMN NAMES
# ------------------------------------------------------------

data.columns = data.columns.str.strip()


# ------------------------------------------------------------
# 3. DEFINE TARGET
# ------------------------------------------------------------

target = "Yield"

if target not in data.columns:
    raise ValueError(
        f"Target column '{target}' not found.\n"
        f"Available columns: {list(data.columns)}"
    )


# Remove rows where Yield is missing
data = data.dropna(subset=[target])


# ------------------------------------------------------------
# 4. SEPARATE FEATURES AND TARGET
# ------------------------------------------------------------

X = data.drop(columns=[target])
y = data[target]


# ------------------------------------------------------------
# 5. IDENTIFY FEATURE TYPES
# ------------------------------------------------------------

categorical_features = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

numerical_features = X.select_dtypes(
    include=["int64", "float64", "int32", "float32"]
).columns.tolist()


print("\nCategorical features:")
print(categorical_features)

print("\nNumerical features:")
print(numerical_features)


# ------------------------------------------------------------
# 6. PREPROCESSING
# ------------------------------------------------------------

numerical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)


categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numerical_transformer,
            numerical_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ],
    remainder="drop"
)


# ------------------------------------------------------------
# 7. RANDOM FOREST MODEL
# ------------------------------------------------------------

rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)


# ------------------------------------------------------------
# 8. COMPLETE ML PIPELINE
# ------------------------------------------------------------

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("random_forest", rf_model)
    ]
)


# ------------------------------------------------------------
# 9. TRAIN-TEST SPLIT
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))


# ------------------------------------------------------------
# 10. TRAIN MODEL
# ------------------------------------------------------------

print("\nTraining Random Forest model...")

model.fit(X_train, y_train)

print("Training completed.")


# ------------------------------------------------------------
# 11. PREDICTIONS
# ------------------------------------------------------------

y_pred = model.predict(X_test)


# ------------------------------------------------------------
# 12. MODEL EVALUATION
# ------------------------------------------------------------

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

r2 = r2_score(y_test, y_pred)


print("\n================ MODEL PERFORMANCE ================")

print(f"Mean Absolute Error (MAE) : {mae:.4f}")
print(f"Root Mean Squared Error   : {rmse:.4f}")
print(f"R² Score                  : {r2:.4f}")

print("====================================================")


# ------------------------------------------------------------
# 13. SAVE MODEL
# ------------------------------------------------------------

joblib.dump(model, "crop_yield_model.pkl")

print("\nModel saved successfully!")
print("File: crop_yield_model.pkl")