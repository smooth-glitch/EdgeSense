import pandas as pd
import numpy as np
import logging
import time
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from skopt import BayesSearchCV

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Start timing
start_time = time.time()

# Load dataset
COLUMN_NAMES = [
    "id", "cycle", "setting1", "setting2", "setting3",
    "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10",
    "s11", "s12", "s13", "s14", "s15", "s16", "s17", "s18", "s19",
    "s20", "s21",
]

train_data = pd.read_csv(
    "data/train_FD001.txt", sep=" ", header=None, names=COLUMN_NAMES, engine="python"
)
train_data.dropna(axis=1, how="all", inplace=True)  # Drop empty columns

# Compute Remaining Useful Life (RUL)
train_data["RUL"] = train_data.groupby("id")["cycle"].transform("last") - train_data["cycle"]

# Create binary labels using the 10th percentile as threshold
threshold = np.percentile(train_data["RUL"], 10)
train_data["label"] = (train_data["RUL"] <= threshold).astype(int)

# Feature Engineering
SENSOR_COLS = [col for col in train_data.columns if col.startswith("s")]

for col in SENSOR_COLS:
    train_data[f"{col}_rolling_mean"] = (
        train_data.groupby("id")[col].rolling(window=5, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    train_data[f"{col}_rolling_std"] = (
        train_data.groupby("id")[col].rolling(window=5, min_periods=1).std().reset_index(level=0, drop=True)
    )
    train_data[f"{col}_rate_of_change"] = train_data[col].diff().fillna(0)

# Drop unnecessary columns
train_data.drop(["id", "cycle", "RUL"], axis=1, inplace=True)

# Handle missing values (Zero-filling instead of backfilling)
train_data.fillna(0, inplace=True)

# Split the data into features and target
X = train_data.drop("label", axis=1)
y = train_data["label"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Apply SMOTE only on training data
smote = SMOTE(sampling_strategy=0.6, k_neighbors=3, random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)


xgb_params = {"tree_method": "hist"}  # Use CPU fallback
logging.info("Falling back to CPU.")

# Define XGBoost classifier
xgb = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
    **xgb_params
)

# Hyperparameter tuning using Bayesian Optimization
param_space = {
    "n_estimators": (200, 600),  # Increased range for better tuning
    "max_depth": (3, 12),
    "learning_rate": (0.01, 0.3),
    "subsample": (0.5, 1.0),
    "colsample_bytree": (0.5, 1.0),
}

search = BayesSearchCV(
    xgb,
    param_space,
    n_iter=30,  # Increased iterations for better tuning
    scoring="f1",
    cv=3,
    random_state=42,
    n_jobs=4,
)

# Train the best model with early stopping
search.fit(X_train_smote, y_train_smote)

# Get best model
best_model = search.best_estimator_

# Evaluate model
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

logging.info("Model Evaluation Completed.")
logging.info(f"Best Hyperparameters: {search.best_params_}")
logging.info(f"Accuracy: {accuracy:.4f}")
logging.info("\n" + classification_report(y_test, y_pred))

# Save the trained model and feature names
joblib.dump(best_model, "models/predictive_maintenance_model.pkl")
joblib.dump(X_train.columns.tolist(), "models/feature_names.pkl")

# Log execution time
end_time = time.time()
execution_time = end_time - start_time
logging.info(f"Model training and tuning completed in {execution_time:.2f} seconds.")
