import pandas as pd
import numpy as np
import logging
import time
import joblib
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from imblearn.combine import SMOTETomek
from xgboost import XGBClassifier
from skopt import BayesSearchCV

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

start_time = time.time()

# Load dataset
COLUMN_NAMES = [
    "id", "cycle", "setting1", "setting2", "setting3"
] + [f"s{i}" for i in range(1, 22)]

train_data = pd.read_csv("data/train_FD001.txt", sep=" ", header=None, names=COLUMN_NAMES, engine="python")
train_data.dropna(axis=1, how="all", inplace=True)

# Compute RUL
train_data["RUL"] = train_data.groupby("id")["cycle"].transform("last") - train_data["cycle"]
threshold = np.percentile(train_data["RUL"], 15)
train_data["label"] = (train_data["RUL"] <= threshold).astype(int)

# Allow user to disable specific sensors
DISABLED_SENSORS = []  # Users can update this list with sensors to disable

# Extract only sensor readings and operational settings
sensor_cols = [col for col in train_data.columns if col.startswith("s") and col not in DISABLED_SENSORS]
feature_cols = ["setting1", "setting2", "setting3"] + sensor_cols

# Handle missing values
imputer = SimpleImputer(strategy="mean")
X = pd.DataFrame(imputer.fit_transform(train_data[feature_cols]), columns=feature_cols)
y = train_data["label"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
logging.info(f"Original class distribution: {Counter(y_train)}")

# Compute scale_pos_weight dynamically
scale_pos_weight = 2 * Counter(y_train)[0] / Counter(y_train)[1]
logging.info(f"Computed scale_pos_weight: {scale_pos_weight:.2f}")

# Apply SMOTETomek for resampling
over = SMOTETomek(sampling_strategy=1, random_state=42)
X_train_res, y_train_res = over.fit_resample(X_train, y_train)
logging.info(f"After SMOTETomek: {Counter(y_train_res)}")

# Convert to NumPy
X_train_res = X_train_res.values
y_train_res = y_train_res.values
X_test_np = X_test.values
y_test_np = y_test.values

# Define XGBoost classifier
xgb = XGBClassifier(
    objective="binary:logistic",
    eval_metric="auc",
    random_state=42,
    tree_method="hist",
    early_stopping_rounds=25,
    scale_pos_weight=scale_pos_weight,
)

# Hyperparameter tuning
param_space = {
    "n_estimators": (500, 1000),
    "max_depth": (10, 15),
    "learning_rate": (0.01, 0.2),
    "subsample": (0.5, 1.0),
    "colsample_bytree": (0.6, 1.0),
    "gamma": (0.1, 5),
    "min_child_weight": (1, 10),
}

search = BayesSearchCV(
    xgb,
    param_space,
    n_iter=50,
    scoring="f1",
    cv=5,
    random_state=42,
    n_jobs=-1,
)

search.fit(X_train_res, y_train_res, eval_set=[(X_test_np, y_test_np)], verbose=False)

# Get best model
best_model = search.best_estimator_

# Evaluate model
y_pred_prob = best_model.predict_proba(X_test_np)[:, 1]
y_pred_adjusted = (y_pred_prob > 0.25).astype(int)

accuracy = accuracy_score(y_test_np, y_pred_adjusted)
f1 = f1_score(y_test_np, y_pred_adjusted)
conf_matrix = confusion_matrix(y_test_np, y_pred_adjusted)

logging.info("Model Evaluation Completed.")
logging.info(f"Best Hyperparameters: {search.best_params_}")
logging.info(f"Accuracy: {accuracy:.4f}")
logging.info(f"F1 Score: {f1:.4f}")
logging.info(f"Confusion Matrix: \n{conf_matrix}")
logging.info("\n" + classification_report(y_test_np, y_pred_adjusted))

# Save model and features
joblib.dump(best_model, "models/predictive_maintenance_model.pkl")
joblib.dump(feature_cols, "models/feature_names.pkl")

end_time = time.time()
logging.info(f"Model training and tuning completed in {end_time - start_time:.2f} seconds.")