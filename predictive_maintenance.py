import pandas as pd
import numpy as np
import logging
import time
import joblib

from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from imblearn.combine import SMOTETomek  # Replacing SMOTE and RandomUnderSampler with SMOTETomek
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

# Create binary labels (15th percentile threshold)
threshold = np.percentile(train_data["RUL"], 15)
train_data["label"] = (train_data["RUL"] <= threshold).astype(int)

# Select only sensor & setting features
FEATURE_COLS = ["setting1", "setting2", "setting3"] + [col for col in train_data.columns if col.startswith("s")]
train_data = train_data[FEATURE_COLS + ["label"]]

# Handle missing values
imputer = SimpleImputer(strategy="mean")
X = pd.DataFrame(imputer.fit_transform(train_data.drop(columns=["label"])), columns=FEATURE_COLS)
y = train_data["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Log original class distribution
logging.info(f"Original class distribution: {Counter(y_train)}")

# Compute scale_pos_weight dynamically
scale_pos_weight = Counter(y_train)[0] / Counter(y_train)[1]
logging.info(f"Computed scale_pos_weight: {scale_pos_weight:.2f}")

# Apply SMOTETomek for resampling
over = SMOTETomek(sampling_strategy=0.8, random_state=42)
X_train_res, y_train_res = over.fit_resample(X_train, y_train)
logging.info(f"After SMOTETomek: {Counter(y_train_res)}")

# Convert to NumPy
X_train_res = X_train_res.values
y_train_res = y_train_res.values
X_test_np = X_test.values
y_test_np = y_test.values

# Define XGBoost classifier with adjusted parameters
xgb = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    tree_method="hist",
    early_stopping_rounds=25,
    scale_pos_weight=scale_pos_weight,
)

# Hyperparameter tuning using Bayesian Optimization
param_space = {
    "n_estimators": (500, 1000),
    "max_depth": (6, 20),
    "learning_rate": (0.05, 0.2),
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

# Train the best model
search.fit(X_train_res, y_train_res, eval_set=[(X_test_np, y_test_np)], verbose=False)

# Get best model
best_model = search.best_estimator_

# Evaluate model
y_pred_prob = best_model.predict_proba(X_test_np)[:, 1]  # Get the probability for class 1
threshold = 0.3  # Adjust the threshold to focus on predicting the minority class more
y_pred_adjusted = (y_pred_prob > threshold).astype(int)  # Apply the adjusted threshold

# Log accuracy and other metrics
accuracy = accuracy_score(y_test_np, y_pred_adjusted)
f1 = f1_score(y_test_np, y_pred_adjusted)
conf_matrix = confusion_matrix(y_test_np, y_pred_adjusted)

logging.info("Model Evaluation Completed.")
logging.info(f"Best Hyperparameters: {search.best_params_}")
logging.info(f"Accuracy: {accuracy:.4f}")
logging.info(f"F1 Score: {f1:.4f}")
logging.info(f"Confusion Matrix: \n{conf_matrix}")
logging.info("\n" + classification_report(y_test_np, y_pred_adjusted))

# Save model
joblib.dump(best_model, "models/predictive_maintenance_model.pkl")
joblib.dump(FEATURE_COLS, "models/feature_names.pkl")

# Log execution time
end_time = time.time()
logging.info(f"Model training and tuning completed in {end_time - start_time:.2f} seconds.")
