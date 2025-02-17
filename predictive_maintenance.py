import pandas as pd
import numpy as np
import logging
import time
import joblib

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from skopt import BayesSearchCV
from sklearn.metrics import confusion_matrix

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

# Create binary labels (10th percentile threshold)
threshold = np.percentile(train_data["RUL"], 10)
train_data["label"] = (train_data["RUL"] <= threshold).astype(int)

# Select only sensor & setting features
FEATURE_COLS = ["setting1", "setting2", "setting3"] + [col for col in train_data.columns if col.startswith("s")]
train_data = train_data[FEATURE_COLS + ["label"]]

# Handle missing values using imputation
imputer = SimpleImputer(strategy="mean")
X = pd.DataFrame(imputer.fit_transform(train_data.drop(columns=["label"])), columns=FEATURE_COLS)
y = train_data["label"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Apply SMOTE to balance classes
smote = SMOTE(sampling_strategy=0.2, k_neighbors=2, random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
X_train_smote = X_train_smote.to_numpy()
y_train_smote = y_train_smote.to_numpy()

# Define XGBoost classifier optimized for AMD GPUs
xgb = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    tree_method="hist",  # Optimized for CPU & AMD GPU
    early_stopping_rounds=10,  # Prevents overfitting
)

# Hyperparameter tuning using Bayesian Optimization
param_space = {
    "n_estimators": (200, 500),
    "max_depth": (3, 12),
    "learning_rate": (0.01, 0.2),
    "subsample": (0.6, 1.0),
    "colsample_bytree": (0.6, 1.0),
}

search = BayesSearchCV(
    xgb,
    param_space,
    n_iter=15,  # Increased iterations for better tuning
    scoring="f1",
    cv=3,
    random_state=42,
    n_jobs=-1,  # Parallel processing
)

# Train the best model
search.fit(X_train_smote, y_train_smote, eval_set=[(X_test.to_numpy(), y_test.to_numpy())], verbose=False)

# Get best model
best_model = search.best_estimator_

# Evaluate model
y_pred = best_model.predict(X_test.to_numpy())

# Get accuracy and other evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

logging.info("Model Evaluation Completed.")
logging.info(f"Best Hyperparameters: {search.best_params_}")
logging.info(f"Accuracy: {accuracy:.4f}")
logging.info(f"F1 Score: {f1:.4f}")
logging.info(f"Confusion Matrix: \n{conf_matrix}")
logging.info("\n" + classification_report(y_test, y_pred))

# Save model and feature names
joblib.dump(best_model, "models/predictive_maintenance_model.pkl")
joblib.dump(FEATURE_COLS, "models/feature_names.pkl")

# Log execution time
end_time = time.time()
logging.info(f"Model training and tuning completed in {end_time - start_time:.2f} seconds.")