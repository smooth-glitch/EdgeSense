import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load the dataset
column_names = [
    'id', 'cycle', 'setting1', 'setting2', 'setting3', 
    's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 
    's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 
    's20', 's21'
]
train_data = pd.read_csv('train_FD001.txt', sep=' ', header=None, names=column_names)
train_data.dropna(axis=1, how='all', inplace=True)  # Drop empty columns

# Calculate RUL (Remaining Useful Life)
train_data['RUL'] = train_data.groupby('id')['cycle'].transform("max") - train_data['cycle']

# Create binary labels using the 90th percentile as the threshold
threshold = train_data['RUL'].quantile(0.90)
train_data['label'] = train_data['RUL'].apply(lambda x: 1 if x <= threshold else 0)

# Feature Engineering: Add rolling averages and rate of change
train_data['s1_rolling_avg'] = train_data.groupby('id')['s1'].transform(lambda x: x.rolling(window=5).mean())
train_data['s2_rate_of_change'] = train_data.groupby('id')['s2'].diff().fillna(0)

# Drop unnecessary columns
train_data.drop(['id', 'cycle', 'RUL'], axis=1, inplace=True)

# Check for missing values
print("Missing Values in Dataset:")
print(train_data.isnull().sum())

# Split the data into features (X) and target (y)
X = train_data.drop('label', axis=1)
y = train_data['label']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with imputation, scaling, SMOTE, and XGBoost
pipeline = ImbPipeline([
    ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values
    ('scaler', StandardScaler()),  # Standardize features
    ('smote', SMOTE(random_state=42)),  # Handle imbalanced data
    ('xgb', XGBClassifier(random_state=42))
])

# Hyperparameter tuning
param_grid = {
    'xgb__n_estimators': [100, 200, 300],
    'xgb__max_depth': [3, 5, 7],
    'xgb__learning_rate': [0.01, 0.1, 0.2]
}

# Use RandomizedSearchCV for hyperparameter tuning
search = RandomizedSearchCV(pipeline, param_grid, n_iter=10, scoring='f1', cv=3, random_state=42)
search.fit(X_train, y_train)

# Best model
best_model = search.best_estimator_

# Evaluate the model
y_pred = best_model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save the model
joblib.dump(best_model, 'predictive_maintenance_model.pkl')

# Save the feature names
joblib.dump(X_train.columns.tolist(), 'feature_names.pkl')

print("Model and feature names saved successfully!")