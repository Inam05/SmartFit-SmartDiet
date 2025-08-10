# scripts/train_workout_recommendation.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

warnings.filterwarnings('ignore')

# =====================
# Config
# =====================
DATA_PATH = r"D:\Projects\SmartFit-SmartDiet\data\processed\pamap2_features_clustered.csv"
MODEL_DIR = r"D:\Projects\SmartFit-SmartDiet\models"
os.makedirs(MODEL_DIR, exist_ok=True)

# =====================
# Load data
# =====================
df = pd.read_csv(DATA_PATH)
print("Data shape:", df.shape)

# =====================
# Map activities to workout intensity
# =====================
intensity_map = {
    "lying": "Light", "sitting": "Light", "standing": "Light",
    "walking": "Moderate", "cycling": "Moderate",
    "running": "Intense", "nordic_walking": "Intense", "playing_soccer": "Intense",
}

df['workout_intensity'] = df['activity'].map(intensity_map).fillna("Moderate")

# =====================
# Features / Target
# =====================
non_feature_cols = [
    'subject_id', 'session_type', 'window_start', 'window_end',
    'activity', 'kmeans_cluster', 'workout_intensity'
]
X = df.drop(columns=non_feature_cols)
y = df['workout_intensity']

print("Features shape:", X.shape)
print("Target distribution:\n", y.value_counts())

# =====================
# Train/test split
# =====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# =====================
# Scaling
# =====================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =====================
# Hyperparameter tuning for Random Forest
# =====================
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5]
}

grid_rf = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=3,
    scoring='f1_weighted',
    n_jobs=-1
)
grid_rf.fit(X_train, y_train)

print(f"Best params RF: {grid_rf.best_params_}")
best_rf = grid_rf.best_estimator_

# =====================
# Evaluation
# =====================
y_pred = best_rf.predict(X_test)
print("Tuned Random Forest Classification Report:")
print(classification_report(y_test, y_pred))

ConfusionMatrixDisplay.from_estimator(best_rf, X_test, y_test, cmap='Blues')
plt.title("Confusion Matrix - Tuned Random Forest")
plt.show()

# =====================
# Feature Importance Plot
# =====================
importances = best_rf.feature_importances_
indices = np.argsort(importances)[::-1]
top_n = 15

plt.figure(figsize=(10, 6))
plt.title("Top 15 Feature Importances (Random Forest)")
plt.bar(range(top_n), importances[indices][:top_n], align='center')
plt.xticks(range(top_n), X.columns[indices][:top_n], rotation=90)
plt.tight_layout()
plt.show()

# =====================
# Save model & scaler
# =====================
joblib.dump(best_rf, os.path.join(MODEL_DIR, 'best_rf_workout_model.joblib'))
joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler_workout.pkl'))
print("Saved tuned Random Forest model and scaler.")
