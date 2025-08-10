import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

def main():
    # Load data
    data_path = r"D:\Projects\SmartFit-SmartDiet\data\processed\preprocessed_diet_data.csv"
    df = pd.read_csv(data_path)
    print(f"Loaded data shape: {df.shape}")

    # Features and target
    feature_cols = ['Age', 'Weight_kg', 'Height_cm', 'BMI', 'BMR', 'TDEE', 'Gender_Male',
                    'Calories (kcal)', 'Protein (g)', 'Carbohydrates (g)', 'Fat (g)', 'Fiber (g)',
                    'Sugars (g)', 'Sodium (mg)', 'Cholesterol (mg)', 'Water_Intake (ml)',
                    'protein_ratio', 'fat_ratio', 'carbs_ratio']
    X = df[feature_cols]
    y = df['calorie_adjustment']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Best hyperparameters after tuning (fixed for speed)
    best_params = {
        'n_estimators': 200,
        'max_depth': 20,
        'min_samples_split': 2,
        'random_state': 42,
        'n_jobs': -1
    }

    # Train final Random Forest model
    rf = RandomForestRegressor(**best_params)
    rf.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = rf.predict(X_test)
    print(f"RF MAE: {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"RF RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
    print(f"RF R2: {r2_score(y_test, y_pred):.3f}")

    # Save model
    model_dir = r"D:\Projects\SmartFit-SmartDiet\models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'best_rf_diet_adjustment_model.joblib')
    joblib.dump(rf, model_path)
    print(f"Saved Random Forest diet adjustment model to {model_path}")

if __name__ == "__main__":
    main()
