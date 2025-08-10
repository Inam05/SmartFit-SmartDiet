import os
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

def diet_recommendation_engine():
    # Paths
    base_dir = os.path.join("D:", os.sep, "Projects", "SmartFit-SmartDiet", "data")
    data_path = os.path.join(base_dir, "processed", "preprocessed_diet_data.csv")
    model_path = os.path.join("D:", os.sep, "Projects", "SmartFit-SmartDiet", "models", "best_rf_diet_adjustment_model.joblib")

    # Load data and model
    df = pd.read_csv(data_path)
    diet_model = joblib.load(model_path)
    print(f"Loaded dataset shape: {df.shape}")

    # Features for prediction
    feature_cols = ['Age', 'Weight_kg', 'Height_cm', 'BMI', 'BMR', 'TDEE', 'Gender_Male',
                    'Calories (kcal)', 'Protein (g)', 'Carbohydrates (g)', 'Fat (g)', 'Fiber (g)',
                    'Sugars (g)', 'Sodium (mg)', 'Cholesterol (mg)', 'Water_Intake (ml)',
                    'protein_ratio', 'fat_ratio', 'carbs_ratio']

    # Predict calorie adjustment
    X = df[feature_cols]
    df['calorie_adjustment_pred'] = diet_model.predict(X)

    # Rule-based diet recommendation function
    def diet_recommendation(row):
        adj = row['calorie_adjustment_pred']
        protein_ratio = row['protein_ratio']
        fat_ratio = row['fat_ratio']
        carbs_ratio = row['carbs_ratio']

        if adj > 200:
            diet_plan = "Increase calories: Focus on protein and healthy fats"
        elif adj < -200:
            diet_plan = "Decrease calories: Reduce carbs and fats"
        else:
            diet_plan = "Maintain calories: Balanced macros"

        if protein_ratio < 0.15:
            diet_plan += "; Increase protein intake"
        if fat_ratio > 0.35:
            diet_plan += "; Monitor fat intake"
        if carbs_ratio > 0.55:
            diet_plan += "; Consider reducing carbs slightly"

        return diet_plan

    df['diet_recommendation'] = df.apply(diet_recommendation, axis=1)

    # Prepare output directory
    output_dir = os.path.join(base_dir, "processed", "recommendations")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "diet_recommendations.csv")

    # Save recommendations
    df.to_csv(output_file, index=False)
    print(f"Diet recommendations saved to: {output_file}")

if __name__ == "__main__":
    diet_recommendation_engine()
