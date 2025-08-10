import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_diet_data():
    # Load dataset
    data_path = r"D:\Projects\SmartFit-SmartDiet\data\raw\daily_food_nutrition_dataset.csv"
    df = pd.read_csv(data_path)

    print(f"Dataset shape: {df.shape}")
    print(df.head())

    # Step 1: Check and handle missing values
    print("Missing values per column:")
    print(df.isnull().sum())

    df.dropna(subset=['Calories (kcal)', 'Protein (g)', 'Carbohydrates (g)', 'Fat (g)', 'Date', 'User_ID'], inplace=True)

    # Step 2: Convert 'Date' to datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Step 3: Aggregate nutrient intake per user per day
    daily_totals = df.groupby(['User_ID', 'Date']).agg({
        'Calories (kcal)': 'sum',
        'Protein (g)': 'sum',
        'Carbohydrates (g)': 'sum',
        'Fat (g)': 'sum',
        'Fiber (g)': 'sum',
        'Sugars (g)': 'sum',
        'Sodium (mg)': 'sum',
        'Cholesterol (mg)': 'sum',
        'Water_Intake (ml)': 'sum'
    }).reset_index()

    print("Daily totals head:")
    print(daily_totals.head())

    # Step 4: Create synthetic user profiles if none available
    user_profiles = pd.DataFrame({
        'User_ID': daily_totals['User_ID'].unique(),
        'Age': np.random.randint(18, 60, size=daily_totals['User_ID'].nunique()),
        'Gender': np.random.choice(['Male', 'Female'], size=daily_totals['User_ID'].nunique()),
        'Weight_kg': np.random.uniform(50, 90, size=daily_totals['User_ID'].nunique()),
        'Height_cm': np.random.uniform(150, 190, size=daily_totals['User_ID'].nunique())
    })

    print("User profiles head:")
    print(user_profiles.head())

    # Step 5: Merge daily totals with user profiles
    data = daily_totals.merge(user_profiles, on='User_ID', how='left')

    # Step 6: Calculate BMI
    data['BMI'] = data['Weight_kg'] / (data['Height_cm'] / 100) ** 2

    # Step 7: Calculate BMR (Mifflin-St Jeor)
    def calc_bmr(row):
        if row['Gender'] == 'Male':
            return 10 * row['Weight_kg'] + 6.25 * row['Height_cm'] - 5 * row['Age'] + 5
        else:
            return 10 * row['Weight_kg'] + 6.25 * row['Height_cm'] - 5 * row['Age'] - 161

    data['BMR'] = data.apply(calc_bmr, axis=1)

    # Step 8: Estimate TDEE assuming sedentary activity (x1.2)
    data['TDEE'] = data['BMR'] * 1.2

    # Step 9: Calculate nutrient ratios (protein, fat, carbs as % calories)
    data['protein_ratio'] = data['Protein (g)'] * 4 / data['Calories (kcal)']
    data['fat_ratio'] = data['Fat (g)'] * 9 / data['Calories (kcal)']
    data['carbs_ratio'] = data['Carbohydrates (g)'] * 4 / data['Calories (kcal)']

    # Step 10: Define calorie adjustment target
    data['calorie_adjustment'] = data['TDEE'] - data['Calories (kcal)']

    # Step 11: Encode gender (drop first for binary encoding)
    data = pd.get_dummies(data, columns=['Gender'], drop_first=True)

    # Step 12: Handle infinite or NaN values
    data.replace([np.inf, -np.inf], 0, inplace=True)
    data.fillna(0, inplace=True)

    # Step 13: Scale numeric features
    features_to_scale = ['Age', 'Weight_kg', 'Height_cm', 'BMI', 'BMR', 'TDEE',
                         'Calories (kcal)', 'Protein (g)', 'Fat (g)', 'Carbohydrates (g)',
                         'protein_ratio', 'fat_ratio', 'carbs_ratio', 'calorie_adjustment']

    scaler = StandardScaler()
    data[features_to_scale] = scaler.fit_transform(data[features_to_scale])

    # Step 14: Save preprocessed data
    processed_dir = r"D:\Projects\SmartFit-SmartDiet\data\processed"
    os.makedirs(processed_dir, exist_ok=True)

    save_path = os.path.join(processed_dir, 'preprocessed_diet_data.csv')
    data.to_csv(save_path, index=False)
    print(f"Preprocessed diet data saved at {save_path}")

if __name__ == "__main__":
    preprocess_diet_data()
