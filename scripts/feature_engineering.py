import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import kurtosis, skew
from scipy.signal import welch

def main():
    data_path = Path(r"D:\Projects\SmartFit-SmartDiet\data\processed\pamap2_clean.csv")
    df = pd.read_csv(data_path)
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
    print("Raw data shape:", df.shape)
    
    # Constants for windowing
    WINDOW_SIZE = 5  # seconds
    SAMPLE_FREQ = 100  # Hz
    SAMPLES_PER_WINDOW = WINDOW_SIZE * SAMPLE_FREQ
    
    # Sort and reset index
    df.sort_values(by=['subject_id', 'session_type', 'datetime'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    def extract_features(window_df):
        features = {}
        # Exclude labels and identifiers
        numeric_cols = window_df.select_dtypes(include=np.number).columns.drop(['activity_id', 'subject_id'])
        
        for col in numeric_cols:
            data = window_df[col].values
            features[f'{col}_mean'] = np.mean(data)
            features[f'{col}_std'] = np.std(data)
            features[f'{col}_min'] = np.min(data)
            features[f'{col}_max'] = np.max(data)
            features[f'{col}_median'] = np.median(data)
            features[f'{col}_kurtosis'] = kurtosis(data)
            features[f'{col}_skew'] = skew(data)
            features[f'{col}_energy'] = np.sum(data**2) / len(data)
            try:
                freqs, psd = welch(data, fs=SAMPLE_FREQ)
                features[f'{col}_dom_freq'] = freqs[np.argmax(psd)]
            except Exception:
                features[f'{col}_dom_freq'] = np.nan
        
        features['activity_id'] = window_df['activity_id'].mode()[0]
        features['subject_id'] = window_df['subject_id'].iloc[0]
        features['session_type'] = window_df['session_type'].iloc[0]
        features['window_start'] = window_df['datetime'].iloc[0]
        features['window_end'] = window_df['datetime'].iloc[-1]
        return features
    
    windowed_features = []
    
    for (subj, session), group in df.groupby(['subject_id', 'session_type']):
        group = group.reset_index(drop=True)
        n_windows = len(group) // SAMPLES_PER_WINDOW
        for w in range(n_windows):
            start_idx = w * SAMPLES_PER_WINDOW
            end_idx = start_idx + SAMPLES_PER_WINDOW
            window_df = group.iloc[start_idx:end_idx]
            if len(window_df) == SAMPLES_PER_WINDOW:
                feat = extract_features(window_df)
                windowed_features.append(feat)
    
    features_df = pd.DataFrame(windowed_features)
    print("Feature matrix shape:", features_df.shape)
    
    activity_map = {
        1: "lying", 2: "sitting", 3: "standing", 4: "walking", 5: "running",
        6: "cycling", 7: "nordic_walking", 9: "watching_tv", 10: "computer_work",
        11: "car_driving", 12: "ascending_stairs", 13: "descending_stairs",
        16: "vacuum_cleaning", 17: "ironing", 18: "folding_laundry",
        19: "house_cleaning", 20: "playing_soccer", 24: "rope_jumping"
    }
    
    features_df['activity'] = features_df['activity_id'].map(activity_map)
    features_df.drop(columns=['activity_id'], inplace=True)
    
    out_path = Path(r"D:\Projects\SmartFit-SmartDiet\data\processed\pamap2_features.csv")
    features_df.to_csv(out_path, index=False)
    print("Features saved to", out_path)

if __name__ == "__main__":
    main()
