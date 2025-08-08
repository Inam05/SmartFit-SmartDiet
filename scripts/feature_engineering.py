import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import kurtosis, skew
from scipy.signal import welch

def extract_features(window_df, sample_freq):
    features = {}
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
            freqs, psd = welch(data, fs=sample_freq)
            features[f'{col}_dom_freq'] = freqs[np.argmax(psd)]
        except Exception:
            features[f'{col}_dom_freq'] = np.nan
    features['activity_id'] = window_df['activity_id'].mode()[0]
    features['subject_id'] = window_df['subject_id'].iloc[0]
    features['session_type'] = window_df['session_type'].iloc[0]
    features['window_start'] = window_df['datetime'].iloc[0]
    features['window_end'] = window_df['datetime'].iloc[-1]
    return features

def main(input_path, output_path, window_size=5, sample_freq=100):
    df = pd.read_csv(input_path)
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
    
    samples_per_window = window_size * sample_freq
    
    df.sort_values(by=['subject_id', 'session_type', 'datetime'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    windowed_features = []
    for (subj, session), group in df.groupby(['subject_id', 'session_type']):
        group = group.reset_index(drop=True)
        n_windows = len(group) // samples_per_window
        for w in range(n_windows):
            start_idx = w * samples_per_window
            end_idx = start_idx + samples_per_window
            window_df = group.iloc[start_idx:end_idx]
            if len(window_df) == samples_per_window:
                feat = extract_features(window_df, sample_freq)
                windowed_features.append(feat)
    
    features_df = pd.DataFrame(windowed_features)
    
    activity_map = {
        1: "lying", 2: "sitting", 3: "standing", 4: "walking", 5: "running",
        6: "cycling", 7: "nordic_walking", 9: "watching_tv", 10: "computer_work",
        11: "car_driving", 12: "ascending_stairs", 13: "descending_stairs",
        16: "vacuum_cleaning", 17: "ironing", 18: "folding_laundry",
        19: "house_cleaning", 20: "playing_soccer", 24: "rope_jumping"
    }
    
    features_df['activity'] = features_df['activity_id'].map(activity_map)
    features_df.drop(columns=['activity_id'], inplace=True)
    
    features_df.to_csv(output_path, index=False)
    print(f"Features saved to {output_path}")

if __name__ == "__main__":
    import sys
    import os
    import argparse

    parser = argparse.ArgumentParser(description="PAMAP2 Feature Engineering")
    parser.add_argument("--input_path", type=str, required=True, help="Path to cleaned pamap2 csv")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save features csv")
    parser.add_argument("--window_size", type=int, default=5, help="Window size in seconds")
    parser.add_argument("--sample_freq", type=int, default=100, help="Sampling frequency Hz")
    args = parser.parse_args()

    main(args.input_path, args.output_path, args.window_size, args.sample_freq)
