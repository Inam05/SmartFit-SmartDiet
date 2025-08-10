import os
import glob
import re
import pandas as pd

# ===== CONFIG =====
BASE_PATH = r"D:\Projects\SmartFit-SmartDiet\data\raw"
PROTOCOL_PATH = os.path.join(BASE_PATH, "Protocol")
OPTIONAL_PATH = os.path.join(BASE_PATH, "Optional")
OUTPUT_PATH = r"D:\Projects\SmartFit-SmartDiet\data\processed\pamap2_clean.csv"

# ===== COLUMN NAMES =====
COLUMNS = [
    "timestamp", "activity_id", "heart_rate",
    "hand_acc_16g_x", "hand_acc_16g_y", "hand_acc_16g_z",
    "hand_acc_6g_x", "hand_acc_6g_y", "hand_acc_6g_z",
    "hand_gyro_x", "hand_gyro_y", "hand_gyro_z",
    "hand_mag_x", "hand_mag_y", "hand_mag_z",
    "hand_temp",
    "chest_acc_16g_x", "chest_acc_16g_y", "chest_acc_16g_z",
    "chest_acc_6g_x", "chest_acc_6g_y", "chest_acc_6g_z",
    "chest_gyro_x", "chest_gyro_y", "chest_gyro_z",
    "chest_mag_x", "chest_mag_y", "chest_mag_z",
    "chest_temp",
    "ankle_acc_16g_x", "ankle_acc_16g_y", "ankle_acc_16g_z",
    "ankle_acc_6g_x", "ankle_acc_6g_y", "ankle_acc_6g_z",
    "ankle_gyro_x", "ankle_gyro_y", "ankle_gyro_z",
    "ankle_mag_x", "ankle_mag_y", "ankle_mag_z",
    "ankle_temp"
]
while len(COLUMNS) < 54:
    COLUMNS.append(f"extra_col_{len(COLUMNS) + 1}")

# ===== FUNCTIONS =====
def extract_subject_id(filepath):
    match = re.search(r"subject(\d+)", filepath)
    if match:
        return int(match.group(1))
    else:
        return None

def extract_session_type(filepath):
    # Simple heuristic: folder name before filename is session type
    if "Protocol" in filepath:
        return "protocol"
    elif "Optional" in filepath:
        return "optional"
    else:
        return "unknown"

def read_pamap_file(filepath, columns):
    df = pd.read_csv(filepath, sep=" ", header=None, names=columns)
    df = df.dropna(axis=1, how="all")  # Remove completely empty columns
    df['subject_id'] = extract_subject_id(filepath)
    df['session_type'] = extract_session_type(filepath)
    return df

def load_and_combine(protocol_path, optional_path, columns):
    protocol_files = sorted(glob.glob(os.path.join(protocol_path, "**", "*.dat"), recursive=True))
    optional_files = sorted(glob.glob(os.path.join(optional_path, "**", "*.dat"), recursive=True))

    print(f"Found {len(protocol_files)} protocol files and {len(optional_files)} optional files.")

    protocol_dfs = [read_pamap_file(f, columns) for f in protocol_files]
    optional_dfs = [read_pamap_file(f, columns) for f in optional_files]

    df_protocol = pd.concat(protocol_dfs, ignore_index=True) if protocol_dfs else pd.DataFrame()
    df_optional = pd.concat(optional_dfs, ignore_index=True) if optional_dfs else pd.DataFrame()

    df_combined = pd.concat([df_protocol, df_optional], ignore_index=True)
    print(f"Combined shape before cleaning: {df_combined.shape}")
    return df_combined

def clean_and_save(df, output_path):
    df = df.dropna(subset=["activity_id"])
    df = df.dropna().reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Cleaned dataset saved to {output_path}")
    print(f"Final shape: {df.shape}")

# ===== MAIN =====
if __name__ == "__main__":
    df_combined = load_and_combine(PROTOCOL_PATH, OPTIONAL_PATH, COLUMNS)
    clean_and_save(df_combined, OUTPUT_PATH)
