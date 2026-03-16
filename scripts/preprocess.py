import os
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold


RAW_PATH = "../data/raw/CICIDS2017"
PROCESSED_PATH = "../data/processed/"

os.makedirs(PROCESSED_PATH, exist_ok=True)


def load_and_merge_data(raw_path):
    """
    Loads all CSV files from the given directory and merges them into a single DataFrame.
    """
    all_files = [
        os.path.join(raw_path, f)
        for f in os.listdir(raw_path)
        if f.endswith(".csv")
    ]
    df_list = []
    for file in all_files:
        print(f"Loading: {file}")
        temp_df = pd.read_csv(file, low_memory=False)
        df_list.append(temp_df)

    merged = pd.concat(df_list, ignore_index=True)
    print(f"Total records after merge: {len(merged)}")
    return merged


def fix_column_names(df):
    """
    Strips leading/trailing whitespace from column names and removes duplicate columns.
    """
    df.columns = df.columns.str.strip()
    df = df.loc[:, ~df.columns.duplicated()]
    return df


def map_labels(df):
    """
    Maps original CICIDS2017 labels into 6 consolidated attack classes.
    """
    df["Label"] = df["Label"].str.strip()

    label_map = {
        "BENIGN": "Normal",

        "DoS Hulk": "DoS",
        "DoS GoldenEye": "DoS",
        "DoS slowloris": "DoS",
        "DoS Slowhttptest": "DoS",
        "DDoS": "DoS",

        "FTP-Patator": "BruteForce",
        "SSH-Patator": "BruteForce",
        "Web Attack - Brute Force": "BruteForce",

        "PortScan": "PortScan",

        "Web Attack - XSS": "WebAttack",
        "Web Attack - Sql Injection": "WebAttack",

        "Bot": "Bot",

        "Infiltration": None,
        "Heartbleed": None,
    }

    df["AttackClass"] = df["Label"].map(label_map)

    unmapped = df[df["AttackClass"].isna()]["Label"].unique()
    if len(unmapped) > 0:
        print(f"[WARNING] Unmapped labels (will be removed): {unmapped}")

    df = df[df["AttackClass"].notna()].copy()

    print("\nClass distribution after mapping:")
    print(df["AttackClass"].value_counts())
    return df


def clean_features(df, feature_cols):
    """
    Replaces infinite values with NaN and removes rows with missing feature values.
    """
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)

    before = len(df)
    df = df.dropna(subset=feature_cols)
    after = len(df)
    print(f"\nRemoved {before - after} rows containing NaN or inf values.")
    print(f"Records after cleaning: {after}")
    return df


def remove_zero_variance(X, feature_cols):
    """
    Removes features with zero variance using sklearn's VarianceThreshold.
    """
    selector = VarianceThreshold(threshold=0.0)
    X_filtered = selector.fit_transform(X)
    kept_cols = [col for col, keep in zip(feature_cols, selector.get_support()) if keep]
    removed = [col for col, keep in zip(feature_cols, selector.get_support()) if not keep]

    print(f"\nRemoved zero-variance features ({len(removed)}): {removed}")
    print(f"Remaining features: {len(kept_cols)}")

    joblib.dump(selector, PROCESSED_PATH + "variance_selector.save")
    return pd.DataFrame(X_filtered, columns=kept_cols), kept_cols


def remove_high_correlation(X_df, threshold=0.95):
    """
    Removes features that are highly correlated with each other.
    """
    corr_matrix = X_df.corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]

    print(f"\nRemoved highly correlated features (>{threshold}) ({len(to_drop)}): {to_drop}")
    X_df = X_df.drop(columns=to_drop)
    print(f"Remaining features after correlation filter: {X_df.shape[1]}")

    return X_df


def scale_and_encode(X_df, y):
    """
    Scales features using StandardScaler and encodes class labels using LabelEncoder.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df)
    joblib.dump(scaler, PROCESSED_PATH + "scaler.save")

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    joblib.dump(le, PROCESSED_PATH + "label_encoder.save")

    print(f"\nEncoded classes: {list(le.classes_)}")
    return X_scaled, y_encoded, X_df.columns.tolist()


def save_results(X_scaled, y_encoded, feature_cols):
    """
    Saves the processed feature matrix and encoded labels to CSV files.
    """
    X_df = pd.DataFrame(X_scaled, columns=feature_cols)
    y_df = pd.DataFrame(y_encoded, columns=["AttackClass"])

    X_df.to_csv(PROCESSED_PATH + "X_scaled.csv", index=False)
    y_df.to_csv(PROCESSED_PATH + "y.csv", index=False)

    print(f"\nSaved outputs:")
    print(f"  {PROCESSED_PATH}X_scaled.csv  ({X_df.shape[0]} rows, {X_df.shape[1]} features)")
    print(f"  {PROCESSED_PATH}y.csv")
    print(f"  {PROCESSED_PATH}scaler.save")
    print(f"  {PROCESSED_PATH}label_encoder.save")
    print(f"  {PROCESSED_PATH}variance_selector.save")


FEATURE_COLS = [
    'Destination Port', 'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
    'Total Length of Fwd Packets', 'Total Length of Bwd Packets', 'Fwd Packet Length Max',
    'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Std',
    'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean', 'Bwd Packet Length Std',
    'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
    'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
    'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min',
    'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Header Length', 'Bwd Header Length',
    'Fwd Packets/s', 'Bwd Packets/s', 'Min Packet Length', 'Max Packet Length', 'Packet Length Mean',
    'Packet Length Std', 'Packet Length Variance', 'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count',
    'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count', 'CWE Flag Count', 'ECE Flag Count',
    'Down/Up Ratio', 'Average Packet Size', 'Avg Fwd Segment Size', 'Avg Bwd Segment Size',
    'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate',
    'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate',
    'Subflow Fwd Packets', 'Subflow Fwd Bytes', 'Subflow Bwd Packets', 'Subflow Bwd Bytes',
    'Init_Win_bytes_forward', 'Init_Win_bytes_backward', 'act_data_pkt_fwd', 'min_seg_size_forward',
    'Active Mean', 'Active Std', 'Active Max', 'Active Min',
    'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min'
]


if __name__ == "__main__":
    print("=" * 50)
    print("PREPROCESSING CICIDS2017")
    print("=" * 50)

    df = load_and_merge_data(RAW_PATH)
    df = fix_column_names(df)
    df = map_labels(df)
    df = clean_features(df, FEATURE_COLS)

    X_df = df[FEATURE_COLS].reset_index(drop=True)
    y = df["AttackClass"].reset_index(drop=True)

    X_df, kept_cols = remove_zero_variance(X_df, FEATURE_COLS)
    X_df = remove_high_correlation(X_df)

    X_scaled, y_encoded, final_cols = scale_and_encode(X_df, y)
    save_results(X_scaled, y_encoded, final_cols)

    print("\nPreprocessing completed successfully.")