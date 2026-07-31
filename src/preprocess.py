import os
import re
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

RAW_PATH = "../data/raw/CICIDS2017"
PROCESSED_PATH = "../data/processed/"
RANDOM_STATE = 42

os.makedirs(PROCESSED_PATH, exist_ok=True)


def load_and_merge_data(raw_path):
    all_files = [
        os.path.join(raw_path, f)
        for f in os.listdir(raw_path)
        if f.endswith(".csv")
    ]
    if not all_files:
        raise FileNotFoundError(f"Brak plików CSV w {raw_path}")

    df_list = []
    for file in all_files:
        print(f"Loading: {file}")
        temp_df = pd.read_csv(file, low_memory=False)
        temp_df["source_file"] = os.path.basename(file)
        df_list.append(temp_df)

    merged = pd.concat(df_list, ignore_index=True)
    print(f"Total records after merge: {len(merged)}")
    return merged


def fix_column_names(df):
    df.columns = df.columns.str.strip()
    df = df.loc[:, ~df.columns.duplicated()]
    return df


def normalize_label_text(label: str) -> str:
    if not isinstance(label, str):
        return label
    label = re.sub(r"[\u2013\u2014\x96]", "-", label)
    label = re.sub(r"\s+", " ", label).strip()
    return label


def map_labels(df):
    df["Label"] = df["Label"].apply(normalize_label_text)

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

        "Infiltration": "Unknown",
        "Heartbleed": "Unknown",
    }

    df["AttackClass"] = df["Label"].map(label_map)

    unmapped = df[df["AttackClass"].isna()]["Label"].unique()
    if len(unmapped) > 0:
        print(f"[UWAGA] Etykiety spoza mapy po normalizacji (zostaną usunięte): {unmapped}")
        n_dropped = df["AttackClass"].isna().sum()
        print(f"[UWAGA] Liczba usuwanych rekordów: {n_dropped} "
              f"({100 * n_dropped / len(df):.4f}% wszystkich danych)")

    df = df[df["AttackClass"].notna()].copy()

    df["Label_binary"] = (df["AttackClass"] != "Normal").astype(int)

    print("\nRozkład klas po mapowaniu:")
    print(df["AttackClass"].value_counts())
    return df


def clean_features(df, feature_cols):
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)

    before = len(df)
    df = df.dropna(subset=feature_cols)
    print(f"\nUsunięto {before - len(df)} wierszy z NaN/Inf. Zostało: {len(df)}")

    dedup_cols = feature_cols + ["AttackClass"]
    before = len(df)
    df = df.drop_duplicates(subset=dedup_cols)
    print(f"Usunięto {before - len(df)} zduplikowanych wierszy. Zostało: {len(df)}")

    return df


def validate_feature_columns(df, feature_cols):
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"Brakuje oczekiwanych kolumn cech w danych: {missing}\n"
            f"Sprawdź nazwy kolumn w plikach źródłowych (możliwe rozjazdy między dniami)."
        )


def split_train_test(df):
    df_known = df[df["AttackClass"] != "Unknown"].copy()
    df_unknown = df[df["AttackClass"] == "Unknown"].copy()

    train_df, test_df = train_test_split(
        df_known, test_size=0.2, stratify=df_known["AttackClass"],
        random_state=RANDOM_STATE
    )

    test_df = pd.concat([test_df, df_unknown], ignore_index=True)

    print(f"\nZbiór treningowy: {len(train_df)} wierszy (bez klasy Unknown)")
    print(f"Zbiór testowy: {len(test_df)} wierszy (w tym {len(df_unknown)} próbek Unknown)")
    return train_df, test_df


def remove_zero_variance(X_train, X_test, feature_cols):
    selector = VarianceThreshold(threshold=0.0)
    selector.fit(X_train)

    kept_cols = [c for c, keep in zip(feature_cols, selector.get_support()) if keep]
    removed = [c for c, keep in zip(feature_cols, selector.get_support()) if not keep]
    print(f"\nUsunięto cechy o zerowej wariancji ({len(removed)}): {removed}")

    X_train_f = pd.DataFrame(selector.transform(X_train), columns=kept_cols)
    X_test_f = pd.DataFrame(selector.transform(X_test), columns=kept_cols)

    joblib.dump(selector, PROCESSED_PATH + "variance_selector.save")
    return X_train_f, X_test_f, kept_cols


def remove_high_correlation(X_train, X_test, threshold=0.95):
    corr_matrix = X_train.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]

    print(f"\nUsunięto silnie skorelowane cechy (>{threshold}) ({len(to_drop)}): {to_drop}")

    X_train = X_train.drop(columns=to_drop)
    X_test = X_test.drop(columns=to_drop)

    joblib.dump(to_drop, PROCESSED_PATH + "dropped_correlated_features.save")
    return X_train, X_test


def scale_features(X_train, X_test):
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    joblib.dump(scaler, PROCESSED_PATH + "scaler.save")
    return X_train_scaled, X_test_scaled


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
    validate_feature_columns(df, FEATURE_COLS)
    df = map_labels(df)
    df = clean_features(df, FEATURE_COLS)

    train_df, test_df = split_train_test(df)

    X_train = train_df[FEATURE_COLS].reset_index(drop=True)
    X_test = test_df[FEATURE_COLS].reset_index(drop=True)
    y_train = train_df[["AttackClass", "Label_binary"]].reset_index(drop=True)
    y_test = test_df[["AttackClass", "Label_binary"]].reset_index(drop=True)

    X_train, X_test, kept_cols = remove_zero_variance(X_train, X_test, FEATURE_COLS)
    X_train, X_test = remove_high_correlation(X_train, X_test)

    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

    X_train_scaled.to_csv(PROCESSED_PATH + "X_train_scaled.csv", index=False)
    X_test_scaled.to_csv(PROCESSED_PATH + "X_test_scaled.csv", index=False)
    y_train.to_csv(PROCESSED_PATH + "y_train.csv", index=False)
    y_test.to_csv(PROCESSED_PATH + "y_test.csv", index=False)

    print(f"\nZapisano do {PROCESSED_PATH}:")
    print(f"  X_train_scaled.csv  {X_train_scaled.shape}")
    print(f"  X_test_scaled.csv   {X_test_scaled.shape}")
    print(f"  y_train.csv / y_test.csv (kolumny: AttackClass, Label_binary)")
    print("\nPreprocessing zakończony.")