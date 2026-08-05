from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import RANDOM_STATE

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "CICIDS2017"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed"

LEAVE_ONE_ATTACKS = ("Bot", "BruteForce", "DoS", "PortScan", "WebAttack")

FEATURE_COLS = [
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Std",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Total",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Bwd IAT Total",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",
    "Fwd PSH Flags",
    "Bwd PSH Flags",
    "Fwd URG Flags",
    "Bwd URG Flags",
    "Fwd Header Length",
    "Bwd Header Length",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "Min Packet Length",
    "Max Packet Length",
    "Packet Length Mean",
    "Packet Length Std",
    "Packet Length Variance",
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
    "CWE Flag Count",
    "ECE Flag Count",
    "Down/Up Ratio",
    "Average Packet Size",
    "Avg Fwd Segment Size",
    "Avg Bwd Segment Size",
    "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate",
    "Subflow Fwd Packets",
    "Subflow Fwd Bytes",
    "Subflow Bwd Packets",
    "Subflow Bwd Bytes",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
    "act_data_pkt_fwd",
    "min_seg_size_forward",
    "Active Mean",
    "Active Std",
    "Active Max",
    "Active Min",
    "Idle Mean",
    "Idle Std",
    "Idle Max",
    "Idle Min",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocessing CICIDS2017 dla podziału standardowego, "
            "leave-one-attack-class-out i eksperymentu międzydniowego."
        )
    )
    parser.add_argument("--raw-path", type=Path, default=DEFAULT_RAW_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--mode",
        choices=["standard", "leave-one-out", "cross-day"],
        default="standard",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--correlation-threshold", type=float, default=0.95)
    parser.add_argument(
        "--excluded-attacks",
        nargs="+",
        default=["all"],
        help=(
            "Klasy wyłączane w leave-one-out. Użyj 'all' albo np. "
            "--excluded-attacks WebAttack Bot."
        ),
    )
    parser.add_argument(
        "--train-files",
        nargs="+",
        help="Dokładne nazwy plików używanych do treningu cross-day.",
    )
    parser.add_argument(
        "--test-files",
        nargs="+",
        help="Dokładne nazwy plików używanych do testu cross-day.",
    )
    parser.add_argument(
        "--scenario-name",
        default="cross_day",
        help="Nazwa katalogu scenariusza międzydniowego.",
    )
    return parser.parse_args()


def safe_slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return value.strip("_") or "scenario"


def load_and_merge_data(raw_path: Path) -> pd.DataFrame:
    if not raw_path.exists():
        raise FileNotFoundError(f"Nie istnieje katalog danych surowych: {raw_path}")
    if not raw_path.is_dir():
        raise NotADirectoryError(f"Ścieżka danych surowych nie jest katalogiem: {raw_path}")

    all_files = sorted(
        path
        for path in raw_path.iterdir()
        if path.is_file() and path.suffix.lower() == ".csv"
    )
    if not all_files:
        raise FileNotFoundError(f"Brak plików CSV w {raw_path}")

    frames: list[pd.DataFrame] = []
    for path in all_files:
        print(f"Loading: {path}")
        frame = pd.read_csv(path, low_memory=False)
        frame["source_file"] = path.name
        frames.append(frame)

    merged = pd.concat(frames, ignore_index=True)
    print(f"Total records after merge: {len(merged)}")
    return merged


def fix_column_names(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result.columns = result.columns.str.strip()
    result = result.loc[:, ~result.columns.duplicated()]
    return result


def normalize_label_text(label: object) -> object:
    if not isinstance(label, str):
        return label
    label = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2015\x96\ufffd]", "-", label)
    label = re.sub(r"\s+", " ", label).strip()
    return label


def map_labels(df: pd.DataFrame) -> pd.DataFrame:
    if "Label" not in df.columns:
        raise KeyError("Brak wymaganej kolumny 'Label'.")

    result = df.copy()
    result["Label"] = result["Label"].apply(normalize_label_text)

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

    result["AttackClass"] = result["Label"].map(label_map)
    unmapped = result.loc[result["AttackClass"].isna(), "Label"].dropna().unique()
    if len(unmapped) > 0:
        n_dropped = int(result["AttackClass"].isna().sum())
        print(f"[UWAGA] Etykiety spoza mapy (zostaną usunięte): {unmapped}")
        print(
            f"[UWAGA] Liczba usuwanych rekordów: {n_dropped} "
            f"({100.0 * n_dropped / len(result):.4f}% danych)"
        )

    result = result[result["AttackClass"].notna()].copy()
    result["Label_binary"] = result["AttackClass"].ne("Normal").astype(np.int32)

    print("\nRozkład klas po mapowaniu:")
    print(result["AttackClass"].value_counts())
    return result


def validate_feature_columns(df: pd.DataFrame, feature_cols: Iterable[str]) -> None:
    missing = [column for column in feature_cols if column not in df.columns]
    if missing:
        raise KeyError("Brakuje oczekiwanych kolumn cech:\n- " + "\n- ".join(missing))


def clean_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    result = df.copy()
    result[feature_cols] = result[feature_cols].apply(pd.to_numeric, errors="coerce")
    result[feature_cols] = result[feature_cols].replace([np.inf, -np.inf], np.nan)

    before = len(result)
    result = result.dropna(subset=feature_cols)
    print(f"\nUsunięto {before - len(result)} wierszy z NaN/Inf. Zostało: {len(result)}")

    # Usunięcie duplikatów przed podziałem ogranicza ryzyko umieszczenia
    # identycznych rekordów w zbiorze treningowym i testowym.
    dedup_cols = feature_cols + ["AttackClass"]
    before = len(result)
    result = result.drop_duplicates(subset=dedup_cols)
    print(
        f"Usunięto {before - len(result)} zduplikowanych wierszy. "
        f"Zostało: {len(result)}"
    )
    return result.reset_index(drop=True)


def _validate_stratified_split(labels: pd.Series, test_size: float) -> None:
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size musi należeć do przedziału (0, 1).")
    counts = labels.value_counts()
    too_small = counts[counts < 2]
    if not too_small.empty:
        raise ValueError(
            "Nie można wykonać podziału stratyfikowanego. Klasy z mniej niż "
            f"2 próbkami:\n{too_small}"
        )


def split_standard(
        df: pd.DataFrame,
        test_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    known = df[df["AttackClass"].ne("Unknown")].copy()
    unknown = df[df["AttackClass"].eq("Unknown")].copy()
    _validate_stratified_split(known["AttackClass"], test_size)

    train_df, test_known = train_test_split(
        known,
        test_size=test_size,
        stratify=known["AttackClass"],
        random_state=RANDOM_STATE,
    )
    test_df = pd.concat([test_known, unknown], ignore_index=True)
    test_df = test_df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

    metadata = {
        "experiment_type": "standard",
        "scenario_name": "standard_random_stratified",
        "test_size": test_size,
        "held_out_attack": None,
        "train_files": sorted(train_df["source_file"].astype(str).unique().tolist()),
        "test_files": sorted(test_df["source_file"].astype(str).unique().tolist()),
    }
    return train_df.reset_index(drop=True), test_df, metadata


def split_leave_one_attack_out(
        df: pd.DataFrame,
        excluded_attack: str,
        test_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    if excluded_attack not in LEAVE_ONE_ATTACKS:
        raise ValueError(
            f"Nieobsługiwana klasa '{excluded_attack}'. Dozwolone: {LEAVE_ONE_ATTACKS}."
        )

    held_out = df[df["AttackClass"].eq(excluded_attack)].copy()
    if held_out.empty:
        raise ValueError(f"Brak próbek klasy {excluded_attack}.")

    known_pool = df[~df["AttackClass"].isin(["Unknown", excluded_attack])].copy()
    _validate_stratified_split(known_pool["AttackClass"], test_size)

    train_df, known_test = train_test_split(
        known_pool,
        test_size=test_size,
        stratify=known_pool["AttackClass"],
        random_state=RANDOM_STATE,
    )
    test_df = pd.concat([known_test, held_out], ignore_index=True)
    test_df = test_df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

    metadata = {
        "experiment_type": "leave_one_attack_out",
        "scenario_name": f"leave_out_{safe_slug(excluded_attack)}",
        "test_size": test_size,
        "held_out_attack": excluded_attack,
        "held_out_samples": int(len(held_out)),
        "train_files": sorted(train_df["source_file"].astype(str).unique().tolist()),
        "test_files": sorted(test_df["source_file"].astype(str).unique().tolist()),
        "recommended_target": "Label_binary",
    }
    return train_df.reset_index(drop=True), test_df, metadata


def _validate_requested_files(
        available: set[str],
        requested: Iterable[str],
        argument_name: str,
) -> list[str]:
    values = list(dict.fromkeys(str(value) for value in requested))
    missing = [value for value in values if value not in available]
    if missing:
        raise ValueError(
            f"{argument_name} zawiera nieistniejące pliki:\n- "
            + "\n- ".join(missing)
        )
    return values


def split_cross_day(
        df: pd.DataFrame,
        train_files: Iterable[str],
        test_files: Iterable[str],
        scenario_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    available = set(df["source_file"].astype(str).unique())
    train_names = _validate_requested_files(available, train_files, "--train-files")
    test_names = _validate_requested_files(available, test_files, "--test-files")

    overlap = sorted(set(train_names) & set(test_names))
    if overlap:
        raise ValueError(
            "Te same pliki nie mogą należeć do train i test:\n- " + "\n- ".join(overlap)
        )

    train_df = df[df["source_file"].isin(train_names)].copy()
    test_df = df[df["source_file"].isin(test_names)].copy()
    if train_df.empty or test_df.empty:
        raise ValueError("Zbiór treningowy lub testowy eksperymentu cross-day jest pusty.")

    # Klasa Unknown pozostaje wyłącznie w teście, zgodnie z założeniami projektu.
    unknown_train_count = int(train_df["AttackClass"].eq("Unknown").sum())
    if unknown_train_count:
        print(
            f"[UWAGA] Usuwam z treningu cross-day {unknown_train_count} "
            "próbek klasy Unknown."
        )
        train_df = train_df[train_df["AttackClass"].ne("Unknown")].copy()

    if train_df["AttackClass"].nunique() < 2:
        raise ValueError("Trening cross-day musi zawierać co najmniej dwie klasy.")

    metadata = {
        "experiment_type": "cross_day",
        "scenario_name": safe_slug(scenario_name),
        "held_out_attack": None,
        "train_files": train_names,
        "test_files": test_names,
        "removed_unknown_from_train": unknown_train_count,
        "recommended_target": "Label_binary",
    }
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True), metadata


def _class_distribution(frame: pd.DataFrame) -> dict[str, int]:
    return {
        str(label): int(count)
        for label, count in frame["AttackClass"].value_counts().sort_index().items()
    }


def preprocess_and_save_split(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        output_dir: Path,
        metadata: dict[str, object],
        correlation_threshold: float,
) -> None:
    if not 0.0 <= correlation_threshold <= 1.0:
        raise ValueError("correlation_threshold musi należeć do przedziału [0, 1].")

    output_dir.mkdir(parents=True, exist_ok=True)

    X_train = train_df[FEATURE_COLS].reset_index(drop=True)
    X_test = test_df[FEATURE_COLS].reset_index(drop=True)
    y_train = train_df[["AttackClass", "Label_binary"]].reset_index(drop=True)
    y_test = test_df[["AttackClass", "Label_binary"]].reset_index(drop=True)

    selector = VarianceThreshold(threshold=0.0)
    selector.fit(X_train)
    kept_after_variance = [
        column for column, keep in zip(FEATURE_COLS, selector.get_support()) if keep
    ]
    removed_zero_variance = [
        column for column, keep in zip(FEATURE_COLS, selector.get_support()) if not keep
    ]
    print(
        f"\nUsunięto cechy o zerowej wariancji ({len(removed_zero_variance)}): "
        f"{removed_zero_variance}"
    )

    X_train_selected = pd.DataFrame(
        selector.transform(X_train), columns=kept_after_variance
    )
    X_test_selected = pd.DataFrame(
        selector.transform(X_test), columns=kept_after_variance
    )

    corr_matrix = X_train_selected.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    dropped_correlated = [
        column
        for column in upper.columns
        if bool((upper[column] > correlation_threshold).any())
    ]
    print(
        f"Usunięto silnie skorelowane cechy (>{correlation_threshold}) "
        f"({len(dropped_correlated)}): {dropped_correlated}"
    )
    X_train_selected = X_train_selected.drop(columns=dropped_correlated)
    X_test_selected = X_test_selected.drop(columns=dropped_correlated)

    scaler = StandardScaler()
    scaler.fit(X_train_selected)
    X_train_scaled = pd.DataFrame(
        scaler.transform(X_train_selected), columns=X_train_selected.columns
    ).astype(np.float32)
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test_selected), columns=X_test_selected.columns
    ).astype(np.float32)

    X_train_scaled.to_csv(output_dir / "X_train_scaled.csv", index=False)
    X_test_scaled.to_csv(output_dir / "X_test_scaled.csv", index=False)
    y_train.to_csv(output_dir / "y_train.csv", index=False)
    y_test.to_csv(output_dir / "y_test.csv", index=False)

    metadata_cols = ["source_file", "Label", "AttackClass", "Label_binary"]
    train_df[metadata_cols].reset_index(drop=True).to_csv(
        output_dir / "train_metadata.csv", index=False
    )
    test_df[metadata_cols].reset_index(drop=True).to_csv(
        output_dir / "test_metadata.csv", index=False
    )

    joblib.dump(selector, output_dir / "variance_selector.save")
    joblib.dump(dropped_correlated, output_dir / "dropped_correlated_features.save")
    joblib.dump(scaler, output_dir / "scaler.save")

    final_feature_columns = X_train_scaled.columns.tolist()
    with (output_dir / "final_feature_columns.json").open("w", encoding="utf-8") as file:
        json.dump(final_feature_columns, file, indent=2, ensure_ascii=False)

    complete_metadata = {
        **metadata,
        "random_state": RANDOM_STATE,
        "correlation_threshold": correlation_threshold,
        "original_feature_count": len(FEATURE_COLS),
        "removed_zero_variance_features": removed_zero_variance,
        "dropped_correlated_features": dropped_correlated,
        "final_feature_count": len(final_feature_columns),
        "final_feature_columns": final_feature_columns,
        "train_samples": int(len(train_df)),
        "test_samples": int(len(test_df)),
        "train_class_distribution": _class_distribution(train_df),
        "test_class_distribution": _class_distribution(test_df),
    }
    with (output_dir / "experiment_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(complete_metadata, file, indent=2, ensure_ascii=False)

    print(f"\nZapisano scenariusz do: {output_dir}")
    print(f"  X_train_scaled.csv: {X_train_scaled.shape}")
    print(f"  X_test_scaled.csv:  {X_test_scaled.shape}")
    print(f"  y_train.csv:        {y_train.shape}")
    print(f"  y_test.csv:         {y_test.shape}")


def expand_excluded_attacks(values: Iterable[str]) -> list[str]:
    selected = list(values)
    if "all" in selected:
        return list(LEAVE_ONE_ATTACKS)
    unknown = [value for value in selected if value not in LEAVE_ONE_ATTACKS]
    if unknown:
        raise ValueError(f"Nieznane klasy leave-one-out: {unknown}")
    return list(dict.fromkeys(selected))


def list_source_files(raw_path: Path) -> None:
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Nie istnieje katalog danych: {raw_path}"
        )

    files = sorted(
        path.name
        for path in raw_path.iterdir()
        if path.is_file() and path.suffix.lower() == ".csv"
    )

    if not files:
        print(f"Brak plików CSV w: {raw_path}")
        return

    print(f"\nDostępne pliki CSV w {raw_path}:")
    for file_name in files:
        print(f"  - {file_name}")

def main() -> None:
    args = parse_args()

    print("=" * 70)
    print("PREPROCESSING CICIDS2017")
    print(f"Tryb: {args.mode}")
    print("=" * 70)

    df = load_and_merge_data(args.raw_path)
    df = fix_column_names(df)
    validate_feature_columns(df, FEATURE_COLS)
    df = map_labels(df)
    df = clean_features(df, FEATURE_COLS)

    if args.mode == "standard":
        train_df, test_df, metadata = split_standard(df, args.test_size)
        preprocess_and_save_split(
            train_df=train_df,
            test_df=test_df,
            output_dir=args.output_root,
            metadata=metadata,
            correlation_threshold=args.correlation_threshold,
        )
        return

    if args.mode == "leave-one-out":
        for excluded_attack in expand_excluded_attacks(args.excluded_attacks):
            print("\n" + "=" * 70)
            print(f"LEAVE-ONE-OUT: {excluded_attack}")
            print("=" * 70)

            train_df, test_df, metadata = split_leave_one_attack_out(
                df=df,
                excluded_attack=excluded_attack,
                test_size=args.test_size,
            )
            output_dir = args.output_root / "leave_one_out" / safe_slug(excluded_attack)
            preprocess_and_save_split(
                train_df=train_df,
                test_df=test_df,
                output_dir=output_dir,
                metadata=metadata,
                correlation_threshold=args.correlation_threshold,
            )
        return

    if not args.train_files or not args.test_files:
        raise ValueError(
            "Dla --mode cross-day podaj --train-files oraz --test-files. "
        )

    train_df, test_df, metadata = split_cross_day(
        df=df,
        train_files=args.train_files,
        test_files=args.test_files,
        scenario_name=args.scenario_name,
    )
    output_dir = args.output_root / "cross_day" / safe_slug(args.scenario_name)
    preprocess_and_save_split(
        train_df=train_df,
        test_df=test_df,
        output_dir=output_dir,
        metadata=metadata,
        correlation_threshold=args.correlation_threshold,
    )


if __name__ == "__main__":
    main()
