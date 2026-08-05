from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import RANDOM_STATE


@dataclass(frozen=True)
class TrainData:
    X: pd.DataFrame
    labels: pd.DataFrame
    experiment_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainTestData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.DataFrame
    y_test: pd.DataFrame
    experiment_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationTarget:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_eval: pd.DataFrame
    y_eval: pd.Series
    X_unknown: pd.DataFrame
    unknown_mask_in_eval: pd.Series | None
    unseen_attack_classes: tuple[str, ...] = ()
    unseen_attack_samples: int = 0


def _validate_features(X: pd.DataFrame, name: str) -> None:
    if X.isna().any().any():
        raise ValueError(f"{name} zawiera wartości NaN.")
    if not np.isfinite(X.to_numpy(copy=False)).all():
        raise ValueError(f"{name} zawiera wartości Inf lub -Inf.")


def _validate_labels(labels: pd.DataFrame, name: str) -> None:
    required = {"AttackClass", "Label_binary"}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"{name} nie zawiera kolumn: {sorted(missing)}")
    if labels[list(required)].isna().any().any():
        raise ValueError(f"{name} zawiera brakujące etykiety.")


def _load_experiment_metadata(processed_path: Path) -> dict[str, Any]:
    metadata_path = processed_path / "experiment_metadata.json"
    if not metadata_path.exists():
        return {
            "experiment_type": "standard",
            "scenario_name": processed_path.name or "standard",
        }

    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    if not isinstance(metadata, dict):
        raise ValueError(f"Plik {metadata_path} nie zawiera obiektu JSON.")
    return metadata


def load_train_data(processed_path: Path) -> TrainData:
    x_path = processed_path / "X_train_scaled.csv"
    y_path = processed_path / "y_train.csv"
    missing = [str(path) for path in (x_path, y_path) if not path.exists()]
    if missing:
        raise FileNotFoundError("Brakuje plików:\n- " + "\n- ".join(missing))

    X = pd.read_csv(x_path, dtype=np.float32)
    labels = pd.read_csv(y_path)
    if len(X) != len(labels):
        raise ValueError(f"Niezgodna liczba wierszy: X={len(X)}, y={len(labels)}.")

    _validate_features(X, "X_train")
    _validate_labels(labels, "y_train")
    if labels["AttackClass"].astype(str).eq("Unknown").any():
        raise ValueError("Zbiór treningowy nie może zawierać klasy Unknown.")

    return TrainData(
        X=X.reset_index(drop=True),
        labels=labels.reset_index(drop=True),
        experiment_metadata=_load_experiment_metadata(processed_path),
    )


def load_train_test_data(processed_path: Path) -> TrainTestData:
    paths = {
        "X_train": processed_path / "X_train_scaled.csv",
        "X_test": processed_path / "X_test_scaled.csv",
        "y_train": processed_path / "y_train.csv",
        "y_test": processed_path / "y_test.csv",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Brakuje plików:\n- " + "\n- ".join(missing))

    X_train = pd.read_csv(paths["X_train"], dtype=np.float32)
    X_test = pd.read_csv(paths["X_test"], dtype=np.float32)
    y_train = pd.read_csv(paths["y_train"])
    y_test = pd.read_csv(paths["y_test"])

    if len(X_train) != len(y_train):
        raise ValueError(f"Niezgodna liczba wierszy train: X={len(X_train)}, y={len(y_train)}.")
    if len(X_test) != len(y_test):
        raise ValueError(f"Niezgodna liczba wierszy test: X={len(X_test)}, y={len(y_test)}.")
    if not X_train.columns.equals(X_test.columns):
        raise ValueError("X_train i X_test mają inne kolumny lub inną kolejność.")

    _validate_features(X_train, "X_train")
    _validate_features(X_test, "X_test")
    _validate_labels(y_train, "y_train")
    _validate_labels(y_test, "y_test")
    if y_train["AttackClass"].astype(str).eq("Unknown").any():
        raise ValueError("Zbiór treningowy nie może zawierać klasy Unknown.")

    return TrainTestData(
        X_train=X_train.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        experiment_metadata=_load_experiment_metadata(processed_path),
    )


def target_series(labels: pd.DataFrame, target: str) -> pd.Series:
    if target not in {"AttackClass", "Label_binary"}:
        raise ValueError(f"Nieobsługiwany target: {target}")
    if target == "AttackClass":
        return labels[target].astype(str).reset_index(drop=True)
    return labels[target].astype(np.int32).reset_index(drop=True)


def unseen_attack_mask(data: TrainTestData) -> pd.Series:
    train_classes = set(data.y_train["AttackClass"].astype(str).unique())
    test_classes = data.y_test["AttackClass"].astype(str).reset_index(drop=True)
    return (
        ~test_classes.isin(train_classes)
        & test_classes.ne("Normal")
    ).reset_index(drop=True)


def prepare_evaluation_target(data: TrainTestData, target: str) -> EvaluationTarget:
    y_train = target_series(data.y_train, target)
    test_attack_classes = data.y_test["AttackClass"].astype(str).reset_index(drop=True)
    unseen_mask = unseen_attack_mask(data)
    unseen_classes = tuple(sorted(test_attack_classes.loc[unseen_mask].unique().tolist()))

    if target == "AttackClass":
        train_classes = set(data.y_train["AttackClass"].astype(str).unique())
        known_mask = test_attack_classes.isin(train_classes)
        if not known_mask.any():
            raise ValueError(
                "Po odrzuceniu klas niewidzianych w treningu zbiór testowy jest pusty."
            )
        if unseen_mask.any():
            warnings.warn(
                "W teście wieloklasowym pominięto próbki klas niewidzianych podczas "
                f"treningu: {list(unseen_classes)} ({int(unseen_mask.sum())} próbek). "
                "Ich detekcję oceniaj w trybie Label_binary."
            )
        return EvaluationTarget(
            X_train=data.X_train,
            y_train=y_train,
            X_eval=data.X_test.loc[known_mask].reset_index(drop=True),
            y_eval=test_attack_classes.loc[known_mask].reset_index(drop=True),
            X_unknown=data.X_test.loc[unseen_mask].reset_index(drop=True),
            unknown_mask_in_eval=None,
            unseen_attack_classes=unseen_classes,
            unseen_attack_samples=int(unseen_mask.sum()),
        )

    return EvaluationTarget(
        X_train=data.X_train,
        y_train=y_train,
        X_eval=data.X_test,
        y_eval=target_series(data.y_test, target),
        X_unknown=data.X_test.loc[unseen_mask].reset_index(drop=True),
        unknown_mask_in_eval=unseen_mask,
        unseen_attack_classes=unseen_classes,
        unseen_attack_samples=int(unseen_mask.sum()),
    )


def stratified_subsample(
    X: pd.DataFrame,
    y: pd.Series,
    n_samples: int,
    seed: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.Series]:
    if n_samples <= 0 or len(X) <= n_samples:
        return X.reset_index(drop=True), y.reset_index(drop=True)
    X_sub, _, y_sub, _ = train_test_split(
        X,
        y,
        train_size=n_samples,
        stratify=y,
        random_state=seed,
    )
    return X_sub.reset_index(drop=True), y_sub.reset_index(drop=True)


def stratified_subsample_with_labels(
    X: pd.DataFrame,
    labels: pd.DataFrame,
    n_samples: int,
    seed: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if n_samples <= 0 or len(X) <= n_samples:
        return X.reset_index(drop=True), labels.reset_index(drop=True)
    indices = np.arange(len(X))
    selected, _ = train_test_split(
        indices,
        train_size=n_samples,
        stratify=labels["AttackClass"].astype(str),
        random_state=seed,
    )
    return X.iloc[selected].reset_index(drop=True), labels.iloc[selected].reset_index(drop=True)


def split_with_labels(
    X: pd.DataFrame,
    labels: pd.DataFrame,
    validation_size: float,
    seed: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    validate_fraction(validation_size, "validation_size")
    indices = np.arange(len(X))
    train_idx, validation_idx = train_test_split(
        indices,
        test_size=validation_size,
        stratify=labels["AttackClass"].astype(str),
        random_state=seed,
    )
    return (
        X.iloc[train_idx].reset_index(drop=True),
        labels.iloc[train_idx].reset_index(drop=True),
        X.iloc[validation_idx].reset_index(drop=True),
        labels.iloc[validation_idx].reset_index(drop=True),
    )


def determine_cv_folds(y: pd.Series, requested_folds: int) -> int:
    min_count = int(y.value_counts().min())
    folds = min(requested_folds, min_count)
    if folds < 2:
        raise ValueError("Najrzadsza klasa ma mniej niż 2 próbki.")
    if folds != requested_folds:
        warnings.warn(
            f"Zmniejszono liczbę foldów z {requested_folds} do {folds}, "
            "ponieważ najrzadsza klasa ma za mało próbek."
        )
    return folds


def validate_fraction(value: float, name: str) -> None:
    if not 0.0 < value < 1.0:
        raise ValueError(f"{name} musi należeć do przedziału (0, 1).")
