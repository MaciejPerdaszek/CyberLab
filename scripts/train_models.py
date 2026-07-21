"""
train_models_approach2.py
=========================

Trening i porównanie modeli dla CICIDS2017:
- Random Forest,
- MLP,
- Linear SVM,
- KNN,
- Autoenkoder anomalii + Random Forest (wyłącznie podejście 2).

Podejście 2:
1. Autoenkoder jest trenowany wyłącznie na ruchu Normal.
2. Próg anomalii jest wyznaczany z błędu rekonstrukcji walidacji Normal.
3. Próbki powyżej progu są traktowane jako potencjalne ataki.
4. Random Forest klasyfikuje typ znanego ataku dla próbek zaflagowanych.
5. Klasa Unknown pozostaje wyłącznie w teście zdolności zero-day.

Kod nie wymaga osobnego pliku autoencoder_anomaly_rf.py.
"""

from __future__ import annotations

import argparse
import json
import random
import time
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC


RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Trening MLP, SVM, Random Forest, KNN oraz "
            "dwuetapowego Autoenkodera anomalii + RF."
        )
    )

    parser.add_argument(
        "--target",
        choices=["AttackClass", "Label_binary"],
        default="AttackClass",
        help="Rodzaj zadania: wieloklasowe lub binarne.",
    )
    parser.add_argument(
        "--processed-path",
        type=Path,
        default=Path("../data/processed"),
        help="Katalog z przetworzonymi plikami CSV.",
    )
    parser.add_argument(
        "--models-path",
        type=Path,
        default=Path("../models"),
        help="Katalog zapisu modeli.",
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=Path("../results"),
        help="Katalog zapisu wyników.",
    )

    parser.add_argument("--rf-trees", type=int, default=300)
    parser.add_argument("--mlp-max-iter", type=int, default=100)
    parser.add_argument(
        "--knn-max-train",
        type=int,
        default=100_000,
        help="Maksymalna liczba próbek treningowych KNN; 0 oznacza cały zbiór.",
    )
    parser.add_argument("--knn-neighbors", type=int, default=7)

    parser.add_argument("--ae-epochs", type=int, default=100)
    parser.add_argument("--ae-batch-size", type=int, default=2048)
    parser.add_argument(
        "--ae-latent-dim",
        type=int,
        default=0,
        help="Wymiar latentny; 0 oznacza dobór automatyczny.",
    )
    parser.add_argument(
        "--ae-threshold-percentile",
        type=int,
        default=95,
        help=(
            "Percentyl błędu rekonstrukcji na walidacji Normal, używany jako "
            "próg anomalii w dwuetapowym podejściu AE + RF. "
            "Wartość można dobrać na wewnętrznym zbiorze walidacyjnym."
        ),
    )

    parser.add_argument("--skip-rf", action="store_true")
    parser.add_argument("--skip-mlp", action="store_true")
    parser.add_argument("--skip-svm", action="store_true")
    parser.add_argument("--skip-knn", action="store_true")
    parser.add_argument("--skip-autoencoder", action="store_true")

    return parser.parse_args()


def set_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def validate_required_files(processed_path: Path) -> dict[str, Path]:
    files = {
        "X_train": processed_path / "X_train_scaled.csv",
        "X_test": processed_path / "X_test_scaled.csv",
        "y_train": processed_path / "y_train.csv",
        "y_test": processed_path / "y_test.csv",
    }

    missing = [str(path) for path in files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Brakuje wymaganych plików:\n- " + "\n- ".join(missing)
        )

    return files


def load_data(processed_path: Path) -> tuple[pd.DataFrame, ...]:
    files = validate_required_files(processed_path)

    print("Wczytywanie danych...")
    X_train = pd.read_csv(files["X_train"], dtype=np.float32)
    X_test = pd.read_csv(files["X_test"], dtype=np.float32)
    y_train_df = pd.read_csv(files["y_train"])
    y_test_df = pd.read_csv(files["y_test"])

    if len(X_train) != len(y_train_df):
        raise ValueError(
            f"Niezgodna liczba wierszy train: X={len(X_train)}, y={len(y_train_df)}."
        )

    if len(X_test) != len(y_test_df):
        raise ValueError(
            f"Niezgodna liczba wierszy test: X={len(X_test)}, y={len(y_test_df)}."
        )

    if not X_train.columns.equals(X_test.columns):
        raise ValueError("X_train i X_test mają inne kolumny lub inną kolejność.")

    if X_train.isna().any().any() or X_test.isna().any().any():
        raise ValueError("X_train lub X_test zawiera wartości NaN.")

    if not np.isfinite(X_train.to_numpy(copy=False)).all():
        raise ValueError("X_train zawiera wartości Inf/-Inf.")

    if not np.isfinite(X_test.to_numpy(copy=False)).all():
        raise ValueError("X_test zawiera wartości Inf/-Inf.")

    print(f"X_train: {X_train.shape}")
    print(f"X_test:  {X_test.shape}")

    return X_train, X_test, y_train_df, y_test_df


def prepare_target(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train_df: pd.DataFrame,
        y_test_df: pd.DataFrame,
        target: str,
) -> tuple[
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series | None,
]:
    required_columns = {"AttackClass", "Label_binary"}
    missing = required_columns - set(y_train_df.columns)
    missing |= required_columns - set(y_test_df.columns)

    if missing:
        raise ValueError(f"Brak kolumn etykiet: {sorted(missing)}")

    if target == "AttackClass":
        if (y_train_df["AttackClass"] == "Unknown").any():
            raise ValueError(
                "Zbiór treningowy zawiera klasę Unknown, choć nie powinien."
            )

        known_mask = y_test_df["AttackClass"].ne("Unknown")
        unknown_mask = ~known_mask

        X_eval = X_test.loc[known_mask].reset_index(drop=True)
        y_eval = (
            y_test_df.loc[known_mask, "AttackClass"]
            .astype(str)
            .reset_index(drop=True)
        )
        y_train = y_train_df["AttackClass"].astype(str).reset_index(drop=True)

        X_unknown = X_test.loc[unknown_mask].reset_index(drop=True)
        unknown_binary_labels = None

        print("\nTryb wieloklasowy:")
        print(f"Znane próbki testowe: {len(X_eval)}")
        print(f"Próbki Unknown poza standardową oceną: {len(X_unknown)}")

    else:
        X_eval = X_test.reset_index(drop=True)
        y_train = y_train_df["Label_binary"].astype(int).reset_index(drop=True)
        y_eval = y_test_df["Label_binary"].astype(int).reset_index(drop=True)

        unknown_mask = y_test_df["AttackClass"].eq("Unknown").reset_index(drop=True)
        X_unknown = X_eval.loc[unknown_mask].reset_index(drop=True)
        unknown_binary_labels = unknown_mask

        print("\nTryb binarny:")
        print(f"Wszystkie próbki testowe: {len(X_eval)}")
        print(f"Próbki Unknown oznaczone jako atak: {int(unknown_mask.sum())}")

    print("\nRozkład klas treningowych:")
    print(y_train.value_counts())

    print("\nRozkład klas testowych używanych w standardowej ocenie:")
    print(y_eval.value_counts())

    return (
        X_train.reset_index(drop=True),
        y_train,
        X_eval,
        y_eval,
        X_unknown,
        unknown_binary_labels,
    )


def stratified_sample(
        X: pd.DataFrame,
        y: pd.Series,
        max_samples: int,
        seed: int,
) -> tuple[pd.DataFrame, pd.Series]:
    if max_samples <= 0 or len(X) <= max_samples:
        return X, y

    X_sample, _, y_sample, _ = train_test_split(
        X,
        y,
        train_size=max_samples,
        stratify=y,
        random_state=seed,
    )

    return X_sample.reset_index(drop=True), y_sample.reset_index(drop=True)


def safe_name(name: str) -> str:
    return name.lower().replace(" ", "_").replace("+", "plus")


def save_classification_outputs(
        model_name: str,
        y_true: pd.Series,
        predictions: np.ndarray,
        results_path: Path,
) -> None:
    stem = safe_name(model_name)

    labels = sorted(
        set(pd.Series(y_true).tolist()) | set(pd.Series(predictions).tolist()),
        key=str,
    )

    matrix = confusion_matrix(y_true, predictions, labels=labels)
    matrix_df = pd.DataFrame(
        matrix,
        index=[f"true_{label}" for label in labels],
        columns=[f"pred_{label}" for label in labels],
    )
    matrix_df.to_csv(
        results_path / f"confusion_matrix_{stem}.csv",
        index=True,
    )

    report = classification_report(
        y_true,
        predictions,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).T.to_csv(
        results_path / f"classification_report_{stem}.csv",
        index=True,
    )


def evaluate_model(
        model_name: str,
        model: Any,
        X_fit: pd.DataFrame | np.ndarray,
        y_fit: pd.Series,
        X_eval: pd.DataFrame | np.ndarray,
        y_eval: pd.Series,
        models_path: Path,
        results_path: Path,
        target: str,
        unknown_mask: pd.Series | None = None,
        save_model: bool = True,
) -> tuple[Any, np.ndarray, dict[str, Any]]:
    print("\n" + "=" * 78)
    print(f"MODEL: {model_name}")
    print("=" * 78)

    train_start = time.perf_counter()
    model.fit(X_fit, y_fit)
    training_time = time.perf_counter() - train_start

    predict_start = time.perf_counter()
    predictions = model.predict(X_eval)
    prediction_time = time.perf_counter() - predict_start

    accuracy = accuracy_score(y_eval, predictions)
    balanced_accuracy = balanced_accuracy_score(y_eval, predictions)
    macro_f1 = f1_score(
        y_eval,
        predictions,
        average="macro",
        zero_division=0,
    )
    weighted_f1 = f1_score(
        y_eval,
        predictions,
        average="weighted",
        zero_division=0,
    )

    result: dict[str, Any] = {
        "model": model_name,
        "target": target,
        "train_samples": len(X_fit),
        "test_samples": len(X_eval),
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "training_time_seconds": training_time,
        "prediction_time_seconds": prediction_time,
        "unknown_attack_detection_rate": np.nan,
    }

    if (
            target == "Label_binary"
            and unknown_mask is not None
            and bool(unknown_mask.any())
    ):
        aligned_mask = unknown_mask.to_numpy(dtype=bool)
        unknown_predictions = np.asarray(predictions)[aligned_mask]
        result["unknown_attack_detection_rate"] = float(
            np.mean(unknown_predictions == 1)
        )

    print(f"Accuracy:          {accuracy:.6f}")
    print(f"Balanced accuracy: {balanced_accuracy:.6f}")
    print(f"Macro F1:          {macro_f1:.6f}")
    print(f"Weighted F1:       {weighted_f1:.6f}")
    print(f"Czas treningu:     {training_time:.2f} s")
    print(f"Czas predykcji:    {prediction_time:.2f} s")

    if not np.isnan(result["unknown_attack_detection_rate"]):
        print(
            "Wykrywanie Unknown jako atak: "
            f"{result['unknown_attack_detection_rate']:.6f}"
        )

    print("\nClassification report:")
    print(classification_report(y_eval, predictions, zero_division=0))

    save_classification_outputs(
        model_name=model_name,
        y_true=y_eval,
        predictions=predictions,
        results_path=results_path,
    )

    if save_model:
        joblib.dump(
            model,
            models_path / f"{safe_name(model_name)}.joblib",
        )

    return model, predictions, result


def train_random_forest(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_eval: pd.DataFrame,
        y_eval: pd.Series,
        args: argparse.Namespace,
        unknown_mask: pd.Series | None,
) -> dict[str, Any]:
    model = RandomForestClassifier(
        n_estimators=args.rf_trees,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    _, _, result = evaluate_model(
        model_name="Random Forest",
        model=model,
        X_fit=X_train,
        y_fit=y_train,
        X_eval=X_eval,
        y_eval=y_eval,
        models_path=args.models_path,
        results_path=args.results_path,
        target=args.target,
        unknown_mask=unknown_mask,
    )
    return result


def train_mlp(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_eval: pd.DataFrame,
        y_eval: pd.Series,
        args: argparse.Namespace,
        unknown_mask: pd.Series | None,
) -> dict[str, Any]:
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        batch_size=1024,
        learning_rate="adaptive",
        learning_rate_init=0.001,
        max_iter=args.mlp_max_iter,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=RANDOM_STATE,
        verbose=True,
    )

    _, _, result = evaluate_model(
        model_name="MLP",
        model=model,
        X_fit=X_train,
        y_fit=y_train,
        X_eval=X_eval,
        y_eval=y_eval,
        models_path=args.models_path,
        results_path=args.results_path,
        target=args.target,
        unknown_mask=unknown_mask,
    )
    return result


def train_linear_svm(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_eval: pd.DataFrame,
        y_eval: pd.Series,
        args: argparse.Namespace,
        unknown_mask: pd.Series | None,
) -> dict[str, Any]:
    model = LinearSVC(
        C=1.0,
        class_weight="balanced",
        dual=False,
        max_iter=5000,
        random_state=RANDOM_STATE,
    )

    _, _, result = evaluate_model(
        model_name="Linear SVM",
        model=model,
        X_fit=X_train,
        y_fit=y_train,
        X_eval=X_eval,
        y_eval=y_eval,
        models_path=args.models_path,
        results_path=args.results_path,
        target=args.target,
        unknown_mask=unknown_mask,
    )
    return result


def train_knn(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_eval: pd.DataFrame,
        y_eval: pd.Series,
        args: argparse.Namespace,
        unknown_mask: pd.Series | None,
) -> dict[str, Any]:
    X_fit, y_fit = stratified_sample(
        X_train,
        y_train,
        max_samples=args.knn_max_train,
        seed=RANDOM_STATE,
    )

    if len(X_fit) < len(X_train):
        print(
            f"\nKNN: użycie stratified próbki {len(X_fit)} "
            f"z {len(X_train)} rekordów treningowych."
        )

    model = KNeighborsClassifier(
        n_neighbors=args.knn_neighbors,
        weights="distance",
        metric="minkowski",
        p=2,
        n_jobs=-1,
    )

    _, _, result = evaluate_model(
        model_name="KNN",
        model=model,
        X_fit=X_fit,
        y_fit=y_fit,
        X_eval=X_eval,
        y_eval=y_eval,
        models_path=args.models_path,
        results_path=args.results_path,
        target=args.target,
        unknown_mask=unknown_mask,
    )
    return result


def build_anomaly_autoencoder(input_dimension: int, latent_dimension: int):
    """Buduje autoenkoder używany wyłącznie do detekcji anomalii."""
    try:
        from tensorflow import keras
        from tensorflow.keras import layers
    except ImportError as error:
        raise ImportError(
            "TensorFlow nie jest zainstalowany. "
            "Zainstaluj pakiet tensorflow, aby uruchomić AE + RF."
        ) from error

    inputs = keras.Input(
        shape=(input_dimension,),
        name="network_features",
    )

    x = layers.Dense(128, activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(32, activation="relu")(x)

    latent = layers.Dense(
        latent_dimension,
        activation="relu",
        name="latent_vector",
    )(x)

    x = layers.Dense(32, activation="relu")(latent)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(128, activation="relu")(x)

    outputs = layers.Dense(
        input_dimension,
        activation="linear",
        name="reconstruction",
    )(x)

    autoencoder = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="anomaly_autoencoder",
    )
    autoencoder.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
    )

    return autoencoder


def reconstruction_error(
        model: Any,
        X: pd.DataFrame | np.ndarray,
        batch_size: int = 4096,
) -> np.ndarray:
    """Zwraca średni błąd rekonstrukcji MSE osobno dla każdej próbki."""
    if isinstance(X, pd.DataFrame):
        X_array = X.to_numpy(dtype=np.float32, copy=False)
    else:
        X_array = np.asarray(X, dtype=np.float32)

    reconstructed = model.predict(
        X_array,
        verbose=0,
        batch_size=batch_size,
    )

    return np.mean(np.square(X_array - reconstructed), axis=1)


def save_stage_report(
        name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: list[Any],
        target_names: list[str],
        results_path: Path,
) -> None:
    """Zapisuje raport i macierz pomyłek dla pojedynczego etapu pipeline'u."""
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).T.to_csv(
        results_path / f"classification_report_{name}.csv",
        index=True,
    )

    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    matrix_df = pd.DataFrame(
        matrix,
        index=[f"true_{label}" for label in target_names],
        columns=[f"pred_{label}" for label in target_names],
    )
    matrix_df.to_csv(
        results_path / f"confusion_matrix_{name}.csv",
        index=True,
    )


def train_autoencoder_anomaly_rf(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train_df: pd.DataFrame,
        y_test_df: pd.DataFrame,
        args: argparse.Namespace,
) -> dict[str, Any] | None:
    """
    Dwuetapowy pipeline:

    Etap 1:
        Autoenkoder trenowany tylko na klasie Normal wykrywa anomalie na
        podstawie błędu rekonstrukcji.

    Etap 2:
        Random Forest klasyfikuje typ znanego ataku. Jako dodatkową cechę
        otrzymuje błąd rekonstrukcji autoenkodera.

    Klasa Unknown nie występuje w treningu i jest używana do oceny zero-day.
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    except ImportError:
        warnings.warn(
            "TensorFlow nie jest zainstalowany. "
            "Pomijam Autoenkoder(anomaly) + Random Forest."
        )
        return None

    required_columns = {"AttackClass", "Label_binary"}
    missing_columns = required_columns - set(y_train_df.columns)
    missing_columns |= required_columns - set(y_test_df.columns)
    if missing_columns:
        raise ValueError(
            f"Brak kolumn etykiet wymaganych przez AE + RF: "
            f"{sorted(missing_columns)}"
        )

    if not 0 < args.ae_threshold_percentile <= 100:
        raise ValueError("--ae-threshold-percentile musi należeć do (0, 100].")

    tf.random.set_seed(RANDOM_STATE)

    y_train_multi = y_train_df["AttackClass"].astype(str).to_numpy()
    y_test_multi = y_test_df["AttackClass"].astype(str).to_numpy()
    y_test_binary = y_test_df["Label_binary"].astype(np.int32).to_numpy()

    normal_train_mask = y_train_multi == "Normal"
    known_attack_train_mask = (
        (y_train_multi != "Normal")
        & (y_train_multi != "Unknown")
    )

    if normal_train_mask.sum() < 2:
        raise ValueError(
            "Za mało próbek Normal do treningu i walidacji autoenkodera."
        )
    if known_attack_train_mask.sum() == 0:
        raise ValueError(
            "Brak znanych ataków do treningu drugiego etapu Random Forest."
        )

    X_train_normal = X_train.loc[normal_train_mask].reset_index(drop=True)

    print("\n" + "=" * 78)
    print("AUTOENKODER ANOMALII + RANDOM FOREST — PODEJŚCIE 2")
    print("=" * 78)
    print(
        f"AE: {len(X_train_normal)} próbek Normal "
        f"z {len(X_train)} wszystkich próbek treningowych"
    )
    print(
        f"RF etap 2: {int(known_attack_train_mask.sum())} "
        "próbek znanych ataków"
    )

    X_ae_train, X_ae_validation = train_test_split(
        X_train_normal,
        test_size=0.1,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    input_dimension = X_train.shape[1]
    latent_dimension = (
        args.ae_latent_dim
        if args.ae_latent_dim > 0
        else min(16, max(4, input_dimension // 4))
    )

    print(f"Liczba cech wejściowych: {input_dimension}")
    print(f"Wymiar latentny:         {latent_dimension}")

    autoencoder = build_anomaly_autoencoder(
        input_dimension=input_dimension,
        latent_dimension=latent_dimension,
    )

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=7,
            restore_best_weights=True,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
        ),
    ]

    ae_start = time.perf_counter()
    history = autoencoder.fit(
        X_ae_train.to_numpy(dtype=np.float32, copy=False),
        X_ae_train.to_numpy(dtype=np.float32, copy=False),
        validation_data=(
            X_ae_validation.to_numpy(dtype=np.float32, copy=False),
            X_ae_validation.to_numpy(dtype=np.float32, copy=False),
        ),
        epochs=args.ae_epochs,
        batch_size=args.ae_batch_size,
        shuffle=True,
        callbacks=callbacks,
        verbose=1,
    )
    ae_training_time = time.perf_counter() - ae_start

    pd.DataFrame(history.history).to_csv(
        args.results_path / "autoencoder_anomaly_training_history.csv",
        index=False,
    )
    autoencoder.save(args.models_path / "autoencoder_anomaly.keras")

    validation_errors = reconstruction_error(
        autoencoder,
        X_ae_validation,
        batch_size=max(args.ae_batch_size, 4096),
    )
    threshold = float(
        np.percentile(
            validation_errors,
            args.ae_threshold_percentile,
        )
    )

    print(
        f"Próg anomalii ({args.ae_threshold_percentile} percentyl "
        f"walidacji Normal): {threshold:.8f}"
    )

    train_errors = reconstruction_error(
        autoencoder,
        X_train,
        batch_size=max(args.ae_batch_size, 4096),
    )

    prediction_start = time.perf_counter()
    test_errors = reconstruction_error(
        autoencoder,
        X_test,
        batch_size=max(args.ae_batch_size, 4096),
    )
    stage1_predictions = (test_errors > threshold).astype(np.int32)

    stage1_accuracy = accuracy_score(y_test_binary, stage1_predictions)
    stage1_balanced_accuracy = balanced_accuracy_score(
        y_test_binary,
        stage1_predictions,
    )
    stage1_macro_f1 = f1_score(
        y_test_binary,
        stage1_predictions,
        average="macro",
        zero_division=0,
    )
    stage1_weighted_f1 = f1_score(
        y_test_binary,
        stage1_predictions,
        average="weighted",
        zero_division=0,
    )

    try:
        stage1_auc = roc_auc_score(y_test_binary, test_errors)
    except ValueError:
        stage1_auc = np.nan

    print("\n=== ETAP 1: wykrywanie anomalii na pełnym teście ===")
    print(
        classification_report(
            y_test_binary,
            stage1_predictions,
            labels=[0, 1],
            target_names=["Normal", "Attack"],
            zero_division=0,
        )
    )
    print(f"ROC-AUC błędu rekonstrukcji: {stage1_auc:.6f}")

    save_stage_report(
        name="ae_stage1_binary",
        y_true=y_test_binary,
        y_pred=stage1_predictions,
        labels=[0, 1],
        target_names=["Normal", "Attack"],
        results_path=args.results_path,
    )

    unknown_mask = y_test_multi == "Unknown"
    unknown_detection_rate = np.nan
    if unknown_mask.any():
        unknown_detection_rate = float(
            stage1_predictions[unknown_mask].mean()
        )
        print(
            "Wykrywanie Unknown jako anomalia: "
            f"{unknown_detection_rate:.6f} "
            f"({int(unknown_mask.sum())} próbek)"
        )

    X_train_array = X_train.to_numpy(dtype=np.float32, copy=False)
    X_test_array = X_test.to_numpy(dtype=np.float32, copy=False)

    X_train_rf_all = np.column_stack([X_train_array, train_errors])
    X_test_rf_all = np.column_stack([X_test_array, test_errors])

    X_train_rf = X_train_rf_all[known_attack_train_mask]
    y_train_rf = y_train_multi[known_attack_train_mask]

    rf_model = RandomForestClassifier(
        n_estimators=args.rf_trees,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    rf_start = time.perf_counter()
    rf_model.fit(X_train_rf, y_train_rf)
    rf_training_time = time.perf_counter() - rf_start

    flagged_mask = stage1_predictions == 1
    full_predictions = np.full(
        shape=len(X_test),
        fill_value="Normal",
        dtype=object,
    )

    if flagged_mask.any():
        flagged_predictions = rf_model.predict(
            X_test_rf_all[flagged_mask]
        )
        full_predictions[flagged_mask] = flagged_predictions
    else:
        flagged_predictions = np.array([], dtype=object)

    prediction_time = time.perf_counter() - prediction_start

    known_attack_flagged_mask = (
        flagged_mask
        & (y_test_multi != "Normal")
        & (y_test_multi != "Unknown")
    )

    print(
        "\n=== ETAP 2: klasyfikacja typu znanego ataku "
        f"(zaflagowane znane ataki: "
        f"{int(known_attack_flagged_mask.sum())}) ==="
    )

    if known_attack_flagged_mask.any():
        stage2_true = y_test_multi[known_attack_flagged_mask]
        stage2_pred = full_predictions[known_attack_flagged_mask]
        print(
            classification_report(
                stage2_true,
                stage2_pred,
                zero_division=0,
            )
        )
        save_classification_outputs(
            model_name="AE stage2 RF attack type",
            y_true=pd.Series(stage2_true),
            predictions=np.asarray(stage2_pred),
            results_path=args.results_path,
        )
    else:
        print("Brak zaflagowanych znanych ataków do oceny etapu 2.")

    known_test_mask = y_test_multi != "Unknown"
    known_true = y_test_multi[known_test_mask]
    known_predictions = full_predictions[known_test_mask]

    final_accuracy = accuracy_score(known_true, known_predictions)
    final_balanced_accuracy = balanced_accuracy_score(
        known_true,
        known_predictions,
    )
    final_macro_f1 = f1_score(
        known_true,
        known_predictions,
        average="macro",
        zero_division=0,
    )
    final_weighted_f1 = f1_score(
        known_true,
        known_predictions,
        average="weighted",
        zero_division=0,
    )

    print("\n=== PEŁNY PIPELINE: znane klasy ===")
    print(
        classification_report(
            known_true,
            known_predictions,
            zero_division=0,
        )
    )

    save_classification_outputs(
        model_name="Autoencoder anomaly plus Random Forest",
        y_true=pd.Series(known_true),
        predictions=np.asarray(known_predictions),
        results_path=args.results_path,
    )

    joblib.dump(
        rf_model,
        args.models_path / "autoencoder_anomaly_plus_random_forest.joblib",
    )

    metadata = {
        "pipeline": "autoencoder_anomaly_plus_random_forest_two_stage",
        "threshold": threshold,
        "threshold_percentile": args.ae_threshold_percentile,
        "latent_dimension": latent_dimension,
        "input_features": X_train.columns.tolist(),
        "rf_extra_feature": "reconstruction_error",
        "rf_classes": [str(value) for value in rf_model.classes_],
        "random_state": RANDOM_STATE,
    }
    with open(
        args.models_path / "autoencoder_anomaly_metadata.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)

    if args.target == "Label_binary":
        primary_accuracy = stage1_accuracy
        primary_balanced_accuracy = stage1_balanced_accuracy
        primary_macro_f1 = stage1_macro_f1
        primary_weighted_f1 = stage1_weighted_f1
        primary_test_samples = len(X_test)
    else:
        primary_accuracy = final_accuracy
        primary_balanced_accuracy = final_balanced_accuracy
        primary_macro_f1 = final_macro_f1
        primary_weighted_f1 = final_weighted_f1
        primary_test_samples = int(known_test_mask.sum())

    result: dict[str, Any] = {
        "model": "Autoencoder(anomaly) + Random Forest (2-stage)",
        "target": args.target,
        "train_samples": len(X_train),
        "test_samples": primary_test_samples,
        "accuracy": primary_accuracy,
        "balanced_accuracy": primary_balanced_accuracy,
        "macro_f1": primary_macro_f1,
        "weighted_f1": primary_weighted_f1,
        "training_time_seconds": ae_training_time + rf_training_time,
        "prediction_time_seconds": prediction_time,
        "stage1_accuracy": stage1_accuracy,
        "stage1_balanced_accuracy": stage1_balanced_accuracy,
        "stage1_macro_f1": stage1_macro_f1,
        "stage1_weighted_f1": stage1_weighted_f1,
        "auc_stage1_anomaly": stage1_auc,
        "unknown_attack_detection_rate": unknown_detection_rate,
        "final_multiclass_accuracy": final_accuracy,
        "final_multiclass_balanced_accuracy": final_balanced_accuracy,
        "final_multiclass_macro_f1": final_macro_f1,
        "final_multiclass_weighted_f1": final_weighted_f1,
        "anomaly_threshold": threshold,
        "threshold_percentile": args.ae_threshold_percentile,
        "latent_dimension": latent_dimension,
        "autoencoder_training_time_seconds": ae_training_time,
        "random_forest_training_time_seconds": rf_training_time,
        "flagged_test_samples": int(flagged_mask.sum()),
    }

    print("\n=== PODSUMOWANIE AE(anomaly) + RF ===")
    for key, value in result.items():
        print(f"  {key}: {value}")

    return result


def save_configuration(args: argparse.Namespace) -> None:
    configuration = {
        "target": args.target,
        "processed_path": str(args.processed_path),
        "models_path": str(args.models_path),
        "results_path": str(args.results_path),
        "random_state": RANDOM_STATE,
        "rf_trees": args.rf_trees,
        "mlp_max_iter": args.mlp_max_iter,
        "knn_max_train": args.knn_max_train,
        "knn_neighbors": args.knn_neighbors,
        "ae_epochs": args.ae_epochs,
        "ae_batch_size": args.ae_batch_size,
        "ae_latent_dim": args.ae_latent_dim,
        "ae_threshold_percentile": args.ae_threshold_percentile,
    }

    with open(
            args.results_path / "experiment_config.json",
            "w",
            encoding="utf-8",
    ) as file:
        json.dump(configuration, file, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    set_random_seeds(RANDOM_STATE)

    args.models_path.mkdir(parents=True, exist_ok=True)
    args.results_path.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("TRENING MODELI CICIDS2017")
    print("=" * 78)
    print(f"Target: {args.target}")
    print("Autoenkoder: wyłącznie podejście 2 (anomaly detection + RF)")

    save_configuration(args)

    X_train_raw, X_test_raw, y_train_df, y_test_df = load_data(args.processed_path)

    (
        X_train,
        y_train,
        X_eval,
        y_eval,
        _X_unknown,
        unknown_mask,
    ) = prepare_target(
        X_train=X_train_raw,
        X_test=X_test_raw,
        y_train_df=y_train_df,
        y_test_df=y_test_df,
        target=args.target,
    )

    results: list[dict[str, Any]] = []

    if not args.skip_rf:
        results.append(
            train_random_forest(X_train, y_train, X_eval, y_eval, args, unknown_mask)
        )

    if not args.skip_mlp:
        results.append(
            train_mlp(X_train, y_train, X_eval, y_eval, args, unknown_mask)
        )

    if not args.skip_svm:
        results.append(
            train_linear_svm(X_train, y_train, X_eval, y_eval, args, unknown_mask)
        )

    if not args.skip_knn:
        results.append(
            train_knn(X_train, y_train, X_eval, y_eval, args, unknown_mask)
        )

    if not args.skip_autoencoder:
        ae_result = train_autoencoder_anomaly_rf(
            X_train=X_train_raw,
            X_test=X_test_raw,
            y_train_df=y_train_df,
            y_test_df=y_test_df,
            args=args,
        )
        if ae_result is not None:
            results.append(ae_result)

    if not results:
        raise RuntimeError("Wszystkie modele zostały pominięte.")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        by="macro_f1",
        ascending=False,
    ).reset_index(drop=True)

    comparison_path = args.results_path / f"model_comparison_{args.target}.csv"
    results_df.to_csv(comparison_path, index=False)

    print("\n" + "=" * 78)
    print("PORÓWNANIE MODELI")
    print("=" * 78)
    display_columns = [
        column
        for column in [
            "model",
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "weighted_f1",
            "auc_stage1_anomaly",
            "unknown_attack_detection_rate",
            "training_time_seconds",
            "prediction_time_seconds",
        ]
        if column in results_df.columns
    ]
    print(results_df[display_columns].to_string(index=False))
    print(f"\nWyniki zapisano w: {comparison_path}")
    print(f"Modele zapisano w: {args.models_path}")


if __name__ == "__main__":
    main()