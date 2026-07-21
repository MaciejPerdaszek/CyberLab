from __future__ import annotations

import argparse
import json
import math
import random
import time
import warnings
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    ParameterSampler,
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_sample_weight

RANDOM_STATE = 42
SUPPORTED_MODELS = ("rf", "mlp", "svm", "knn", "ae_rf")

RF_PARAM_DIST: dict[str, list[Any]] = {
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 20, 30, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],
    "class_weight": ["balanced_subsample", "balanced"],
}

MLP_PARAM_DIST: dict[str, list[Any]] = {
    "hidden_layer_sizes": [
        (64,),
        (128, 64),
        (64, 32),
    ],
    "activation": ["relu", "tanh"],
    "alpha": [0.0001, 0.001, 0.01, 0.05],
    "learning_rate_init": [0.0005, 0.001, 0.005],
    "batch_size": [256, 512, 1024],
}

SVM_PARAM_DIST: list[dict[str, list[Any]]] = [
    {
        "C": [0.001, 0.01, 0.1, 1.0, 10.0],
        "loss": ["squared_hinge"],
        "dual": [False],
        "tol": [1e-3, 1e-4],
    },
    {
        "C": [0.001, 0.01, 0.1, 1.0, 10.0],
        "loss": ["hinge"],
        "dual": [True],
        "tol": [1e-3, 1e-4],
    },
]

KNN_PARAM_DIST: list[dict[str, list[Any]]] = [
    {
        "n_neighbors": [5, 7, 9, 11, 15, 21],
        "weights": ["distance", "uniform"],
        "p": [1, 2],
        "algorithm": ["brute"],
    },
    {
        "n_neighbors": [5, 7, 9, 11, 15, 21],
        "weights": ["distance", "uniform"],
        "p": [1, 2],
        "algorithm": ["ball_tree", "kd_tree"],
        "leaf_size": [20, 30, 50],
    },
]

AE_RF_PARAM_DIST: dict[str, list[Any]] = {
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 20, 30, None],
    "min_samples_split": [2, 3, 5],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt", "log2"],
    "class_weight": ["balanced_subsample", "balanced"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tuning RF, MLP, Linear SVM, KNN, AE + RF."
        )
    )

    parser.add_argument(
        "--processed-path",
        type=Path,
        default=Path("../data/processed"),
        help="Katalog zawierający X_train_scaled.csv i y_train.csv.",
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=Path("../results/tuning"),
        help="Katalog zapisu wyników tuningu.",
    )
    parser.add_argument(
        "--target",
        choices=["AttackClass", "Label_binary"],
        default="AttackClass",
        help=(
            "Target klasycznych modeli. Dla ae_rf wpływa na automatyczny wybór "
            "metryki: AttackClass -> pełny pipeline, Label_binary -> etap 1."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["all", *SUPPORTED_MODELS],
        default=["all"],
        help="Modele do strojenia.",
    )

    parser.add_argument(
        "--subsample",
        type=int,
        default=50_000,
        help="Podpróbka do tuningu klasycznych modeli; 0 = pełny train.",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=20,
        help="Maksymalna liczba konfiguracji na model klasyczny.",
    )
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--scoring", type=str, default="f1_macro")
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Równoległe zadania RandomizedSearchCV.",
    )
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument(
        "--mlp-max-iter",
        type=int,
        default=60,
        help="Maksymalna liczba iteracji MLP podczas tuningu.",
    )
    parser.add_argument(
        "--mlp-balanced-sample-weight",
        action="store_true",
        default=True,
        help=(
            "MLPClassifier nie wspiera class_weight, więc domyślnie "
            "przekazujemy sample_weight='balanced' liczony ręcznie, "
            "aby częściowo zrekompensować niezbalansowane klasy."
        ),
    )
    parser.add_argument(
        "--no-mlp-balanced-sample-weight",
        dest="mlp_balanced_sample_weight",
        action="store_false",
        help="Wyłącza sample_weight='balanced' dla MLP.",
    )

    parser.add_argument(
        "--ae-subsample",
        type=int,
        default=100_000,
        help="Podpróbka do tuningu AE+RF; 0 = pełny train.",
    )
    parser.add_argument("--ae-epochs", type=int, default=30)
    parser.add_argument("--ae-batch-size", type=int, default=2048)
    parser.add_argument(
        "--ae-latent-dims",
        nargs="+",
        type=int,
        default=[8, 16, 32],
        help="Sprawdzane wymiary warstwy latentnej.",
    )
    parser.add_argument(
        "--ae-threshold-percentiles",
        nargs="+",
        type=float,
        default=[90.0, 95.0, 99.0],
        help="Percentyle błędu rekonstrukcji Normal używane jako progi.",
    )
    parser.add_argument(
        "--ae-rf-iter",
        type=int,
        default=10,
        help="Liczba losowanych konfiguracji RF drugiego etapu.",
    )
    parser.add_argument(
        "--ae-pipeline-validation-size",
        type=float,
        default=0.2,
        help="Część podpróbki AE przeznaczona do oceny całego pipeline'u.",
    )
    parser.add_argument(
        "--ae-normal-validation-size",
        type=float,
        default=0.1,
        help=(
            "Część treningowych próbek Normal przeznaczona wyłącznie "
            "do early stopping i wyznaczenia progu."
        ),
    )
    parser.add_argument(
        "--ae-selection-metric",
        choices=["auto", "pipeline_macro_f1", "stage1_macro_f1"],
        default="auto",
        help=(
            "Metryka wyboru konfiguracji. auto: pipeline_macro_f1 dla "
            "AttackClass, stage1_macro_f1 dla Label_binary."
        ),
    )

    return parser.parse_args()


def set_random_seeds(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)


def expand_models(selected: Iterable[str]) -> list[str]:
    selected_list = list(selected)
    if "all" in selected_list:
        return list(SUPPORTED_MODELS)
    return list(dict.fromkeys(selected_list))


def validate_fraction(value: float, name: str) -> None:
    if not 0.0 < value < 1.0:
        raise ValueError(f"{name} musi należeć do przedziału (0, 1).")


def load_training_data(
        processed_path: Path,
        target: str,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    x_path = processed_path / "X_train_scaled.csv"
    y_path = processed_path / "y_train.csv"

    missing = [str(path) for path in (x_path, y_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Brakuje wymaganych plików:\n- " + "\n- ".join(missing)
        )

    print("Wczytywanie WYŁĄCZNIE danych treningowych...")
    X_train = pd.read_csv(x_path, dtype=np.float32)
    y_train_df = pd.read_csv(y_path)

    if len(X_train) != len(y_train_df):
        raise ValueError(
            f"Niezgodna liczba wierszy: X_train={len(X_train)}, "
            f"y_train={len(y_train_df)}."
        )

    required_columns = {"AttackClass", "Label_binary", target}
    missing_columns = required_columns - set(y_train_df.columns)
    if missing_columns:
        raise ValueError(
            f"Brak wymaganych kolumn etykiet: {sorted(missing_columns)}"
        )

    if X_train.isna().any().any():
        raise ValueError("X_train zawiera wartości NaN.")

    if not np.isfinite(X_train.to_numpy(copy=False)).all():
        raise ValueError("X_train zawiera wartości Inf lub -Inf.")

    if y_train_df[list(required_columns)].isna().any().any():
        raise ValueError("Plik y_train zawiera brakujące etykiety.")

    if (y_train_df["AttackClass"].astype(str) == "Unknown").any():
        raise ValueError(
            "y_train zawiera klasę Unknown. Podejście 2 zakłada, że Unknown "
            "występuje wyłącznie w prawdziwym zbiorze testowym."
        )

    if target == "AttackClass":
        y_train = y_train_df[target].astype(str)
    else:
        y_train = y_train_df[target].astype(np.int32)

    print(f"X_train: {X_train.shape}")
    print("\nRozkład AttackClass:")
    print(y_train_df["AttackClass"].astype(str).value_counts())

    return (
        X_train.reset_index(drop=True),
        y_train.reset_index(drop=True),
        y_train_df.reset_index(drop=True),
    )


def encode_labels_if_needed(
        y_train: pd.Series,
) -> tuple[pd.Series, LabelEncoder | None]:
    if y_train.dtype == object or y_train.dtype.kind in ("U", "S"):
        encoder = LabelEncoder()
        encoded = encoder.fit_transform(y_train)
        print(
            "Zakodowano etykiety tekstowe na liczby całkowite "
            "(wymagane m.in. przez MLPClassifier + early_stopping):"
        )
        for code, name in enumerate(encoder.classes_):
            print(f"  {code} -> {name}")
        return (
            pd.Series(encoded, name=y_train.name).reset_index(drop=True),
            encoder,
        )
    return y_train.reset_index(drop=True), None


def stratified_subsample(
        X: pd.DataFrame,
        y: pd.Series,
        n_subsample: int,
        seed: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.Series]:
    if n_subsample <= 0 or len(X) <= n_subsample:
        return X.reset_index(drop=True), y.reset_index(drop=True)

    X_sub, _, y_sub, _ = train_test_split(
        X,
        y,
        train_size=n_subsample,
        stratify=y,
        random_state=seed,
    )

    return X_sub.reset_index(drop=True), y_sub.reset_index(drop=True)


def stratified_subsample_with_labels(
        X: pd.DataFrame,
        y_df: pd.DataFrame,
        n_subsample: int,
        seed: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if n_subsample <= 0 or len(X) <= n_subsample:
        return X.reset_index(drop=True), y_df.reset_index(drop=True)

    indices = np.arange(len(X))
    selected_indices, _ = train_test_split(
        indices,
        train_size=n_subsample,
        stratify=y_df["AttackClass"].astype(str),
        random_state=seed,
    )

    return (
        X.iloc[selected_indices].reset_index(drop=True),
        y_df.iloc[selected_indices].reset_index(drop=True),
    )


def determine_cv_folds(y: pd.Series, requested_folds: int) -> int:
    min_class_count = int(y.value_counts().min())
    folds = min(requested_folds, min_class_count)

    if folds < 2:
        raise ValueError(
            "Najrzadsza klasa ma mniej niż 2 próbki w podpróbce. "
            "Zwiększ --subsample lub sprawdź dane."
        )

    if folds != requested_folds:
        warnings.warn(
            f"Zmniejszono liczbę foldów CV z {requested_folds} do {folds}, "
            "ponieważ najrzadsza klasa ma zbyt mało próbek."
        )

    return folds


def count_finite_combinations(
        param_distributions: dict[str, list[Any]] | list[dict[str, list[Any]]],
) -> int:
    distributions = (
        param_distributions
        if isinstance(param_distributions, list)
        else [param_distributions]
    )

    total = 0
    for distribution in distributions:
        combinations = 1
        for values in distribution.values():
            try:
                combinations *= len(values)
            except TypeError:
                return math.inf
        total += combinations

    return total


def make_json_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): make_json_serializable(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [make_json_serializable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_json(data: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(
            make_json_serializable(data),
            file,
            indent=2,
            ensure_ascii=False,
        )


def tune_model(
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        base_estimator: Any,
        param_distributions: dict[str, list[Any]] | list[dict[str, list[Any]]],
        results_path: Path,
        n_subsample: int,
        n_iter: int,
        cv_folds: int,
        scoring: str,
        n_jobs: int,
        verbose: int,
        use_balanced_sample_weight: bool = False,
) -> tuple[dict[str, Any], pd.DataFrame]:
    X_sub, y_sub = stratified_subsample(
        X_train,
        y_train,
        n_subsample=n_subsample,
    )

    actual_cv_folds = determine_cv_folds(y_sub, cv_folds)
    finite_combinations = count_finite_combinations(param_distributions)

    actual_n_iter = n_iter
    if finite_combinations != math.inf:
        actual_n_iter = min(n_iter, int(finite_combinations))

    print("\n" + "=" * 78)
    print(f"TUNING: {model_name}")
    print("=" * 78)
    print(f"Podpróbka:     {len(X_sub)} z {len(X_train)} wierszy")
    print(f"CV folds:      {actual_cv_folds}")
    print(f"Konfiguracje:  {actual_n_iter}")
    print(f"Scoring:       {scoring}")

    cv = StratifiedKFold(
        n_splits=actual_cv_folds,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    fit_params: dict[str, Any] = {}
    if use_balanced_sample_weight:
        sample_weight = compute_sample_weight(
            class_weight="balanced",
            y=y_sub,
        )
        fit_params["sample_weight"] = sample_weight
        print("Użyto sample_weight='balanced' (rekompensata dla MLP).")

    search = RandomizedSearchCV(
        estimator=base_estimator,
        param_distributions=param_distributions,
        n_iter=actual_n_iter,
        scoring=scoring,
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=n_jobs,
        verbose=verbose,
        refit=False,
        return_train_score=True,
        error_score=np.nan,
    )

    start = time.perf_counter()
    search.fit(X_sub, y_sub, **fit_params)
    elapsed = time.perf_counter() - start

    if not hasattr(search, "best_params_"):
        raise RuntimeError(f"Tuning modelu {model_name} nie zwrócił wyniku.")

    results_df = pd.DataFrame(search.cv_results_)
    results_df = results_df.sort_values(
        by="rank_test_score",
        ascending=True,
    ).reset_index(drop=True)

    output_name = model_name.lower().replace(" ", "_")
    results_df.to_csv(
        results_path / f"{output_name}_tuning_results.csv",
        index=False,
    )

    print(f"\nTuning zakończony w {elapsed:.1f} s")
    print(f"Najlepszy wynik ({scoring}): {search.best_score_:.6f}")
    print(f"Najlepsze parametry: {search.best_params_}")

    best_summary = {
        "best_params": search.best_params_,
        "best_cv_score": float(search.best_score_),
        "scoring": scoring,
        "subsample_size": len(X_sub),
        "cv_folds": actual_cv_folds,
        "n_iter": actual_n_iter,
        "tuning_time_seconds": elapsed,
        "used_balanced_sample_weight": use_balanced_sample_weight,
    }

    return best_summary, results_df


def build_classical_model_specs(
        mlp_max_iter: int,
) -> dict[str, dict[str, Any]]:
    return {
        "rf": {
            "display_name": "Random Forest",
            "estimator": RandomForestClassifier(
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
            "params": RF_PARAM_DIST,
            "use_balanced_sample_weight": False,
        },
        "mlp": {
            "display_name": "MLP",
            "estimator": MLPClassifier(
                solver="adam",
                max_iter=mlp_max_iter,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=8,
                random_state=RANDOM_STATE,
                verbose=False,
            ),
            "params": MLP_PARAM_DIST,
            "use_balanced_sample_weight": True,
        },
        "svm": {
            "display_name": "Linear SVM",
            "estimator": LinearSVC(
                class_weight="balanced",
                max_iter=10_000,
                random_state=RANDOM_STATE,
            ),
            "params": SVM_PARAM_DIST,
            "use_balanced_sample_weight": False,
        },
        "knn": {
            "display_name": "KNN",
            "estimator": KNeighborsClassifier(
                metric="minkowski",
                n_jobs=1,
            ),
            "params": KNN_PARAM_DIST,
            "use_balanced_sample_weight": False,
        },
    }


def build_anomaly_autoencoder(
        input_dimension: int,
        latent_dimension: int,
):
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
    except ImportError as error:
        raise ImportError(
            "TensorFlow nie jest zainstalowany. "
            "Zainstaluj go albo pomiń model ae_rf."
        ) from error

    tf.random.set_seed(RANDOM_STATE)

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
        name=f"anomaly_autoencoder_latent_{latent_dimension}",
    )

    autoencoder.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
    )

    return autoencoder, keras


def reconstruction_error(
        model: Any,
        X: pd.DataFrame | np.ndarray,
        batch_size: int,
) -> np.ndarray:
    X_array = (
        X.to_numpy(dtype=np.float32, copy=False)
        if hasattr(X, "to_numpy")
        else np.asarray(X, dtype=np.float32)
    )
    reconstructed = model.predict(
        X_array,
        batch_size=batch_size,
        verbose=0,
    )
    return np.mean(
        np.square(X_array - reconstructed),
        axis=1,
    )


def resolve_ae_selection_metric(
        target: str,
        requested_metric: str,
) -> str:
    if requested_metric != "auto":
        return requested_metric

    if target == "Label_binary":
        return "stage1_macro_f1"

    return "pipeline_macro_f1"


def tune_autoencoder_anomaly_random_forest(
        X_train: pd.DataFrame,
        y_train_df: pd.DataFrame,
        target: str,
        results_path: Path,
        n_subsample: int,
        latent_dimensions: list[int],
        threshold_percentiles: list[float],
        ae_epochs: int,
        ae_batch_size: int,
        rf_n_iter: int,
        pipeline_validation_size: float,
        normal_validation_size: float,
        selection_metric: str,
) -> dict[str, Any]:
    validate_fraction(
        pipeline_validation_size,
        "--ae-pipeline-validation-size",
    )
    validate_fraction(
        normal_validation_size,
        "--ae-normal-validation-size",
    )

    if not latent_dimensions:
        raise ValueError("--ae-latent-dims nie może być puste.")
    if not threshold_percentiles:
        raise ValueError("--ae-threshold-percentiles nie może być puste.")

    if any(value <= 0 for value in latent_dimensions):
        raise ValueError("Każdy wymiar latentny musi być większy od 0.")

    if any(not 0 < value <= 100 for value in threshold_percentiles):
        raise ValueError("Każdy percentyl progu musi należeć do (0, 100].")

    X_sub, y_sub_df = stratified_subsample_with_labels(
        X_train,
        y_train_df,
        n_subsample=n_subsample,
    )

    all_indices = np.arange(len(X_sub))
    train2_indices, validation2_indices = train_test_split(
        all_indices,
        test_size=pipeline_validation_size,
        stratify=y_sub_df["AttackClass"].astype(str),
        random_state=RANDOM_STATE,
    )

    X_train2 = X_sub.iloc[train2_indices].reset_index(drop=True)
    y_train2_df = y_sub_df.iloc[train2_indices].reset_index(drop=True)

    X_validation2 = X_sub.iloc[validation2_indices].reset_index(drop=True)
    y_validation2_df = y_sub_df.iloc[validation2_indices].reset_index(drop=True)

    y_train2_multi = y_train2_df["AttackClass"].astype(str).to_numpy()
    y_validation2_multi = (
        y_validation2_df["AttackClass"].astype(str).to_numpy()
    )
    y_validation2_binary = (
        y_validation2_df["Label_binary"].astype(np.int32).to_numpy()
    )

    normal_train_mask = y_train2_multi == "Normal"
    known_attack_train_mask = (
            (y_train2_multi != "Normal")
            & (y_train2_multi != "Unknown")
    )

    if int(normal_train_mask.sum()) < 2:
        raise ValueError(
            "Za mało próbek Normal w wewnętrznym train2 do treningu AE."
        )

    if int(known_attack_train_mask.sum()) == 0:
        raise ValueError(
            "Brak znanych ataków w train2 do treningu RF drugiego etapu."
        )

    X_normal = X_train2.loc[normal_train_mask].reset_index(drop=True)
    X_ae_fit, X_ae_threshold_validation = train_test_split(
        X_normal,
        test_size=normal_validation_size,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    if len(X_ae_fit) == 0 or len(X_ae_threshold_validation) == 0:
        raise ValueError(
            "Podział próbek Normal dał pusty zbiór. "
            "Zwiększ --ae-subsample lub zmniejsz walidację."
        )

    rf_combination_count = count_finite_combinations(
        AE_RF_PARAM_DIST
    )
    actual_rf_n_iter = min(rf_n_iter, int(rf_combination_count))

    rf_candidates = list(
        ParameterSampler(
            AE_RF_PARAM_DIST,
            n_iter=actual_rf_n_iter,
            random_state=RANDOM_STATE,
        )
    )

    resolved_selection_metric = resolve_ae_selection_metric(
        target=target,
        requested_metric=selection_metric,
    )

    print("\n" + "=" * 78)
    print("TUNING: AUTOENKODER ANOMALII + RANDOM FOREST — PODEJŚCIE 2")
    print("=" * 78)
    print(f"Podpróbka:                 {len(X_sub)} z {len(X_train)}")
    print(f"Train2 całego pipeline'u:  {len(X_train2)}")
    print(f"Walidacja całego pipeline: {len(X_validation2)}")
    print(f"AE fit — tylko Normal:     {len(X_ae_fit)}")
    print(
        "AE próg — osobna walidacja Normal: "
        f"{len(X_ae_threshold_validation)}"
    )
    print(f"Wymiary latentne:          {latent_dimensions}")
    print(f"Percentyle progu:          {threshold_percentiles}")
    print(f"Konfiguracje RF:           {len(rf_candidates)}")
    print(f"Metryka wyboru:            {resolved_selection_metric}")

    y_train2_rf = y_train2_multi[known_attack_train_mask]

    X_train2_array = X_train2.to_numpy(dtype=np.float32, copy=False)
    X_validation2_array = X_validation2.to_numpy(
        dtype=np.float32,
        copy=False,
    )

    best_key: tuple[float, float] | None = None
    best_configuration: dict[str, Any] | None = None
    all_results: list[dict[str, Any]] = []

    total_start = time.perf_counter()

    for latent_dimension in latent_dimensions:
        print("\n" + "-" * 78)
        print(
            f"AE trenowany tylko na Normal: latent_dim={latent_dimension}"
        )
        print("-" * 78)

        autoencoder, keras = build_anomaly_autoencoder(
            input_dimension=X_train.shape[1],
            latent_dimension=latent_dimension,
        )

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=2,
                min_lr=1e-6,
            ),
        ]

        ae_start = time.perf_counter()
        history = autoencoder.fit(
            X_ae_fit.to_numpy(dtype=np.float32, copy=False),
            X_ae_fit.to_numpy(dtype=np.float32, copy=False),
            validation_data=(
                X_ae_threshold_validation.to_numpy(
                    dtype=np.float32,
                    copy=False,
                ),
                X_ae_threshold_validation.to_numpy(
                    dtype=np.float32,
                    copy=False,
                ),
            ),
            epochs=ae_epochs,
            batch_size=ae_batch_size,
            shuffle=True,
            callbacks=callbacks,
            verbose=1,
        )
        ae_training_time = time.perf_counter() - ae_start

        pd.DataFrame(history.history).to_csv(
            results_path
            / f"ae_anomaly_history_latent_{latent_dimension}.csv",
            index=False,
        )

        prediction_batch_size = max(ae_batch_size, 4096)

        normal_validation_errors = reconstruction_error(
            autoencoder,
            X_ae_threshold_validation,
            batch_size=prediction_batch_size,
        )
        train2_errors = reconstruction_error(
            autoencoder,
            X_train2,
            batch_size=prediction_batch_size,
        )
        validation2_errors = reconstruction_error(
            autoencoder,
            X_validation2,
            batch_size=prediction_batch_size,
        )

        try:
            stage1_auc = roc_auc_score(
                y_validation2_binary,
                validation2_errors,
            )
        except ValueError:
            stage1_auc = np.nan

        X_train2_rf_all = np.column_stack(
            [X_train2_array, train2_errors]
        )
        X_validation2_rf_all = np.column_stack(
            [X_validation2_array, validation2_errors]
        )

        X_train2_rf = X_train2_rf_all[known_attack_train_mask]

        for rf_index, rf_params in enumerate(rf_candidates, start=1):
            print(
                f"RF {rf_index}/{len(rf_candidates)}, "
                f"latent_dim={latent_dimension}: {rf_params}"
            )

            rf_model = RandomForestClassifier(
                **rf_params,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )

            rf_start = time.perf_counter()
            rf_model.fit(X_train2_rf, y_train2_rf)

            validation2_rf_predictions = rf_model.predict(
                X_validation2_rf_all
            )
            rf_time = time.perf_counter() - rf_start

            for threshold_percentile in threshold_percentiles:
                threshold = float(
                    np.percentile(
                        normal_validation_errors,
                        threshold_percentile,
                    )
                )

                stage1_predictions = (
                        validation2_errors > threshold
                ).astype(np.int32)

                stage1_macro_f1 = f1_score(
                    y_validation2_binary,
                    stage1_predictions,
                    average="macro",
                    zero_division=0,
                )
                stage1_accuracy = accuracy_score(
                    y_validation2_binary,
                    stage1_predictions,
                )
                stage1_balanced_accuracy = balanced_accuracy_score(
                    y_validation2_binary,
                    stage1_predictions,
                )

                attack_mask = y_validation2_binary == 1
                normal_mask = y_validation2_binary == 0

                attack_detection_rate = (
                    float(stage1_predictions[attack_mask].mean())
                    if attack_mask.any()
                    else np.nan
                )
                normal_false_positive_rate = (
                    float(stage1_predictions[normal_mask].mean())
                    if normal_mask.any()
                    else np.nan
                )

                flagged_mask = stage1_predictions == 1
                full_predictions = np.full(
                    len(X_validation2),
                    "Normal",
                    dtype=object,
                )
                full_predictions[flagged_mask] = (
                    validation2_rf_predictions[flagged_mask]
                )

                pipeline_macro_f1 = f1_score(
                    y_validation2_multi,
                    full_predictions,
                    average="macro",
                    zero_division=0,
                )
                pipeline_accuracy = accuracy_score(
                    y_validation2_multi,
                    full_predictions,
                )
                pipeline_balanced_accuracy = balanced_accuracy_score(
                    y_validation2_multi,
                    full_predictions,
                )

                flagged_known_attack_mask = (
                        flagged_mask
                        & (y_validation2_multi != "Normal")
                        & (y_validation2_multi != "Unknown")
                )
                if flagged_known_attack_mask.any():
                    stage2_macro_f1 = f1_score(
                        y_validation2_multi[flagged_known_attack_mask],
                        full_predictions[flagged_known_attack_mask],
                        average="macro",
                        zero_division=0,
                    )
                else:
                    stage2_macro_f1 = np.nan

                if resolved_selection_metric == "stage1_macro_f1":
                    selection_score = stage1_macro_f1
                else:
                    selection_score = pipeline_macro_f1

                candidate_key = (
                    float(selection_score),
                    float(pipeline_macro_f1),
                )

                row = {
                    "latent_dimension": latent_dimension,
                    "threshold_percentile": threshold_percentile,
                    "internal_anomaly_threshold": threshold,
                    "selection_metric": resolved_selection_metric,
                    "selection_score": selection_score,
                    "pipeline_macro_f1": pipeline_macro_f1,
                    "pipeline_accuracy": pipeline_accuracy,
                    "pipeline_balanced_accuracy": (
                        pipeline_balanced_accuracy
                    ),
                    "stage1_macro_f1": stage1_macro_f1,
                    "stage1_accuracy": stage1_accuracy,
                    "stage1_balanced_accuracy": (
                        stage1_balanced_accuracy
                    ),
                    "stage1_roc_auc": stage1_auc,
                    "known_attack_detection_rate": (
                        attack_detection_rate
                    ),
                    "normal_false_positive_rate": (
                        normal_false_positive_rate
                    ),
                    "stage2_macro_f1_flagged_known_attacks": (
                        stage2_macro_f1
                    ),
                    "flagged_validation_samples": int(
                        flagged_mask.sum()
                    ),
                    "ae_epochs_used": len(
                        history.history.get("loss", [])
                    ),
                    "ae_training_time_seconds": (
                        ae_training_time
                    ),
                    "rf_training_and_prediction_time_seconds": (
                        rf_time
                    ),
                    **rf_params,
                }
                all_results.append(row)

                if best_key is None or candidate_key > best_key:
                    best_key = candidate_key
                    best_configuration = {
                        "latent_dimension": latent_dimension,
                        "threshold_percentile": (
                            threshold_percentile
                        ),
                        "rf_params": rf_params,
                        "selection_metric": (
                            resolved_selection_metric
                        ),
                        "best_selection_score": (
                            selection_score
                        ),
                        "best_pipeline_macro_f1": (
                            pipeline_macro_f1
                        ),
                        "best_stage1_macro_f1": (
                            stage1_macro_f1
                        ),
                        "internal_reference_threshold": threshold,
                        "ae_epochs_used": len(
                            history.history.get("loss", [])
                        ),
                    }

        keras.backend.clear_session()

    total_time = time.perf_counter() - total_start

    if best_configuration is None:
        raise RuntimeError(
            "Nie udało się znaleźć konfiguracji podejścia 2."
        )

    results_df = pd.DataFrame(all_results).sort_values(
        by=["selection_score", "pipeline_macro_f1"],
        ascending=False,
    )
    results_df.to_csv(
        results_path / "ae_anomaly_rf_tuning_results.csv",
        index=False,
    )

    best_configuration.update(
        {
            "approach": (
                "autoencoder_trained_only_on_normal_plus_"
                "threshold_plus_stage2_random_forest"
            ),
            "subsample_size": len(X_sub),
            "pipeline_train_size": len(X_train2),
            "pipeline_validation_size": len(X_validation2),
            "normal_ae_fit_size": len(X_ae_fit),
            "normal_threshold_validation_size": len(
                X_ae_threshold_validation
            ),
            "tuning_time_seconds": total_time,
            "important_note": (
                "internal_reference_threshold nie może być użyty "
                "bezpośrednio w finalnym modelu; finalny trening "
                "musi ponownie policzyć próg z walidacji Normal."
            ),
        }
    )

    print("\nNajlepsza konfiguracja podejścia 2:")
    print(json.dumps(
        make_json_serializable(best_configuration),
        indent=2,
        ensure_ascii=False,
    ))

    return best_configuration


def main() -> None:
    args = parse_args()
    set_random_seeds()

    args.results_path.mkdir(parents=True, exist_ok=True)

    models = expand_models(args.models)
    print(f"Modele do strojenia: {models}")
    print(f"Target: {args.target}")

    X_train, y_train, y_train_df = load_training_data(
        processed_path=args.processed_path,
        target=args.target,
    )

    y_train, label_encoder = encode_labels_if_needed(y_train)

    best_params_all: dict[str, Any] = {
        "target": args.target,
        "random_state": RANDOM_STATE,
        "scoring": args.scoring,
        "autoencoder_mode": "approach_2_anomaly_only",
        "models": {},
    }

    if label_encoder is not None:
        best_params_all["label_encoding"] = {
            int(code): str(name)
            for code, name in enumerate(label_encoder.classes_)
        }

    model_specs = build_classical_model_specs(
        mlp_max_iter=args.mlp_max_iter,
    )

    for model_key in models:
        if model_key == "ae_rf":
            try:
                ae_best = tune_autoencoder_anomaly_random_forest(
                    X_train=X_train,
                    y_train_df=y_train_df,
                    target=args.target,
                    results_path=args.results_path,
                    n_subsample=args.ae_subsample,
                    latent_dimensions=args.ae_latent_dims,
                    threshold_percentiles=(
                        args.ae_threshold_percentiles
                    ),
                    ae_epochs=args.ae_epochs,
                    ae_batch_size=args.ae_batch_size,
                    rf_n_iter=args.ae_rf_iter,
                    pipeline_validation_size=(
                        args.ae_pipeline_validation_size
                    ),
                    normal_validation_size=(
                        args.ae_normal_validation_size
                    ),
                    selection_metric=args.ae_selection_metric,
                )
                best_params_all["models"]["ae_rf"] = ae_best
            except ImportError as error:
                warnings.warn(str(error))
                best_params_all["models"]["ae_rf"] = {
                    "skipped": True,
                    "reason": str(error),
                }

            save_json(
                best_params_all,
                args.results_path / "best_params_all.json",
            )
            continue

        spec = model_specs[model_key]

        use_balanced_sample_weight = (
                spec["use_balanced_sample_weight"]
                and (
                        model_key != "mlp"
                        or args.mlp_balanced_sample_weight
                )
        )

        best_summary, _ = tune_model(
            model_name=spec["display_name"],
            X_train=X_train,
            y_train=y_train,
            base_estimator=spec["estimator"],
            param_distributions=spec["params"],
            results_path=args.results_path,
            n_subsample=args.subsample,
            n_iter=args.n_iter,
            cv_folds=args.cv_folds,
            scoring=args.scoring,
            n_jobs=args.n_jobs,
            verbose=args.verbose,
            use_balanced_sample_weight=use_balanced_sample_weight,
        )

        best_params_all["models"][model_key] = best_summary

        save_json(
            best_params_all,
            args.results_path / "best_params_all.json",
        )

    final_path = args.results_path / "best_params_all.json"
    save_json(best_params_all, final_path)

    print("\n" + "=" * 78)
    print("TUNING ZAKOŃCZONY")
    print("=" * 78)
    print(f"Najlepsze parametry: {final_path}")
    print(f"Pełne wyniki CSV:    {args.results_path}")

    for model_key, model_result in best_params_all["models"].items():
        print(f"\n{model_key}:")
        print(json.dumps(
            make_json_serializable(model_result),
            indent=2,
            ensure_ascii=False,
        ))


if __name__ == "__main__":
    main()