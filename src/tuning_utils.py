from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import ParameterSampler, RandomizedSearchCV, StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

from src.ae import augment_with_ae_features, fit_autoencoder_on_normal, reconstruct_and_encode
from src.config import AE_FEATURE_RF_PARAM_DIST, RANDOM_STATE
from src.data import (
    determine_cv_folds,
    split_with_labels,
    stratified_subsample,
    stratified_subsample_with_labels,
    target_series,
    validate_fraction,
)
from src.io_utils import count_finite_combinations
from src.metrics import choose_best_anomaly_threshold
from src.models import ModelSpec, build_random_forest


def tune_classical_model(
    spec: ModelSpec,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    results_path: Path,
    n_subsample: int,
    n_iter: int,
    cv_folds: int,
    scoring: str,
    n_jobs: int,
    verbose: int,
    use_balanced_sample_weight: bool,
) -> dict[str, Any]:
    X_sub, y_sub = stratified_subsample(X_train, y_train, n_subsample)
    actual_cv = determine_cv_folds(y_sub, cv_folds)
    combinations = count_finite_combinations(spec.param_distributions)
    actual_n_iter = n_iter if combinations == math.inf else min(n_iter, int(combinations))

    print("\n" + "=" * 78)
    print(f"TUNING: {spec.display_name}")
    print("=" * 78)
    print(f"Podpróbka:    {len(X_sub)} z {len(X_train)}")
    print(f"CV folds:     {actual_cv}")
    print(f"Konfiguracje: {actual_n_iter}")
    print(f"Scoring:      {scoring}")

    cv = StratifiedKFold(
        n_splits=actual_cv,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    fit_params: dict[str, Any] = {}
    if use_balanced_sample_weight:
        fit_params["sample_weight"] = compute_sample_weight("balanced", y_sub)

    search = RandomizedSearchCV(
        estimator=spec.estimator,
        param_distributions=spec.param_distributions,
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

    started = time.perf_counter()
    search.fit(X_sub, y_sub, **fit_params)
    elapsed = time.perf_counter() - started
    if not hasattr(search, "best_params_"):
        raise RuntimeError(f"Tuning modelu {spec.display_name} nie zwrócił wyniku.")

    results = pd.DataFrame(search.cv_results_).sort_values(
        "rank_test_score"
    ).reset_index(drop=True)
    results_path.mkdir(parents=True, exist_ok=True)
    results.to_csv(
        results_path / f"{spec.key}_tuning_results.csv",
        index=False,
    )

    print(f"Najlepszy wynik: {search.best_score_:.6f}")
    print(f"Najlepsze parametry: {search.best_params_}")
    return {
        "model_name": spec.display_name,
        "best_params": search.best_params_,
        "tuning": {
            "selection_metric": scoring,
            "best_score": float(search.best_score_),
            "validation_method": f"StratifiedKFold({actual_cv})",
            "subsample_size": len(X_sub),
            "n_iter": actual_n_iter,
            "time_seconds": elapsed,
            "used_balanced_sample_weight": use_balanced_sample_weight,
        },
    }


def tune_ae_feature_rf(
    X_train: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    target: str,
    results_path: Path,
    n_subsample: int,
    latent_dimensions: list[int],
    threshold_percentiles: list[float],
    ae_epochs: int,
    ae_batch_size: int,
    rf_n_iter: int,
    validation_size: float,
    normal_validation_size: float,
    ae_verbose: int = 1,
) -> dict[str, Any]:
    validate_fraction(validation_size, "ae_pipeline_validation_size")
    validate_fraction(normal_validation_size, "ae_normal_validation_size")
    if not latent_dimensions or any(value <= 0 for value in latent_dimensions):
        raise ValueError("Wymiary latentne muszą być dodatnią, niepustą listą.")
    if not threshold_percentiles or any(not 0 < value <= 100 for value in threshold_percentiles):
        raise ValueError("Percentyle AE muszą należeć do (0, 100].")

    X_sub, labels_sub = stratified_subsample_with_labels(X_train, labels, n_subsample)
    X_fit, labels_fit, X_validation, labels_validation = split_with_labels(
        X_sub,
        labels_sub,
        validation_size,
    )
    y_fit = target_series(labels_fit, target).to_numpy()
    y_validation = target_series(labels_validation, target).to_numpy()
    y_validation_binary = labels_validation["Label_binary"].astype(np.int32).to_numpy()

    normal_mask = labels_fit["AttackClass"].astype(str).eq("Normal").to_numpy()
    X_normal = X_fit.loc[normal_mask].reset_index(drop=True)

    combinations = count_finite_combinations(AE_FEATURE_RF_PARAM_DIST)
    actual_rf_n_iter = min(rf_n_iter, int(combinations))
    rf_candidates = list(
        ParameterSampler(
            AE_FEATURE_RF_PARAM_DIST,
            n_iter=actual_rf_n_iter,
            random_state=RANDOM_STATE,
        )
    )

    print("\n" + "=" * 78)
    print("TUNING: AE JAKO GENERATOR CECH + RANDOM FOREST")
    print("=" * 78)
    print(f"Podpróbka:       {len(X_sub)} z {len(X_train)}")
    print(f"Train pipeline:  {len(X_fit)}")
    print(f"Walidacja:       {len(X_validation)}")
    print(f"Normal dla AE:   {len(X_normal)}")
    print(f"Latent dims:     {latent_dimensions}")
    print(f"Konfiguracje RF: {len(rf_candidates)}")

    best_key: tuple[float, float] | None = None
    best: dict[str, Any] | None = None
    rows: list[dict[str, Any]] = []
    total_started = time.perf_counter()

    for latent_dimension in latent_dimensions:
        print(f"\nAE latent_dimension={latent_dimension}")
        ae_result = fit_autoencoder_on_normal(
            X_normal,
            latent_dimension=latent_dimension,
            epochs=ae_epochs,
            batch_size=ae_batch_size,
            validation_size=normal_validation_size,
            verbose=ae_verbose,
        )
        pd.DataFrame(ae_result.history.history).to_csv(
            results_path / f"ae_history_latent_{latent_dimension}.csv",
            index=False,
        )

        prediction_batch = max(ae_batch_size, 4096)
        normal_validation_errors, _ = reconstruct_and_encode(
            ae_result.autoencoder,
            ae_result.encoder,
            ae_result.normal_validation,
            prediction_batch,
        )
        X_fit_augmented, _, _ = augment_with_ae_features(
            ae_result.autoencoder,
            ae_result.encoder,
            X_fit,
            prediction_batch,
        )
        X_validation_augmented, validation_errors, _ = augment_with_ae_features(
            ae_result.autoencoder,
            ae_result.encoder,
            X_validation,
            prediction_batch,
        )
        anomaly_metrics = choose_best_anomaly_threshold(
            normal_validation_errors,
            validation_errors,
            y_validation_binary,
            threshold_percentiles,
        )

        for index, rf_params in enumerate(rf_candidates, start=1):
            print(f"RF {index}/{len(rf_candidates)}: {rf_params}")
            model = build_random_forest(rf_params, n_jobs=-1)
            started = time.perf_counter()
            model.fit(X_fit_augmented, y_fit)
            predictions = model.predict(X_validation_augmented)
            rf_elapsed = time.perf_counter() - started

            macro_f1 = float(
                f1_score(y_validation, predictions, average="macro", zero_division=0)
            )
            balanced_accuracy = float(
                balanced_accuracy_score(y_validation, predictions)
            )
            candidate_key = (macro_f1, balanced_accuracy)
            row = {
                "latent_dimension": latent_dimension,
                "pipeline_macro_f1": macro_f1,
                "pipeline_balanced_accuracy": balanced_accuracy,
                "original_features": X_fit.shape[1],
                "latent_features": latent_dimension,
                "rf_input_features": X_fit_augmented.shape[1],
                "ae_epochs_used": len(ae_result.history.history.get("loss", [])),
                "ae_training_time_seconds": ae_result.training_time_seconds,
                "rf_training_prediction_time_seconds": rf_elapsed,
                **anomaly_metrics,
                **rf_params,
            }
            rows.append(row)

            if best_key is None or candidate_key > best_key:
                best_key = candidate_key
                best = {
                    "model_name": "AE feature generator + Random Forest",
                    "best_params": {
                        "latent_dimension": latent_dimension,
                        "rf_params": rf_params,
                        "threshold_percentile": anomaly_metrics["threshold_percentile"],
                    },
                    "tuning": {
                        "selection_metric": "f1_macro",
                        "best_score": macro_f1,
                        "balanced_accuracy": balanced_accuracy,
                        "validation_method": f"stratified holdout {1-validation_size:.2f}/{validation_size:.2f}",
                        "subsample_size": len(X_sub),
                        "train_size": len(X_fit),
                        "validation_size": len(X_validation),
                    },
                    "features": {
                        "original": X_fit.shape[1],
                        "reconstruction_error": 1,
                        "latent": latent_dimension,
                        "total": X_fit_augmented.shape[1],
                    },
                    "auxiliary_ae_metrics": anomaly_metrics,
                    "ae_epochs_used": len(ae_result.history.history.get("loss", [])),
                    "normal_ae_fit_size": ae_result.fit_samples,
                    "normal_ae_validation_size": ae_result.validation_samples,
                }

        ae_result.keras.backend.clear_session()

    if best is None:
        raise RuntimeError("Nie udało się znaleźć konfiguracji AE+RF.")

    total_elapsed = time.perf_counter() - total_started
    results_path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values(
        ["pipeline_macro_f1", "pipeline_balanced_accuracy"],
        ascending=False,
    ).to_csv(results_path / "ae_rf_tuning_results.csv", index=False)
    best["tuning"]["time_seconds"] = total_elapsed
    best["approach"] = "autoencoder_normal_feature_generator_plus_rf_all_classes"
    return best
