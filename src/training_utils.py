from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight

from .ae import (
    anomaly_threshold,
    augment_with_ae_features,
    fit_autoencoder_on_normal,
    reconstruct_and_encode,
)
from .benchmark import benchmark_predictor
from .config import RANDOM_STATE
from .data import (
    EvaluationTarget,
    TrainTestData,
    stratified_subsample,
    target_series,
    unseen_attack_mask,
)
from .io_utils import safe_name, save_json
from .metrics import (
    anomaly_metrics_for_threshold,
    evaluate_classification,
    get_binary_scores,
    save_classification_outputs,
)
from .models import build_classical_model, build_random_forest


@dataclass(frozen=True)
class TrainingRun:
    result: dict[str, Any]
    benchmark_rows: list[dict[str, Any]]


def train_classical_model(
    model_key: str,
    display_name: str,
    params: dict[str, Any],
    evaluation: EvaluationTarget,
    *,
    target: str,
    models_path: Path,
    results_path: Path,
    mlp_max_iter: int,
    knn_max_train: int,
    use_balanced_sample_weight: bool,
    benchmark_batch_sizes: Sequence[int],
    benchmark_repeats: int,
    benchmark_warmup_runs: int,
    run_benchmark: bool,
    verbose: bool = False,
) -> TrainingRun:
    X_fit = evaluation.X_train
    y_fit = evaluation.y_train
    if model_key == "knn":
        X_fit, y_fit = stratified_subsample(X_fit, y_fit, knn_max_train)

    model = build_classical_model(
        model_key,
        params,
        mlp_max_iter=mlp_max_iter,
        n_jobs=-1,
        verbose=verbose,
    )
    fit_kwargs: dict[str, Any] = {}
    if model_key == "mlp" and use_balanced_sample_weight:
        fit_kwargs["sample_weight"] = compute_sample_weight("balanced", y_fit)

    print("\n" + "=" * 78)
    print(f"MODEL: {display_name}")
    print("=" * 78)

    started = time.perf_counter()
    model.fit(X_fit, y_fit, **fit_kwargs)
    training_time = time.perf_counter() - started

    started = time.perf_counter()
    predictions = model.predict(evaluation.X_eval)
    prediction_time = time.perf_counter() - started

    binary_scores = (
        get_binary_scores(model, evaluation.X_eval)
        if target == "Label_binary"
        else None
    )
    metrics = evaluate_classification(
        evaluation.y_eval,
        predictions,
        target=target,
        binary_scores=binary_scores,
    )

    unknown_rate = np.nan
    if target == "Label_binary" and evaluation.unknown_mask_in_eval is not None:
        mask = evaluation.unknown_mask_in_eval.to_numpy(dtype=bool)
        if mask.any():
            unknown_rate = float(np.mean(np.asarray(predictions)[mask] == 1))

    output_name = f"{display_name}_{target}"
    save_classification_outputs(
        output_name,
        evaluation.y_eval,
        predictions,
        results_path,
    )
    models_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        model,
        models_path / f"{safe_name(output_name)}.joblib",
    )

    benchmark_rows: list[dict[str, Any]] = []
    if run_benchmark:
        benchmark_rows = benchmark_predictor(
            model_name=display_name,
            target=target,
            predict_function=model.predict,
            X=evaluation.X_eval,
            batch_sizes=benchmark_batch_sizes,
            repeats=benchmark_repeats,
            warmup_runs=benchmark_warmup_runs,
            scope="model_prediction_on_preprocessed_flow_features",
        )

    result = {
        "model": display_name,
        "model_key": model_key,
        "target": target,
        "train_samples": len(X_fit),
        "test_samples": len(evaluation.X_eval),
        "unseen_attack_classes": list(evaluation.unseen_attack_classes),
        "unseen_attack_samples": evaluation.unseen_attack_samples,
        **metrics,
        "training_time_seconds": training_time,
        "prediction_time_seconds": prediction_time,
        "full_test_throughput_flows_per_second": (
            float(len(evaluation.X_eval) / prediction_time)
            if prediction_time > 0.0
            else np.nan
        ),
        "unseen_attack_detection_rate": unknown_rate,
        "unknown_attack_detection_rate": unknown_rate,
        "best_params": params,
    }
    print(pd.Series(result).to_string())
    return TrainingRun(result=result, benchmark_rows=benchmark_rows)


def train_ae_feature_rf(
    data: TrainTestData,
    *,
    target: str,
    latent_dimension: int,
    rf_params: dict[str, Any],
    threshold_percentile: float,
    ae_epochs: int,
    ae_batch_size: int,
    normal_validation_size: float,
    models_path: Path,
    results_path: Path,
    benchmark_batch_sizes: Sequence[int],
    benchmark_repeats: int,
    benchmark_warmup_runs: int,
    run_benchmark: bool,
    ae_verbose: int = 1,
) -> TrainingRun:
    y_train = target_series(data.y_train, target).to_numpy()
    y_test_binary = data.y_test["Label_binary"].astype(np.int32).to_numpy()
    normal_mask = data.y_train["AttackClass"].astype(str).eq("Normal").to_numpy()
    X_normal = data.X_train.loc[normal_mask].reset_index(drop=True)

    print("\n" + "=" * 78)
    print("MODEL: AE JAKO GENERATOR CECH + RANDOM FOREST")
    print("=" * 78)

    ae_result = fit_autoencoder_on_normal(
        X_normal,
        latent_dimension=latent_dimension,
        epochs=ae_epochs,
        batch_size=ae_batch_size,
        validation_size=normal_validation_size,
        verbose=ae_verbose,
    )
    results_path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(ae_result.history.history).to_csv(
        results_path / f"ae_training_history_{target}.csv",
        index=False,
    )

    prediction_batch = max(ae_batch_size, 4096)
    normal_validation_errors, _ = reconstruct_and_encode(
        ae_result.autoencoder,
        ae_result.encoder,
        ae_result.normal_validation,
        prediction_batch,
    )
    threshold = anomaly_threshold(normal_validation_errors, threshold_percentile)

    train_features_started = time.perf_counter()
    X_train_augmented, _, _ = augment_with_ae_features(
        ae_result.autoencoder,
        ae_result.encoder,
        data.X_train,
        prediction_batch,
    )
    train_feature_time = time.perf_counter() - train_features_started

    test_features_started = time.perf_counter()
    X_test_augmented, test_errors, _ = augment_with_ae_features(
        ae_result.autoencoder,
        ae_result.encoder,
        data.X_test,
        prediction_batch,
    )
    test_feature_time = time.perf_counter() - test_features_started

    anomaly_metrics = anomaly_metrics_for_threshold(
        y_test_binary,
        test_errors,
        threshold,
    )
    anomaly_predictions = (test_errors > threshold).astype(np.int32)
    unseen_mask = unseen_attack_mask(data).to_numpy(dtype=bool)
    unseen_classes = sorted(
        data.y_test.loc[unseen_mask, "AttackClass"].astype(str).unique().tolist()
    )
    ae_unknown_detection_rate = (
        float(anomaly_predictions[unseen_mask].mean()) if unseen_mask.any() else np.nan
    )

    rf_model = build_random_forest(rf_params, n_jobs=-1)
    started = time.perf_counter()
    rf_model.fit(X_train_augmented, y_train)
    rf_training_time = time.perf_counter() - started

    if target == "AttackClass":
        train_classes = set(data.y_train["AttackClass"].astype(str).unique())
        eval_mask = (
            data.y_test["AttackClass"].astype(str).isin(train_classes).to_numpy()
        )
        y_eval = (
            data.y_test.loc[eval_mask, "AttackClass"]
            .astype(str)
            .reset_index(drop=True)
        )
    else:
        eval_mask = np.ones(len(data.X_test), dtype=bool)
        y_eval = data.y_test["Label_binary"].astype(np.int32).reset_index(drop=True)

    X_eval_augmented = X_test_augmented[eval_mask]
    X_eval_raw = data.X_test.loc[eval_mask].reset_index(drop=True)

    started = time.perf_counter()
    predictions = rf_model.predict(X_eval_augmented)
    rf_prediction_time = time.perf_counter() - started

    binary_scores = (
        get_binary_scores(rf_model, X_eval_augmented)
        if target == "Label_binary"
        else None
    )
    metrics = evaluate_classification(
        y_eval,
        predictions,
        target=target,
        binary_scores=binary_scores,
    )

    final_unknown_detection_rate = np.nan
    if target == "Label_binary" and unseen_mask.any():
        final_unknown_detection_rate = float(
            np.mean(np.asarray(predictions)[unseen_mask] == 1)
        )

    output_name = f"AE feature generator plus Random Forest_{target}"
    save_classification_outputs(
        output_name,
        y_eval,
        predictions,
        results_path,
    )

    models_path.mkdir(parents=True, exist_ok=True)
    ae_result.autoencoder.save(
        models_path / f"ae_feature_autoencoder_{target}.keras"
    )
    ae_result.encoder.save(
        models_path / f"ae_feature_encoder_{target}.keras"
    )
    joblib.dump(
        rf_model,
        models_path / f"ae_feature_random_forest_{target}.joblib",
    )
    save_json(
        {
            "pipeline": "autoencoder_normal_feature_generator_plus_rf_all_classes",
            "random_state": RANDOM_STATE,
            "target": target,
            "experiment_metadata": data.experiment_metadata,
            "latent_dimension": latent_dimension,
            "threshold_percentile": threshold_percentile,
            "anomaly_threshold": threshold,
            "original_feature_names": data.X_train.columns.tolist(),
            "original_features": data.X_train.shape[1],
            "latent_features": latent_dimension,
            "reconstruction_error_features": 1,
            "total_rf_features": X_train_augmented.shape[1],
            "rf_params": rf_params,
            "rf_classes": [str(value) for value in rf_model.classes_],
        },
        models_path / f"ae_feature_rf_metadata_{target}.json",
    )

    def full_pipeline_predict(batch: Any) -> np.ndarray:
        augmented, _, _ = augment_with_ae_features(
            ae_result.autoencoder,
            ae_result.encoder,
            batch,
            prediction_batch,
        )
        return rf_model.predict(augmented)

    benchmark_rows: list[dict[str, Any]] = []
    if run_benchmark:
        benchmark_rows = benchmark_predictor(
            model_name="AE feature generator + Random Forest",
            target=target,
            predict_function=full_pipeline_predict,
            X=X_eval_raw,
            batch_sizes=benchmark_batch_sizes,
            repeats=benchmark_repeats,
            warmup_runs=benchmark_warmup_runs,
            scope="full_ae_feature_generation_plus_rf_prediction_on_preprocessed_flows",
        )

    total_prediction_time = test_feature_time + rf_prediction_time
    result = {
        "model": "AE feature generator + Random Forest",
        "model_key": "ae_rf",
        "target": target,
        "train_samples": len(data.X_train),
        "test_samples": int(eval_mask.sum()),
        "unseen_attack_classes": unseen_classes,
        "unseen_attack_samples": int(unseen_mask.sum()),
        **metrics,
        "training_time_seconds": (
            ae_result.training_time_seconds + train_feature_time + rf_training_time
        ),
        "prediction_time_seconds": total_prediction_time,
        "full_test_throughput_flows_per_second": (
            float(int(eval_mask.sum()) / total_prediction_time)
            if total_prediction_time > 0.0
            else np.nan
        ),
        "unseen_attack_detection_rate": final_unknown_detection_rate,
        "unknown_attack_detection_rate": final_unknown_detection_rate,
        "ae_unseen_anomaly_detection_rate": ae_unknown_detection_rate,
        "ae_unknown_anomaly_detection_rate": ae_unknown_detection_rate,
        "latent_dimension": latent_dimension,
        "threshold_percentile": threshold_percentile,
        "rf_input_features": X_train_augmented.shape[1],
        "original_features": data.X_train.shape[1],
        "ae_epochs_used": len(ae_result.history.history.get("loss", [])),
        "autoencoder_training_time_seconds": ae_result.training_time_seconds,
        "ae_train_feature_generation_time_seconds": train_feature_time,
        "ae_test_feature_generation_time_seconds": test_feature_time,
        "rf_training_time_seconds": rf_training_time,
        "rf_prediction_time_seconds": rf_prediction_time,
        "rf_params": rf_params,
        **anomaly_metrics,
    }
    print(pd.Series(result).to_string())
    ae_result.keras.backend.clear_session()
    return TrainingRun(result=result, benchmark_rows=benchmark_rows)
