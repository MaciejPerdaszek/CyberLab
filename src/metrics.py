from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from src.ae import anomaly_threshold
from src.io_utils import safe_name


def classification_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def anomaly_metrics_for_threshold(
    y_binary: np.ndarray,
    errors: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    predictions = (errors > threshold).astype(np.int32)
    attack_mask = y_binary == 1
    normal_mask = y_binary == 0

    try:
        auc = float(roc_auc_score(y_binary, errors))
    except ValueError:
        auc = np.nan

    return {
        "ae_anomaly_threshold": float(threshold),
        "ae_anomaly_macro_f1": float(
            f1_score(y_binary, predictions, average="macro", zero_division=0)
        ),
        "ae_anomaly_accuracy": float(accuracy_score(y_binary, predictions)),
        "ae_anomaly_balanced_accuracy": float(
            balanced_accuracy_score(y_binary, predictions)
        ),
        "ae_anomaly_roc_auc": auc,
        "ae_attack_detection_rate": (
            float(predictions[attack_mask].mean()) if attack_mask.any() else np.nan
        ),
        "ae_normal_false_positive_rate": (
            float(predictions[normal_mask].mean()) if normal_mask.any() else np.nan
        ),
        "ae_flagged_samples": int(predictions.sum()),
    }


def choose_best_anomaly_threshold(
    normal_validation_errors: np.ndarray,
    validation_errors: np.ndarray,
    y_validation_binary: np.ndarray,
    percentiles: list[float],
) -> dict[str, Any]:
    if not percentiles:
        raise ValueError("Lista percentyli nie może być pusta.")

    best_key: tuple[float, float, float] | None = None
    best: dict[str, Any] | None = None
    for percentile in percentiles:
        threshold = anomaly_threshold(normal_validation_errors, percentile)
        metrics = anomaly_metrics_for_threshold(
            y_validation_binary,
            validation_errors,
            threshold,
        )
        key = (
            float(metrics["ae_anomaly_macro_f1"]),
            float(metrics["ae_anomaly_balanced_accuracy"]),
            -float(metrics["ae_normal_false_positive_rate"]),
        )
        if best_key is None or key > best_key:
            best_key = key
            best = {"threshold_percentile": float(percentile), **metrics}

    if best is None:
        raise RuntimeError("Nie udało się dobrać progu anomalii.")
    return best


def save_classification_outputs(
    model_name: str,
    y_true: Any,
    predictions: Any,
    results_path: Path,
) -> None:
    results_path.mkdir(parents=True, exist_ok=True)
    stem = safe_name(model_name)
    true_series = pd.Series(y_true)
    pred_series = pd.Series(predictions)
    labels = sorted(set(true_series.tolist()) | set(pred_series.tolist()), key=str)

    matrix = confusion_matrix(true_series, pred_series, labels=labels)
    pd.DataFrame(
        matrix,
        index=[f"true_{label}" for label in labels],
        columns=[f"pred_{label}" for label in labels],
    ).to_csv(results_path / f"confusion_matrix_{stem}.csv", index=True)

    report = classification_report(
        true_series,
        pred_series,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).T.to_csv(
        results_path / f"classification_report_{stem}.csv",
        index=True,
    )
