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
    precision_score,
    recall_score,
    roc_auc_score,
)

from .ae import anomaly_threshold
from .io_utils import safe_name


def get_binary_scores(
        model: Any,
        X: Any,
        *,
        positive_label: int = 1,
) -> np.ndarray | None:
    classes = np.asarray(getattr(model, "classes_", []))

    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(X))
        if probabilities.ndim != 2 or probabilities.shape[1] < 2:
            return None

        matches = np.flatnonzero(classes == positive_label)
        if len(matches) != 1:
            return None
        return probabilities[:, int(matches[0])].astype(np.float64, copy=False)

    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X))

        if scores.ndim == 1:
            if len(classes) != 2:
                return None
            if classes[1] == positive_label:
                return scores.astype(np.float64, copy=False)
            if classes[0] == positive_label:
                return (-scores).astype(np.float64, copy=False)
            return None

        if scores.ndim == 2:
            matches = np.flatnonzero(classes == positive_label)
            if len(matches) != 1:
                return None
            return scores[:, int(matches[0])].astype(np.float64, copy=False)

    return None


def false_alarm_metrics(
        y_true: Any,
        y_pred: Any,
        *,
        target: str,
) -> dict[str, float | int]:
    true_values = np.asarray(y_true)
    predicted_values = np.asarray(y_pred)

    if len(true_values) != len(predicted_values):
        raise ValueError("y_true i y_pred mają różną liczbę elementów.")

    normal_label: int | str = 0 if target == "Label_binary" else "Normal"
    normal_mask = true_values == normal_label
    normal_count = int(normal_mask.sum())

    if normal_count == 0:
        return {
            "normal_samples": 0,
            "false_alarm_count": 0,
            "false_alarm_rate": np.nan,
        }

    false_alarm_count = int(np.sum(predicted_values[normal_mask] != normal_label))
    return {
        "normal_samples": normal_count,
        "false_alarm_count": false_alarm_count,
        "false_alarm_rate": float(false_alarm_count / normal_count),
    }


def evaluate_classification(
        y_true: Any,
        y_pred: Any,
        *,
        target: str,
        binary_scores: np.ndarray | None = None,
) -> dict[str, float | int]:
    true_values = np.asarray(y_true)
    predicted_values = np.asarray(y_pred)

    result: dict[str, float | int] = {
        "accuracy": float(accuracy_score(true_values, predicted_values)),
        "balanced_accuracy": float(
            balanced_accuracy_score(true_values, predicted_values)
        ),
        "macro_precision": float(
            precision_score(
                true_values,
                predicted_values,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                true_values,
                predicted_values,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                true_values,
                predicted_values,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                true_values,
                predicted_values,
                average="weighted",
                zero_division=0,
            )
        ),
        "roc_auc": np.nan,
    }
    result.update(
        false_alarm_metrics(
            true_values,
            predicted_values,
            target=target,
        )
    )

    if target == "Label_binary":
        result.update(
            {
                "attack_precision": float(
                    precision_score(
                        true_values,
                        predicted_values,
                        pos_label=1,
                        average="binary",
                        zero_division=0,
                    )
                ),
                "attack_recall": float(
                    recall_score(
                        true_values,
                        predicted_values,
                        pos_label=1,
                        average="binary",
                        zero_division=0,
                    )
                ),
                "attack_f1": float(
                    f1_score(
                        true_values,
                        predicted_values,
                        pos_label=1,
                        average="binary",
                        zero_division=0,
                    )
                ),
            }
        )

        matrix = confusion_matrix(true_values, predicted_values, labels=[0, 1])
        tn, fp, fn, tp = matrix.ravel()
        result.update(
            {
                "true_negative_count": int(tn),
                "false_positive_count": int(fp),
                "false_negative_count": int(fn),
                "true_positive_count": int(tp),
            }
        )

        if binary_scores is not None:
            scores = np.asarray(binary_scores, dtype=np.float64)
            if len(scores) != len(true_values):
                raise ValueError(
                    "binary_scores i y_true mają różną liczbę elementów."
                )
            try:
                result["roc_auc"] = float(roc_auc_score(true_values, scores))
            except ValueError:
                result["roc_auc"] = np.nan

    return result


def classification_metrics(
        y_true: Any,
        y_pred: Any,
        *,
        target: str,
        binary_scores: np.ndarray | None = None,
) -> dict[str, float | int]:
    return evaluate_classification(
        y_true,
        y_pred,
        target=target,
        binary_scores=binary_scores,
    )


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
