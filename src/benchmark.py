from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd


def _first_rows(X: Any, count: int) -> Any:
    if isinstance(X, pd.DataFrame):
        return X.iloc[:count]
    if isinstance(X, pd.Series):
        return X.iloc[:count]
    return np.asarray(X)[:count]


def benchmark_predictor(
    *,
    model_name: str,
    target: str,
    predict_function: Callable[[Any], Any],
    X: Any,
    batch_sizes: Sequence[int] = (1, 32, 128, 1024),
    repeats: int = 10,
    warmup_runs: int = 3,
    scope: str = "prediction_on_preprocessed_flow_features",
) -> list[dict[str, Any]]:
    sample_count = len(X)
    if sample_count == 0:
        raise ValueError("Zbiór użyty do benchmarku jest pusty.")
    if repeats <= 0:
        raise ValueError("repeats musi być większe od 0.")
    if warmup_runs < 0:
        raise ValueError("warmup_runs nie może być ujemne.")

    normalized_sizes: list[int] = []
    for requested_size in batch_sizes:
        if requested_size <= 0:
            raise ValueError("Wielkości batchy muszą być większe od 0.")
        actual_size = min(int(requested_size), sample_count)
        if actual_size not in normalized_sizes:
            normalized_sizes.append(actual_size)

    rows: list[dict[str, Any]] = []
    for batch_size in normalized_sizes:
        batch = _first_rows(X, batch_size)

        for _ in range(warmup_runs):
            predict_function(batch)

        durations: list[float] = []
        for _ in range(repeats):
            started = time.perf_counter()
            predict_function(batch)
            durations.append(time.perf_counter() - started)

        values = np.asarray(durations, dtype=np.float64)
        total_time = float(values.sum())
        total_predictions = batch_size * repeats

        rows.append(
            {
                "model": model_name,
                "target": target,
                "benchmark_scope": scope,
                "batch_size": batch_size,
                "repeats": repeats,
                "warmup_runs": warmup_runs,
                "mean_batch_latency_ms": float(values.mean() * 1000.0),
                "p50_batch_latency_ms": float(
                    np.percentile(values, 50) * 1000.0
                ),
                "p95_batch_latency_ms": float(
                    np.percentile(values, 95) * 1000.0
                ),
                "p99_batch_latency_ms": float(
                    np.percentile(values, 99) * 1000.0
                ),
                "mean_sample_latency_ms": float(
                    values.mean() / batch_size * 1000.0
                ),
                "throughput_flows_per_second": (
                    float(total_predictions / total_time)
                    if total_time > 0.0
                    else np.nan
                ),
            }
        )

    return rows
