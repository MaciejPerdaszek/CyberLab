from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np


def set_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def count_finite_combinations(
    distributions: dict[str, list[Any]] | list[dict[str, list[Any]]],
) -> int | float:
    groups = distributions if isinstance(distributions, list) else [distributions]
    total = 0
    for group in groups:
        combinations = 1
        for values in group.values():
            try:
                combinations *= len(values)
            except TypeError:
                return math.inf
        total += combinations
    return total


def make_json_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): make_json_serializable(item) for key, item in value.items()}
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(make_json_serializable(data), file, indent=2, ensure_ascii=False)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku konfiguracji: {path}")
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Plik {path} nie zawiera obiektu JSON.")
    return data


def safe_name(name: str) -> str:
    return name.lower().replace(" ", "_").replace("+", "plus")


def model_tuning_entry(config: dict[str, Any], model_key: str) -> dict[str, Any]:
    models = config.get("models", {})
    entry = models.get(model_key)
    if not isinstance(entry, dict):
        raise KeyError(f"Brak wyników tuningu modelu '{model_key}'.")
    if entry.get("skipped"):
        raise ValueError(f"Model '{model_key}' został pominięty podczas tuningu: {entry.get('reason')}")
    return entry


def classical_best_params(config: dict[str, Any], model_key: str) -> dict[str, Any]:
    entry = model_tuning_entry(config, model_key)
    params = entry.get("best_params", {})
    if not isinstance(params, dict):
        raise ValueError(f"Niepoprawne best_params dla modelu '{model_key}'.")
    return params


def ae_best_params(config: dict[str, Any]) -> tuple[int, dict[str, Any], float]:
    entry = model_tuning_entry(config, "ae_rf")

    best = entry.get("best_params")
    if isinstance(best, dict):
        latent = int(best["latent_dimension"])
        rf_params = dict(best["rf_params"])
        percentile = float(best.get("threshold_percentile", 95.0))
        return latent, rf_params, percentile

    latent = int(entry["latent_dimension"])
    rf_params = dict(entry["rf_params"])
    auxiliary = entry.get("auxiliary_ae_metrics", {})
    percentile = float(auxiliary.get("threshold_percentile", 95.0))
    return latent, rf_params, percentile
