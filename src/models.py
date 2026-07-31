from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

from src.config import PARAM_DISTRIBUTIONS, RANDOM_STATE


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    estimator: Any
    param_distributions: Any
    use_balanced_sample_weight: bool = False


def _merged(base: dict[str, Any], params: dict[str, Any] | None) -> dict[str, Any]:
    values = dict(base)
    values.update(params or {})
    return values


def build_classical_model(
    model_key: str,
    params: dict[str, Any] | None = None,
    *,
    mlp_max_iter: int = 100,
    n_jobs: int = -1,
    verbose: bool = False,
) -> Any:
    if model_key == "rf":
        return RandomForestClassifier(
            **_merged(
                {"random_state": RANDOM_STATE, "n_jobs": n_jobs},
                params,
            )
        )
    if model_key == "mlp":
        return MLPClassifier(
            **_merged(
                {
                    "solver": "adam",
                    "max_iter": mlp_max_iter,
                    "early_stopping": True,
                    "validation_fraction": 0.1,
                    "n_iter_no_change": 8,
                    "random_state": RANDOM_STATE,
                    "verbose": verbose,
                },
                params,
            )
        )
    if model_key == "svm":
        return LinearSVC(
            **_merged(
                {
                    "class_weight": "balanced",
                    "max_iter": 10_000,
                    "random_state": RANDOM_STATE,
                },
                params,
            )
        )
    if model_key == "knn":
        return KNeighborsClassifier(
            **_merged(
                {"metric": "minkowski", "n_jobs": n_jobs},
                params,
            )
        )
    raise ValueError(f"Nieobsługiwany model klasyczny: {model_key}")


def build_random_forest(params: dict[str, Any], *, n_jobs: int = -1) -> RandomForestClassifier:
    return RandomForestClassifier(
        **_merged(
            {"random_state": RANDOM_STATE, "n_jobs": n_jobs},
            params,
        )
    )


def classical_model_specs(mlp_max_iter: int) -> dict[str, ModelSpec]:
    names = {
        "rf": "Random Forest",
        "mlp": "MLP",
        "svm": "Linear SVM",
        "knn": "KNN",
    }
    return {
        key: ModelSpec(
            key=key,
            display_name=names[key],
            estimator=build_classical_model(
                key,
                mlp_max_iter=mlp_max_iter,
                n_jobs=1,
                verbose=False,
            ),
            param_distributions=PARAM_DISTRIBUTIONS[key],
            use_balanced_sample_weight=(key == "mlp"),
        )
        for key in names
    }
