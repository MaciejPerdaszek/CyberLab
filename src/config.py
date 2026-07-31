from __future__ import annotations

from typing import Any

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
    "hidden_layer_sizes": [(64,), (128, 64), (64, 32)],
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

AE_FEATURE_RF_PARAM_DIST: dict[str, list[Any]] = {
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 20, 30, None],
    "min_samples_split": [2, 3, 5],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt", "log2"],
    "class_weight": ["balanced_subsample", "balanced"],
}

PARAM_DISTRIBUTIONS: dict[str, Any] = {
    "rf": RF_PARAM_DIST,
    "mlp": MLP_PARAM_DIST,
    "svm": SVM_PARAM_DIST,
    "knn": KNN_PARAM_DIST,
    "ae_rf": AE_FEATURE_RF_PARAM_DIST,
}
