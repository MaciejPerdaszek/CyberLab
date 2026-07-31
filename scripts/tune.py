from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Iterable

from src.config import RANDOM_STATE, SUPPORTED_MODELS
from src.data import load_train_data, target_series
from src.io_utils import save_json, set_random_seeds
from src.models import classical_model_specs
from src.tuning_utils import (tune_ae_feature_rf, tune_classical_model)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tuning RF, MLP, Linear SVM, KNN oraz AE+RF."
    )
    parser.add_argument("--processed-path", type=Path, default=Path("data/processed"))
    parser.add_argument("--results-path", type=Path, default=Path("results/tuning"))
    parser.add_argument(
        "--target",
        choices=["AttackClass", "LabelBinary"],
        default="AttackClass",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["all", *SUPPORTED_MODELS],
        default=["all"],
    )
    parser.add_argument("--subsample", type=int, default=50_000)
    parser.add_argument("--n-iter", type=int, default=20)
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--scoring", default="f1_macro")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--mlp-max-iter", type=int, default=60)
    parser.add_argument(
        "--no-mlp-balanced-sample-weight",
        dest="mlp_balanced_sample_weight",
        action="store_false",
    )
    parser.set_defaults(mlp_balanced_sample_weight=True)

    parser.add_argument("--ae-subsample", type=int, default=100_000)
    parser.add_argument("--ae-epochs", type=int, default=100)
    parser.add_argument("--ae-batch-size", type=int, default=2048)
    parser.add_argument("--ae-latent-dims", nargs="+", type=int, default=[8, 16, 32])
    parser.add_argument(
        "--ae-threshold-percentiles",
        nargs="+",
        type=float,
        default=[90.0, 95.0, 99.0],
    )
    parser.add_argument("--ae-rf-iter", type=int, default=10)
    parser.add_argument("--ae-validation-size", type=float, default=0.2)
    parser.add_argument("--ae-normal-validation-size", type=float, default=0.1)
    parser.add_argument("--ae-verbose", type=int, default=1)
    return parser.parse_args()


def expand_models(selected: Iterable[str]) -> list[str]:
    values = list(selected)
    if "all" in values:
        return list(SUPPORTED_MODELS)
    return list(dict.fromkeys(values))


def main() -> None:
    args = parse_args()
    set_random_seeds(RANDOM_STATE)
    args.results_path.mkdir(parents=True, exist_ok=True)

    train_data = load_train_data(args.processed_path)
    y_train = target_series(train_data.labels, args.target)
    models = expand_models(args.models)
    specs = classical_model_specs(args.mlp_max_iter)

    summary = {
        "target": args.target,
        "random_state": RANDOM_STATE,
        "scoring": args.scoring,
        "pipeline_version": "ae_feature_generator_plus_rf_all_classes",
        "models": {},
    }
    output_path = args.results_path / "best_params_all.json"

    for model_key in models:
        if model_key == "ae_rf":
            try:
                result = tune_ae_feature_rf(
                    train_data.X,
                    train_data.labels,
                    target=args.target,
                    results_path=args.results_path,
                    n_subsample=args.ae_subsample,
                    latent_dimensions=args.ae_latent_dims,
                    threshold_percentiles=args.ae_threshold_percentiles,
                    ae_epochs=args.ae_epochs,
                    ae_batch_size=args.ae_batch_size,
                    rf_n_iter=args.ae_rf_iter,
                    validation_size=args.ae_validation_size,
                    normal_validation_size=args.ae_normal_validation_size,
                    ae_verbose=args.ae_verbose,
                )
            except ImportError as error:
                warnings.warn(str(error))
                result = {"skipped": True, "reason": str(error)}
        else:
            spec = specs[model_key]
            use_weight = spec.use_balanced_sample_weight
            if model_key == "mlp" and not args.mlp_balanced_sample_weight:
                use_weight = False
            result = tune_classical_model(
                spec,
                train_data.X,
                y_train,
                results_path=args.results_path,
                n_subsample=args.subsample,
                n_iter=args.n_iter,
                cv_folds=args.cv_folds,
                scoring=args.scoring,
                n_jobs=args.n_jobs,
                verbose=args.verbose,
                use_balanced_sample_weight=use_weight,
            )

        summary["models"][model_key] = result
        save_json(summary, output_path)

    print(f"\nNajlepsze parametry zapisano w: {output_path}")


if __name__ == "__main__":
    main()