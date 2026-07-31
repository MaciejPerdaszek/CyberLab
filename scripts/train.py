from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.config import RANDOM_STATE, SUPPORTED_MODELS
from src.data import load_train_test_data, prepare_evaluation_target
from src.io_utils import (
    ae_best_params,
    classical_best_params,
    load_json,
    model_tuning_entry,
    save_json,
    set_random_seeds,
)
from src.training_utils import TrainingRun, train_ae_feature_rf, train_classical_model

DISPLAY_NAMES = {
    "rf": "Random Forest",
    "mlp": "MLP",
    "svm": "Linear SVM",
    "knn": "KNN",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Finalny trening, wspólna ocena metryk oraz benchmark modeli "
            "CICIDS2017."
        )
    )
    parser.add_argument("--processed-path", type=Path, default=Path("data/processed"))
    parser.add_argument("--models-path", type=Path, default=Path("models"))
    parser.add_argument("--results-path", type=Path, default=Path("results/final"))
    parser.add_argument(
        "--best-params-path",
        type=Path,
        default=Path("results/tuning/best_params_all.json"),
    )
    parser.add_argument(
        "--target",
        choices=["AttackClass", "Label_binary"],
        default="AttackClass",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["all", *SUPPORTED_MODELS],
        default=["all"],
    )
    parser.add_argument("--mlp-max-iter", type=int, default=200)
    parser.add_argument("--knn-max-train", type=int, default=100_000)
    parser.add_argument("--model-verbose", action="store_true")
    parser.add_argument("--ae-epochs", type=int, default=100)
    parser.add_argument("--ae-batch-size", type=int, default=2048)
    parser.add_argument("--ae-normal-validation-size", type=float, default=0.1)
    parser.add_argument("--ae-verbose", type=int, default=1)

    parser.add_argument(
        "--benchmark-batch-sizes",
        nargs="+",
        type=int,
        default=[1, 32, 128, 1024],
        help="Wielkości partii używane w benchmarku latency/throughput.",
    )
    parser.add_argument("--benchmark-repeats", type=int, default=10)
    parser.add_argument("--benchmark-warmup-runs", type=int, default=3)
    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="Pomija benchmark latency i throughput.",
    )
    return parser.parse_args()


def expand_models(selected: Iterable[str]) -> list[str]:
    values = list(selected)
    if "all" in values:
        return list(SUPPORTED_MODELS)
    return list(dict.fromkeys(values))


def main() -> None:
    args = parse_args()
    set_random_seeds(RANDOM_STATE)
    args.models_path.mkdir(parents=True, exist_ok=True)
    args.results_path.mkdir(parents=True, exist_ok=True)

    tuning_config = load_json(args.best_params_path)
    tuned_target = tuning_config.get("target")
    if tuned_target and tuned_target != args.target:
        raise ValueError(
            f"Parametry były strojone dla target={tuned_target}, "
            f"a trening uruchomiono dla target={args.target}."
        )

    data = load_train_test_data(args.processed_path)
    evaluation = prepare_evaluation_target(data, args.target)
    results: list[dict] = []
    benchmark_rows: list[dict] = []

    for model_key in expand_models(args.models):
        run: TrainingRun | None = None

        if model_key == "ae_rf":
            try:
                latent, rf_params, percentile = ae_best_params(tuning_config)
                run = train_ae_feature_rf(
                    data,
                    target=args.target,
                    latent_dimension=latent,
                    rf_params=rf_params,
                    threshold_percentile=percentile,
                    ae_epochs=args.ae_epochs,
                    ae_batch_size=args.ae_batch_size,
                    normal_validation_size=args.ae_normal_validation_size,
                    models_path=args.models_path,
                    results_path=args.results_path,
                    benchmark_batch_sizes=args.benchmark_batch_sizes,
                    benchmark_repeats=args.benchmark_repeats,
                    benchmark_warmup_runs=args.benchmark_warmup_runs,
                    run_benchmark=not args.skip_benchmark,
                    ae_verbose=args.ae_verbose,
                )
            except ImportError as error:
                warnings.warn(str(error))
        else:
            params = classical_best_params(tuning_config, model_key)
            entry = model_tuning_entry(tuning_config, model_key)
            tuning_meta = entry.get("tuning", {})
            use_balanced_weight = bool(
                tuning_meta.get(
                    "used_balanced_sample_weight",
                    entry.get("used_balanced_sample_weight", False),
                )
            )
            run = train_classical_model(
                model_key,
                DISPLAY_NAMES[model_key],
                params,
                evaluation,
                target=args.target,
                models_path=args.models_path,
                results_path=args.results_path,
                mlp_max_iter=args.mlp_max_iter,
                knn_max_train=args.knn_max_train,
                use_balanced_sample_weight=use_balanced_weight,
                benchmark_batch_sizes=args.benchmark_batch_sizes,
                benchmark_repeats=args.benchmark_repeats,
                benchmark_warmup_runs=args.benchmark_warmup_runs,
                run_benchmark=not args.skip_benchmark,
                verbose=args.model_verbose,
            )

        if run is not None:
            results.append(run.result)
            benchmark_rows.extend(run.benchmark_rows)

    if not results:
        raise RuntimeError("Nie wytrenowano żadnego modelu.")

    comparison = pd.DataFrame(results).sort_values(
        "macro_f1",
        ascending=False,
    ).reset_index(drop=True)
    comparison_path = args.results_path / f"model_comparison_{args.target}.csv"
    comparison.to_csv(comparison_path, index=False)
    save_json(
        {
            "target": args.target,
            "random_state": RANDOM_STATE,
            "best_params_path": args.best_params_path,
            "models": results,
        },
        args.results_path / f"model_comparison_{args.target}.json",
    )

    benchmark_path: Path | None = None
    if benchmark_rows:
        benchmark_df = pd.DataFrame(benchmark_rows).sort_values(
            ["model", "batch_size"],
            ascending=True,
        )
        benchmark_path = (
            args.results_path / f"realtime_benchmark_{args.target}.csv"
        )
        benchmark_df.to_csv(benchmark_path, index=False)

    columns = [
        "model",
        "accuracy",
        "balanced_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "roc_auc",
        "false_alarm_count",
        "false_alarm_rate",
        "unknown_attack_detection_rate",
        "training_time_seconds",
        "prediction_time_seconds",
        "full_test_throughput_flows_per_second",
    ]
    print("\n" + "=" * 78)
    print("PORÓWNANIE MODELI")
    print("=" * 78)
    print(
        comparison[
            [column for column in columns if column in comparison.columns]
        ].to_string(index=False)
    )
    print(f"\nWyniki metryk: {comparison_path}")
    if benchmark_path is not None:
        print(f"Benchmark latency/throughput: {benchmark_path}")
        print(
            "Uwaga: benchmark zaczyna się od gotowych, przeskalowanych cech "
            "przepływu i nie obejmuje przechwytywania pakietów ani CICFlowMeter."
        )


if __name__ == "__main__":
    main()