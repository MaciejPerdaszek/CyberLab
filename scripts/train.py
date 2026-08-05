from __future__ import annotations

import argparse
import gc
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Iterable

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


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DISPLAY_NAMES = {
    "rf": "Random Forest",
    "mlp": "MLP",
    "svm": "Linear SVM",
    "knn": "KNN",
}

REQUIRED_DATA_FILES = {
    "X_train_scaled.csv",
    "X_test_scaled.csv",
    "y_train.csv",
    "y_test.csv",
}

EXPERIMENT_FOLDERS = {
    "leave-one-out": "leave_one_out",
    "cross-day": "cross_day",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Finalny trening, wspólna ocena metryk i benchmark modeli "
            "CICIDS2017. Skrypt może wykonać jeden standardowy podział albo "
            "automatycznie przejść przez wszystkie scenariusze leave-one-out "
            "lub cross-day."
        )
    )

    parser.add_argument(
        "--experiment",
        choices=["standard", "leave-one-out", "cross-day"],
        default="standard",
        help="Rodzaj uruchamianego eksperymentu.",
    )

    parser.add_argument(
        "--processed-path",
        type=Path,
        default=Path("data/processed"),
        help="Katalog danych dla --experiment standard.",
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=Path("results/final"),
        help="Katalog wyników dla --experiment standard.",
    )
    parser.add_argument(
        "--models-path",
        type=Path,
        default=Path("models"),
        help="Katalog modeli dla --experiment standard.",
    )

    parser.add_argument(
        "--processed-root",
        type=Path,
        help=(
            "Katalog zawierający podkatalogi leave-one-out albo cross-day. "
            "Bez podania zostanie użyte data/processed/<typ_eksperymentu>."
        ),
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        help=(
            "Katalog zbiorczy wyników eksperymentów dodatkowych. "
            "Bez podania zostanie użyte results/experiments/<typ_eksperymentu>."
        ),
    )
    parser.add_argument(
        "--models-root",
        type=Path,
        help=(
            "Katalog zbiorczy modeli eksperymentów dodatkowych. "
            "Bez podania zostanie użyte models/experiments/<typ_eksperymentu>."
        ),
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        help=(
            "Opcjonalna lista nazw podkatalogów scenariuszy do uruchomienia. "
            "Bez tego parametru uruchamiane są wszystkie kompletne scenariusze."
        ),
    )

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
        help="Wielkości partii w benchmarku latency/throughput.",
    )
    parser.add_argument("--benchmark-repeats", type=int, default=10)
    parser.add_argument("--benchmark-warmup-runs", type=int, default=3)
    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        help=(
            "Pomija benchmark. Dla standardowego eksperymentu benchmark jest "
            "domyślnie włączony."
        ),
    )
    parser.add_argument(
        "--with-benchmark",
        action="store_true",
        help=(
            "Włącza benchmark również dla leave-one-out lub cross-day. "
            "Domyślnie eksperymenty dodatkowe nie wykonują benchmarku."
        ),
    )

    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help=(
            "W trybie wieloscenariuszowym przechodzi do kolejnego scenariusza "
            "po błędzie i zapisuje failed_scenarios.csv."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Wyświetla wykryte scenariusze bez uruchamiania treningu.",
    )

    return parser.parse_args()


def resolve_project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def expand_models(selected: Iterable[str]) -> list[str]:
    values = list(selected)
    if "all" in values:
        return list(SUPPORTED_MODELS)
    return list(dict.fromkeys(values))


def _serialize_metadata_value(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def experiment_result_fields(
    metadata: dict[str, Any],
    processed_path: Path,
) -> dict[str, Any]:
    selected_keys = (
        "experiment_type",
        "scenario_name",
        "held_out_attack",
        "train_files",
        "test_files",
    )
    result = {
        key: _serialize_metadata_value(metadata.get(key))
        for key in selected_keys
    }
    result["processed_path"] = str(processed_path)
    return result


def validate_experiment_target(
    metadata: dict[str, Any],
    target: str,
) -> None:
    experiment_type = metadata.get("experiment_type", "standard")
    if experiment_type == "leave_one_attack_out" and target != "Label_binary":
        raise ValueError(
            "Leave-one-attack-class-out należy uruchamiać z "
            "--target Label_binary. Model wieloklasowy nie może przewidzieć "
            "klasy usuniętej z treningu."
        )


def is_processed_scenario(path: Path) -> bool:
    if not path.is_dir():
        return False

    existing_files = {
        child.name
        for child in path.iterdir()
        if child.is_file()
    }
    return REQUIRED_DATA_FILES.issubset(existing_files)


def discover_scenarios(
    root: Path,
    selected: Iterable[str] | None,
) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(
            f"Nie istnieje katalog scenariuszy: {root}"
        )

    if selected:
        scenarios = [root / name for name in selected]
        missing = [
            str(path)
            for path in scenarios
            if not is_processed_scenario(path)
        ]
        if missing:
            raise FileNotFoundError(
                "Nie znaleziono kompletnych danych scenariuszy:\n- "
                + "\n- ".join(missing)
            )
        return scenarios

    scenarios = sorted(
        path
        for path in root.iterdir()
        if is_processed_scenario(path)
    )

    if not scenarios:
        raise FileNotFoundError(
            f"W {root} nie znaleziono podkatalogów zawierających: "
            f"{sorted(REQUIRED_DATA_FILES)}"
        )

    return scenarios


def read_scenario_metadata(path: Path) -> dict[str, Any]:
    metadata_path = path / "experiment_metadata.json"
    if not metadata_path.exists():
        return {"scenario_name": path.name}

    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    if not isinstance(metadata, dict):
        raise ValueError(
            f"Plik {metadata_path} nie zawiera obiektu JSON."
        )

    metadata.setdefault("scenario_name", path.name)
    return metadata


def benchmark_enabled(args: argparse.Namespace) -> bool:
    if args.skip_benchmark and args.with_benchmark:
        raise ValueError(
            "Nie można jednocześnie użyć --skip-benchmark i --with-benchmark."
        )

    if args.skip_benchmark:
        return False

    if args.experiment == "standard":
        return True

    return bool(args.with_benchmark)


def train_single_scenario(
    *,
    args: argparse.Namespace,
    tuning_config: dict[str, Any],
    processed_path: Path,
    results_path: Path,
    models_path: Path,
    run_benchmark: bool,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Trenuje wszystkie wybrane modele dla jednego przygotowanego podziału."""

    models_path.mkdir(parents=True, exist_ok=True)
    results_path.mkdir(parents=True, exist_ok=True)

    data = load_train_test_data(processed_path)
    validate_experiment_target(data.experiment_metadata, args.target)
    evaluation = prepare_evaluation_target(data, args.target)
    experiment_fields = experiment_result_fields(
        data.experiment_metadata,
        processed_path,
    )

    scenario_name = str(
        data.experiment_metadata.get("scenario_name")
        or processed_path.name
    )

    print("\n" + "=" * 78)
    print("SCENARIUSZ EKSPERYMENTU")
    print("=" * 78)
    print(
        "Typ: "
        f"{data.experiment_metadata.get('experiment_type', 'standard')}"
    )
    print(f"Nazwa: {scenario_name}")
    print(f"Dane: {processed_path}")
    print(f"Benchmark: {'tak' if run_benchmark else 'nie'}")

    if data.experiment_metadata.get("held_out_attack"):
        print(
            "Wyłączona klasa: "
            f"{data.experiment_metadata['held_out_attack']}"
        )

    if evaluation.unseen_attack_classes:
        print(
            "Klasy ataków niewidziane w treningu: "
            f"{list(evaluation.unseen_attack_classes)}"
        )
        print(
            "Liczba próbek niewidzianych ataków: "
            f"{evaluation.unseen_attack_samples}"
        )

    results: list[dict[str, Any]] = []
    benchmark_rows: list[dict[str, Any]] = []

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
                    models_path=models_path,
                    results_path=results_path,
                    benchmark_batch_sizes=args.benchmark_batch_sizes,
                    benchmark_repeats=args.benchmark_repeats,
                    benchmark_warmup_runs=args.benchmark_warmup_runs,
                    run_benchmark=run_benchmark,
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
                models_path=models_path,
                results_path=results_path,
                mlp_max_iter=args.mlp_max_iter,
                knn_max_train=args.knn_max_train,
                use_balanced_sample_weight=use_balanced_weight,
                benchmark_batch_sizes=args.benchmark_batch_sizes,
                benchmark_repeats=args.benchmark_repeats,
                benchmark_warmup_runs=args.benchmark_warmup_runs,
                run_benchmark=run_benchmark,
                verbose=args.model_verbose,
            )

        if run is None:
            continue

        run.result.update(experiment_fields)
        results.append(run.result)

        for row in run.benchmark_rows:
            row.update(experiment_fields)
        benchmark_rows.extend(run.benchmark_rows)

    if not results:
        raise RuntimeError(
            f"Nie wytrenowano żadnego modelu dla scenariusza {scenario_name}."
        )

    comparison = pd.DataFrame(results).sort_values(
        "macro_f1",
        ascending=False,
    ).reset_index(drop=True)

    comparison_path = (
        results_path
        / f"model_comparison_{args.target}.csv"
    )
    comparison.to_csv(comparison_path, index=False)

    save_json(
        {
            "target": args.target,
            "random_state": RANDOM_STATE,
            "best_params_path": args.best_params_path,
            "processed_path": processed_path,
            "experiment_metadata": data.experiment_metadata,
            "models": results,
        },
        results_path / f"model_comparison_{args.target}.json",
    )

    benchmark_df: pd.DataFrame | None = None
    benchmark_path: Path | None = None

    if benchmark_rows:
        benchmark_df = pd.DataFrame(benchmark_rows).sort_values(
            ["model", "batch_size"],
            ascending=True,
        ).reset_index(drop=True)
        benchmark_path = (
            results_path
            / f"realtime_benchmark_{args.target}.csv"
        )
        benchmark_df.to_csv(benchmark_path, index=False)

    columns = [
        "experiment_type",
        "scenario_name",
        "held_out_attack",
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
        "unseen_attack_classes",
        "unseen_attack_samples",
        "unseen_attack_detection_rate",
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
            "Uwaga: benchmark zaczyna się od gotowych, przeskalowanych "
            "cech przepływu i nie obejmuje przechwytywania pakietów ani "
            "generowania cech przez CICFlowMeter."
        )

    return comparison, benchmark_df


def cleanup_after_scenario() -> None:
    """Ogranicza kumulowanie pamięci między kolejnymi scenariuszami AE."""

    tensorflow_module = sys.modules.get("tensorflow")
    if tensorflow_module is not None:
        try:
            tensorflow_module.keras.backend.clear_session()
        except Exception:
            pass

    gc.collect()


def validate_tuning_target(
    tuning_config: dict[str, Any],
    target: str,
) -> None:
    tuned_target = tuning_config.get("target")
    if tuned_target and tuned_target != target:
        raise ValueError(
            f"Parametry były strojone dla target={tuned_target}, "
            f"a trening uruchomiono dla target={target}."
        )


def run_standard(
    args: argparse.Namespace,
    tuning_config: dict[str, Any],
    run_benchmark: bool,
) -> None:
    processed_path = resolve_project_path(args.processed_path)
    results_path = resolve_project_path(args.results_path)
    models_path = resolve_project_path(args.models_path)

    if args.dry_run:
        print("Tryb dry-run")
        print(f"Scenariusz: {processed_path}")
        print(f"Wyniki: {results_path}")
        print(f"Modele: {models_path}")
        return

    train_single_scenario(
        args=args,
        tuning_config=tuning_config,
        processed_path=processed_path,
        results_path=results_path,
        models_path=models_path,
        run_benchmark=run_benchmark,
    )


def default_multi_paths(
    args: argparse.Namespace,
) -> tuple[Path, Path, Path, str]:
    folder = EXPERIMENT_FOLDERS[args.experiment]

    processed_root = resolve_project_path(
        args.processed_root
        or Path("data/processed") / folder
    )
    results_root = resolve_project_path(
        args.results_root
        or Path("results/experiments") / folder
    )
    models_root = resolve_project_path(
        args.models_root
        or Path("models/experiments") / folder
    )

    return processed_root, results_root, models_root, folder


def run_multiple_scenarios(
    args: argparse.Namespace,
    tuning_config: dict[str, Any],
    run_benchmark: bool,
) -> None:
    if args.experiment == "leave-one-out" and args.target != "Label_binary":
        raise ValueError(
            "Leave-one-out należy uruchamiać z --target Label_binary."
        )

    processed_root, results_root, models_root, folder = default_multi_paths(args)
    scenarios = discover_scenarios(
        processed_root,
        args.scenarios,
    )

    print("\n" + "=" * 90)
    print("EKSPERYMENT WIELOSCENARIUSZOWY")
    print("=" * 90)
    print(f"Typ: {args.experiment}")
    print(f"Target: {args.target}")
    print(f"Liczba scenariuszy: {len(scenarios)}")
    print(f"Katalog danych: {processed_root}")
    print(f"Benchmark: {'tak' if run_benchmark else 'nie'}")

    for index, path in enumerate(scenarios, start=1):
        metadata = read_scenario_metadata(path)
        scenario_name = str(
            metadata.get("scenario_name")
            or path.name
        )
        print(f"  {index}. {scenario_name} -> {path}")

    if args.dry_run:
        print("\nTryb dry-run: nie uruchomiono treningu.")
        return

    summary_frames: list[pd.DataFrame] = []
    benchmark_frames: list[pd.DataFrame] = []
    failures: list[dict[str, str]] = []

    for index, scenario_path in enumerate(scenarios, start=1):
        metadata = read_scenario_metadata(scenario_path)
        scenario_name = str(
            metadata.get("scenario_name")
            or scenario_path.name
        )

        scenario_results_path = results_root / scenario_name
        scenario_models_path = models_root / scenario_name

        print("\n" + "#" * 90)
        print(
            f"URUCHOMIENIE {index}/{len(scenarios)}: "
            f"{scenario_name}"
        )
        print("#" * 90)

        try:
            comparison, benchmark_df = train_single_scenario(
                args=args,
                tuning_config=tuning_config,
                processed_path=scenario_path,
                results_path=scenario_results_path,
                models_path=scenario_models_path,
                run_benchmark=run_benchmark,
            )
            summary_frames.append(comparison)
            if benchmark_df is not None:
                benchmark_frames.append(benchmark_df)
        except Exception as error:
            failures.append(
                {
                    "scenario_name": scenario_name,
                    "processed_path": str(scenario_path),
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            print(
                f"[BŁĄD] Scenariusz {scenario_name}: "
                f"{type(error).__name__}: {error}"
            )
            if not args.continue_on_error:
                raise
        finally:
            cleanup_after_scenario()

    results_root.mkdir(parents=True, exist_ok=True)

    if summary_frames:
        summary = pd.concat(
            summary_frames,
            ignore_index=True,
        )
        sort_columns = [
            column
            for column in ("scenario_name", "macro_f1")
            if column in summary.columns
        ]
        if sort_columns:
            ascending = [True, False][: len(sort_columns)]
            summary = summary.sort_values(
                sort_columns,
                ascending=ascending,
            ).reset_index(drop=True)

        summary_path = (
            results_root
            / f"{folder}_summary_{args.target}.csv"
        )
        summary.to_csv(summary_path, index=False)
        print(f"\nZbiorcze wyniki: {summary_path}")
    else:
        print("\nNie powstały żadne poprawne wyniki scenariuszy.")

    if benchmark_frames:
        benchmark_summary = pd.concat(
            benchmark_frames,
            ignore_index=True,
        )
        benchmark_path = (
            results_root
            / f"{folder}_benchmark_{args.target}.csv"
        )
        benchmark_summary.to_csv(
            benchmark_path,
            index=False,
        )
        print(f"Zbiorczy benchmark: {benchmark_path}")

    if failures:
        failures_path = results_root / "failed_scenarios.csv"
        pd.DataFrame(failures).to_csv(
            failures_path,
            index=False,
        )
        print(f"Nieudane scenariusze: {failures_path}")

    if not summary_frames:
        raise RuntimeError(
            "Wszystkie scenariusze zakończyły się błędem."
        )


def main() -> None:
    args = parse_args()
    set_random_seeds(RANDOM_STATE)

    args.best_params_path = resolve_project_path(args.best_params_path)
    tuning_config = load_json(args.best_params_path)
    validate_tuning_target(tuning_config, args.target)

    run_benchmark = benchmark_enabled(args)

    if args.experiment == "standard":
        run_standard(
            args,
            tuning_config,
            run_benchmark,
        )
        return

    run_multiple_scenarios(
        args,
        tuning_config,
        run_benchmark,
    )


if __name__ == "__main__":
    main()