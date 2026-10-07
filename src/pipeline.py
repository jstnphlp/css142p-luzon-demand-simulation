from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .comparison import classify_profile_recovery, normalized_period_profiles, pairwise_profile_rmse
from .config import (
    DEFAULT_RANDOM_SEED,
    DEFAULT_SIMULATION_RUNS,
    DEFAULT_TEST_FRACTION,
    FIGURES_DIR,
    MODELS_DIR,
    PERIOD_ORDER,
    PROCESSED_DIR,
    RESULTS_DIR,
)
from .data_preparation import data_quality_report, prepare_dataset
from .eda import save_eda_outputs
from .modeling import fit_period_model, save_model_bundle
from .normalization import add_daily_normalization
from .simulation import save_simulation, simulate_days
from .validation import validate_model


def run_pipeline(
    input_path: Path,
    *,
    sheet_name: str = "Luzon",
    header: int = 0,
    n_runs: int = DEFAULT_SIMULATION_RUNS,
    seed: int = DEFAULT_RANDOM_SEED,
    test_fraction: float = DEFAULT_TEST_FRACTION,
) -> None:
    for directory in [PROCESSED_DIR, FIGURES_DIR, RESULTS_DIR, MODELS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    print("[1/7] Preparing dataset...")
    df = prepare_dataset(input_path, sheet_name=sheet_name, header=header, study_only=True)
    df = add_daily_normalization(df)
    df.to_csv(PROCESSED_DIR / "luzon_hourly_clean.csv", index=False)
    data_quality_report(df).to_csv(RESULTS_DIR / "data_quality_report.csv", index=False)

    print("[2/7] Running exploratory analysis...")
    save_eda_outputs(df, FIGURES_DIR, RESULTS_DIR)
    normalized_period_profiles(df).to_csv(RESULTS_DIR / "normalized_hourly_profiles.csv", index=False)

    validation_rows = []

    print("[3/7] Fitting common statistical model separately for each period...")
    for i, period in enumerate(PERIOD_ORDER):
        print(f"      - {period}")
        bundle = fit_period_model(df, period, test_fraction=test_fraction)
        save_model_bundle(bundle, MODELS_DIR)

        print(f"[4/7] Simulating {period} ({n_runs:,} runs)...")
        simulation = simulate_days(
            bundle.model,
            bundle.residual_library,
            bundle.train,
            n_runs=n_runs,
            seed=seed + i,
            period=period,
        )
        save_simulation(simulation, RESULTS_DIR, period)

        print(f"[5/7] Validating {period}...")
        metrics, _ = validate_model(
            bundle.model,
            bundle.residual_library,
            bundle.test,
            n_runs=n_runs,
            seed=seed + 100 + i,
            period=period,
        )
        metrics.insert(0, "period", period)
        validation_rows.append(metrics)

    pd.concat(validation_rows, ignore_index=True).to_csv(RESULTS_DIR / "validation_metrics.csv", index=False)

    print("[6/7] Comparing disruption and recovery...")
    pairwise_profile_rmse(df).to_csv(RESULTS_DIR / "pairwise_normalized_profile_rmse.csv", index=False)
    classify_profile_recovery(df, seed=seed).to_csv(RESULTS_DIR / "recovery_classification.csv", index=False)

    print("[7/7] Done.")
    print(f"Processed data: {PROCESSED_DIR / 'luzon_hourly_clean.csv'}")
    print(f"Results:        {RESULTS_DIR}")
    print(f"Figures:        {FIGURES_DIR}")
    print(f"Models:         {MODELS_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete Luzon electricity-demand modeling and simulation pipeline.")
    parser.add_argument("input", type=Path, help="Path to the NGCP Hourly Demand per Grid Excel workbook.")
    parser.add_argument("--sheet", default="Luzon")
    parser.add_argument("--header", type=int, default=0)
    parser.add_argument("--runs", type=int, default=DEFAULT_SIMULATION_RUNS)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--test-fraction", type=float, default=DEFAULT_TEST_FRACTION)
    args = parser.parse_args()

    run_pipeline(
        args.input,
        sheet_name=args.sheet,
        header=args.header,
        n_runs=args.runs,
        seed=args.seed,
        test_fraction=args.test_fraction,
    )


if __name__ == "__main__":
    main()
