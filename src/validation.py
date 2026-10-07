from __future__ import annotations

import numpy as np
import pandas as pd

from .normalization import add_daily_normalization
from .simulation import simulate_days, simulation_daily_summary


def _profile(df: pd.DataFrame, value_col: str) -> pd.Series:
    return df.groupby("hour")[value_col].mean().reindex(range(24))


def rmse(a: pd.Series | np.ndarray, b: pd.Series | np.ndarray) -> float:
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    mask = np.isfinite(aa) & np.isfinite(bb)
    return float(np.sqrt(np.mean((aa[mask] - bb[mask]) ** 2)))


def validate_model(
    model,
    residual_library: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    n_runs: int = 1000,
    seed: int = 42,
    period: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    simulated = simulate_days(
        model,
        residual_library,
        test_df,
        n_runs=n_runs,
        seed=seed,
        period=period,
    )

    actual = add_daily_normalization(test_df.dropna(subset=["demand_mw"]))
    sim_summary = simulation_daily_summary(simulated)

    actual_daily = actual.groupby("date").agg(
        mean_demand_mw=("demand_mw", "mean"),
        max_demand_mw=("demand_mw", "max"),
        std_demand_mw=("demand_mw", "std"),
    )
    peak_idx = actual.groupby("date")["demand_mw"].idxmax()
    actual_peak_hours = actual.loc[peak_idx, "hour"]

    actual_abs_profile = _profile(actual, "demand_mw")
    sim_abs_profile = _profile(simulated, "demand_mw")
    actual_norm_profile = _profile(actual, "normalized_demand")
    sim_norm_profile = _profile(simulated, "normalized_demand")

    metrics = pd.DataFrame(
        [
            {"metric": "overall_mean_mw", "actual": actual["demand_mw"].mean(), "simulated": simulated["demand_mw"].mean()},
            {"metric": "overall_std_mw", "actual": actual["demand_mw"].std(), "simulated": simulated["demand_mw"].std()},
            {"metric": "mean_daily_peak_mw", "actual": actual_daily["max_demand_mw"].mean(), "simulated": sim_summary["max_demand_mw"].mean()},
            {"metric": "mean_daily_variability_mw", "actual": actual_daily["std_demand_mw"].mean(), "simulated": sim_summary["std_demand_mw"].mean()},
            {"metric": "modal_peak_hour", "actual": actual_peak_hours.mode().iloc[0], "simulated": sim_summary["peak_hour"].mode().iloc[0]},
            {"metric": "absolute_24h_profile_rmse_mw", "actual": 0.0, "simulated": rmse(actual_abs_profile, sim_abs_profile)},
            {"metric": "normalized_24h_profile_rmse", "actual": 0.0, "simulated": rmse(actual_norm_profile, sim_norm_profile)},
        ]
    )
    return metrics, simulated
