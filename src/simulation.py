from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import RegressionResultsWrapper


def _calendar_days(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df[["date", "month", "day_type"]]
        .drop_duplicates("date")
        .sort_values("date")
        .reset_index(drop=True)
    )


def simulate_days(
    model: RegressionResultsWrapper,
    residual_library: pd.DataFrame,
    calendar_source: pd.DataFrame,
    *,
    n_runs: int = 1000,
    seed: int = 42,
    period: str | None = None,
) -> pd.DataFrame:
    """
    Simulate synthetic 24-hour demand profiles.

    Each run samples a month/day-type condition from calendar_source, predicts the
    expected 24-hour profile, then adds an entire historical residual-day vector.
    Sampling residuals by day (rather than hour) preserves realistic within-day
    correlation and curve shape better than independent hourly noise.
    """
    rng = np.random.default_rng(seed)
    calendar = _calendar_days(calendar_source)
    if calendar.empty:
        raise ValueError("calendar_source contains no days.")

    residual_dates = residual_library[["date", "month", "day_type"]].drop_duplicates("date")
    if residual_dates.empty:
        raise ValueError("Residual library contains no complete 24-hour days.")

    simulated = []
    sampled_calendar_idx = rng.integers(0, len(calendar), size=n_runs)

    for sim_id, idx in enumerate(sampled_calendar_idx, start=1):
        cal = calendar.iloc[int(idx)]
        month = int(cal["month"])
        day_type = str(cal["day_type"])

        design = pd.DataFrame(
            {
                "hour": np.arange(24, dtype=int),
                "month": month,
                "day_type": day_type,
            }
        )
        expected = np.asarray(model.predict(design), dtype=float)

        candidates = residual_dates[residual_dates["day_type"] == day_type]
        month_candidates = candidates[candidates["month"] == month]
        if not month_candidates.empty:
            candidates = month_candidates
        if candidates.empty:
            candidates = residual_dates

        residual_date = candidates.iloc[int(rng.integers(0, len(candidates)))]["date"]
        residuals = (
            residual_library[residual_library["date"] == residual_date]
            .set_index("hour")
            .reindex(range(24))["residual_mw"]
            .to_numpy(dtype=float)
        )
        if np.isnan(residuals).any():
            raise ValueError("Residual day is incomplete after reindexing.")

        demand = np.clip(expected + residuals, a_min=0.0, a_max=None)
        daily_avg = float(demand.mean())
        normalized = demand / daily_avg if daily_avg else np.nan

        run = design.copy()
        run["simulation_id"] = sim_id
        run["period"] = period
        run["expected_mw"] = expected
        run["demand_mw"] = demand
        run["normalized_demand"] = normalized
        simulated.append(run)

    return pd.concat(simulated, ignore_index=True)


def simulation_daily_summary(sim_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sim_id, g in sim_df.groupby("simulation_id"):
        peak_idx = g["demand_mw"].idxmax()
        rows.append(
            {
                "simulation_id": sim_id,
                "period": g["period"].iloc[0] if "period" in g else None,
                "mean_demand_mw": g["demand_mw"].mean(),
                "max_demand_mw": g["demand_mw"].max(),
                "peak_hour": int(g.loc[peak_idx, "hour"]),
                "std_demand_mw": g["demand_mw"].std(ddof=1),
            }
        )
    return pd.DataFrame(rows)


def save_simulation(sim_df: pd.DataFrame, output_dir: str | Path, period: str) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = period.lower().replace("-", "_").replace(" ", "_")
    sim_df.to_csv(output_dir / f"{slug}_simulated_hourly.csv", index=False)
    simulation_daily_summary(sim_df).to_csv(output_dir / f"{slug}_simulation_summary.csv", index=False)
