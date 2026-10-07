from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PERIOD_ORDER
from .normalization import add_daily_normalization
from .validation import rmse


def normalized_period_profiles(df: pd.DataFrame) -> pd.DataFrame:
    d = add_daily_normalization(df)
    return (
        d.groupby(["period", "hour"], as_index=False)["normalized_demand"]
        .mean()
        .sort_values(["period", "hour"])
    )


def pairwise_profile_rmse(df: pd.DataFrame) -> pd.DataFrame:
    profiles = normalized_period_profiles(df)
    vectors = {
        p: profiles[profiles["period"] == p].set_index("hour")["normalized_demand"].reindex(range(24))
        for p in PERIOD_ORDER
    }
    pairs = [
        ("Pre-pandemic", "Pandemic"),
        ("Pandemic", "Recovery"),
        ("Pre-pandemic", "Recovery"),
    ]
    return pd.DataFrame(
        [{"period_a": a, "period_b": b, "normalized_profile_rmse": rmse(vectors[a], vectors[b])} for a, b in pairs]
    )


def estimate_pre_baseline_tolerance(
    df: pd.DataFrame,
    *,
    n_bootstrap: int = 500,
    quantile: float = 0.95,
    seed: int = 42,
) -> float:
    rng = np.random.default_rng(seed)
    pre = add_daily_normalization(df[df["period"] == "Pre-pandemic"].copy())
    dates = np.array(sorted(pre["date"].unique()))
    if len(dates) < 30:
        raise ValueError("Not enough pre-pandemic days to estimate baseline tolerance.")

    values = []
    half = len(dates) // 2
    for _ in range(n_bootstrap):
        shuffled = rng.permutation(dates)
        a_dates, b_dates = shuffled[:half], shuffled[half : 2 * half]
        a = pre[pre["date"].isin(a_dates)].groupby("hour")["normalized_demand"].mean().reindex(range(24))
        b = pre[pre["date"].isin(b_dates)].groupby("hour")["normalized_demand"].mean().reindex(range(24))
        values.append(rmse(a, b))
    return float(np.quantile(values, quantile))


def classify_profile_recovery(df: pd.DataFrame, *, seed: int = 42) -> pd.DataFrame:
    pairwise = pairwise_profile_rmse(df)
    disruption = float(
        pairwise.loc[
            (pairwise["period_a"] == "Pre-pandemic") & (pairwise["period_b"] == "Pandemic"),
            "normalized_profile_rmse",
        ].iloc[0]
    )
    recovery = float(
        pairwise.loc[
            (pairwise["period_a"] == "Pre-pandemic") & (pairwise["period_b"] == "Recovery"),
            "normalized_profile_rmse",
        ].iloc[0]
    )
    tolerance = estimate_pre_baseline_tolerance(df, seed=seed)

    if recovery <= tolerance:
        classification = "Full recovery"
        rationale = "Recovery-period normalized profile falls within empirical pre-pandemic profile variability."
    elif recovery < disruption:
        classification = "Partial recovery"
        rationale = "Recovery is closer to the pre-pandemic profile than the pandemic profile was, but remains outside baseline variability."
    else:
        classification = "Structural change"
        rationale = "Recovery remains at least as different from the pre-pandemic profile as the pandemic period, using normalized-profile RMSE."

    ratio = recovery / disruption if disruption > 0 else np.nan
    return pd.DataFrame(
        [
            {
                "classification": classification,
                "pre_vs_pandemic_rmse": disruption,
                "pre_vs_recovery_rmse": recovery,
                "recovery_to_disruption_ratio": ratio,
                "pre_baseline_95pct_tolerance": tolerance,
                "rationale": rationale,
            }
        ]
    )
