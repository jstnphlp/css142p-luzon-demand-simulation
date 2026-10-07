from __future__ import annotations

import pandas as pd


def add_daily_normalization(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    daily_avg = out.groupby("date")["demand_mw"].transform("mean")
    out["daily_average_mw"] = daily_avg
    out["normalized_demand"] = out["demand_mw"] / daily_avg
    return out


def normalized_hourly_profile(df: pd.DataFrame) -> pd.DataFrame:
    if "normalized_demand" not in df.columns:
        df = add_daily_normalization(df)
    return (
        df.groupby(["period", "hour"], as_index=False)["normalized_demand"]
        .mean()
        .sort_values(["period", "hour"])
    )
