from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .config import PERIOD_ORDER


def period_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("period")["demand_mw"]
        .agg(observations="count", mean_mw="mean", median_mw="median", std_mw="std", min_mw="min", max_mw="max")
        .reindex(PERIOD_ORDER)
        .reset_index()
    )
    return summary


def hourly_profile(df: pd.DataFrame, value_col: str = "demand_mw") -> pd.DataFrame:
    return (
        df.groupby(["period", "hour"], as_index=False)[value_col]
        .mean()
        .sort_values(["period", "hour"])
    )


def day_type_profile(df: pd.DataFrame, value_col: str = "demand_mw") -> pd.DataFrame:
    return (
        df.groupby(["period", "day_type", "hour"], as_index=False)[value_col]
        .mean()
        .sort_values(["period", "day_type", "hour"])
    )


def daily_peaks(df: pd.DataFrame) -> pd.DataFrame:
    clean = df.dropna(subset=["demand_mw"]).copy()
    idx = clean.groupby("date")["demand_mw"].idxmax()
    peaks = clean.loc[idx, ["date", "period", "day_type", "hour", "demand_mw"]].copy()
    return peaks.rename(columns={"hour": "peak_hour", "demand_mw": "peak_demand_mw"}).sort_values("date")


def peak_hour_distribution(df: pd.DataFrame) -> pd.DataFrame:
    peaks = daily_peaks(df)
    counts = peaks.groupby(["period", "peak_hour"]).size().rename("days").reset_index()
    totals = counts.groupby("period")["days"].transform("sum")
    counts["share"] = counts["days"] / totals
    return counts


def save_eda_outputs(df: pd.DataFrame, figures_dir: str | Path, results_dir: str | Path) -> None:
    figures_dir = Path(figures_dir)
    results_dir = Path(results_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    period_summary(df).to_csv(results_dir / "period_summary.csv", index=False)
    hourly_profile(df).to_csv(results_dir / "hourly_profile_absolute.csv", index=False)
    day_type_profile(df).to_csv(results_dir / "day_type_profile_absolute.csv", index=False)
    daily_peaks(df).to_csv(results_dir / "daily_peaks.csv", index=False)
    peak_hour_distribution(df).to_csv(results_dir / "peak_hour_distribution.csv", index=False)

    monthly = df.set_index("datetime").groupby(pd.Grouper(freq="MS"))["demand_mw"].mean()
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(monthly.index, monthly.values)
    ax.set(title="Luzon Grid Monthly Average Electricity Demand", xlabel="Date", ylabel="Average demand (MW)")
    fig.tight_layout()
    fig.savefig(figures_dir / "01_monthly_demand_trend.png", dpi=160)
    plt.close(fig)

    prof = hourly_profile(df)
    fig, ax = plt.subplots(figsize=(9, 5))
    for period in PERIOD_ORDER:
        part = prof[prof["period"] == period]
        ax.plot(part["hour"], part["demand_mw"], marker="o", markersize=3, label=period)
    ax.set(title="Average 24-hour Demand Profile", xlabel="Hour", ylabel="Average demand (MW)", xticks=range(0, 24, 2))
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "02_average_24h_profile.png", dpi=160)
    plt.close(fig)

    dt = day_type_profile(df)
    for day_type in ["Weekday", "Weekend"]:
        fig, ax = plt.subplots(figsize=(9, 5))
        for period in PERIOD_ORDER:
            part = dt[(dt["period"] == period) & (dt["day_type"] == day_type)]
            ax.plot(part["hour"], part["demand_mw"], marker="o", markersize=3, label=period)
        ax.set(title=f"{day_type} 24-hour Demand Profile", xlabel="Hour", ylabel="Average demand (MW)", xticks=range(0, 24, 2))
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures_dir / f"03_{day_type.lower()}_24h_profile.png", dpi=160)
        plt.close(fig)

    phd = peak_hour_distribution(df)
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.25
    for i, period in enumerate(PERIOD_ORDER):
        shares = phd[phd["period"] == period].set_index("peak_hour")["share"].reindex(range(24), fill_value=0)
        ax.bar([h + (i - 1) * width for h in range(24)], shares, width=width, label=period)
    ax.set(title="Distribution of Daily Peak-Demand Hour", xlabel="Peak hour", ylabel="Share of days", xticks=range(0, 24, 2))
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "04_peak_hour_distribution.png", dpi=160)
    plt.close(fig)
