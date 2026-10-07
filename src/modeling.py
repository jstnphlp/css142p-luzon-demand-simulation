from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.regression.linear_model import RegressionResultsWrapper

from .config import MODEL_FORMULA


@dataclass
class PeriodModelBundle:
    period: str
    model: RegressionResultsWrapper
    train: pd.DataFrame
    test: pd.DataFrame
    residual_library: pd.DataFrame


def chronological_day_split(df: pd.DataFrame, test_fraction: float = 0.20) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.Series(pd.to_datetime(df["date"]).dt.normalize().unique()).sort_values().reset_index(drop=True)
    if len(dates) < 10:
        raise ValueError("Not enough unique dates for a chronological train/test split.")
    cut = max(1, min(len(dates) - 1, int(round(len(dates) * (1 - test_fraction)))))
    train_dates = set(dates.iloc[:cut])
    test_dates = set(dates.iloc[cut:])
    train = df[df["date"].isin(train_dates)].copy()
    test = df[df["date"].isin(test_dates)].copy()
    return train, test


def build_residual_library(model: RegressionResultsWrapper, train: pd.DataFrame) -> pd.DataFrame:
    lib = train[["date", "hour", "month", "day_type", "demand_mw"]].copy()
    lib["expected_mw"] = model.predict(train)
    lib["residual_mw"] = lib["demand_mw"] - lib["expected_mw"]

    complete = lib.groupby("date")["hour"].nunique()
    complete_dates = complete[complete == 24].index
    return lib[lib["date"].isin(complete_dates)].sort_values(["date", "hour"]).reset_index(drop=True)


def fit_period_model(
    df: pd.DataFrame,
    period: str,
    *,
    formula: str = MODEL_FORMULA,
    test_fraction: float = 0.20,
) -> PeriodModelBundle:
    part = df[(df["period"] == period) & df["demand_mw"].notna()].copy()
    if part.empty:
        raise ValueError(f"No data found for period: {period}")
    train, test = chronological_day_split(part, test_fraction=test_fraction)
    model = smf.ols(formula=formula, data=train).fit()
    residual_library = build_residual_library(model, train)
    return PeriodModelBundle(period=period, model=model, train=train, test=test, residual_library=residual_library)


def save_model_bundle(bundle: PeriodModelBundle, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = bundle.period.lower().replace("-", "_").replace(" ", "_")
    bundle.model.save(output_dir / f"{slug}_ols_model.pkl")
    bundle.residual_library.to_csv(output_dir / f"{slug}_residual_library.csv", index=False)
    with open(output_dir / f"{slug}_model_summary.txt", "w", encoding="utf-8") as f:
        f.write(bundle.model.summary().as_text())
