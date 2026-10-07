from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import PERIOD_RANGES


def classify_period(year: int) -> str:
    for name, (start, end) in PERIOD_RANGES.items():
        if start <= year <= end:
            return name
    return "Out of scope"


def _detect_date_column(df: pd.DataFrame) -> str:
    preferred = [c for c in df.columns if str(c).strip().lower() in {"date", "day", "datetime"}]
    if preferred:
        return preferred[0]

    best_col = None
    best_rate = 0.0
    for col in df.columns[: min(8, len(df.columns))]:
        converted = pd.to_datetime(df[col], errors="coerce")
        rate = float(converted.notna().mean())
        if rate > best_rate:
            best_col, best_rate = col, rate
    if best_col is None or best_rate < 0.7:
        raise ValueError(
            "Could not reliably identify the date column. Pass date_col explicitly after inspecting the workbook."
        )
    return best_col


def _clock_hour_from_label(label: object) -> int | None:
    """Parse common hour labels when they clearly represent a clock hour."""
    s = str(label).strip().lower().replace(" ", "")

    m = re.fullmatch(r"(\d{1,2}):00(?:h)?", s)
    if m:
        h = int(m.group(1))
        return h if 0 <= h <= 23 else None

    m = re.fullmatch(r"(\d{2})00h?", s)
    if m:
        h = int(m.group(1))
        return h if 0 <= h <= 23 else None

    m = re.fullmatch(r"(?:hour|h)[_-]?(\d{1,2})", s)
    if m:
        h = int(m.group(1))
        return h if 0 <= h <= 23 else None

    if re.fullmatch(r"\d{1,2}", s):
        h = int(s)
        return h if 0 <= h <= 23 else None

    return None


def _detect_hour_columns(df: pd.DataFrame, date_col: str) -> tuple[list[str], dict[str, int]]:
    candidates = [c for c in df.columns if c != date_col]
    parsed = {c: _clock_hour_from_label(c) for c in candidates}
    clear = [(c, h) for c, h in parsed.items() if h is not None]

    if len(clear) >= 24:
        by_hour: dict[int, str] = {}
        for c, h in clear:
            if h not in by_hour:
                by_hour[h] = c
        if set(by_hour) == set(range(24)):
            cols = [by_hour[h] for h in range(24)]
            return cols, {c: h for h, c in by_hour.items()}

    numeric_named = []
    for c in candidates:
        s = str(c).strip()
        if re.fullmatch(r"\d{1,2}", s):
            numeric_named.append((c, int(s)))
    if {v for _, v in numeric_named} >= set(range(1, 25)):
        ordered = [next(c for c, v in numeric_named if v == i) for i in range(1, 25)]
        warnings.warn(
            "Detected hour columns labeled 1..24. They are mapped sequentially to analysis hours 0..23. "
            "Verify the NGCP workbook's hour convention before interpreting peak-clock labels.",
            stacklevel=2,
        )
        return ordered, {c: i for i, c in enumerate(ordered)}

    numeric_candidates = []
    for c in candidates:
        rate = pd.to_numeric(df[c], errors="coerce").notna().mean()
        if rate >= 0.8:
            numeric_candidates.append(c)
    if len(numeric_candidates) == 24:
        warnings.warn(
            "Hour labels could not be parsed confidently. Using the 24 numeric columns in workbook order as hours 0..23. "
            "Verify the mapping before reporting peak times.",
            stacklevel=2,
        )
        return numeric_candidates, {c: i for i, c in enumerate(numeric_candidates)}

    raise ValueError(
        f"Expected 24 hourly demand columns but could not identify them reliably. Found {len(numeric_candidates)} numeric candidates."
    )


def reshape_luzon_to_hourly(
    daily_df: pd.DataFrame,
    *,
    date_col: str | None = None,
    hour_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    df = daily_df.copy()
    date_col = date_col or _detect_date_column(df)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    df = df[df[date_col].notna()].copy()

    if hour_columns is None:
        hour_columns, hour_map = _detect_hour_columns(df, date_col)
    else:
        hour_columns = list(hour_columns)
        if len(hour_columns) != 24:
            raise ValueError("hour_columns must contain exactly 24 columns.")
        hour_map = {c: i for i, c in enumerate(hour_columns)}

    long_df = df.melt(
        id_vars=[date_col],
        value_vars=list(hour_columns),
        var_name="source_hour_label",
        value_name="demand_mw",
    )
    long_df["hour"] = long_df["source_hour_label"].map(hour_map).astype(int)
    long_df["demand_mw"] = pd.to_numeric(long_df["demand_mw"], errors="coerce")
    long_df = long_df.rename(columns={date_col: "date"})
    long_df["datetime"] = long_df["date"] + pd.to_timedelta(long_df["hour"], unit="h")

    return long_df.sort_values("datetime").reset_index(drop=True)


def add_time_features(hourly_df: pd.DataFrame) -> pd.DataFrame:
    df = hourly_df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["year"] = df["date"].dt.year.astype(int)
    df["month"] = df["date"].dt.month.astype(int)
    df["day_of_week"] = df["date"].dt.day_name()
    df["day_type"] = np.where(df["date"].dt.dayofweek < 5, "Weekday", "Weekend")
    df["period"] = df["year"].map(classify_period)
    return df


def clean_hourly_data(hourly_df: pd.DataFrame) -> pd.DataFrame:
    df = hourly_df.copy()
    df = df.drop_duplicates(subset=["datetime"], keep="first")
    df.loc[df["demand_mw"] < 0, "demand_mw"] = np.nan

    complete_days = df.groupby("date")["hour"].nunique()
    bad_days = complete_days[complete_days != 24]
    if not bad_days.empty:
        warnings.warn(
            f"{len(bad_days)} day(s) do not contain all 24 hourly slots and will remain flagged through missingness checks.",
            stacklevel=2,
        )

    return df.sort_values("datetime").reset_index(drop=True)


def prepare_dataset(
    excel_path: str | Path,
    *,
    sheet_name: str = "Luzon",
    header: int = 0,
    date_col: str | None = None,
    study_only: bool = True,
) -> pd.DataFrame:
    raw = pd.read_excel(excel_path, sheet_name=sheet_name, header=header)
    hourly = reshape_luzon_to_hourly(raw, date_col=date_col)
    hourly = add_time_features(hourly)
    hourly = clean_hourly_data(hourly)
    if study_only:
        hourly = hourly[hourly["period"] != "Out of scope"].copy()
    return hourly.reset_index(drop=True)


def data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    unique_dates = df["date"].nunique()
    duplicate_datetimes = int(df["datetime"].duplicated().sum())
    missing_demand = int(df["demand_mw"].isna().sum())
    negative_demand = int((df["demand_mw"] < 0).sum())
    complete_days = int((df.groupby("date")["hour"].nunique() == 24).sum())
    return pd.DataFrame(
        {
            "metric": [
                "rows",
                "unique_dates",
                "complete_24_hour_days",
                "missing_demand_values",
                "duplicate_datetimes",
                "negative_demand_values",
            ],
            "value": [len(df), unique_dates, complete_days, missing_demand, duplicate_datetimes, negative_demand],
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare NGCP Luzon hourly demand data.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--sheet", default="Luzon")
    parser.add_argument("--header", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("data/processed/luzon_hourly_clean.csv"))
    args = parser.parse_args()

    df = prepare_dataset(args.input, sheet_name=args.sheet, header=args.header)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(data_quality_report(df).to_string(index=False))
    print(f"\nSaved {len(df):,} rows to {args.output}")


if __name__ == "__main__":
    main()
