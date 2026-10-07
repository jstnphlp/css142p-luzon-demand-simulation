# Simulation Design

## Research question

How far did Luzon electricity-demand behavior move away from its pre-pandemic pattern during COVID-19, and how far did it move back afterward?

## Study periods

- **Pre-pandemic:** 2017–2019
- **Pandemic:** 2020–2021
- **Recovery:** 2022–2024

These are analytical periods for this study, not official declarations of pandemic start/end dates.

## Data structure

The NGCP Luzon worksheet is transformed from one row per date with 24 hourly demand columns into one row per hourly observation.

Core fields:

- `date`
- `datetime`
- `hour`
- `day_of_week`
- `day_type` (`Weekday` / `Weekend`)
- `month`
- `year`
- `period`
- `demand_mw`

## Common statistical model

The same OLS model structure is calibrated separately for all three periods:

```text
demand_mw ~ C(hour) * C(day_type) + C(month)
```

This models hour-of-day effects, weekday/weekend differences, different hourly patterns on weekdays vs weekends, monthly seasonality, and residual random variation.

## Normalization

For every day:

```text
normalized_demand = hourly_demand / daily_average_demand
```

This separates **demand level** from **demand pattern**.

## Calibration / validation split

Each period is split chronologically by complete days:

- first 80% of days: calibration/training,
- last 20% of days: validation/testing.

## Monte Carlo simulation

For each period, default `n = 1,000` synthetic daily profiles are generated.

For each simulation run:

1. Sample a historical calendar condition (`month`, `day_type`) from that period.
2. Use the fitted OLS model to compute the expected 24-hour demand profile.
3. Sample a **complete historical residual-day vector** from the calibration period, preferring the same month and day type when available.
4. Add the sampled 24-hour residual vector to the expected profile.
5. Clip impossible negative MW values to zero.
6. Record simulated average demand, maximum demand, peak-demand time, 24-hour profile, normalized profile, and variability.

Residuals are sampled as complete daily vectors rather than as 24 independent random values. This preserves realistic within-day dependence and avoids jagged simulated curves.

## Validation metrics

For each period, simulated validation profiles are compared with withheld historical observations using:

- overall mean MW,
- overall standard deviation,
- mean daily peak MW,
- mean within-day variability,
- modal daily peak hour,
- RMSE of the absolute 24-hour profile,
- RMSE of the normalized 24-hour profile.

## Recovery comparison

Three normalized-profile comparisons are calculated:

1. Pre-pandemic vs Pandemic — disruption
2. Pandemic vs Recovery — movement away from pandemic behavior
3. Pre-pandemic vs Recovery — extent of return toward baseline

## Recovery classification

To avoid inventing an arbitrary RMSE threshold, the code estimates ordinary **within-pre-pandemic variability** using bootstrap splits of pre-pandemic days.

- **Full recovery:** pre-vs-recovery RMSE falls within the 95th-percentile pre-pandemic internal variability tolerance.
- **Partial recovery:** recovery is closer to pre-pandemic than the pandemic period was, but remains outside that tolerance.
- **Structural change:** recovery remains at least as different from pre-pandemic as pandemic demand was.

This classification should be interpreted together with demand magnitude, peak timing, weekday/weekend behavior, and variability—not as a causal claim that COVID-19 alone produced every observed difference.

## Important verification before final analysis

The NGCP workbook's exact 24-hour column convention must be checked before interpreting clock-time labels. The data loader detects common conventions and warns on ambiguous layouts.
