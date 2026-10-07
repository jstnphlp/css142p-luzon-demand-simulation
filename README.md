# Modeling and Simulation of Hourly Electricity Demand in the Luzon Grid

CSS142P / Modeling and Simulation Theory project based on the study proposal for comparing Luzon Grid electricity-demand behavior before, during, and after the COVID-19 disruption.

## Research goal

The project compares three analytical periods:

- **Pre-pandemic:** 2017–2019
- **Pandemic:** 2020–2021
- **Recovery:** 2022–2024

The analysis distinguishes **demand-level recovery** from **demand-pattern recovery**, uses the same temporal statistical model for all periods, and uses Monte Carlo simulation to reproduce realistic daily demand variation.

## Model

Common model structure, calibrated independently per period:

```text
demand_mw ~ C(hour) * C(day_type) + C(month)
```

Monte Carlo simulation adds bootstrapped full-day historical residual patterns to model-predicted 24-hour profiles. This preserves within-day correlation better than independent hourly random noise.

See [`docs/simulation_design.md`](docs/simulation_design.md) for the full design.

## Repository structure

```text
.
├── data/
│   ├── raw/                 # Put NGCP Excel workbook here (not committed)
│   └── processed/           # Generated cleaned hourly CSV
├── docs/
│   └── simulation_design.md
├── figures/                 # Generated plots
├── models/                  # Saved OLS models and residual libraries
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_exploratory_analysis.ipynb
│   ├── 03_normalization.ipynb
│   ├── 04_model_development.ipynb
│   ├── 05_monte_carlo_simulation.ipynb
│   ├── 06_validation.ipynb
│   └── 07_period_comparison.ipynb
├── results/                 # Generated CSV results
├── src/
│   ├── config.py
│   ├── data_preparation.py
│   ├── eda.py
│   ├── normalization.py
│   ├── modeling.py
│   ├── simulation.py
│   ├── validation.py
│   ├── comparison.py
│   └── pipeline.py
├── tests/
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS/Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Add the NGCP dataset

Place the official workbook at:

```text
data/raw/Hourly Demand per Grid.xlsx
```

The project intentionally does **not** commit the raw public dataset into Git so the source file can be downloaded/verified independently.

## Run the complete pipeline

```bash
python -m src.pipeline "data/raw/Hourly Demand per Grid.xlsx"
```

Optional:

```bash
python -m src.pipeline "data/raw/Hourly Demand per Grid.xlsx" --sheet Luzon --runs 1000 --seed 42
```

The pipeline will:

1. reshape and clean the Luzon data,
2. add time features and study-period labels,
3. calculate normalized demand,
4. produce exploratory analysis outputs,
5. calibrate one common model separately for all three periods,
6. run Monte Carlo simulation,
7. validate against chronologically withheld data,
8. compare normalized profiles and classify recovery behavior.

## Important data check

Before reporting peak-demand clock times, inspect the source workbook's 24 hourly column labels. The loader handles common formats and emits a warning when it must assume that a `1..24` layout maps sequentially to analysis hours `0..23`.

## Main generated results

Expected CSV outputs include:

- `data_quality_report.csv`
- `period_summary.csv`
- `hourly_profile_absolute.csv`
- `day_type_profile_absolute.csv`
- `daily_peaks.csv`
- `peak_hour_distribution.csv`
- `normalized_hourly_profiles.csv`
- one simulation output set per period
- `validation_metrics.csv`
- `pairwise_normalized_profile_rmse.csv`
- `recovery_classification.csv`
