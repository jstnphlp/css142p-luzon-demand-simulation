import numpy as np
import pandas as pd

from src.data_preparation import add_time_features, reshape_luzon_to_hourly
from src.modeling import fit_period_model
from src.simulation import simulate_days


def _synthetic_wide(start="2017-01-01", days=120):
    dates = pd.date_range(start, periods=days, freq="D")
    rows = {"Date": dates}
    rng = np.random.default_rng(7)
    for h in range(24):
        base = 9000 + 1400 * np.sin((h - 7) / 24 * 2 * np.pi)
        rows[f"{h:02d}:00"] = base + rng.normal(0, 150, size=days)
    return pd.DataFrame(rows)


def test_reshape_creates_24_rows_per_day():
    wide = _synthetic_wide(days=5)
    long = reshape_luzon_to_hourly(wide, date_col="Date")
    assert len(long) == 5 * 24
    assert sorted(long["hour"].unique().tolist()) == list(range(24))


def test_model_and_simulation_generate_complete_days():
    wide = _synthetic_wide(days=120)
    long = add_time_features(reshape_luzon_to_hourly(wide, date_col="Date"))
    long = long[long["period"] == "Pre-pandemic"].copy()
    bundle = fit_period_model(long, "Pre-pandemic", test_fraction=0.2)
    sim = simulate_days(bundle.model, bundle.residual_library, bundle.train, n_runs=20, seed=3, period="Pre-pandemic")
    assert len(sim) == 20 * 24
    assert sim.groupby("simulation_id")["hour"].nunique().eq(24).all()
    assert (sim["demand_mw"] >= 0).all()
