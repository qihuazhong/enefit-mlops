"""No feature of day D may depend on an input that was not available when D was scored.

Availability (enefit.data.available_at, verified against the Kaggle example files):
  targets            date <= D-2          (revealed with a 2-day delay)
  client             date <= D-2          (usable 2 days after its date)
  electricity price  forecast_date <= D-1 (usable 24 h after forecast_date)
  historical weather only through lags >= 48 h, i.e. datetime <= D-2 23:00 (stricter than the API's D-1 10:00)
  forecast weather   data_block_id <= block(D), i.e. forecasts issued up to D-1 02:00; forecast hours of D
                     itself are allowed, the forecast issued ON D (block D+1) is not.

We perturb (large noise / NaN / row deletion) everything outside these rules in the full-information batch
build and require D's feature rows to be unchanged; a positive control proves the test can fail.
"""
import numpy as np
import pandas as pd
import pytest

from enefit import data, features

from _util import diff_columns, feature_columns
from conftest import SCORING_DAYS

DAYS = SCORING_DAYS[::2][:2] + [SCORING_DAYS[-1]]      # one test-block day, two live-run days
DAY = pd.Timedelta(days=1)


def _window(raw, d):
    return data.time_slice(raw, d - pd.Timedelta(days=48), d + pd.Timedelta(days=3))


def _noise(s: pd.Series, mode: str) -> pd.Series:
    if mode == "nan":
        return s * np.nan
    rng = np.random.default_rng(7)
    return s + rng.normal(0, 1, len(s)) * (np.abs(s).mean() + 1.0) * 1e3


def _perturb(w: data.RawTables, d: pd.Timestamp, mode: str) -> data.RawTables:
    """Corrupt every row not available at D. mode: noise | nan | delete (targets cannot be deleted: the
    train rows are the keys of the output, so for them 'delete' means NaN)."""
    b = data.day_index(d)

    def hit(df, mask, cols):
        df = df.copy()
        if mode == "delete":
            return df[~mask]
        for c in cols:
            df.loc[mask, c] = _noise(df.loc[mask, c].astype("float64"), mode)
        return df

    tr = w.train.copy()
    m = tr.datetime >= d - DAY
    tr.loc[m, "target"] = _noise(tr.loc[m, "target"], "nan" if mode == "delete" else mode)

    cl = hit(w.client, w.client.date >= d - DAY, ["eic_count", "installed_capacity"])
    ep = hit(w.electricity_prices, w.electricity_prices.forecast_date >= d, ["euros_per_mwh"])
    hw = hit(w.historical_weather, w.historical_weather.datetime >= d - DAY,
             ["temperature", "dewpoint", "cloudcover_total", "direct_solar_radiation", "diffuse_radiation"])
    fmask = w.forecast_weather.data_block_id > b
    fcols = ["temperature", "dewpoint", "cloudcover_low", "cloudcover_total", "direct_solar_radiation",
             "10_metre_u_wind_component", "total_precipitation", "surface_solar_radiation_downwards", "snowfall"]
    fw = hit(w.forecast_weather, fmask, fcols)
    return w._replace(train=tr, client=cl, electricity_prices=ep, historical_weather=hw, forecast_weather=fw)


def _rows(df, d):
    return df[df.date == d].reset_index(drop=True)


@pytest.fixture(scope="module", params=DAYS)
def case(request, raw):
    d = pd.Timestamp(request.param)
    w = _window(raw, d)
    return d, w, _rows(features.build_features(w), d)


@pytest.mark.parametrize("mode", ["noise", "nan", "delete"])
def test_unavailable_inputs_do_not_change_day_d(case, mode):
    d, w, base = case
    pert = _rows(features.build_features(_perturb(w, d, mode)), d)
    bad = diff_columns(base, pert, feature_columns(base))
    assert not bad, f"features of {d.date()} depend on unavailable inputs ({mode}): {bad}"


# ---- positive controls: the same machinery must flag a real dependency -------------------------------
def test_positive_control_revealed_target_matters(raw):
    d = pd.Timestamp(DAYS[1])
    w = _window(raw, d)
    base = _rows(features.build_features(w), d)
    tr = w.train.copy()
    m = (tr.datetime >= d - 2 * DAY) & (tr.datetime < d - DAY)               # D-2: revealed
    tr.loc[m, "target"] = tr.loc[m, "target"] * 5 + 10
    pert = _rows(features.build_features(w._replace(train=tr)), d)
    assert {"lag_2_day", "rolling_mean_24"} <= set(diff_columns(base, pert, feature_columns(base)))
