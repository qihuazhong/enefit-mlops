"""Weather: 112-point grid -> 16 counties, then the weather features of v6 (cells 14 and 30).

Port notes
- Coordinates come from the pixel-weight file (the notebook read the *global* historical-weather
  frame for them); we assert both agree so the matmul columns cannot be mis-aligned.
- Hours are sorted explicitly (the notebook relied on `unique()` appearance order matching the
  sorted groupby labels, which holds for the CSV but not for API-ordered frames).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

_EXTENSION_HOURS = 48  # notebook: pad each county with max_time + 1..48 h so lags reach beyond the data


def grid_coordinates(weights: pd.DataFrame) -> pd.MultiIndex:
    """(latitude, longitude) of the 112 pixels, in the column order of county_pixel_weights.csv."""
    cols = [c for c in weights.columns if c != "county_id"]
    coords = pd.MultiIndex.from_tuples(
        [tuple(round(float(x), 1) for x in c.split()) for c in cols], names=["latitude", "longitude"]
    )
    if not coords.is_monotonic_increasing:
        raise ValueError("pixel-weight columns are not sorted by (latitude, longitude)")
    if list(weights["county_id"]) != list(range(C.N_COUNTIES)):
        raise ValueError("pixel-weight rows must be counties 0..15")
    return coords


def _shift(frame, periods: int):
    """Shift within each county of a (county, datetime)-sorted frame (v6 shifted across counties)."""
    return frame.groupby(level="county", sort=False).shift(periods)


def county_hourly(df: pd.DataFrame, time_column: str, value_columns: list[str], weights: pd.DataFrame) -> pd.DataFrame:
    """Weighted grid -> county aggregation. Returns a frame indexed (county, datetime), sorted.

    Duplicate grid points are averaged, missing ones (and missing hours) become NaN, which then
    propagates through the matmul exactly as in the notebook.
    """
    coords = grid_coordinates(weights)
    df = df[[time_column, "latitude", "longitude"] + value_columns].copy()
    df[["latitude", "longitude"]] = df[["latitude", "longitude"]].astype(np.float64).round(1)

    times = np.sort(df[time_column].unique())
    if len(times):   # complete hourly grid, so lags are time shifts across DST holes
        times = pd.date_range(times[0], times[-1], freq="h").values
    times = pd.DatetimeIndex(times)

    mean = df.groupby([time_column, "latitude", "longitude"])[value_columns].mean()
    full = pd.MultiIndex.from_arrays(
        [np.repeat(times.values, len(coords)),
         np.tile(coords.get_level_values("latitude"), len(times)),
         np.tile(coords.get_level_values("longitude"), len(times))],
        names=[time_column, "latitude", "longitude"],
    )
    arr = mean.reindex(full).to_numpy().reshape(len(times), len(coords), len(value_columns))
    w = weights.set_index("county_id").to_numpy().T                     # (112, 16)
    county_arr = np.swapaxes(np.swapaxes(arr, 1, 2) @ w, 1, 2)           # (T, 16, n)

    index = pd.MultiIndex.from_arrays(
        [np.tile(np.arange(C.N_COUNTIES), len(times)), np.repeat(times.values, C.N_COUNTIES)],
        names=["county", "datetime"],
    )
    out = pd.DataFrame(county_arr.reshape(-1, len(value_columns)), columns=value_columns, index=index)
    return out.sort_index()


def _pad_after_max(frame: pd.DataFrame, max_time: pd.Timestamp) -> pd.DataFrame:
    extra = pd.MultiIndex.from_tuples(
        [(county, max_time + pd.Timedelta(hours=i)) for county in range(C.N_COUNTIES)
         for i in range(1, _EXTENSION_HOURS + 1)],
        names=["county", "datetime"],
    )
    return frame.reindex(frame.index.append(extra)).sort_index()


def historical_weather_features(hist: pd.DataFrame, weights: pd.DataFrame, cfg: C.FeatureConfig) -> pd.DataFrame:
    """Observed weather, only ever exposed through lags >= 2 days (the observation of hour t-48h is
    the newest one the scorer holds). Columns carry the `_h` suffix."""
    w = county_hourly(hist, "datetime", C.HISTORICAL_WEATHER_COLUMNS, weights)
    w = _pad_after_max(w, hist["datetime"].max())

    pf = C.HISTORICAL_POINT_FEATURES
    w[pf] = (_shift(w[pf], C.POINT_SMOOTHING_SHIFT) + w[pf]) / 2

    lag_features = C.HISTORICAL_LAG_FEATURES
    out = pd.DataFrame(index=w.index)
    for lag_day in C.HISTORICAL_LAG_DAYS:
        shifted = _shift(w[lag_features], 24 * lag_day)
        for ft in lag_features:
            out[f"{ft}_lag_{lag_day}_day_h"] = shifted[ft]
    mean7 = sum(_shift(w[lag_features], 24 * i) for i in range(2, 9)) / 7
    for ft in lag_features:
        out[f"{ft}_7_day_mean_h"] = mean7[ft]

    # NB: `cloudcover_total` is divided by 100 in the notebook *after* these lags were taken, so the
    # `_h` cloud-cover lags stay in percent while the forecast cloud cover is a fraction. Ported as-is.
    keep = [
        "direct_solar_radiation_lag_2_day_h", "direct_solar_radiation_lag_7_day_h", "direct_solar_radiation_7_day_mean_h",
        "cloudcover_total_lag_2_day_h", "cloudcover_total_lag_7_day_h", "cloudcover_total_7_day_mean_h",
        "diffuse_radiation_lag_2_day_h", "diffuse_radiation_lag_7_day_h",
    ]
    return out[keep].reset_index()


def select_forecast_rows(forecast: pd.DataFrame, cfg: C.FeatureConfig) -> pd.DataFrame:
    """`forecast_weather` holds, for every hour, one forecast per issue time (usually two:
    issued the previous and the same day at 02:00). The same-day one is delivered by the API only
    with the *next* call (data_block_id = day(forecast_datetime) + 1), after the day was scored."""
    if cfg.forecast_origin == "all_origins":
        return forecast
    origin_day = forecast["origin_datetime"].dt.normalize()
    prev_day = forecast["forecast_datetime"].dt.normalize() - pd.Timedelta(days=1)
    # Every origin the API has delivered when day D is predicted (issued on D-1 or earlier); hours
    # covered by more than one such origin (D 00:00-02:00: issued D-2 and D-1) are averaged, as in v6.
    return forecast[origin_day <= prev_day]


def forecast_weather_features(forecast: pd.DataFrame, hist_features: pd.DataFrame, weights: pd.DataFrame,
                              cfg: C.FeatureConfig) -> pd.DataFrame:
    """County-level forecast weather and its lag / difference features (v6 `get_weighted_forecast_by_county`)."""
    forecast = select_forecast_rows(forecast, cfg)
    w = county_hourly(forecast, "forecast_datetime", C.FORECAST_WEATHER_COLUMNS, weights)

    pf = C.FORECAST_POINT_FEATURES
    w[pf] = (_shift(w[pf], C.POINT_SMOOTHING_SHIFT) + w[pf]) / 2

    new: dict[str, pd.Series | np.ndarray] = {}
    lag_features = C.FORECAST_LAG_FEATURES
    for lag_day in C.FORECAST_LAG_DAYS:
        lagged = _shift(w[lag_features], 24 * lag_day)
        for ft in lag_features:
            new[f"{ft}_lag_{lag_day}_day"] = lagged[ft]
            new[f"{ft}_{lag_day}_day_diff"] = w[ft].values - lagged[ft].values

    special = C.FORECAST_SPECIAL_LAG_FEATURES
    for lag_day in C.FORECAST_SPECIAL_LAG_DAYS:
        lagged = _shift(w[special], 24 * lag_day + 1)
        for ft in special:
            new[f"{ft}_lag_{lag_day}_day_1h"] = lagged[ft]
            new[f"{ft}_{lag_day}_day_1h_diff"] = w[ft].values - lagged[ft].values

    new["temperature_lag_1h"] = _shift(w["temperature"], 1)

    for suffix, days in C.FORECAST_MEAN_WINDOWS.items():
        mean = sum(_shift(w[lag_features], 24 * i) for i in days) / 7
        for ft in lag_features:
            new[f"{ft}_{suffix}"] = mean[ft]
            new[f"{ft}_{suffix}_diff"] = w[ft].values - mean[ft].values

    snow = w["snowfall"]
    rolled = snow.groupby(level="county", sort=False).transform(lambda s: s.rolling(24).sum())
    new["snowfall_1_day_rolling_sum"] = rolled.values
    snow_roll = pd.Series(new["snowfall_1_day_rolling_sum"], index=w.index)
    for i in (2, 7):
        new[f"snowfall_1_day_rolling_sum_lag_{i}_day_diff"] = _shift(snow_roll, 24 * i).values - snow_roll.values

    new["temp_minus_dewpoint"] = w["temperature"] - w["dewpoint"]

    out = pd.concat([w, pd.DataFrame(new, index=w.index)], axis=1).reset_index()
    out = out.merge(hist_features, how="left", on=["county", "datetime"])

    # Override the forecast-only differences with ones against the *observed* history
    for ft, hist_ft in (("direct_solar_radiation", "direct_solar_radiation"), ("cloudcover_total", "cloudcover_total")):
        out[f"{ft}_7_day_mean_diff"] = out[ft].values - out[f"{hist_ft}_7_day_mean_h"].values
        out[f"{ft}_2_day_diff"] = out[ft].values - out[f"{hist_ft}_lag_2_day_h"].values
        out[f"{ft}_7_day_diff"] = out[ft].values - out[f"{hist_ft}_lag_7_day_h"].values
    return out
