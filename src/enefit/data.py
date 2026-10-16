"""Raw data access: typed CSV loading, Parquet cache, availability slicing.

The Parquet cache keeps the raw columns in their native dtypes (float64): any downcasting is a
feature-builder decision (config.FeatureConfig.float_dtype), never a storage decision.
"""
from __future__ import annotations

import hashlib
import json
import time
from typing import NamedTuple

import numpy as np
import pandas as pd

from . import config as C

_DATETIME_COLUMNS = {
    "train": ["datetime"],
    "client": ["date"],
    "electricity_prices": ["forecast_date", "origin_date"],
    "forecast_weather": ["origin_datetime", "forecast_datetime"],
    "historical_weather": ["datetime"],
}
_TABLES = tuple(_DATETIME_COLUMNS)


class RawTables(NamedTuple):
    train: pd.DataFrame
    client: pd.DataFrame
    electricity_prices: pd.DataFrame
    forecast_weather: pd.DataFrame
    historical_weather: pd.DataFrame
    county_weights: pd.DataFrame


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(C.RAW_FILES[name], parse_dates=_DATETIME_COLUMNS[name])


def parquet_path(name: str):
    return C.PROCESSED_DIR / f"{name}.parquet"


def build_parquet_cache(force: bool = False) -> None:
    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name in _TABLES:
        out = parquet_path(name)
        if out.exists() and not force and out.stat().st_mtime >= C.RAW_FILES[name].stat().st_mtime:
            continue
        t = time.time()
        df = read_csv(name)
        df.to_parquet(out, index=False)
        print(f"[data] cached {name}: {len(df):,} rows in {time.time() - t:.1f}s -> {out.name}", flush=True)


def load_county_weights() -> pd.DataFrame:
    return pd.read_csv(C.COUNTY_WEIGHTS_FILE)


def load_raw_tables() -> RawTables:
    """All raw tables (train, client, prices, forecast/historical weather, county weights), via the parquet cache."""
    build_parquet_cache()
    return RawTables(county_weights=load_county_weights(), **{n: pd.read_parquet(parquet_path(n)) for n in _TABLES})


# -------------------------------------------------------------------------------- provenance
def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def input_hashes() -> dict[str, str]:
    """sha256 of every raw input file (cached by size+mtime in data/processed/.hashes.json)."""
    cache_file = C.PROCESSED_DIR / ".hashes.json"
    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    files = {**C.RAW_FILES, "county_weights": C.COUNTY_WEIGHTS_FILE}
    out = {}
    for name, path in files.items():
        st = path.stat()
        key = f"{st.st_size}:{st.st_mtime_ns}"
        if cache.get(name, {}).get("key") != key:
            cache[name] = {"key": key, "sha256": file_sha256(path)}
        out[name] = cache[name]["sha256"]
    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(cache, indent=1))
    return out


def combined_input_hash(hashes: dict[str, str] | None = None) -> str:
    hashes = hashes or input_hashes()
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


# ------------------------------------------------------------------------------- availability
def day_index(day) -> int:
    """data_block_id of a calendar day in train.csv (2021-09-01 -> 0)."""
    return (pd.Timestamp(day).normalize() - pd.Timestamp(C.DATA_START)).days


def as_ts(day) -> pd.Timestamp:
    return pd.Timestamp(day).normalize()


def available_at(raw: RawTables, day, history_days: int = C.SCORING_HISTORY_DAYS) -> RawTables:
    """The inputs a scorer holds when it must predict calendar day ``day`` (the Kaggle API's call
    for data_block b = day_index(day)), cut to ``history_days`` of history.

    client / prices / forecast / historical weather: ``data_block_id <= b`` ("available at that block").
    train.csv is different: there ``data_block_id`` is the block in which the row was *predicted*,
    so a target revealed with call b has block <= b - 2. Rows of days D-1 and D are kept only as
    skeletons (target = NaN): they carry the keys and the client/weather merge targets, exactly like the
    notebook's `placeholder_previous_day` + test rows.
    """
    d = as_ts(day)
    b = day_index(d)
    start = d - pd.Timedelta(days=history_days)

    tr = raw.train
    tr = tr[(tr.datetime >= start) & (tr.datetime < d + pd.Timedelta(days=1))].copy()
    unrevealed = tr.datetime >= d - pd.Timedelta(days=C.TARGET_DELAY_DAYS - 1)
    tr.loc[unrevealed, "target"] = np.nan

    cl = raw.client
    cl = cl[(cl.data_block_id <= b) & (cl.date >= start - pd.Timedelta(days=3))]
    ep = raw.electricity_prices
    ep = ep[(ep.data_block_id <= b) & (ep.forecast_date >= start - pd.Timedelta(days=1))]
    fw = raw.forecast_weather
    fw = fw[(fw.data_block_id <= b) & (fw.forecast_datetime >= start)]
    hw = raw.historical_weather
    hw = hw[(hw.data_block_id <= b) & (hw.datetime >= start)]
    return raw._replace(train=tr, client=cl, electricity_prices=ep, forecast_weather=fw, historical_weather=hw)


def time_slice(raw: RawTables, start, end) -> RawTables:
    """Plain calendar slice [start, end] of every table (no availability logic): test helper to keep
    batch builds small. Weather/price tables are cut with the same margins as `available_at`."""
    s, e = as_ts(start), as_ts(end) + pd.Timedelta(days=1)
    return raw._replace(
        train=raw.train[(raw.train.datetime >= s) & (raw.train.datetime < e)],
        client=raw.client[(raw.client.date >= s - pd.Timedelta(days=3)) & (raw.client.date < e)],
        electricity_prices=raw.electricity_prices[
            (raw.electricity_prices.forecast_date >= s - pd.Timedelta(days=1)) & (raw.electricity_prices.forecast_date < e)],
        forecast_weather=raw.forecast_weather[
            (raw.forecast_weather.forecast_datetime >= s) & (raw.forecast_weather.forecast_datetime < e + pd.Timedelta(days=2))],
        historical_weather=raw.historical_weather[
            (raw.historical_weather.datetime >= s) & (raw.historical_weather.datetime < e + pd.Timedelta(days=2))],
    )
