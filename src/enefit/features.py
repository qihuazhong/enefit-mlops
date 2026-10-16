"""The ONE feature builder, used for training (batch over the whole period) and for scoring.

Scoring day D is not a separate code path:

    features_for_day(raw, D) == build_features(available_at(raw, D)) restricted to the rows of D

so the only thing that differs between training and serving is *which inputs are visible*, and
that is decided in one place (`enefit.data.available_at`). This replaces the notebook's training
cells (35-47) and `append_lag` (a second copy of the same logic).

Port of v6 (cells 11, 14-16, 30-47); comments explain *why* (availability) rather than *what*.
Run ``python -m enefit.features --build [--feature-config NAME|all]`` to write
data/features/features_<name>_<confighash8>.parquet (+ .json sidecar), see ``config.features_path``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import data, targets, weather

SEG = C.SEGMENT_KEYS
_TARGET_COLUMNS = ["target"] + C.DERIVED_TARGET_COLUMNS
CODE_FILES = ["config.py", "data.py", "weather.py", "targets.py", "features.py"]   # the code a feature table depends on


def code_hash() -> str:
    """sha256 over CODE_FILES: a table built by other code is stale (train.feature_table rebuilds it)."""
    h = hashlib.sha256()
    for name in CODE_FILES:
        h.update((Path(__file__).parent / name).read_bytes())
    return h.hexdigest()


@contextmanager
def _stage(name: str, quiet: bool = False):
    t = time.time()
    yield
    if not quiet:
        print(f"[features] {name}: {time.time() - t:.1f}s", flush=True)


# --------------------------------------------------------------------------------- prices (cell 32)
def process_elec_prices(prices: pd.DataFrame, cfg: C.FeatureConfig) -> pd.DataFrame:
    """Price of forecast_date f is usable from f + 24 h (the row for hour t carries the price of t - 24 h).

    Reindexed to a complete hourly grid first so lags are time shifts, not row counts.
    """
    df = prices[["forecast_date", "euros_per_mwh"]].drop_duplicates("forecast_date", keep="last")
    df = df.sort_values("forecast_date")
    if len(df):
        grid = pd.date_range(df["forecast_date"].min(), df["forecast_date"].max(), freq="h", name="forecast_date")
        df = df.set_index("forecast_date").reindex(grid).reset_index()
    df = df.reset_index(drop=True)
    df["data_available_datetime"] = df["forecast_date"] + pd.Timedelta(hours=C.PRICE_DELAY_HOURS)

    x = df["euros_per_mwh"]
    h = cfg.price_hours_per_day
    lags = {i: x.shift(h * i) for i in C.PRICE_LAG_DAYS}
    mean7 = sum(lags[i] for i in C.PRICE_LAG_DAYS) / len(C.PRICE_LAG_DAYS)
    df["euros_per_mwh_lag_7_day_mean_diff"] = x - mean7
    df["euros_per_mwh_lag_7_day_diff"] = x - lags[7]
    df["euros_per_mwh_lag_2_day_diff"] = x - lags[2]
    return df.drop(columns="forecast_date")


# ----------------------------------------------------------------- missing hours (cell 36, vectorised)
def fill_missing_time(df: pd.DataFrame) -> pd.DataFrame:
    """Complete hourly grid for every segment that has at least one row (so positional shifts are
    time shifts), plus `consumption_counterpart` = target of the other side of the same unit.
    Rows added here have NaN everywhere (also in client / weather / price columns, as in v6 where
    the merges happen before this step)."""
    hours = pd.date_range(df["datetime"].min(), df["datetime"].max(), freq="h")
    segs = df[SEG].drop_duplicates().sort_values(SEG).to_numpy()
    n_h = len(hours)
    full = pd.MultiIndex.from_arrays(
        [np.repeat(segs[:, k], n_h) for k in range(4)] + [np.tile(hours.values, len(segs))],
        names=SEG + ["datetime"],
    )
    out = df.set_index(SEG + ["datetime"])
    if not out.index.is_unique:
        raise ValueError("duplicate (segment, datetime) rows in the input")
    out = out.reindex(full)

    flipped = pd.MultiIndex.from_arrays(
        [full.get_level_values(0), full.get_level_values(1), full.get_level_values(2),
         1 - full.get_level_values(3).to_numpy(), full.get_level_values(4)], names=full.names)
    out["consumption_counterpart"] = out["target"].reindex(flipped).to_numpy()
    out = out.reset_index()
    out["date"] = out["datetime"].dt.normalize()
    return out


# --------------------------------------------------------------------------------- calendar (cells 44-47)
def _daytype(dt: pd.Series) -> pd.Series:
    dow = dt.dt.dayofweek
    day = dt.dt.normalize()
    one = pd.Timedelta(days=1)
    nat = day.isin(C.NATIONAL_HOLIDAY_DATES)
    pre_nat = day.isin([d - one for d in C.NATIONAL_HOLIDAY_DATES])
    post_nat = day.isin([d + one for d in C.NATIONAL_HOLIDAY_DATES])
    out = pd.Series(np.nan, index=dt.index, dtype="object")
    out[(dow >= 1) & (dow <= 3)] = "inner_weekday"
    out[(dow == 4) | pre_nat] = "pre_holiday"
    out[(dow == 0) | post_nat] = "post_holiday"
    out[(dow >= 5) | nat] = "holiday"
    return out.astype(pd.CategoricalDtype(C.CATEGORIES["daytype"]))


def add_calendar(df: pd.DataFrame, cfg: C.FeatureConfig) -> pd.DataFrame:
    dt = df["datetime"]
    df["dayofweek"] = dt.dt.dayofweek.astype(pd.CategoricalDtype(C.CATEGORIES["dayofweek"]))
    df["day"] = dt.dt.day

    df["daytype"] = _daytype(dt)
    for k in (2, 3, 7):
        # calendar fact: no need to shift anything, hence no boundary leak
        df[f"daytype_lag_{k}_day"] = _daytype(dt - pd.Timedelta(days=k))

    hour = dt.dt.hour
    df["sin_hour"] = np.sin(2 * np.pi * hour / 24)
    df["cos_hour"] = np.cos(2 * np.pi * hour / 24)
    df["hour"] = hour.astype(pd.CategoricalDtype(C.CATEGORIES["hour"]))
    df["county"] = df["county"].astype(pd.CategoricalDtype(C.CATEGORIES["county"]))
    df["product_type"] = df["product_type"].astype(pd.CategoricalDtype(C.CATEGORIES["product_type"]))
    return df



def _exact_rolling_stats(values: np.ndarray, sizes: np.ndarray, window: int = 24) -> dict[str, np.ndarray]:
    """mean / std (ddof=1) / skew / excess kurtosis of every full 24-row window, computed from scratch.

    Same estimators as pandas' `rolling().skew()/.kurt()` (adjusted Fisher-Pearson, NaN if the window holds
    a NaN or has ~zero variance) but without pandas' running sums, whose rounding error depends on the data
    that precedes the window. `values` is the concatenation of contiguous per-segment series.
    """
    n = window
    out = {k: np.full(len(values), np.nan) for k in ("mean", "std", "skew", "kurt")}
    start = 0
    # NaN targets (skeleton rows of the scored day) must propagate silently
    with np.errstate(invalid="ignore", divide="ignore"):
        for size in sizes:
            if size >= n:
                v = np.lib.stride_tricks.sliding_window_view(values[start:start + size], n)
                mean = v.mean(axis=1)
                d = v - mean[:, None]
                d2 = d * d
                m2, m3, m4 = d2.mean(axis=1), (d2 * d).mean(axis=1), (d2 * d2).mean(axis=1)
                sl = slice(start + n - 1, start + size)
                ok = m2 > 1e-14                      # pandas also returns NaN for ~zero variance
                out["mean"][sl] = mean
                out["std"][sl] = np.sqrt(m2 * n / (n - 1))
                out["skew"][sl] = np.where(ok, np.sqrt(n * (n - 1.0)) / (n - 2) * m3 / m2 ** 1.5, np.nan)
                out["kurt"][sl] = np.where(
                    ok, ((n * n - 1.0) * m4 / (m2 * m2) - 3 * (n - 1.0) ** 2) / ((n - 2.0) * (n - 3.0)), np.nan)
            start += size
    return out


# ----------------------------------------------------------------------------- lag features (cell 41)
def add_lags(df: pd.DataFrame, cfg: C.FeatureConfig) -> pd.DataFrame:
    """`df` must be sorted by (segment, datetime) with a complete hourly grid per segment.

    Targets are revealed with a 2-day delay, so no target lag is shorter than BASE_LAG_HOURS (48 h)."""
    grouped = df.groupby(SEG, sort=False)
    group_id = grouped.ngroup()
    new: dict[str, pd.Series | np.ndarray] = {}
    fill = (lambda s: s.fillna(cfg.lag_fill_value)) if cfg.lag_fill_value is not None else (lambda s: s)

    def shifted(series: pd.Series, n: int) -> pd.Series:
        return series.groupby(group_id, sort=False).shift(n)

    # client: usable 2 days after its date (merged on available_date); diffs are against the same series
    for col in C.CLIENT_DIFF_COLUMNS:
        g = grouped[col]
        new[f"{col}_7_day_diff"] = g.shift(0) - g.shift(C.CLIENT_DIFF_LAG_HOURS)
        new[f"{col}_7_day_mean_diff"] = g.shift(0) - np.mean([g.shift(24 * i) for i in C.CLIENT_MEAN_DIFF_DAYS], axis=0)

    gt = grouped["target"]
    for i in C.TARGET_LAG_DAYS:
        new[f"lag_{i}_day"] = fill(gt.shift(24 * i))
    for i in C.TARGET_LAG_1H_DAYS:
        new[f"lag_{i}_day_1h"] = fill(gt.shift(24 * i + 1))
    new["lag_7_day_mean"] = np.mean([new[f"lag_{i}_day"] for i in C.LAG_MEAN_7_DAYS], axis=0)
    new["lag_7_day_1h_mean"] = np.mean([new[f"lag_{i}_day_1h"] for i in C.LAG_MEAN_7_DAYS], axis=0)
    new["lag_8_14_day_mean"] = np.mean([new[f"lag_{i}_day"] for i in C.LAG_MEAN_8_14_DAYS], axis=0)
    # (c) these three deliberately have NO fill, as in v6 training *and* serving
    for name, hours in C.UNFILLED_LAG_HOURS.items():
        new[name] = gt.shift(hours)

    sizes = np.bincount(group_id.to_numpy())
    target_values = df["target"].to_numpy("float64")
    windows = _exact_rolling_stats(target_values, sizes, C.ROLLING_WINDOW)   # exact per-window stats

    def rolling(stat: str, shift: int) -> np.ndarray:
        return shifted(pd.Series(windows[stat], index=df.index), shift).to_numpy()

    for stat, name in (("mean", "rolling_mean_24"), ("std", "rolling_std_24"),
                       ("kurt", "rolling_kurt_24"), ("skew", "rolling_skew_24")):
        new[name] = rolling(stat, C.ROLLING_SHIFT_BASE)
    new["rolling_mean_24_lag_7_day"] = rolling("mean", C.ROLLING_SHIFT_WEEK)

    gc = grouped["consumption_counterpart" if cfg.counterpart_source == "counterpart" else "target"]
    for i in C.COUNTERPART_LAG_DAYS:
        new[f"lag_{i}_day_consumption_counterpart"] = gc.shift(24 * i)
    new["lag_2_day_1h_consumption_counterpart"] = gc.shift(C.COUNTERPART_1H_LAG_HOURS)
    new["lag_7_day_mean_consumption_counterpart"] = np.mean(
        [gc.shift(24 * i) for i in C.COUNTERPART_MEAN_DAYS], axis=0)

    df = pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)

    # target summed over product types of a (county, is_business, side), 2 days late
    key = ["county", "is_business", "is_consumption", "datetime"]
    tot = df[key + ["target"]].groupby(key)[["target"]].sum()
    tot = tot.groupby(level=[0, 1, 2]).shift(C.TOTAL_TARGET_SHIFT_HOURS)
    tot = tot.rename(columns={"target": "target_all_type"}).reset_index()
    return df.merge(tot, on=key, how="left")


def clip_features(df: pd.DataFrame, bound: float) -> pd.DataFrame:
    """`test[col].clip(-1e6, 1e6)` in the serving notebook discarded its result. Here the clipped
    column is assigned back. float16 columns cannot exceed 65504 and are skipped, as in the notebook."""
    for col in df.columns:
        if df[col].dtype in (np.float32, np.float64):
            df[col] = df[col].clip(-bound, bound)
    return df


# ------------------------------------------------------------------------------------ main builder
def build_features(raw: data.RawTables, cfg: C.FeatureConfig = C.SERVING_FEATURE_CONFIG,
                   quiet: bool = True) -> pd.DataFrame:
    """Features (and `target`) for every row of ``raw.train``, using only information that is visible
    through the tables passed in. One row per row of train.csv; rows whose target is NaN (skeleton rows
    of the day being scored, or genuinely missing labels) are kept.

    Timestamps in a row's feature never reach beyond what `available_at` would have exposed: the rules are
    in `enefit.data.available_at` and enforced by tests/test_no_lookahead.py.
    """
    st = lambda name: _stage(name, quiet)  # noqa: E731
    with st("rows"):
        df = raw.train[["county", "is_business", "product_type", "target", "is_consumption", "datetime",
                        "data_block_id", "row_id", "prediction_unit_id"]].copy()
        df["date"] = df["datetime"].dt.normalize()

    with st("prices"):
        df = df.merge(process_elec_prices(raw.electricity_prices, cfg), how="left",
                      left_on="datetime", right_on="data_available_datetime")

    with st("client"):
        client = raw.client.copy()
        client["available_date"] = client["date"] + pd.Timedelta(days=C.CLIENT_DELAY_DAYS)
        client = client.drop(columns=["data_block_id", "date"]).drop_duplicates(
            ["county", "is_business", "product_type", "available_date"], keep="last")
        df = df.merge(client, how="left", left_on=["county", "is_business", "product_type", "date"],
                      right_on=["county", "is_business", "product_type", "available_date"])
        # a NaN-free window would keep eic_count int64 and change the dtype path of every diff below
        df["eic_count"] = df["eic_count"].astype("float64")

    with st("weather"):
        hist = weather.historical_weather_features(raw.historical_weather, raw.county_weights, cfg)
        fc = weather.forecast_weather_features(raw.forecast_weather, hist, raw.county_weights, cfg)
        df = df.merge(fc, how="left", on=["county", "datetime"])

    if cfg.float_dtype is not None:
        with st("cast_floats"):
            for col in df.columns:
                if df[col].dtype == np.float64:
                    df[col] = df[col].astype(np.float32).astype(cfg.float_dtype)

    with st("fill_missing_time"):
        df = fill_missing_time(df)
    with st("lags"):
        df = add_lags(df, cfg)
    with st("calendar"):
        df = add_calendar(df, cfg)

    df = df[df["row_id"].notna()].copy()
    for col in ("row_id", "data_block_id", "prediction_unit_id"):   # ids first: clipping must never touch them
        df[col] = df[col].astype("int64")
    df = clip_features(df, C.CLIP_BOUND)
    df = df.drop(columns=["data_available_datetime", "available_date"])
    return df.sort_values("row_id").reset_index(drop=True)


def features_for_day(raw: data.RawTables, day, cfg: C.FeatureConfig = C.SERVING_FEATURE_CONFIG,
                     history_days: int = C.SCORING_HISTORY_DAYS) -> pd.DataFrame:
    """The scoring path: same builder, inputs cut at D's availability cutoff, keep D's rows."""
    feats = build_features(data.available_at(raw, day, history_days), cfg)
    return feats[feats["date"] == data.as_ts(day)].reset_index(drop=True)


# ---------------------------------------------------------------------------------------------- CLI
def build_and_write(cfg: C.FeatureConfig = C.SERVING_FEATURE_CONFIG) -> pd.DataFrame:
    """Full period (config.FEATURE_START..FEATURE_END) -> config.features_path(cfg) + .json sidecar."""
    out, start, end = C.features_path(cfg), C.FEATURE_START, C.FEATURE_END
    t0 = time.time()
    with _stage("load_raw"):
        raw = data.load_raw_tables()
    raw = raw._replace(train=raw.train[(raw.train.datetime >= data.as_ts(start))
                                      & (raw.train.datetime < data.as_ts(end) + pd.Timedelta(days=1))])
    print(f"[features] building {cfg.name} for {start}..{end}: {len(raw.train):,} train rows", flush=True)
    df = build_features(raw, cfg, quiet=False)
    with _stage("derived_targets"):
        df = targets.add_derived_targets(df)
    with _stage("select_and_write"):
        cols = [c for c in C.OUTPUT_COLUMNS if c in df.columns]
        missing = set(C.OUTPUT_COLUMNS) - set(cols)
        if missing:
            raise RuntimeError(f"builder did not produce {sorted(missing)}")
        out_df = df[cols].copy()
        for c in out_df.columns:                       # float16 -> float32 is lossless; parquet has no float16
            if out_df[c].dtype == np.float16:
                out_df[c] = out_df[c].astype(np.float32)
        out.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_parquet(out, index=False)
    with _stage("input_hash"):
        hashes = data.input_hashes()
    meta = {
        "rows": int(len(out_df)),
        "columns": list(out_df.columns),
        "date_range": [str(out_df["datetime"].min()), str(out_df["datetime"].max())],
        "feature_config": cfg.__dict__,
        "feature_config_hash": C.feature_config_hash(cfg),
        "parquet_file": out.name,
        "parquet_sha256": data.file_sha256(out),
        "input_hash": data.combined_input_hash(hashes),
        "input_file_hashes": hashes,
        "code_hash": code_hash(),
        "git_commit": C.git_commit(),
        "build_seconds": round(time.time() - t0, 1),
        "versions": {"pandas": pd.__version__, "numpy": np.__version__},
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"[features] wrote {out} shape={out_df.shape} in {meta['build_seconds']}s", flush=True)
    return out_df


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Build the v6 feature table")
    ap.add_argument("--build", action="store_true", help="build features for the full period")
    ap.add_argument("--feature-config", default="serving", choices=sorted(C.FEATURE_CONFIGS) + ["all"],
                    help="all: every table any milestone uses (the AWS processing job)")
    args = ap.parse_args(argv)
    if not args.build:
        ap.print_help()
        return
    if args.feature_config == "all":
        cfgs = {V[k] for V in map(C.variant, C.VARIANTS) for k in ("train_cfg", "serving_cfg")}
    else:
        cfgs = {C.FEATURE_CONFIGS[args.feature_config]}
    for cfg in sorted(cfgs, key=lambda c: C.features_path(c).name):
        build_and_write(cfg)


if __name__ == "__main__":
    main(sys.argv[1:])
