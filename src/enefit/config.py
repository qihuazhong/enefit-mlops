"""Single source of truth for everything the Kaggle notebook (v6) hard-codes.

Values here were read from ``reference/kaggle_notebooks/v6.ipynb`` (cell numbers in comments).
Milestones are numbered v1-v6 in order; the Kaggle notebook versions they come from: v1 = Kaggle version 84, v2 = 125, v3 = 158, v4 = 190, v5 = 217, v6 = 238.
"""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------------------
# Paths (the notebook's /kaggle/input/... locations map onto these)
# --------------------------------------------------------------------------------------
ROOT = Path(os.environ.get("ENEFIT_ROOT", Path(__file__).resolve().parents[2]))
# On AWS, DATA_DIR is the job's read-only S3 mount and the caches go to local disk (Dockerfile sets ENEFIT_CACHE_DIR).
DATA_DIR = Path(os.environ.get("ENEFIT_DATA_DIR", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"                      # predict-energy-behavior-of-prosumers/
EXT_DIR = DATA_DIR / "ext" / "county-weights"   # enefit-county-coordinate-weights/
PROCESSED_DIR = Path(os.environ.get("ENEFIT_CACHE_DIR", DATA_DIR / "processed"))   # parquet cache of the raw CSVs
FEATURES_DIR = DATA_DIR / "features"

RAW_FILES = {
    "train": RAW_DIR / "train.csv",
    "client": RAW_DIR / "client.csv",
    "electricity_prices": RAW_DIR / "electricity_prices.csv",
    "forecast_weather": RAW_DIR / "forecast_weather.csv",
    "historical_weather": RAW_DIR / "historical_weather.csv",
    "weather_station_to_county_mapping": RAW_DIR / "weather_station_to_county_mapping.csv",
}
COUNTY_WEIGHTS_FILE = EXT_DIR / "county_pixel_weights.csv"

# --------------------------------------------------------------------------------------
# Fixed split (README) and timeline
# --------------------------------------------------------------------------------------
DATA_START = date(2021, 9, 1)            # data_block_id 0 in train.csv
TRAIN_START, TRAIN_END = date(2021, 9, 1), date(2023, 2, 28)
TEST_START, TEST_END = date(2023, 3, 1), date(2023, 4, 15)
LIVE_START, LIVE_END = date(2023, 4, 16), date(2023, 5, 31)    # held-out days the live run walks through
FEATURE_START, FEATURE_END = TRAIN_START, LIVE_END

# --------------------------------------------------------------------------------------
# Availability rules (no look-ahead). Days are relative to the day D being scored.
# --------------------------------------------------------------------------------------
TARGET_DELAY_DAYS = 2        # target of day D-2 is revealed with the call for D  (cell 41: base_lag = 48)
CLIENT_DELAY_DAYS = 2        # client row of date d is usable from d + 2          (cell 11: available_date)
PRICE_DELAY_HOURS = 24       # price row forecast_date f is usable from f + 1 day (cell 32)
BASE_LAG_HOURS = 48          # = TARGET_DELAY_DAYS * 24
SCORING_HISTORY_DAYS = 45    # >= 28 d price lags (+1 d price delay, + DST hole), 16 d target lags, 15 d weather lags

SEGMENT_KEYS = ["county", "is_business", "product_type", "is_consumption"]
SIDE_NAMES = {0: "production", 1: "consumption"}


def git_commit() -> str:
    """Commit of the code that runs: $GIT_COMMIT (set in the Docker image, which has no .git), else git, + "-dirty"."""
    if os.environ.get("GIT_COMMIT"):
        return os.environ["GIT_COMMIT"]
    def run(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return run("rev-parse", "HEAD") + ("-dirty" if run("status", "--porcelain", "--", "src", "tests") else "")

# --------------------------------------------------------------------------------------
# Feature-builder policy. The porting fixes (segment-safe shifts, DST-safe lags, clipping, exact rolling stats) are always on (v6 reproduced bit-for-bit
# with them switched off, verified at commit 48d1b05); the fields below are what model variants change.
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class FeatureConfig:
    name: str = "serving"
    # Forecast origins: v6 averages *every* forecast issued for an hour, including the one issued on the
    # day itself (block D+1), which the live API only delivers after D was scored.
    #   "all_origins" = v6 behaviour (training); "available_origins" = average of every origin issued
    #   by D-1, i.e. what the API has delivered when D is predicted (evaluation and serving).
    forecast_origin: str = "available_origins"
    # Precision of all float64 inputs right after the merges, before any lag is built.
    # "float16" = v6 (cell 38, via float32), "float32" = the later notebook, None = v5 (no cast).
    float_dtype: str | None = "float16"
    # D7: v2+ fill missing daily target lags with 0 (v6); None keeps NaN (v1).
    lag_fill_value: float | None = 0.0
    # D4: "counterpart" = target of the other side of the unit (v1, v5+); "own_target" = the bug of v2-v4
    counterpart_source: str = "counterpart"
    # D5: hours per "day" in the electricity-price lags: 24 (v3+) or 28 (typo in v2, log "Version 150")
    price_hours_per_day: int = 24


# The serving notebook meant to clip float features to +-1e6 but never assigned the result.
CLIP_BOUND = 1e6

# --------------------------------------------------------------------------------------
# Fixed categorical levels (inferred levels would change codes with the data window and break
# training/serving parity of the LightGBM categorical encoding)
# --------------------------------------------------------------------------------------
N_COUNTIES = 16
CATEGORIES = {
    "county": list(range(N_COUNTIES)),
    "product_type": [0, 1, 2, 3],
    "dayofweek": list(range(7)),
    "hour": list(range(24)),
    "daytype": ["holiday", "inner_weekday", "post_holiday", "pre_holiday"],
}

# --------------------------------------------------------------------------------------
# Weather (cells 14, 30)
# --------------------------------------------------------------------------------------
HISTORICAL_WEATHER_COLUMNS = [
    "temperature", "dewpoint", "cloudcover_high", "cloudcover_low", "cloudcover_mid",
    "cloudcover_total", "direct_solar_radiation", "diffuse_radiation", "shortwave_radiation",
    "snowfall", "rain",
]
HISTORICAL_POINT_FEATURES = [
    "temperature", "dewpoint", "cloudcover_high", "cloudcover_low", "cloudcover_mid", "cloudcover_total",
]
HISTORICAL_LAG_FEATURES = ["direct_solar_radiation", "cloudcover_total", "diffuse_radiation"]
HISTORICAL_LAG_DAYS = [2, 7]

FORECAST_WEATHER_COLUMNS = [
    "temperature", "dewpoint", "cloudcover_high", "cloudcover_low", "cloudcover_mid", "cloudcover_total",
    "10_metre_u_wind_component", "10_metre_v_wind_component", "direct_solar_radiation",
    "surface_solar_radiation_downwards", "snowfall", "total_precipitation",
]
FORECAST_POINT_FEATURES = [
    "temperature", "dewpoint", "cloudcover_high", "cloudcover_low", "cloudcover_mid", "cloudcover_total",
    "10_metre_u_wind_component", "10_metre_v_wind_component",
]
FORECAST_LAG_FEATURES = [
    "direct_solar_radiation", "temperature", "cloudcover_low", "dewpoint", "total_precipitation",
    "surface_solar_radiation_downwards",
]
FORECAST_LAG_DAYS = [2, 3, 7]
FORECAST_SPECIAL_LAG_FEATURES = ["direct_solar_radiation", "temperature"]
FORECAST_SPECIAL_LAG_DAYS = [2, 7]       # 1-hour-offset lags: shift(24*d + 1)
FORECAST_MEAN_WINDOWS = {"7_day_mean": range(2, 9), "8_14_day_mean": range(9, 16)}
POINT_SMOOTHING_SHIFT = 1   # (x[t-1] + x[t]) / 2 for point-in-time variables (v2 used -1, forward)

# --------------------------------------------------------------------------------------
# Lag structure (cells 32, 41). Anything a notebook version changed between v1 and v6 lives here
# or in FeatureConfig.
# --------------------------------------------------------------------------------------
TARGET_LAG_DAYS = list(range(2, 17))           # lag_{i}_day, filled with lag_fill_value
TARGET_LAG_1H_DAYS = list(range(2, 9))         # lag_{i}_day_1h (filled; feeds lag_7_day_1h_mean)
LAG_MEAN_7_DAYS = list(range(2, 9))            # lag_7_day_mean, lag_7_day_1h_mean
LAG_MEAN_8_14_DAYS = list(range(9, 16))        # lag_8_14_day_mean
UNFILLED_LAG_HOURS = {"lag_2_day_1h": 24 * 2 + 1, "lag_7_day_1h": 24 * 7 + 1, "lag_7_day_m1h": 24 * 7 - 1}  # never filled (as in v6)
ROLLING_WINDOW = 24
ROLLING_SHIFT_BASE = BASE_LAG_HOURS            # rolling_{mean,std,kurt,skew}_24 end 48 h before the row
ROLLING_SHIFT_WEEK = 24 * 7                    # rolling_mean_24_lag_7_day
COUNTERPART_LAG_DAYS = [2, 3, 7]
COUNTERPART_1H_LAG_HOURS = 24 * 2 + 1
COUNTERPART_MEAN_DAYS = list(range(2, 9))
CLIENT_DIFF_COLUMNS = ["eic_count", "installed_capacity"]
CLIENT_DIFF_LAG_HOURS = 24 * 5                 # `_7_day_diff` (sic: 5 days)
CLIENT_MEAN_DIFF_DAYS = list(range(0, 7))
PRICE_LAG_DAYS = list(range(1, 8))
TOTAL_TARGET_SHIFT_HOURS = BASE_LAG_HOURS      # target_all_type

# --------------------------------------------------------------------------------------
# Calendar (cells 44-46)
# --------------------------------------------------------------------------------------
NATIONAL_HOLIDAYS = [
    (2021, 1, 1), (2021, 2, 24), (2021, 4, 2), (2021, 4, 4), (2021, 5, 1), (2021, 5, 23),
    (2021, 6, 23), (2021, 6, 24), (2021, 8, 20), (2021, 12, 24), (2021, 12, 25), (2021, 12, 26),
    (2022, 1, 1), (2022, 2, 24), (2022, 4, 15), (2022, 4, 17), (2022, 5, 1), (2022, 6, 5),
    (2022, 6, 23), (2022, 6, 24), (2022, 8, 20), (2022, 12, 24), (2022, 12, 25), (2022, 12, 26),
    (2023, 1, 1), (2023, 2, 24), (2023, 4, 7), (2023, 4, 9), (2023, 5, 1), (2023, 5, 28),
    (2023, 6, 23), (2023, 6, 24), (2023, 8, 20), (2023, 12, 24), (2023, 12, 25), (2023, 12, 26),
    (2024, 1, 1), (2024, 2, 24), (2024, 3, 29), (2024, 3, 31), (2024, 5, 1), (2024, 5, 19),
    (2024, 6, 23), (2024, 6, 24), (2024, 8, 20), (2024, 12, 24), (2024, 12, 25), (2024, 12, 26),
]
NATIONAL_HOLIDAY_DATES = [pd.Timestamp(*h) for h in NATIONAL_HOLIDAYS]

# --------------------------------------------------------------------------------------
# Targets, scaling and ensemble
# --------------------------------------------------------------------------------------
# Derived targets: name -> lag column the (target - lag) difference is taken against (cell 62).
# Positive part is `<lag>_diff`, negative part `<lag>_diff_neg`, both clipped at 0.
DIFF_LAG_COLUMNS = ["lag_7_day_mean", "lag_2_day", "lag_7_day"]
# Training order of the models per side (cell 72)
TARGETS = [
    "target",
    "lag_7_day_mean_diff", "lag_7_day_mean_diff_neg",
    "lag_2_day_diff", "lag_2_day_diff_neg",
    "lag_7_day_diff", "lag_7_day_diff_neg",
]


@dataclass(frozen=True)
class EnsembleComponent:
    """One reconstructed prediction: direct model, or lag + (pos model - neg model)."""
    name: str
    lag_column: str | None        # None => direct model on `target`
    weight: float

    @property
    def models(self) -> tuple[str, ...]:
        if self.lag_column is None:
            return ("target",)
        return (f"{self.lag_column}_diff", f"{self.lag_column}_diff_neg")


# cell 77: coef_0..3 = 2/7, 2/7, 2/7, 1/7 for [direct, lag_7_day_mean, lag_2_day, lag_7_day], both sides
ENSEMBLE = {
    side: (
        EnsembleComponent("direct", None, 2 / 7),
        EnsembleComponent("lag_7_day_mean", "lag_7_day_mean", 2 / 7),
        EnsembleComponent("lag_2_day", "lag_2_day", 2 / 7),
        EnsembleComponent("lag_7_day", "lag_7_day", 1 / 7),
    )
    for side in (0, 1)
}


# --------------------------------------------------------------------------------------
# Feature lists (cells 28, 52, 55) -- ported verbatim, but returned *sorted* so the order does not
# depend on PYTHONHASHSEED (the notebook returned list(set(...))).
# --------------------------------------------------------------------------------------
CLIENT_FEATURES = ["eic_count", "installed_capacity"]
WEATHER_FEATURES = [
    "temperature", "cloudcover_low", "cloudcover_mid", "cloudcover_total",
    "10_metre_u_wind_component", "10_metre_v_wind_component", "direct_solar_radiation",
    "surface_solar_radiation_downwards", "total_precipitation",
]
RAW_FEATURES = [
    "county", "is_business", "product_type", "is_consumption", "dayofweek", "daytype",
    "daytype_lag_3_day", "sin_hour", "cos_hour", "lag_2_day", "lag_3_day", "lag_4_day",
    "lag_7_day_mean", "lag_5_day", "lag_6_day", "lag_7_day", "lag_2_day_1h", "rolling_mean_24",
    "rolling_std_24", "lag_2_day_consumption_counterpart", "lag_7_day_mean_consumption_counterpart",
    "snowfall_1_day_rolling_sum", "eic_count_7_day_mean_diff", "installed_capacity_7_day_mean_diff",
    "cloudcover_total_7_day_mean_diff", "direct_solar_radiation_2_day_1h_diff",
]


def select_features(is_consumption: int, target: str = "target") -> list[str]:
    """Port of v6 cell 55 `select_features`. Same list operations (note `list.remove` drops only
    the first occurrence, so duplicates survive a removal exactly as in the notebook)."""
    features = RAW_FEATURES + WEATHER_FEATURES + CLIENT_FEATURES
    features.remove("is_consumption")
    to_remove: list[str] = []
    to_add: list[str] = []

    if is_consumption == 0:
        to_remove += ["lag_4_day", "lag_5_day", "lag_6_day"]
    elif is_consumption == 1:
        to_remove += [
            "cloudcover_mid", "cloudcover_total", "10_metre_u_wind_component",
            "10_metre_v_wind_component", "lag_5_day", "lag_6_day", "sin_hour", "cos_hour",
        ]
        to_add += ["hour", "rolling_kurt_24", "rolling_skew_24", "lag_3_day_consumption_counterpart",
                   "temperature_2_day_1h_diff"]
    else:
        raise ValueError(is_consumption)

    if is_consumption == 0:
        to_add += [
            "temp_minus_dewpoint",
            "direct_solar_radiation_7_day_mean_diff", "cloudcover_low_7_day_mean_diff",
            "total_precipitation_7_day_mean_diff", "surface_solar_radiation_downwards_7_day_mean_diff",
            "direct_solar_radiation_2_day_diff", "cloudcover_low_2_day_diff",
            "total_precipitation_2_day_diff", "surface_solar_radiation_downwards_2_day_diff",
            "snowfall_1_day_rolling_sum_lag_2_day_diff",
        ]
        if target in ["lag_2_day_diff", "lag_2_day_diff_neg"]:
            to_add += ["diffuse_radiation_lag_2_day_h"]
        if target in ["lag_7_day_diff", "lag_7_day_diff_neg"]:
            to_add += ["diffuse_radiation_lag_7_day_h"]
    else:
        to_add += [
            "euros_per_mwh", "euros_per_mwh_lag_7_day_mean_diff", "lag_8_14_day_mean",
            "direct_solar_radiation_7_day_mean_diff",
        ]
        if target in ["target"]:
            to_add += ["temperature_lag_1h", "temperature_7_day_diff",
                       "temperature_8_14_day_mean_diff", "temperature_7_day_diff"]
        if target in ["target", "lag_7_day_mean_diff", "lag_7_day_mean_diff_neg"]:
            to_add += ["temperature_7_day_mean_diff", "rolling_mean_24_lag_7_day"]
        if target in ["lag_7_day_diff", "lag_7_day_diff_neg"]:
            to_add += ["direct_solar_radiation_7_day_1h_diff", "temperature_7_day_1h_diff"]

    if target in ["lag_2_day_diff", "lag_2_day_diff_neg"]:
        to_add += [
            "daytype_lag_2_day", "temperature_2_day_diff", "direct_solar_radiation_2_day_diff",
            "surface_solar_radiation_downwards_2_day_diff", "total_precipitation_2_day_diff",
            "cloudcover_total_2_day_diff", "euros_per_mwh_lag_2_day_diff",
        ]
        to_remove += ["euros_per_mwh_lag_7_day_mean_diff", "eic_count_7_day_mean_diff",
                      "installed_capacity_7_day_mean_diff"]
    elif target in ["lag_7_day_diff", "lag_7_day_diff_neg"]:
        to_add += [
            "daytype_lag_7_day", "lag_7_day_consumption_counterpart", "temperature_7_day_diff",
            "direct_solar_radiation_7_day_diff", "surface_solar_radiation_downwards_7_day_diff",
            "total_precipitation_7_day_diff", "cloudcover_total_7_day_diff",
            "euros_per_mwh_lag_7_day_diff", "eic_count_7_day_diff", "installed_capacity_7_day_diff",
            "lag_7_day_1h", "lag_7_day_m1h",
        ]
        to_remove += ["euros_per_mwh_lag_7_day_mean_diff", "temperature_2_day_1h_diff"]

    features += to_add
    for ft in to_remove:
        try:
            features.remove(ft)
        except ValueError:
            pass
    return sorted(set(features))


def _monotone_map(features: list[str], is_consumption: int) -> dict[str, int]:
    m = {ft: 0 for ft in features}
    if is_consumption == 0:
        m["installed_capacity"] = 1
        m["direct_solar_radiation"] = 1
        m["cloudcover_total"] = -1
        m["cloudcover_mid"] = -1
        m["cloudcover_low"] = -1
    else:
        m["eic_count"] = 1
    return m


def monotone_constraints(features: list[str], is_consumption: int, target: str, override: dict | None = None) -> list[int]:
    """Constraint per feature, aligned with ``features``, as v6 effectively ran it: the intended sign flip
    for `*_neg` targets never happened in v6, so `target` does not change the constraints.
    ``override``: {side: {feature: sign}} replaces the v6 map (variant v1)."""
    m = override[is_consumption] if override else _monotone_map(features, is_consumption)
    return [m.get(f, 0) for f in features]


# Frozen at import: (side, target) -> sorted feature tuple
FEATURES: dict[tuple[int, str], tuple[str, ...]] = {
    (side, target): tuple(select_features(side, target)) for side in (0, 1) for target in TARGETS
}
ALL_MODEL_FEATURES: tuple[str, ...] = tuple(sorted({f for fs in FEATURES.values() for f in fs}))


def categorical_features(is_consumption: int) -> list[str]:
    """cell 72."""
    cats = ["county", "product_type", "daytype", "dayofweek"]
    return cats + ["hour"] if is_consumption else cats


# --------------------------------------------------------------------------------------
# LightGBM (cell 72)
# --------------------------------------------------------------------------------------
def lgb_params(is_consumption: int, target: str, overrides: dict | None = None) -> dict:
    """v6 parameters; ``overrides`` = {"*": {...}, side: {...}} of a variant (explicit values, e.g. 1.5 for an unset
    tweedie_variance_power)."""
    p = {
        "boosting_type": "gbdt",
        "objective": "tweedie",
        "metric": "mae",
        "learning_rate": 0.1,
        "feature_fraction": 0.9 - 0.2 + 0.2 * (target == "target"),
        "cat_l2": 0.1 + 1 * is_consumption,
        "reg_sqrt": True,
        "max_depth": 7,
        "num_leaves": 127,
        "monotone_constraints_method": "advanced",
        "random_state": 42,
        "reg_alpha": 0.2 + 1 * is_consumption,
        "reg_lambda": 0.2 + 1 * is_consumption,
        "verbose": -1,
        "tweedie_variance_power": 1.25 if is_consumption else 1.4,
    }
    if overrides:
        p.update(overrides.get("*", {}) | overrides.get(is_consumption, {}))
    return p


LGB_CV = {"num_boost_round": 1600, "early_stopping_rounds": 100}

# --------------------------------------------------------------------------------------
# Growth augmentation (cell 64): last 400k rows by datetime, duplicated with scale-like
# columns multiplied by U(1, 1.6) (np.random.seed(42); rand(400000) * 0.6 + 1)
# --------------------------------------------------------------------------------------
AUGMENTATION = {
    "n_rows": 400_000,
    "low": 1.0,
    "width": 0.6,
    "seed": 42,
    "columns": [
        "target", "lag_7_day_mean_diff", "lag_7_day_mean_diff_neg", "lag_2_day_diff", "lag_2_day_diff_neg",
        "lag_7_day_diff", "lag_7_day_diff_neg",
        "lag_2_day_consumption_counterpart", "lag_2_day_1h_consumption_counterpart",
        "lag_3_day_consumption_counterpart", "lag_7_day_consumption_counterpart",
        "lag_7_day_mean_consumption_counterpart",
        "installed_capacity", "eic_count", "eic_count_7_day_mean_diff", "installed_capacity_7_day_mean_diff",
        "lag_2_day", "lag_2_day_1h", "lag_3_day", "lag_4_day", "lag_7_day_mean", "lag_5_day", "lag_6_day",
        "lag_7_day", "lag_7_day_1h", "lag_7_day_m1h", "lag_14_day",
        "rolling_mean_24", "rolling_std_24", "lag_8_14_day_mean", "lag_7_day_1h_mean", "target_all_type",
    ],
}

# --------------------------------------------------------------------------------------
# Output of the feature build
# --------------------------------------------------------------------------------------
ID_COLUMNS = ["row_id", "data_block_id", "prediction_unit_id", "datetime", "date", "day",
              "county", "is_business", "product_type", "is_consumption"]
DERIVED_TARGET_COLUMNS = [t for t in TARGETS if t != "target"]
OUTPUT_COLUMNS = list(dict.fromkeys(
    ID_COLUMNS + ["target"] + DERIVED_TARGET_COLUMNS + list(ALL_MODEL_FEATURES) + AUGMENTATION["columns"]
    + ["dewpoint"]     # v1 only
))
# Recipe: train on forecasts averaged over every origin (v6's smoothing, measured to help);
# evaluate and serve on the origins available at prediction time.
SERVING_FEATURE_CONFIG = FeatureConfig()   # what the API has delivered at prediction time
TRAIN_FEATURE_CONFIG = replace(SERVING_FEATURE_CONFIG, name="train", forecast_origin="all_origins")
FEATURE_CONFIGS: dict[str, FeatureConfig] = {"serving": SERVING_FEATURE_CONFIG, "train": TRAIN_FEATURE_CONFIG}


def feature_config_hash(cfg: FeatureConfig) -> str:
    """sha256 over the policy flags (the human label ``name`` is excluded, so a rename keeps the cache)."""
    import dataclasses
    import hashlib
    import json
    flags = {k: v for k, v in dataclasses.asdict(cfg).items() if k != "name"}
    return hashlib.sha256(json.dumps(flags, sort_keys=True).encode()).hexdigest()


def features_path(cfg: FeatureConfig = SERVING_FEATURE_CONFIG) -> Path:
    """Feature cache per config: data/features/features_<name>_<hash8>.parquet (+ .json sidecar), so builds
    of different configs coexist and a trainer can never read a table built with other flags."""
    return FEATURES_DIR / f"features_{cfg.name}_{feature_config_hash(cfg)[:8]}.parquet"


# --------------------------------------------------------------------------------------
# Model variants: the six notebook milestones (docs/milestone_variants.md) as differences from the v6 constants
# above; one recipe for all (train.py). "base": variant whose feature lists "edits" start from (default v6);
# edit = (side or None=both, targets, "-"|"+", names), targets "*" or base target names (`_neg` follows its positive twin).
# "approx": what the current builder cannot express (milestone_variants.md section 3: D1 nearest-grid weather,
# D2 EET + forward shift, D3 forecast bias correction, D6 client join on data_block_id, D8 per-county snowfall
# sum); absent = notebook_fidelity "exact-config".
# --------------------------------------------------------------------------------------
_AUG_30 = [c for c in AUGMENTATION["columns"] if c not in ("lag_7_day_1h_mean", "target_all_type")]
_AUG_27 = [c for c in _AUG_30 if c not in ("lag_7_day_1h", "lag_7_day_m1h", "lag_14_day")]
_AUG_13 = ["installed_capacity", "eic_count", *TARGETS, "lag_2_day_consumption_counterpart",
           "lag_7_day_mean_consumption_counterpart", "eic_count_7_day_mean_diff", "installed_capacity_7_day_mean_diff"]
_LGB_V2 = {"*": {"feature_fraction": 0.9, "reg_sqrt": False, "reg_alpha": 0.2, "reg_lambda": 0.2, "cat_l2": 0.1},
             1: {"tweedie_variance_power": 1.3}}
_FP16 = "float16 cast as in v6 (notebook: float64)"   # v1-v5 run in float16 too: float64 tables need ~16 GiB to build

VARIANTS: dict[str, dict] = {
    "v1": {
        "kaggle_version": 84,
        "notebook_date": "2023-11-08",
        "approx": "D1 nearest-grid-point forecast weather, D6 client join on data_block_id, dropna(any) training "
                  "filter not applied; " + _FP16,
        "base": "v2",
        "targets": ["target"],
        "ensemble": {s: (EnsembleComponent("direct", None, 1.0),) for s in (0, 1)},
        "feature_config": {"lag_fill_value": None},
        "lgb": {"*": {"feature_fraction": 0.9, "reg_sqrt": False, "reg_alpha": 0.0, "reg_lambda": 0.0,
                      "cat_l2": 0.1, "tweedie_variance_power": 1.5}},
        "monotone": {0: {"euros_per_mwh": 1, "installed_capacity": 1, "direct_solar_radiation": 1,
                         "cloudcover_total": -1},
                     1: {"euros_per_mwh": -1, "eic_count": 1}},
        "augmentation": {"n_rows": 0},
        "edits": [
            (0, "*", "-", ("cloudcover_low_2_day_diff cloudcover_low_7_day_mean_diff "
                 "direct_solar_radiation_2_day_diff direct_solar_radiation_7_day_mean_diff "
                 "snowfall_1_day_rolling_sum surface_solar_radiation_downwards_2_day_diff "
                 "surface_solar_radiation_downwards_7_day_mean_diff temp_minus_dewpoint temperature "
                 "total_precipitation total_precipitation_2_day_diff "
                 "total_precipitation_7_day_mean_diff")),
            (0, "*", "+", ("dewpoint euros_per_mwh lag_4_day lag_5_day lag_6_day "
                 "lag_7_day_consumption_counterpart")),
            (1, "*", "-", ("euros_per_mwh_lag_7_day_mean_diff lag_8_14_day_mean snowfall_1_day_rolling_sum "
                 "temperature total_precipitation")),
            (1, "*", "+", ("10_metre_u_wind_component 10_metre_v_wind_component cloudcover_mid cloudcover_total "
                 "dewpoint lag_5_day lag_6_day lag_7_day_consumption_counterpart")),
        ],
    },
    "v2": {
        "kaggle_version": 125,
        "notebook_date": "2023-11-15",
        "approx": "D2 EET weather timestamps + forward shift, D6 client join on data_block_id, D8 per-county "
                  "snowfall rolling sum; " + _FP16,
        "base": "v3",
        "feature_config": {"counterpart_source": "own_target", "price_hours_per_day": 28},
        "lgb": _LGB_V2,
        "augmentation": {"n_rows": 0},
        "edits": [
            (0, "*", "-", "eic_count_7_day_mean_diff installed_capacity_7_day_mean_diff"),
            (None, "lag_2_day_diff", "-", "euros_per_mwh_lag_2_day_diff"),
            (None, "lag_7_day_diff", "-", "eic_count_7_day_diff euros_per_mwh_lag_7_day_diff installed_capacity_7_day_diff"),
            (1, "*", "-", "eic_count_7_day_mean_diff hour installed_capacity_7_day_mean_diff"),
            (1, "*", "+", "cos_hour sin_hour"),
            (1, "lag_2_day_diff lag_7_day_diff", "+", "euros_per_mwh_lag_7_day_mean_diff"),
        ],
    },
    "v3": {
        "kaggle_version": 158,
        "notebook_date": "2023-12-31",
        "approx": "D8 per-county snowfall rolling sum; augmentation drawn as U[1,1.6) seed 42 on the latest 200k rows "
                  "(notebook: unseeded N(1.2,0.1) on the last 200k rows in index order); " + _FP16,
        "base": "v4",
        "feature_config": {"counterpart_source": "own_target"},
        "lgb": _LGB_V2,
        "augmentation": {"n_rows": 200_000, "columns": _AUG_13},
        "edits": [
            (0, "*", "-", "cloudcover_total_7_day_mean_diff"),
            (None, "lag_2_day_diff", "-", "cloudcover_total_2_day_diff"),
            (None, "lag_7_day_diff", "-", "cloudcover_total_7_day_diff"),
            (1, "*", "-", ("cloudcover_total_7_day_mean_diff lag_3_day_consumption_counterpart rolling_kurt_24 "
                 "rolling_skew_24")),
        ],
    },
    "v4": {
        "kaggle_version": 190,
        "notebook_date": "2024-01-05",
        "approx": "D3 forecast bias correction (on in v4) and D8 per-county snowfall rolling sum; " + _FP16,
        "base": "v5",
        "feature_config": {"counterpart_source": "own_target"},
        "lgb": _LGB_V2,
        "augmentation": {"n_rows": 200_000, "columns": _AUG_27},
        "edits": [
            (0, "*", "-", "daytype_lag_3_day snowfall_1_day_rolling_sum_lag_2_day_diff"),
            (0, "lag_7_day_diff", "-", "lag_7_day_1h lag_7_day_m1h"),
            (None, "lag_2_day_diff", "+", "eic_count_7_day_mean_diff installed_capacity_7_day_mean_diff"),
            (1, "*", "-", "daytype_lag_3_day direct_solar_radiation_7_day_mean_diff"),
            (1, "target lag_7_day_mean_diff lag_2_day_diff", "-", "temperature_2_day_1h_diff"),
            (1, "target lag_7_day_mean_diff", "-", "rolling_mean_24_lag_7_day temperature_7_day_mean_diff"),
            (1, "target", "-", "temperature_7_day_diff temperature_8_14_day_mean_diff temperature_lag_1h"),
            (1, "lag_7_day_diff", "-", ("direct_solar_radiation_7_day_1h_diff lag_7_day_1h lag_7_day_m1h "
                 "temperature_7_day_1h_diff")),
        ],
    },
    "v5": {
        "kaggle_version": 217,
        "notebook_date": "2024-01-09",
        "approx": _FP16,
        "lgb": {"*": {"feature_fraction": 0.9, "reg_sqrt": False}, 1: {"tweedie_variance_power": 1.3}},
        "augmentation": {"columns": _AUG_30},
        "edits": [
            (None, "*", "-", "direct_solar_radiation_2_day_1h_diff"),
            (0, "lag_2_day_diff", "-", "diffuse_radiation_lag_2_day_h"),
            (0, "lag_7_day_diff", "-", "diffuse_radiation_lag_7_day_h"),
        ],
    },
    "v6": {"kaggle_version": 238, "notebook_date": "2024-01-15"},    # final training notebook
}


def _variant_features(name: str) -> dict[tuple[int, str], set[str]]:
    v = VARIANTS[name]
    feats = _variant_features(v["base"]) if "base" in v else {k: set(f) for k, f in FEATURES.items()}
    for side, targets, sign, names in sorted(v.get("edits", ()), key=lambda e: e[2] != "-"):   # removals first
        for (s, t), f in feats.items():
            if side in (None, s) and (targets == "*" or t.removesuffix("_neg") in targets.split()):
                feats[(s, t)] = f - set(names.split()) if sign == "-" else f | set(names.split())
    return feats


def variant(name: str) -> dict:
    """VARIANTS[name] resolved against the v6 constants: targets, ensemble, feature lists (side, target) -> tuple,
    augmentation, the train / serving FeatureConfigs; plus lgb / monotone overrides, notebook_date, approx."""
    v = VARIANTS[name]
    targets = v.get("targets", TARGETS)
    feats = _variant_features(name)
    fc = v.get("feature_config", {})
    return {**v, "name": name, "targets": targets, "ensemble": v.get("ensemble", ENSEMBLE),
            "features": {(s, t): tuple(sorted(feats[(s, t)])) for s in (0, 1) for t in targets},
            "augmentation": {**AUGMENTATION, **v.get("augmentation", {})},
            "train_cfg": replace(TRAIN_FEATURE_CONFIG, **fc), "serving_cfg": replace(SERVING_FEATURE_CONFIG, **fc),
            "notebook_fidelity": "approximate: " + v["approx"] if "approx" in v else "exact-config"}
