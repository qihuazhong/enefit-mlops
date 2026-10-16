"""Regression tests for the bugs found while porting the notebook (README, "What broke")."""
import dataclasses

import numpy as np
import pandas as pd
import pytest

from enefit import config as C
from enefit import data, features, targets, weather

DAY = pd.Timedelta(days=1)


def test_forecast_origins():
    """Hours of day D: serving has the origins issued by D-1 (D-2 covers D 00:00-02:00); training also averages D's."""
    o = pd.to_datetime(["2023-05-26 02:00", "2023-05-27 02:00", "2023-05-28 02:00", "2023-05-27 02:00", "2023-05-28 02:00"])
    t = pd.to_datetime(["2023-05-28 01:00"] * 3 + ["2023-05-28 12:00"] * 2)
    fw = pd.DataFrame({"origin_datetime": o, "forecast_datetime": t})
    assert list(weather.select_forecast_rows(fw, C.SERVING_FEATURE_CONFIG).index) == [0, 1, 3]
    assert list(weather.select_forecast_rows(fw, C.TRAIN_FEATURE_CONFIG).index) == [0, 1, 2, 3, 4]


@pytest.fixture(scope="module")
def early(raw):
    return features.build_features(data.time_slice(raw, "2021-09-01", "2021-09-12"))


def test_shifts_do_not_cross_group_boundaries(early):
    assert early.loc[early.datetime < pd.Timestamp("2021-09-08"), "temperature_lag_7_day"].isna().all()
    assert early.loc[early.datetime < pd.Timestamp("2021-09-03"), "rolling_mean_24"].isna().all()
    assert early.loc[early.datetime < pd.Timestamp("2021-09-03"), "target_all_type"].isna().all()
    assert early["daytype_lag_7_day"].notna().all()           # calendar fact, no shift needed


def test_price_lags_are_time_lags_across_gaps():
    n = 24 * 40
    df = pd.DataFrame({"forecast_date": pd.date_range("2022-01-01", periods=n, freq="h"),
                       "euros_per_mwh": np.arange(n, dtype=float)}).drop(index=[300])   # one missing hour (DST)
    out = features.process_elec_prices(df.reset_index(drop=True), C.SERVING_FEATURE_CONFIG)
    assert (out["euros_per_mwh_lag_7_day_diff"].dropna() == 168).all()


def test_fillna_as_in_v6(early):
    seg = early[(early.county == 0) & (early.is_business == 0) & (early.product_type == 1) & (early.is_consumption == 0)]
    head = seg.iloc[:40]
    assert (head["lag_2_day"] == 0).all() and head["lag_2_day_1h"].isna().all()


def test_monotone_constraints_same_for_pos_and_neg_targets():
    feats = list(C.FEATURES[(0, "lag_2_day_diff")])
    pos = C.monotone_constraints(feats, 0, "lag_2_day_diff")
    assert any(pos) and pos == C.monotone_constraints(feats, 0, "lag_2_day_diff_neg")


def test_derived_targets_come_from_the_frame_passed_in():
    a = pd.DataFrame({"target": [10.0, 5.0, 0.0], "lag_2_day": [4.0, 9.0, 1.0], "lag_7_day": [1.0] * 3,
                      "lag_7_day_mean": [2.0] * 3}, index=[100, 101, 102])
    out = targets.add_derived_targets(a.sample(frac=1.0, random_state=1)).sort_index()
    np.testing.assert_array_equal(out["lag_2_day_diff"], [6.0, 0.0, 0.0])
    np.testing.assert_array_equal(out["lag_2_day_diff_neg"], [0.0, 4.0, 1.0])


def test_clipping_is_assigned_and_never_touches_ids():
    df = pd.DataFrame({"row_id": np.array([5_000_000, 7], dtype="int64"), "x": [5e7, -5e7]})
    out = features.clip_features(df, C.CLIP_BOUND)
    assert out["x"].abs().max() == 1e6 and out["row_id"].tolist() == [5_000_000, 7]


def test_rolling_stats_do_not_depend_on_history():
    rng = np.random.default_rng(3)
    x = np.concatenate([rng.gamma(2.0, 5000.0, 400), rng.gamma(2.0, 0.05, 100)])
    full = features._exact_rolling_stats(x, np.array([500]))
    tail = features._exact_rolling_stats(x[400:], np.array([100]))
    for stat in ("mean", "std", "skew", "kurt"):
        np.testing.assert_array_equal(full[stat][423:], tail[stat][23:])


def test_milestone_feature_switches(raw):
    """The FeatureConfig fields that older notebook versions differ in change what they say."""
    d = pd.Timestamp("2023-04-20")
    w = data.time_slice(raw, d - 30 * DAY, d)
    base_cfg = dataclasses.replace(C.SERVING_FEATURE_CONFIG, float_dtype=None)
    base = features.build_features(w, base_cfg)
    own = features.build_features(w, dataclasses.replace(base_cfg, counterpart_source="own_target"))
    ok = own["lag_2_day_consumption_counterpart"].notna()
    np.testing.assert_array_equal(own.loc[ok, "lag_2_day_consumption_counterpart"], own.loc[ok, "lag_2_day"])
    p28 = features.build_features(w, dataclasses.replace(base_cfg, price_hours_per_day=28))
    assert (p28["euros_per_mwh_lag_7_day_diff"] != base["euros_per_mwh_lag_7_day_diff"]).any()
    assert features.build_features(w, C.SERVING_FEATURE_CONFIG)["lag_2_day"].dtype == np.float16
    nofill = features.build_features(data.time_slice(raw, "2021-09-01", "2021-09-05"),
                                     dataclasses.replace(base_cfg, lag_fill_value=None))
    assert nofill["lag_2_day"].iloc[:50].isna().all()
