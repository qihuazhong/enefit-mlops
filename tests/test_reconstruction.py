"""Derived targets and the reconstruction of the final prediction (targets.py)."""
import numpy as np
import pandas as pd
import pytest

from enefit import config as C
from enefit import targets


def _frame(n=400, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "is_consumption": rng.integers(0, 2, n),
        "target": np.where(rng.random(n) < 0.3, 0.0, rng.gamma(2.0, 50.0, n)),
        "lag_2_day": rng.gamma(2.0, 50.0, n),
        "lag_7_day": rng.gamma(2.0, 50.0, n),
        "lag_7_day_mean": rng.gamma(2.0, 50.0, n),
    })
    return df


def _true_components(df):
    """What a perfect model would output for every config.TARGETS entry."""
    return {t: df[t].to_numpy(dtype="float64") for t in C.TARGETS}


def test_diff_targets_are_clipped_at_zero_and_exclusive():
    df = targets.add_derived_targets(_frame())
    for lag in C.DIFF_LAG_COLUMNS:
        pos, neg = df[f"{lag}_diff"], df[f"{lag}_diff_neg"]
        assert (pos >= 0).all() and (neg >= 0).all()
        assert ((pos == 0) | (neg == 0)).all()
        np.testing.assert_array_equal(pos - neg, df["target"] - df[lag])


@pytest.mark.parametrize("side", [0, 1])
def test_exact_reconstruction_from_true_components(side):
    df = targets.add_derived_targets(_frame())
    df = df[df.is_consumption == side]
    pred = targets.final_prediction(df, _true_components(df), side)
    np.testing.assert_allclose(pred, df["target"].to_numpy(), rtol=0, atol=1e-9)


def test_final_prediction_is_clipped_at_zero():
    df = targets.add_derived_targets(_frame())
    df = df[df.is_consumption == 0]
    preds = {t: np.full(len(df), -1000.0) for t in C.TARGETS}   # absurdly negative component models
    assert (targets.final_prediction(df, preds, 0) == 0).all()


def test_neg_and_pos_models_enter_with_opposite_signs():
    df = targets.add_derived_targets(_frame()).query("is_consumption == 0")
    base = {t: np.zeros(len(df)) for t in C.TARGETS}
    up = {**base, "lag_2_day_diff": np.full(len(df), 10.0)}
    down = {**base, "lag_2_day_diff_neg": np.full(len(df), 10.0)}
    w = {c.name: c.weight for c in C.ENSEMBLE[0]}["lag_2_day"]
    a = targets.final_prediction(df, base, 0)
    # use rows where the clip at 0 is inactive in both directions
    mask = (a > 10 * w) & (a < 1e9)
    np.testing.assert_allclose((targets.final_prediction(df, up, 0) - a)[mask], 10 * w)
    np.testing.assert_allclose((targets.final_prediction(df, down, 0) - a)[mask], -10 * w)


def test_nan_targets_stay_nan_not_zero():
    df = _frame(50)
    df.loc[3, "target"] = np.nan
    out = targets.add_derived_targets(df)
    assert np.isnan(out.loc[3, "lag_2_day_diff"]) and np.isnan(out.loc[3, "lag_2_day_diff_neg"])


def test_weights_match_notebook():
    for comps in C.ENSEMBLE.values():
        assert [round(c.weight * 7) for c in comps] == [2, 2, 2, 1]
        assert sum(c.weight for c in comps) == pytest.approx(1.0, abs=1e-15)
    assert len(C.TARGETS) == 7


def test_reconstruction_on_real_features_in_float64(raw):
    """Real frame, recomputed in float64 (the float16 build rounds the stored diffs, which would only
    show up as a ~1e-3 relative error: that is the precision policy, not the reconstruction)."""
    from enefit import data, features
    d = pd.Timestamp("2022-11-15")
    cfg = __import__("dataclasses").replace(C.SERVING_FEATURE_CONFIG, float_dtype=None)
    f = features.build_features(data.time_slice(raw, d - pd.Timedelta(days=30), d), cfg)
    f = f[f.target.notna() & (f.date == d)]
    f = targets.add_derived_targets(f)
    for side in (0, 1):
        df = f[f.is_consumption == side]
        pred = targets.final_prediction(df, _true_components(df), side)
        np.testing.assert_allclose(pred, df.target.to_numpy(), rtol=0, atol=1e-9)
