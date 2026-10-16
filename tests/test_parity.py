"""Training/serving parity: one builder, so the features of day D from the full batch build must equal
the features from the scoring path (only what was available at D's cutoff, 45 days of history).

Exact equality (no tolerance) is required for every output column: all arithmetic is element-wise or
per-window, and rolling statistics are recomputed per window.
"""
import pandas as pd
import pytest

from enefit import config as C
from enefit import data, features

from _util import diff_columns, feature_columns
from conftest import SCORING_DAYS


def _day_rows(df, d):
    return df[df.date == d].reset_index(drop=True)


@pytest.mark.parametrize("day", SCORING_DAYS)
def test_scoring_path_equals_windowed_batch(raw, day):
    d = pd.Timestamp(day)
    scored = features.features_for_day(raw, d)
    # batch with MORE history (60 d) and full information after D: proves independence of window length
    batch = features.build_features(data.time_slice(raw, d - pd.Timedelta(days=60), d + pd.Timedelta(days=3)))
    batch = _day_rows(batch, d)
    assert len(scored) == len(batch) > 3000
    bad = diff_columns(scored, batch, feature_columns(scored))
    assert not bad, f"parity broken on {day}: {bad}"


def test_one_day_scoring_covers_all_rows_of_the_day(raw):
    d = pd.Timestamp("2023-05-31")                     # last day of the live-run block
    scored = features.features_for_day(raw, d)
    expected = raw.train[(raw.train.datetime >= d) & (raw.train.datetime < d + pd.Timedelta(days=1))]
    assert set(scored.row_id) == set(expected.row_id)
    # a day being scored has no label in the frame handed to the builder
    assert scored["target"].isna().all()


def test_scoring_features_are_independent_of_history_length(raw):
    d = pd.Timestamp("2023-04-20")
    a = features.features_for_day(raw, d, history_days=40)
    b = features.features_for_day(raw, d, history_days=70)
    assert not diff_columns(a, b, feature_columns(a))


# The brief's literal requirement: full-period batch build == scoring path. Skipped until the parquet exists.
@pytest.mark.skipif(not C.features_path(C.SERVING_FEATURE_CONFIG).exists(),
                    reason="run `python -m enefit.features --build` first")
@pytest.mark.parametrize("day", SCORING_DAYS)
def test_scoring_path_equals_full_parquet(raw, day):
    d = pd.Timestamp(day)
    full = pd.read_parquet(C.features_path(C.SERVING_FEATURE_CONFIG), filters=[("date", "==", d)])
    full = full.sort_values("row_id").reset_index(drop=True)
    scored = features.features_for_day(raw, d)
    cols = [c for c in feature_columns(scored) if c in full.columns]
    bad = diff_columns(scored.assign(**{c: scored[c] for c in cols}), full, cols)
    assert not bad, bad
