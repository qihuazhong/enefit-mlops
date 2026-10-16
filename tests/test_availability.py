"""`data.available_at` encodes the availability rules; check it against the Kaggle example block
(data/raw/example_test_files = what the API served for the first test call, test day 2023-05-28)."""
import pandas as pd
import pytest

from enefit import config as C
from enefit import data

EX = C.RAW_DIR / "example_test_files"
pytestmark = pytest.mark.skipif(not EX.exists(), reason="example_test_files not present")

D = pd.Timestamp("2023-05-28")


def test_day_index_matches_data_block_ids():
    assert data.day_index("2021-09-01") == 0
    # the API call for test day D carries data_block_id == day_index(D)
    assert data.day_index(D) == int(pd.read_csv(EX / "test.csv", nrows=1).data_block_id[0])


def test_available_at_reproduces_the_api_block(raw):
    a = data.available_at(raw, D, history_days=60)
    b = data.day_index(D)

    def first_call(name, dates):          # the example files hold several consecutive API calls
        df = pd.read_csv(EX / name, parse_dates=dates)
        return df[df.data_block_id == b]

    rt = first_call("revealed_targets.csv", ["datetime"])
    cl = first_call("client.csv", ["date"])
    ep = first_call("electricity_prices.csv", ["forecast_date"])
    fw = first_call("forecast_weather.csv", ["origin_datetime", "forecast_datetime"])
    hw = first_call("historical_weather.csv", ["datetime"])

    revealed = a.train[a.train.target.notna()]
    assert revealed.datetime.max() == rt.datetime.max() == D - pd.Timedelta(days=2) + pd.Timedelta(hours=23)
    assert a.client.date.max() == cl.date.max() == D - pd.Timedelta(days=2)
    assert a.electricity_prices.forecast_date.max() == ep.forecast_date.max()          # D-1 23:00
    assert a.forecast_weather.origin_datetime.max() == fw.origin_datetime.max()         # D-1 02:00
    assert a.forecast_weather.forecast_datetime.max() == fw.forecast_datetime.max()     # D+1 02:00
    assert a.historical_weather.datetime.max() == hw.datetime.max()                     # D-1 10:00


def test_skeleton_rows_of_d_minus_1_and_d_carry_no_target(raw):
    a = data.available_at(raw, D)
    last = a.train[a.train.datetime >= D - pd.Timedelta(days=1)]
    assert len(last) > 5000 and last.target.isna().all()
    assert a.train.datetime.max() == D + pd.Timedelta(hours=23)


def test_nothing_after_the_block_is_visible(raw):
    b = data.day_index(D)
    a = data.available_at(raw, D)
    for t in (a.client, a.electricity_prices, a.forecast_weather, a.historical_weather):
        assert (t.data_block_id <= b).all()
