"""Tests for forecasting benchmark suite."""
import numpy as np
import pandas as pd
import pytest
from enefit.benchmarks import base, basic_lgbm, croston_tsb, runner


def test_base_load_test_data():
    test_df = base.load_test_data(columns=["row_id", "datetime", "is_consumption", "target"])
    assert len(test_df) == 145248
    assert (test_df["datetime"] >= base.TEST_START).all()
    assert (test_df["datetime"] < base.TEST_END).all()


def test_basic_lgbm_produces_valid_predictions(tmp_path):
    pred_df, metrics = basic_lgbm.run_basic_lgbm_benchmark(
        n_estimators=10,
        train_history_start="2023-02-15",
        out_dir=tmp_path / "basic_lgbm",
    )
    assert "row_id" in pred_df.columns and "pred" in pred_df.columns
    assert len(pred_df) == 145248
    assert np.isfinite(pred_df["pred"]).all()
    assert (pred_df["pred"] >= 0).all()
    assert "basic_lgbm/all/mae" in metrics


def test_tsb_produces_valid_predictions(tmp_path):
    # Run on a short slice / default
    pred_df, metrics = croston_tsb.run_tsb_benchmark(
        method="tsb",
        train_history_start="2023-01-01",
        out_dir=tmp_path / "tsb",
    )
    assert "row_id" in pred_df.columns and "pred" in pred_df.columns
    assert len(pred_df) == 145248
    assert np.isfinite(pred_df["pred"]).all()
    assert (pred_df["pred"] >= 0).all()
    assert "tsb_intermittent/all/mae" in metrics


def test_deepar_model_forward():
    torch = pytest.importorskip("torch")      # "benchmarks" extra: installed locally, not in the AWS image
    from enefit.benchmarks import deepar

    model = deepar.DeepAR(hidden_size=32, num_layers=1)
    B, T = 4, 24
    stat = torch.zeros(B, 4, dtype=torch.long)
    dyn = torch.randn(B, T, 10)
    lag_y = torch.randn(B, T, 1)

    mu, sigma, (h_n, c_n) = model(stat, dyn, lag_y)
    assert mu.shape == (B, T, 1)
    assert sigma.shape == (B, T, 1)
    assert (sigma > 0).all()
    assert h_n.shape == (1, B, 32)
    assert c_n.shape == (1, B, 32)


def test_comparison_summary_schema(tmp_path, monkeypatch):
    # fake result files in tmp_path: the real ones (outputs/, not committed) are neither needed nor overwritten
    sides = pd.DataFrame({"side": ["all", "production", "consumption"], "mae": [3.0, 2.0, 4.0],
                          "wape": [0.3, 0.2, 0.4], "relative_bias": [0.0, -0.1, 0.1],
                          "naive_lag7_mae": [5.0, 4.0, 6.0], "naive_lag7_wape": [0.5, 0.4, 0.6]})
    sides.to_csv(tmp_path / "test_by_side.csv", index=False)
    (tmp_path / "benchmarks" / "basic_lgbm").mkdir(parents=True)
    sides.to_csv(tmp_path / "benchmarks" / "basic_lgbm" / "basic_lgbm_by_side.csv", index=False)
    monkeypatch.setattr(runner, "find_latest_lgbm_eval", lambda: tmp_path / "test_by_side.csv")
    monkeypatch.setattr(runner, "BENCHMARK_DIR", tmp_path / "benchmarks")

    summary = runner.compile_comparison_summary()
    assert len(summary) == 3      # v6, naive baseline, basic LightGBM
    expected_cols = [
        "model", "category", "total_mae", "total_wape",
        "production_mae", "production_wape",
        "consumption_mae", "consumption_wape"
    ]
    for col in expected_cols:
        assert col in summary.columns
