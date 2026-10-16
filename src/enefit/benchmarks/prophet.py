"""Prophet benchmark for Enefit energy forecasting using shared features.

Fits decomposable additive/multiplicative models (trend + daily/weekly seasonality + shared lag & weather regressors)
per segment using Prophet with multi-process parallelization.
"""
from __future__ import annotations

import argparse
import logging
import os
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd

from . import base
from .. import config as C

# Silence noisy cmdstanpy and prophet logs
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

REGRESSORS = [
    "lag_2_day",
    "lag_7_day",
    "lag_7_day_mean",
    "installed_capacity",
    "temperature",
    "direct_solar_radiation",
]


def _fit_predict_one_segment(args: tuple) -> tuple[np.ndarray, np.ndarray]:
    """Worker function to fit and predict one segment with shared regressors."""
    from prophet import Prophet

    key, train_sub, test_sub = args
    is_consumption = key[3]

    if len(train_sub) == 0 or len(test_sub) == 0:
        return test_sub["row_id"].to_numpy(dtype="int64"), np.zeros(len(test_sub), dtype=float)

    # Prepare Prophet format (ds, y) with shared regressors
    df_train = pd.DataFrame({
        "ds": train_sub["datetime"],
        "y": train_sub["target"].to_numpy(dtype=float),
    })
    df_test = pd.DataFrame({
        "ds": test_sub["datetime"],
    })

    for r in REGRESSORS:
        df_train[r] = train_sub[r].fillna(0).to_numpy(dtype=float)
        df_test[r] = test_sub[r].fillna(0).to_numpy(dtype=float)

    df_train = df_train.dropna(subset=["ds", "y"])

    if len(df_train) < 24:
        return test_sub["row_id"].to_numpy(dtype="int64"), np.zeros(len(test_sub), dtype=float)

    try:
        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=False,
            seasonality_mode="multiplicative" if is_consumption == 1 else "additive",
        )
        for r in REGRESSORS:
            model.add_regressor(r)

        model.fit(df_train)
        forecast = model.predict(df_test)
        pred = np.clip(forecast["yhat"].to_numpy(dtype=float), 0.0, None)

        # For solar production, enforce zero output at night when solar radiation is zero
        if is_consumption == 0:
            night_mask = (df_test["direct_solar_radiation"] == 0) & (
                test_sub["hour"].isin([20, 21, 22, 23, 0, 1, 2, 3, 4, 5])
            )
            pred[night_mask.to_numpy()] = 0.0

    except Exception:
        # Fallback to mean positive target
        fallback = max(0.0, float(df_train["y"].mean()))
        pred = np.full(len(test_sub), fallback, dtype=float)

    return test_sub["row_id"].to_numpy(dtype="int64"), pred


def run_prophet_benchmark(
    workers: int = 6,
    train_history_start: str = "2022-10-01",
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Run Prophet benchmark across all segments using shared features."""
    train_df = base.load_train_data(start=train_history_start, columns=base.SHARED_FEATURE_COLS)
    test_df = base.load_test_data(columns=base.SHARED_FEATURE_COLS)

    tasks = []
    for key in test_df[C.SEGMENT_KEYS].drop_duplicates().itertuples(index=False):
        key_tuple = tuple(key)
        t_mask = (
            (train_df["county"] == key_tuple[0])
            & (train_df["is_business"] == key_tuple[1])
            & (train_df["product_type"] == key_tuple[2])
            & (train_df["is_consumption"] == key_tuple[3])
        )
        te_mask = (
            (test_df["county"] == key_tuple[0])
            & (test_df["is_business"] == key_tuple[1])
            & (test_df["product_type"] == key_tuple[2])
            & (test_df["is_consumption"] == key_tuple[3])
        )
        tasks.append((key_tuple, train_df[t_mask].copy(), test_df[te_mask].copy()))

    print(f"[prophet] Fitting {len(tasks)} segments with shared regressors on {workers} workers...")
    row_ids, preds = [], []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for r_ids, p_vals in executor.map(_fit_predict_one_segment, tasks):
            row_ids.append(r_ids)
            preds.append(p_vals)

    all_row_ids = np.concatenate(row_ids)
    all_preds = np.concatenate(preds)
    pred_df = pd.DataFrame({"row_id": all_row_ids, "pred": all_preds})

    pred_df = test_df[["row_id"]].merge(pred_df, on="row_id", how="left")
    metrics = base.evaluate_predictions(pred_df, "prophet")
    return pred_df, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Prophet benchmark on shared features.")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    parser.add_argument("--start", type=str, default="2022-10-01")
    args = parser.parse_args()

    pred_df, metrics = run_prophet_benchmark(workers=args.workers, train_history_start=args.start)
    print("\n[prophet] Results on Shared Feature Set:")
    for k, v in metrics.items():
        if k.endswith("/mae") or k.endswith("/wape"):
            print(f"  {k}: {v:.2f}")


if __name__ == "__main__":
    main()
