"""SARIMAX benchmark for Enefit energy forecasting using shared features.

Fits Seasonal Autoregressive Integrated Moving Average with Exogenous regressors
(SARIMAX) per segment using statsmodels on the standardized shared feature set:
- Exogenous regressors: lag_2_day, lag_7_day, lag_7_day_mean, installed_capacity, temperature, direct_solar_radiation
"""
from __future__ import annotations

import argparse
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from . import base
from .. import config as C

EXOG_COLS = [
    "lag_2_day",
    "lag_7_day",
    "lag_7_day_mean",
    "installed_capacity",
    "temperature",
    "direct_solar_radiation",
]


def _fit_predict_one_sarimax(args: tuple) -> tuple[np.ndarray, np.ndarray]:
    """Fit SARIMAX on a single segment with shared exogenous features."""
    key, train_sub, test_sub = args
    is_consumption = key[3]

    if len(train_sub) == 0 or len(test_sub) == 0:
        return test_sub["row_id"].to_numpy(dtype="int64"), np.zeros(len(test_sub), dtype=float)

    train_sub = train_sub.sort_values("datetime")
    test_sub = test_sub.sort_values("datetime")

    y_train = train_sub["target"].fillna(0).to_numpy(dtype=float)
    exog_train = train_sub[EXOG_COLS].fillna(0).to_numpy(dtype=float)
    exog_test = test_sub[EXOG_COLS].fillna(0).to_numpy(dtype=float)

    if len(y_train) < 48 or (y_train == 0).all():
        return test_sub["row_id"].to_numpy(dtype="int64"), np.zeros(len(test_sub), dtype=float)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Auto-regressive state-space conditioned on shared dynamic features
            order = (1, 0, 1) if is_consumption == 0 else (1, 1, 1)

            mod = SARIMAX(
                y_train,
                exog=exog_train,
                order=order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            res = mod.fit(disp=False, maxiter=25)
            forecast = res.forecast(steps=len(test_sub), exog=exog_test)
            pred = np.clip(np.asarray(forecast, dtype=float), 0.0, None)
            pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)

            if is_consumption == 0:
                solar_rad = test_sub["direct_solar_radiation"].fillna(0).to_numpy()
                night_mask = (solar_rad == 0) & test_sub["hour"].isin(
                    [20, 21, 22, 23, 0, 1, 2, 3, 4, 5]
                ).to_numpy()
                pred[night_mask] = 0.0

    except Exception:
        fallback = max(0.0, float(np.nanmean(y_train)))
        pred = np.full(len(test_sub), fallback, dtype=float)

    return test_sub["row_id"].to_numpy(dtype="int64"), pred


def run_sarimax_benchmark(
    workers: int = 6,
    train_history_start: str = "2023-01-01",
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Run SARIMAX benchmark across all segments using shared features."""
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

    print(f"[sarimax] Fitting {len(tasks)} segments with shared exog on {workers} workers...")
    row_ids, preds = [], []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for r_ids, p_vals in executor.map(_fit_predict_one_sarimax, tasks):
            row_ids.append(r_ids)
            preds.append(p_vals)

    all_row_ids = np.concatenate(row_ids)
    all_preds = np.concatenate(preds)
    pred_df = pd.DataFrame({"row_id": all_row_ids, "pred": all_preds})

    pred_df = test_df[["row_id"]].merge(pred_df, on="row_id", how="left")
    metrics = base.evaluate_predictions(pred_df, "sarimax")
    return pred_df, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SARIMAX benchmark on shared features.")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    parser.add_argument("--start", type=str, default="2023-01-01")
    args = parser.parse_args()

    pred_df, metrics = run_sarimax_benchmark(workers=args.workers, train_history_start=args.start)
    print("\n[sarimax] Results on Shared Feature Set:")
    for k, v in metrics.items():
        if k.endswith("/mae") or k.endswith("/wape"):
            print(f"  {k}: {v:.2f}")


if __name__ == "__main__":
    main()
