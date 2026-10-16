"""Basic LightGBM benchmark predicting raw target directly on the shared feature set.

Unlike the complex v6 recipe (14 positive/negative difference models, capacity augmentation,
and float16 quantization), this model trains a single standard LightGBM Regressor per side
directly on the standardized shared feature set.
"""
from __future__ import annotations

import argparse
import lightgbm as lgb
import numpy as np
import pandas as pd

from . import base
from .. import config as C

FEATURES = [
    "county", "is_business", "product_type",
    "lag_2_day", "lag_7_day", "lag_7_day_mean",
    "installed_capacity", "temperature", "direct_solar_radiation",
    "hour", "dayofweek",
]
CATEGORICALS = ["county", "is_business", "product_type", "hour", "dayofweek"]


def run_basic_lgbm_benchmark(
    n_estimators: int = 300,
    learning_rate: float = 0.08,
    num_leaves: int = 63,
    train_history_start: str = "2022-09-01",
    out_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Train basic LightGBM per side on shared features."""
    train_df = base.load_train_data(start=train_history_start, columns=base.SHARED_FEATURE_COLS)
    test_df = base.load_test_data(columns=base.SHARED_FEATURE_COLS)

    for c in CATEGORICALS:
        train_df[c] = train_df[c].astype("int32")
        test_df[c] = test_df[c].astype("int32")

    preds = []
    for side in (0, 1):
        tr = train_df[train_df["is_consumption"] == side].dropna(subset=["target"])
        te = test_df[test_df["is_consumption"] == side]

        # Tweedie power: 1.4 for production, 1.25 for consumption (matching v6 defaults)
        tweedie_p = 1.4 if side == 0 else 1.25
        params = {
            "objective": "tweedie",
            "tweedie_variance_power": tweedie_p,
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "seed": 42,
            "num_threads": 2,
            "verbose": -1,
        }
        dtrain = lgb.Dataset(
            tr[FEATURES],
            label=tr["target"].to_numpy("float64"),
            categorical_feature=CATEGORICALS,
        )
        booster = lgb.train(params, dtrain, num_boost_round=n_estimators)
        pred_vals = np.clip(booster.predict(te[FEATURES]), 0.0, None)

        # Enforce zero night solar production
        if side == 0:
            solar_rad = te["direct_solar_radiation"].fillna(0).to_numpy()
            hours = te["hour"].to_numpy()
            night = (solar_rad == 0) & np.isin(hours, [20, 21, 22, 23, 0, 1, 2, 3, 4, 5])
            pred_vals[night] = 0.0

        preds.append(pd.DataFrame({"row_id": te["row_id"].to_numpy(dtype="int64"), "pred": pred_vals}))

    pred_df = pd.concat(preds, ignore_index=True)
    pred_df = test_df[["row_id"]].merge(pred_df, on="row_id", how="left")
    metrics = base.evaluate_predictions(pred_df, "basic_lgbm", out_dir=out_dir)
    return pred_df, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Basic LightGBM benchmark.")
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.08)
    args = parser.parse_args()

    pred_df, metrics = run_basic_lgbm_benchmark(n_estimators=args.n_estimators, learning_rate=args.lr)
    print("\n[basic_lgbm] Results on Shared Feature Set:")
    for k, v in metrics.items():
        if k.endswith("/mae") or k.endswith("/wape"):
            print(f"  {k}: {v:.2f}")


if __name__ == "__main__":
    main()
