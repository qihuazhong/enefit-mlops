"""Croston's Method and TSB (Teunter-Syntetos-Babai) intermittent demand benchmarks.

Specifically designed for zero-inflated time series (such as solar production with 35% exact zeros).
Implements:
1. Croston's method: estimates smoothed non-zero demand size and smoothed inter-demand intervals.
2. TSB method: updates non-zero demand probability at every period and demand level when positive.
Both methods are fitted per (segment, hour) to capture diurnal solar/consumption cycles, with online
updates as targets are revealed 2 days prior (respecting the D-2 availability rule).
"""
from __future__ import annotations

import argparse
from typing import Literal
import numpy as np
import pandas as pd

from . import base
from .. import config as C

GRP_COLS = ["county", "is_business", "product_type", "is_consumption", "hour"]


def run_tsb_benchmark(
    alpha: float = 0.2,
    beta: float = 0.2,
    method: Literal["tsb", "croston"] = "tsb",
    train_history_start: str = "2022-09-01",
    out_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Fit and predict Croston / TSB across all segments."""
    train_df = base.load_train_data(
        start=train_history_start,
        columns=["datetime", "target", "hour"] + C.SEGMENT_KEYS,
    )
    test_df = base.load_test_data(
        columns=["row_id", "datetime", "target", "hour"] + C.SEGMENT_KEYS,
    )

    # Initialize states per segment and hour from training data
    states: dict[tuple, dict[str, float]] = {}
    for key, g in train_df.groupby(GRP_COLS, observed=True):
        vals = g["target"].to_numpy(dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0 or (vals == 0).all():
            states[key] = {"p": 0.0, "z": 0.0, "q": 1.0}
            continue

        p = float(np.mean(vals > 0)) if method == "tsb" else 1.0
        pos = vals[vals > 0]
        z = float(np.mean(pos)) if len(pos) > 0 else 0.0
        q = 1.0

        for y in vals:
            if y > 0:
                if method == "tsb":
                    p = alpha * 1.0 + (1 - alpha) * p
                    z = beta * y + (1 - beta) * z
                else:  # croston
                    z = alpha * y + (1 - alpha) * z
                    p = alpha * q + (1 - alpha) * p
                    q = 1.0
            else:
                if method == "tsb":
                    p = alpha * 0.0 + (1 - alpha) * p
                else:
                    q += 1.0

        states[key] = {"p": p, "z": z, "q": q}

    # Predict test rows chronologically by day
    # On day D, targets through D-2 are revealed and used to update state
    test_df["day"] = test_df["datetime"].dt.normalize()
    days = sorted(test_df["day"].unique())
    preds = np.zeros(len(test_df), dtype=float)

    # Create mapping from (day, key) to test row indices
    test_groups = test_df.groupby(["day"] + GRP_COLS, observed=True)
    day_key_indices = {k: np.asarray(idx) for k, idx in test_groups.indices.items()}

    # Also keep a lookup of revealed targets by day
    revealed_targets = test_df.groupby(["day"] + GRP_COLS, observed=True)["target"].mean().to_dict()

    for d in days:
        # 1. Update state using targets revealed from d - 2 days
        reveal_day = d - pd.Timedelta(days=2)
        for key in states:
            val = revealed_targets.get((reveal_day, *key))
            if val is not None and not np.isnan(val):
                st = states[key]
                if val > 0:
                    if method == "tsb":
                        st["p"] = alpha * 1.0 + (1 - alpha) * st["p"]
                        st["z"] = beta * val + (1 - beta) * st["z"]
                    else:
                        st["z"] = alpha * val + (1 - alpha) * st["z"]
                        st["p"] = alpha * st["q"] + (1 - alpha) * st["p"]
                        st["q"] = 1.0
                else:
                    if method == "tsb":
                        st["p"] = alpha * 0.0 + (1 - alpha) * st["p"]
                    else:
                        st["q"] += 1.0

        # 2. Predict day d
        for key in states:
            idx = day_key_indices.get((d, *key))
            if idx is not None:
                st = states[key]
                if method == "tsb":
                    forecast = st["p"] * st["z"]
                else:
                    forecast = st["z"] / max(st["p"], 1e-4) if st["p"] > 0 else 0.0
                preds[idx] = max(0.0, forecast)

    pred_df = pd.DataFrame({"row_id": test_df["row_id"].to_numpy(dtype="int64"), "pred": preds})
    model_name = f"{method}_intermittent"
    metrics = base.evaluate_predictions(pred_df, model_name, out_dir=out_dir)
    return pred_df, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Croston/TSB intermittent demand benchmarks.")
    parser.add_argument("--method", choices=["tsb", "croston"], default="tsb")
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--beta", type=float, default=0.2)
    args = parser.parse_args()

    print(f"[benchmark] Running {args.method.upper()} with alpha={args.alpha}, beta={args.beta}...")
    pred_df, metrics = run_tsb_benchmark(alpha=args.alpha, beta=args.beta, method=args.method)
    print(f"[benchmark] {args.method.upper()} Results:")
    for k, v in metrics.items():
        if k.endswith("/mae") or k.endswith("/wape"):
            print(f"  {k}: {v:.2f}")


if __name__ == "__main__":
    main()
