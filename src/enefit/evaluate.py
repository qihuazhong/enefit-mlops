"""Score final predictions against the raw float64 target of train.csv and a naive baseline
(same segment and hour 7 days earlier). Writes per-side / per-segment / per-day CSVs and a daily-MAE plot.

MAE is in kWh per segment-hour (used for training and model selection). WAPE = sum |error| / sum actual: the share of
the energy the forecast missed, for non-technical readers (no division by zero on night-time solar hours, unlike MAPE).
relative_bias = sum (prediction - actual) / sum actual: negative = under-forecast, as a share of the energy."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config as C
from . import data

SIDE_NAMES = C.SIDE_NAMES


def attach_truth(pred: pd.DataFrame) -> pd.DataFrame:
    """``pred``: row_id, pred. Adds y (raw target), segment keys, datetime, naive_lag7; drops rows without a target."""
    data.build_parquet_cache()   # a fresh container (SageMaker job) has the feature tables but not this cache
    raw = pd.read_parquet(data.parquet_path("train"), columns=["row_id", "datetime", "target"] + C.SEGMENT_KEYS)
    out = pred[["row_id", "pred"]].merge(raw.rename(columns={"target": "y"}), on="row_id", how="left", validate="one_to_one")
    lag = raw[C.SEGMENT_KEYS + ["datetime", "target"]].rename(columns={"target": "naive_lag7"})
    lag["datetime"] = lag["datetime"] + pd.Timedelta(days=7)
    out = out.merge(lag, on=C.SEGMENT_KEYS + ["datetime"], how="left")
    return out[out["y"].notna()].reset_index(drop=True)


def wape(pred, y) -> float:
    return float((pred - y).abs().sum() / y.abs().sum())


def relative_bias(pred, y) -> float:
    return float((pred - y).sum() / y.abs().sum())


def _summary(df: pd.DataFrame, by) -> pd.DataFrame:
    df = df.assign(err=df["pred"] - df["y"], abs_err=(df["pred"] - df["y"]).abs(), abs_y=df["y"].abs(),
                   naive_abs_err=(df["naive_lag7"] - df["y"]).abs(),
                   naive_abs_y=df["y"].abs().where(df["naive_lag7"].notna()))
    out = df.groupby(by, observed=True).agg(n=("y", "size"), mae=("abs_err", "mean"), bias=("err", "mean"),
                                            naive_lag7_mae=("naive_abs_err", "mean"), sum_err=("err", "sum"), sum_abs_err=("abs_err", "sum"),
                                            sum_abs_y=("abs_y", "sum"), naive_sum_abs_err=("naive_abs_err", "sum"),
                                            naive_sum_abs_y=("naive_abs_y", "sum"))
    out["wape"] = out["sum_abs_err"] / out["sum_abs_y"]
    out["relative_bias"] = out["sum_err"] / out["sum_abs_y"]
    out["naive_lag7_wape"] = out["naive_sum_abs_err"] / out["naive_sum_abs_y"]
    return out.drop(columns=["sum_err", "sum_abs_err", "sum_abs_y", "naive_sum_abs_err", "naive_sum_abs_y"]).reset_index()


def evaluate(scored: pd.DataFrame, out_dir: Path, prefix: str) -> dict[str, float]:
    """Writes <prefix>_by_side/_by_segment/_by_day CSVs + <prefix>_daily_mae.png; returns headline metrics."""
    out_dir.mkdir(parents=True, exist_ok=True)
    df = scored.assign(side=scored["is_consumption"].map(SIDE_NAMES), day=scored["datetime"].dt.normalize(), all="all")
    by_side = pd.concat([_summary(df, "all").rename(columns={"all": "side"}), _summary(df, "side")])
    by_side.to_csv(out_dir / f"{prefix}_by_side.csv", index=False)
    _summary(df, C.SEGMENT_KEYS).to_csv(out_dir / f"{prefix}_by_segment.csv", index=False)
    by_day = _summary(df, ["side", "day"])
    by_day.to_csv(out_dir / f"{prefix}_by_day.csv", index=False)
    plot_daily(by_day, out_dir / f"{prefix}_daily_mae.png", prefix)
    return {f"{prefix}/{r.side}/{k}": float(getattr(r, k)) for r in by_side.itertuples()
            for k in ("mae", "bias", "naive_lag7_mae", "wape", "relative_bias", "naive_lag7_wape")}


def plot_daily(by_day: pd.DataFrame, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for ax, side in zip(axes, ["production", "consumption"]):
        g = by_day[by_day["side"] == side]
        ax.plot(g["day"], g["mae"], label="model", color="#1f5fa8", lw=1.8)
        ax.plot(g["day"], g["naive_lag7_mae"], label="naive: same hour 7 days ago", color="#999999", lw=1.2)
        ax.set_ylabel("daily MAE (kWh)")
        ax.set_title(side, loc="left", fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle(f"{title}: daily MAE, model vs naive", fontsize=11)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
