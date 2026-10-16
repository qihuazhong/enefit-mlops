"""Base utilities and evaluation harness for benchmark models."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence
import pandas as pd

from .. import config as C
from .. import evaluate as E

DAY = pd.Timedelta(days=1)
TEST_START = pd.Timestamp(C.TEST_START)
TEST_END = pd.Timestamp(C.TEST_END) + DAY
DEFAULT_BENCHMARK_DIR = C.ROOT / "outputs" / "benchmarks"

SHARED_FEATURE_COLS = [
    "row_id", "datetime", "target",
    "county", "is_business", "product_type", "is_consumption",
    "lag_2_day", "lag_7_day", "lag_7_day_mean",
    "installed_capacity",
    "temperature", "direct_solar_radiation",
    "hour", "dayofweek",
]


def get_feature_path() -> Path:
    V = C.variant("v6")
    return C.features_path(V["serving_cfg"])


def load_test_data(columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Load test rows (2023-03-01 to 2023-04-15) from serving features."""
    feat_path = get_feature_path()
    filters = [("datetime", ">=", TEST_START), ("datetime", "<", TEST_END)]
    return pd.read_parquet(feat_path, columns=columns, filters=filters)


def load_train_data(
    start: pd.Timestamp | str | None = None,
    end: pd.Timestamp | str | None = None,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Load training rows up to TEST_START from serving features."""
    feat_path = get_feature_path()
    start_ts = pd.Timestamp(start) if start is not None else pd.Timestamp(C.TRAIN_START)
    end_ts = pd.Timestamp(end) if end is not None else TEST_START
    filters = [("datetime", ">=", start_ts), ("datetime", "<", end_ts)]
    return pd.read_parquet(feat_path, columns=columns, filters=filters)


def evaluate_predictions(
    preds: pd.DataFrame,
    model_name: str,
    out_dir: Path | None = None,
) -> dict[str, float]:
    """Score model predictions with the repo's evaluation suite.

    Args:
        preds: DataFrame with ['row_id', 'pred']
        model_name: identifier for saved files
        out_dir: directory to write evaluation artifacts
    """
    target_dir = (out_dir or DEFAULT_BENCHMARK_DIR / model_name)
    target_dir.mkdir(parents=True, exist_ok=True)
    preds[["row_id", "pred"]].to_parquet(target_dir / "predictions.parquet", index=False)
    scored = E.attach_truth(preds[["row_id", "pred"]])
    metrics = E.evaluate(scored, target_dir, model_name)
    return metrics
