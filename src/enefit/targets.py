"""Derived targets and the reconstruction of the final prediction.

All functions take the frame they are given and nothing else (the notebook's serving /
online-training path read the *global* `train` inside `append_diffs`).
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from . import config as C


# ------------------------------------------------------------------------------ derived targets
def add_derived_targets(df: pd.DataFrame, lag_columns: Sequence[str] = C.DIFF_LAG_COLUMNS) -> pd.DataFrame:
    """`<lag>_diff = max(target - lag, 0)` and `<lag>_diff_neg = max(lag - target, 0)` (v6 cell 62).

    Computed from ``df['target']`` and ``df[lag]`` of *this* frame, in the dtype of the inputs. Returns a new frame.
    """
    out = df.copy()
    for lag in lag_columns:
        out[f"{lag}_diff"] = np.clip(out["target"] - out[lag], 0, np.inf)
        out[f"{lag}_diff_neg"] = np.clip(out[lag] - out["target"], 0, np.inf)
    return out


# ------------------------------------------------------------------------------ reconstruction
def reconstruct_component(component: C.EnsembleComponent, preds: Mapping[str, np.ndarray],
                          df: pd.DataFrame) -> np.ndarray:
    """Direct model -> its prediction; lag model -> lag + (pos - neg) (cell 77)."""
    if component.lag_column is None:
        return np.asarray(preds["target"], dtype="float64")
    pos, neg = component.models
    return np.asarray(preds[pos], dtype="float64") - np.asarray(preds[neg], dtype="float64") \
        + df[component.lag_column].to_numpy(dtype="float64")


def final_prediction(df: pd.DataFrame, preds: Mapping[str, np.ndarray], is_consumption: int,
                     ensemble: Sequence[C.EnsembleComponent] | None = None) -> np.ndarray:
    """Weighted ensemble of the reconstructed components, clipped at 0 (cell 77).

    ``df`` holds the rows of ONE side; ``preds`` maps model name (config.TARGETS entry) to the model's
    raw output for those rows. ``ensemble`` defaults to config.ENSEMBLE (v6); variants pass their own.
    """
    total = 0.0
    for comp in ensemble or C.ENSEMBLE[is_consumption]:
        total = total + comp.weight * reconstruct_component(comp, preds, df)
    return np.clip(total, 0, None)
