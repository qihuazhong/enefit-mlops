"""Shared helpers for the feature tests."""
from __future__ import annotations

import numpy as np
import pandas as pd

from enefit import config as C

NOT_FEATURES = {"target", "row_id", "data_block_id", "prediction_unit_id", *C.DERIVED_TARGET_COLUMNS}


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in C.OUTPUT_COLUMNS if c in df.columns and c not in NOT_FEATURES]


def diff_columns(a: pd.DataFrame, b: pd.DataFrame, columns=None) -> dict[str, tuple[int, float]]:
    """Columns where the two (row-aligned) frames differ exactly (NaN == NaN): {column: (n_rows, max_abs_diff)}."""
    assert len(a) == len(b) and (a["row_id"].to_numpy() == b["row_id"].to_numpy()).all()
    bad = {}
    for c in columns or feature_columns(a):
        x, y = a[c], b[c]
        if str(x.dtype) == "category" or x.dtype == object:
            neq = (x.astype(object).astype(str).to_numpy() != y.astype(object).astype(str).to_numpy())
            maxd = float("nan")
        elif np.issubdtype(x.dtype, np.datetime64):
            neq = x.to_numpy() != y.to_numpy()
            maxd = float("nan")
        else:
            xf, yf = x.to_numpy("float64"), y.to_numpy("float64")
            neq = ~((np.isnan(xf) & np.isnan(yf)) | (xf == yf))
            with np.errstate(invalid="ignore"):
                d = np.abs(xf - yf)[neq]
            maxd = float(np.nanmax(d)) if neq.any() and not np.isnan(d).all() else float("nan")
        if neq.any():
            bad[c] = (int(neq.sum()), maxd)
    return bad
