"""Evaluation in a fresh container: a SageMaker training job has the feature tables but no parquet cache of the raw
CSVs (the first AWS run of v1 failed there, in attach_truth)."""
import pandas as pd

from enefit import config as C
from enefit import evaluate as E

from conftest import requires_data


@requires_data
def test_attach_truth_without_the_raw_parquet_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "PROCESSED_DIR", tmp_path)
    out = E.attach_truth(pd.DataFrame({"row_id": [0, 1], "pred": [1.0, 2.0]}))
    assert list(out["row_id"]) == [0, 1] and out["y"].notna().all()
