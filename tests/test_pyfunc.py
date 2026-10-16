"""The packaged side model, reloaded through MLflow, predicts on feature-table rows as `features.py` writes them
(no casting by the caller) and returns exactly the direct booster reconstruction."""
import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
import pytest

from enefit import config as C
from enefit import model as M
from enefit import targets as T

PATH = C.features_path(C.SERVING_FEATURE_CONFIG)


@pytest.mark.skipif(not PATH.exists(), reason="run `python -m enefit.features --build` first")
@pytest.mark.parametrize("variant,side", [("v6", 0), ("v6", 1), ("v1", 0)])
def test_logged_model_predicts_on_feature_table_rows(tmp_path, variant, side):
    V = C.variant(variant)
    df = pd.read_parquet(PATH, filters=[("is_consumption", "==", side), ("date", "==", pd.Timestamp("2023-03-10"))])
    files = {}
    for t in V["targets"]:
        X = M.booster_frame(df, V["features"][(side, t)])
        b = lgb.train({"objective": "l2", "verbose": -1, "num_threads": 1}, lgb.Dataset(X, df[t].fillna(0)), 5)
        files[t] = tmp_path / f"{t}.txt"
        b.save_model(str(files[t]))
    model_dir = M.write_model_dir(tmp_path / "model", side, files, {}, variant)

    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")    # throwaway store (MLflow 3 has no file store)
    exp = mlflow.create_experiment("test", artifact_location=(tmp_path / "artifacts").as_uri())
    with mlflow.start_run(experiment_id=exp):
        uri = M.log_side_model(model_dir, side).model_uri
    pred = mlflow.pyfunc.load_model(uri).predict(df)

    direct = {t: lgb.Booster(model_file=str(files[t])).predict(M.booster_frame(df, V["features"][(side, t)]))
              for t in V["targets"]}
    np.testing.assert_array_equal(pred, T.final_prediction(df, direct, side, V["ensemble"][side]))
    if variant == "v1":     # the packaged model uses its own ensemble (the single direct model), not config.ENSEMBLE
        np.testing.assert_array_equal(pred, np.clip(direct["target"], 0, None))


@pytest.mark.skipif(not PATH.exists(), reason="run `python -m enefit.features --build` first")
def test_model_gives_the_same_predictions_on_the_scoring_path_and_the_stored_table(raw, tmp_path):
    """features_for_day returns integer categoricals as pandas categories, the parquet as int64."""
    from enefit import features
    d = pd.Timestamp("2023-04-20")
    stored = pd.read_parquet(PATH, filters=[("is_consumption", "==", 1), ("date", "==", d)]).sort_values("row_id")
    scored = features.features_for_day(raw, d)
    scored = scored[scored["is_consumption"] == 1].sort_values("row_id")
    files = {}
    for t in C.TARGETS:
        X = M.booster_frame(stored, C.FEATURES[(1, t)])
        lgb.train({"objective": "l2", "verbose": -1, "num_threads": 1, "min_data_in_leaf": 5},
                  lgb.Dataset(X, stored["lag_2_day"].fillna(0), categorical_feature=C.categorical_features(1)),
                  5).save_model(str(tmp_path / f"{t}.txt"))
        files[t] = tmp_path / f"{t}.txt"
    m = M.EnefitSideModel.from_dir(M.write_model_dir(tmp_path / "model", 1, files, {}))
    np.testing.assert_array_equal(m.predict(None, scored), m.predict(None, stored))
