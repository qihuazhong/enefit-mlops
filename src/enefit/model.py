"""One deployable model per side: the 7 boosters + feature lists + ensemble weights, as one MLflow pyfunc.

`EnefitSideModel.predict(features_df)` takes rows of the feature table as built by `features.py` (one side) and
returns the final reconstructed prediction (the ensemble stored in meta.json, clipped at 0), so a
scorer can never combine boosters, feature lists or weights from different training runs.
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from mlflow.pyfunc import PythonModel

from . import config as C
from . import targets

SIDE_NAMES = C.SIDE_NAMES
REGISTERED_MODEL_NAMES = {0: "enefit-production", 1: "enefit-consumption"}


def booster_frame(df: pd.DataFrame, features) -> pd.DataFrame:
    """Columns of one booster with the training dtypes (daytype* with the fixed categorical levels)."""
    X = df[list(features)].copy()
    for c in X.columns:
        if c.startswith("daytype"):
            X[c] = pd.Categorical(X[c].astype(str), categories=C.CATEGORIES["daytype"])
        elif isinstance(X[c].dtype, pd.CategoricalDtype):
            # the stored feature table has county/product_type/dayofweek/hour as integers, the in-memory scoring
            # path as categoricals with the same values; the boosters were trained on integers
            X[c] = X[c].astype("int64")
    return X


class EnefitSideModel(PythonModel):
    def __init__(self):
        self.meta: dict | None = None
        self.boosters: dict[str, lgb.Booster] | None = None

    @classmethod
    def from_dir(cls, model_dir: str | Path) -> "EnefitSideModel":
        m = cls()
        m._load(Path(model_dir) / "boosters", Path(model_dir) / "meta.json")
        return m

    def _load(self, boosters_dir: Path, meta_path: Path) -> None:
        self.meta = json.loads(Path(meta_path).read_text())
        self.boosters = {t: lgb.Booster(model_file=str(Path(boosters_dir) / f"{t}.txt")) for t in self.meta["targets"]}

    def load_context(self, context) -> None:
        self._load(Path(context.artifacts["boosters"]), Path(context.artifacts["meta"]))

    def predict(self, context, model_input: pd.DataFrame, params=None) -> np.ndarray:
        df = model_input.reset_index(drop=True)
        side = int(self.meta["side"])
        if not (df["is_consumption"] == side).all():
            raise ValueError(f"{SIDE_NAMES[side]} model got rows of the other side")
        comps = {t: self.boosters[t].predict(booster_frame(df, self.meta["features"][t])) for t in self.meta["targets"]}
        ensemble = [C.EnsembleComponent(**e) for e in self.meta["ensemble"]]
        return targets.final_prediction(df, comps, side, ensemble)


def write_model_dir(model_dir: Path, side: int, booster_files: dict[str, Path], extra_meta: dict,
                    variant: str = "v6") -> Path:
    """boosters/<target>.txt + meta.json (side, variant, targets, feature lists, ensemble, run metadata)."""
    V = C.variant(variant)
    (model_dir / "boosters").mkdir(parents=True, exist_ok=True)
    for t in V["targets"]:
        (model_dir / "boosters" / f"{t}.txt").write_text(Path(booster_files[t]).read_text())
    meta = {"side": side, "variant": variant, "targets": list(V["targets"]),
            "features": {t: list(V["features"][(side, t)]) for t in V["targets"]},
            "ensemble": [{"name": c.name, "lag_column": c.lag_column, "weight": c.weight} for c in V["ensemble"][side]],
            "lightgbm_version": lgb.__version__, **extra_meta}
    (model_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    return model_dir


def log_side_model(model_dir: Path, side: int):
    """Log the side model as an MLflow pyfunc in the active run. No signature: the input is the feature table
    as `features.py` writes it (categoricals included), and `predict` casts what the boosters need. No code copy:
    the `enefit` package is installed where the model runs (a frozen copy shadowed the installed code)."""
    import importlib.metadata as md
    reqs = [f"{p}=={md.version(p)}" for p in ["mlflow", "lightgbm", "pandas", "numpy", "scipy", "pyarrow"]]
    return mlflow.pyfunc.log_model(
        name=f"model_{SIDE_NAMES[side]}", python_model=EnefitSideModel(),
        artifacts={"boosters": str(model_dir / "boosters"), "meta": str(model_dir / "meta.json")},
        pip_requirements=reqs)


def load_registered(name: str, ref: str) -> EnefitSideModel:
    """Registered model by version or alias (``"6"`` or ``"@champion"``), loaded with the installed code."""
    uri = f"models:/{name}{ref if ref.startswith('@') else '/' + ref}"
    return EnefitSideModel.from_dir(Path(mlflow.artifacts.download_artifacts(uri)) / "artifacts")
