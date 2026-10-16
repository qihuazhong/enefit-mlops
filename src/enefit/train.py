"""Train both sides on the train block, evaluate on the test block, log to MLflow, optionally register.

    python -m enefit.train [--variant v6] [--register [--champion]] [--workers 5 --threads 2]

Recipe: training rows use forecast weather averaged over every origin (config.TRAIN_FEATURE_CONFIG);
validation and test rows use only the origins available at prediction time (config.SERVING_FEATURE_CONFIG).
One LightGBM booster per (side, target). Forward-chaining CV (3 windows spread over the seasons, 2-day embargo)
finds the number of rounds by early stopping; the production booster is refit on the whole train block with the
median. Growth augmentation (config.AUGMENTATION) is applied to training rows only. ``--variant`` selects one of the
notebook milestones (config.VARIANTS); the recipe is the same for all.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd

from . import config as C
from . import data
from . import evaluate as E
from . import features as F
from . import model as M
from . import targets as T

SIDE_NAMES = M.SIDE_NAMES
FOLDS = [("f1", "2022-05-01", "2022-06-30"), ("f2", "2022-09-01", "2022-10-31"), ("f3", "2023-01-01", "2023-02-28")]
EMBARGO = pd.Timedelta(days=C.TARGET_DELAY_DAYS)
DAY = pd.Timedelta(days=1)
TRAIN_END = pd.Timestamp(C.TRAIN_END) + DAY
TEST_START, TEST_END = pd.Timestamp(C.TEST_START), pd.Timestamp(C.TEST_END) + DAY
OUTPUTS = C.ROOT / "outputs"


def log(msg: str) -> None:
    print(f"{datetime.now():%H:%M:%S} {msg}", flush=True)


def connect() -> None:
    """MLflow server from $MLFLOW_TRACKING_URI (default: the local server of the README), experiment "enefit"."""
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5050"))
    mlflow.set_experiment("enefit")


def feature_table(cfg: C.FeatureConfig) -> tuple[Path, dict]:
    """The table of ``cfg`` (a variant with other FeatureConfig flags has its own); built when missing or stale,
    i.e. built from other input data or other feature code than what is here now."""
    path, sidecar = C.features_path(cfg), C.features_path(cfg).with_suffix(".json")
    meta = json.loads(sidecar.read_text()) if path.exists() and sidecar.exists() else {}
    now = {"input_hash": data.combined_input_hash(), "code_hash": F.code_hash()}
    if any(meta.get(k) != v for k, v in now.items()):
        log(f"[features] {path.name} {'is stale' if meta else 'is missing'}: building it")
        F.build_and_write(cfg)
        meta = json.loads(sidecar.read_text())
    return path, meta


def read_rows(path: Path, side: int, start, end, columns=None) -> pd.DataFrame:
    return pd.read_parquet(path, columns=columns, filters=[("is_consumption", "==", side),
                                                           ("datetime", ">=", start), ("datetime", "<", end)])


def augmentation_factors(train_path: Path, end, a: dict) -> pd.DataFrame:
    """Last n_rows training rows by datetime (both sides, as v6), factor ~ U(low, low + width), fixed seed."""
    rows = pd.read_parquet(train_path, columns=["row_id", "datetime", "target"], filters=[("datetime", "<", end)])
    rows = rows[rows["target"].notna()].sort_values(["datetime", "row_id"], kind="mergesort")
    rows = rows.iloc[max(len(rows) - a["n_rows"], 0):]
    factor = np.random.RandomState(a["seed"]).rand(len(rows)) * a["width"] + a["low"]
    return pd.DataFrame({"row_id": rows["row_id"].to_numpy(), "factor": factor})


def augment(df: pd.DataFrame, factors: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if factors.empty:
        return df
    f = df[["row_id"]].merge(factors, on="row_id", how="left")["factor"].to_numpy()
    copies = df.loc[~np.isnan(f)].copy()
    for c in columns:
        if c in copies.columns:
            copies[c] = (copies[c].to_numpy("float64") * f[~np.isnan(f)]).astype(df[c].dtype)
    return pd.concat([df, copies], ignore_index=True)


def fit(job: dict) -> dict:
    """One booster: a CV fold (early stopping on its serving-feature validation window) or the refit."""
    t0 = time.time()
    side, target, fold = job["side"], job["target"], job["fold"]
    V = C.variant(job["variant"])     # resolved here: spawned workers re-import config
    feats, aug = list(V["features"][(side, target)]), V["augmentation"]
    cols = list(dict.fromkeys(["row_id", "datetime", target, *feats, *aug["columns"]]))
    end = fold["train_end"] if fold else job.get("train_end", TRAIN_END)
    tr = read_rows(job["train_path"], side, pd.Timestamp(C.TRAIN_START), end, cols)
    tr = augment(tr[tr[target].notna()], job["factors"], aug["columns"])
    params = {**C.lgb_params(side, target, V.get("lgb")), "num_threads": job["threads"],
              "monotone_constraints": C.monotone_constraints(feats, side, target, V.get("monotone"))}
    cats = [c for c in C.categorical_features(side) if c in feats]
    dtrain = lgb.Dataset(M.booster_frame(tr, feats), tr[target].to_numpy("float64"), categorical_feature=cats)
    out = {"side": side, "target": target, "fold": fold["name"] if fold else "refit", "n_train": len(tr)}
    if fold:
        va = read_rows(job["serving_path"], side, fold["val_start"], fold["val_end"], ["row_id", target, *feats])
        va = va[va[target].notna()]
        dval = lgb.Dataset(M.booster_frame(va, feats), va[target].to_numpy("float64"), reference=dtrain)
        b = lgb.train(params, dtrain, C.LGB_CV["num_boost_round"], valid_sets=[dval],
                      callbacks=[lgb.early_stopping(C.LGB_CV["early_stopping_rounds"], verbose=False)])
        pred = b.predict(M.booster_frame(va, feats), num_iteration=b.best_iteration)
        out.update(best_iteration=b.best_iteration, val_mae=float(np.mean(np.abs(pred - va[target]))),
                   val_pred=pd.Series(pred, index=va["row_id"].to_numpy()))
    else:
        b = lgb.train(params, dtrain, job["rounds"])
        b.save_model(job["booster_file"])
        out["rounds"] = job["rounds"]
    out["seconds"] = round(time.time() - t0, 1)
    log(f"[fit] {SIDE_NAMES[side]}/{target}/{out['fold']} {out['seconds']}s"
        + (f" best_iter={out['best_iteration']} val_mae={out['val_mae']:.3f}" if fold else f" rounds={job['rounds']}"))
    return out


def cv_mae(serving_path: Path, folds: list[dict], results: list[dict], ensemble: dict) -> dict[str, float]:
    """MAE of the final reconstructed prediction on the validation windows (row-weighted), per side."""
    out = {}
    for side in (0, 1):
        err = []
        for fold in folds:
            va = read_rows(serving_path, side, fold["val_start"], fold["val_end"])
            va = va[va["target"].notna()].reset_index(drop=True)
            comps = {r["target"]: r["val_pred"].reindex(va["row_id"]).to_numpy()
                     for r in results if r["side"] == side and r["fold"] == fold["name"]}
            ok = ~np.isnan(np.column_stack(list(comps.values()))).any(axis=1)
            err.append(np.abs(T.final_prediction(va, comps, side, ensemble[side]) - va["target"].to_numpy("float64"))[ok])
        out[f"cv/{SIDE_NAMES[side]}/mae"] = float(np.concatenate(err).mean())
    return out


def score_test(models: dict, path: Path) -> pd.DataFrame:
    preds = []
    for side, m in models.items():
        df = read_rows(path, side, TEST_START, TEST_END)
        preds.append(pd.DataFrame({"row_id": df["row_id"], "pred": m.predict(None, df)}))
    return E.attach_truth(pd.concat(preds, ignore_index=True))


def pool(workers: int) -> ProcessPoolExecutor:
    """Worker processes started fresh ("spawn", the macOS default) on every platform. Linux's default "fork" copies the
    parent, and a parent that has already run LightGBM (OpenMP threads) gives workers that wait forever."""
    return ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--register", action="store_true", help="register both side models as new versions")
    ap.add_argument("--champion", action="store_true", help="with --register: point @champion at the new versions")
    ap.add_argument("--variant", default="v6", choices=sorted(C.VARIANTS), help="notebook milestone (config.VARIANTS)")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--threads", type=int, default=2, help="LightGBM threads per worker")
    a = ap.parse_args(argv)
    a.run_name = f"milestone-{a.variant}"
    V = C.variant(a.variant)
    t_start = time.time()
    run_dir = OUTPUTS / "runs" / f"{a.run_name}_{datetime.now():%Y%m%d_%H%M%S}"
    (run_dir / "boosters").mkdir(parents=True)
    train_path, train_meta = feature_table(V["train_cfg"])
    serving_path, serving_meta = feature_table(V["serving_cfg"])
    connect()
    git_commit = C.git_commit()
    with mlflow.start_run(run_name=a.run_name):      # opened before training: the run's duration = training time
        mlflow.log_params({"variant": a.variant, "kaggle_notebook_version": V["kaggle_version"],
                           "train_features": V["train_cfg"].forecast_origin,
                           "eval_features": V["serving_cfg"].forecast_origin, "cv_windows": json.dumps(FOLDS),
                           "augmentation_rows": V["augmentation"]["n_rows"], "workers": a.workers, "threads": a.threads})
        folds = [{"name": n, "train_end": pd.Timestamp(s) - EMBARGO, "val_start": pd.Timestamp(s),
                  "val_end": pd.Timestamp(e) + DAY} for n, s, e in FOLDS]
        factors = {f["name"]: augmentation_factors(train_path, f["train_end"], V["augmentation"]) for f in folds}
        factors["refit"] = augmentation_factors(train_path, TRAIN_END, V["augmentation"])
        common = {"train_path": train_path, "serving_path": serving_path, "threads": a.threads, "variant": a.variant}
        keys = [(s, t) for s in (0, 1) for t in V["targets"]]

        log(f"[train] {len(keys) * len(folds)} CV fits + {len(keys)} refits on {a.workers} workers x {a.threads} threads")
        with pool(a.workers) as ex:
            cv = list(ex.map(fit, [{**common, "side": s, "target": t, "fold": f, "factors": factors[f["name"]]}
                                   for f in reversed(folds) for s, t in keys]))
            rounds = {k: max(1, round(float(np.median([r["best_iteration"] for r in cv if (r["side"], r["target"]) == k]))))
                      for k in keys}
            refit = list(ex.map(fit, [{**common, "side": s, "target": t, "fold": None, "factors": factors["refit"],
                                       "rounds": rounds[(s, t)], "booster_file": str(run_dir / "boosters" / f"{s}_{t}.txt")}
                                      for s, t in keys]))
        pd.DataFrame([{k: v for k, v in r.items() if k != "val_pred"} for r in cv + refit]).to_csv(
            run_dir / "fits.csv", index=False)

        metrics = cv_mae(serving_path, folds, cv, V["ensemble"])
        meta = {"run_name": a.run_name, "variant": a.variant, "train_features": train_meta["parquet_sha256"],
                "serving_features": serving_meta["parquet_sha256"], "input_hash": serving_meta["input_hash"],
                "git_commit": git_commit}
        models, model_dirs = {}, {}
        for side in (0, 1):
            files = {t: run_dir / "boosters" / f"{side}_{t}.txt" for t in V["targets"]}
            model_dirs[side] = M.write_model_dir(run_dir / "models" / SIDE_NAMES[side], side, files, meta, a.variant)
            models[side] = M.EnefitSideModel.from_dir(model_dirs[side])

        metrics.update(E.evaluate(score_test(models, serving_path), run_dir / "eval", "test"))
        offline = E.evaluate(score_test(models, train_path), run_dir / "eval", "test_offline")
        metrics.update({k: v for k, v in offline.items() if k.endswith("/mae")})
        log("[eval] " + ", ".join(f"{k}={v:.2f}" for k, v in metrics.items()))

        mlflow.log_param("refit_rounds", json.dumps({f"{SIDE_NAMES[s]}/{t}": n for (s, t), n in rounds.items()}))
        mlflow.set_tags({"variant": a.variant, "notebook_fidelity": V["notebook_fidelity"],
                         "git_commit": meta["git_commit"], "input_data_hash": meta["input_hash"],
                         "train_features_sha256": meta["train_features"],
                         "serving_features_sha256": meta["serving_features"], "lightgbm": lgb.__version__})
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(run_dir / "fits.csv"))
        mlflow.log_artifacts(str(run_dir / "eval"), artifact_path="eval")
        for side in (0, 1):
            info = M.log_side_model(model_dirs[side], side)
            if a.register:
                name = M.REGISTERED_MODEL_NAMES[side]
                mv = mlflow.register_model(info.model_uri, name, tags={
                    "run_name": a.run_name, "git_commit": meta["git_commit"], "variant": a.variant, "label": a.variant,
                    "kaggle_notebook_version": str(V["kaggle_version"]), "notebook_date": V["notebook_date"], "notebook_fidelity": V["notebook_fidelity"],
                    "test_mae": f"{metrics[f'test/{SIDE_NAMES[side]}/mae']:.2f}",
                    "cv_mae": f"{metrics[f'cv/{SIDE_NAMES[side]}/mae']:.2f}"})
                if a.champion:
                    mlflow.MlflowClient().set_registered_model_alias(name, "champion", mv.version)
                log(f"[registry] {name}: {a.variant} (registry #{mv.version})" + (" @champion" if a.champion else ""))
    log(f"[train] done in {time.time() - t_start:.0f}s, run dir {run_dir}")


if __name__ == "__main__":
    main()
