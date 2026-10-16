"""Live run: walk through the held-out days 2023-04-16 .. 2023-05-31 one at a time, as if deployed.

    python -m enefit.live_run [--start-champion N] [--workers 5 --threads 2]

Each day D: build D's features from what was available at D (features.features_for_day), score them with the
registry's @champion (and with the shadow challenger, if any), reveal the targets of D-2, run the drift checks.
Every 7 days (from 2023-04-23), in this order:
  1. rollback check: a champion promoted a week ago must pass `gate` against the model it replaced (which kept
     scoring in shadow for that week); if it fails, @champion goes back to the previous model;
  2. promotion: the challenger created a week earlier replaces the champion if it passes `gate` on the days it
     scored in shadow;
  3. a new challenger is refit on everything revealed so far (through D-2) with the champion's settings.

Drift checks, per day:
  capacity_segments    production segments whose installed capacity on D exceeds the champion's training maximum
  pred_above_max       share of D's production predictions above the segment's largest training target
  segments_worse_naive segments whose MAE over the last 7 revealed days exceeds the naive same-hour-7-days-ago MAE
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime

import mlflow
import numpy as np
import pandas as pd

from . import config as C
from . import data
from . import evaluate as E
from . import features as F
from . import model as M
from . import train as TR

DAYS = pd.date_range(C.LIVE_START, C.LIVE_END)
RETRAIN_DAYS = set(DAYS[7::7])
DAY = pd.Timedelta(days=1)
SEG = ["county", "is_business", "product_type", "is_consumption"]
NAMES = M.REGISTERED_MODEL_NAMES
MARGIN = 0.02        # non-inferiority margin: a new model may be at most 2 % worse than the model it replaces
BIAS_MARGIN = 0.02   # |relative bias| of a side may grow by at most 2 percentage points
CONFIDENCE = 0.90    # one-sided bound for the non-inferiority check


def newest_milestone(client, variant: str = "v6") -> str:
    """Registry version of the newest `train --variant <variant> --register` (not a live-run retrain). Both sides are
    registered together, so they share the number."""
    found = {max(int(v.version) for v in client.search_model_versions(f"name='{name}'") if v.tags.get("label") == variant)
             for name in NAMES.values()}
    assert len(found) == 1, f"{variant}: the two sides have different registry versions {found}"
    return str(found.pop())


def gate(new: pd.DataFrame, ref: pd.DataFrame, seed: int = 0) -> tuple[bool, str, dict]:
    """Can ``new`` replace ``ref``? Both: predictions (row_id, pred) of the same rows, targets revealed.
    Non-inferiority: the 90 % upper bound of total error(new) / total error(ref), bootstrapping days (errors within a day
    are correlated, so a day is the unit), must be below 1 + MARGIN. Guardrails, point estimates: each side at most
    MARGIN worse; each side's |relative bias| at most BIAS_MARGIN worse; predictions finite, >= 0, same rows."""
    if (len(new) != len(ref) or set(new["row_id"]) != set(ref["row_id"]) or not np.isfinite(new["pred"]).all()
            or (new["pred"] < 0).any()):
        return False, "predictions missing, not finite or negative", {}
    n, r = (E.attach_truth(x).assign(ae=lambda d: (d["pred"] - d["y"]).abs()) for x in (new, ref))
    daily = pd.DataFrame({"new": n.groupby(n["datetime"].dt.normalize())["ae"].sum(),
                          "ref": r.groupby(r["datetime"].dt.normalize())["ae"].sum()})
    idx = np.random.default_rng(seed).integers(0, len(daily), (5000, len(daily)))
    upper = float(np.quantile(daily["new"].to_numpy()[idx].sum(1) / daily["ref"].to_numpy()[idx].sum(1), CONFIDENCE))
    stats = {"error_ratio": float(daily["new"].sum() / daily["ref"].sum()), "error_ratio_upper_90": upper,
             "new_mae": float(n["ae"].mean()), "ref_mae": float(r["ae"].mean()),
             "new_wape": E.wape(n["pred"], n["y"]), "ref_wape": E.wape(r["pred"], r["y"])}
    fails = [f"error up to {upper - 1:+.1%} (90% bound; margin {MARGIN:.0%})"] if upper >= 1 + MARGIN else []
    for side, name in M.SIDE_NAMES.items():
        ns, rs = n[n["is_consumption"] == side], r[r["is_consumption"] == side]
        ratio = float(ns["ae"].sum() / rs["ae"].sum())
        b_new, b_ref = E.relative_bias(ns["pred"], ns["y"]), E.relative_bias(rs["pred"], rs["y"])
        stats.update({f"{name}_error_ratio": ratio, f"{name}_relative_bias_new": b_new, f"{name}_relative_bias_ref": b_ref})
        if ratio > 1 + MARGIN:
            fails.append(f"{name} error {ratio - 1:+.1%}")
        if abs(b_new) > abs(b_ref) + BIAS_MARGIN:
            fails.append(f"{name} bias {b_new:+.1%} vs {b_ref:+.1%}")
    reason = "; ".join(fails) or f"error {stats['error_ratio'] - 1:+.1%} (90% bound {upper - 1:+.1%})"
    return not fails, reason, stats


def training_limits(train_path, end) -> pd.DataFrame:
    """Per segment: largest installed capacity and target seen in training rows before ``end``."""
    df = pd.read_parquet(train_path, columns=SEG + ["installed_capacity", "target"], filters=[("datetime", "<", end)])
    return df.groupby(SEG, observed=True).agg(cap_max=("installed_capacity", "max"), y_max=("target", "max")).reset_index()


def retrain(day, rounds: dict, train_path, workers: int, threads: int, run_dir) -> dict[int, object]:
    """Refit the 14 v6 boosters on every row revealed by ``day`` (through D-2), champion's numbers of rounds."""
    end = day - DAY
    run_dir.mkdir(parents=True)
    factors = TR.augmentation_factors(train_path, end, C.AUGMENTATION)
    jobs = [{"side": s, "target": t, "fold": None, "variant": "v6", "train_path": train_path, "threads": threads,
             "train_end": end, "factors": factors, "rounds": rounds[f"{M.SIDE_NAMES[s]}/{t}"],
             "booster_file": str(run_dir / f"{s}_{t}.txt")} for s in (0, 1) for t in C.TARGETS]
    with TR.pool(workers) as ex:
        list(ex.map(TR.fit, jobs))
    meta = {"trained_through": str((day - 2 * DAY).date())}
    return {s: M.write_model_dir(run_dir / M.SIDE_NAMES[s], s, {t: run_dir / f"{s}_{t}.txt" for t in C.TARGETS}, meta)
            for s in (0, 1)}


def plot(daily: pd.DataFrame, decisions: pd.DataFrame, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for ax, side in zip(axes[:2], ["production", "consumption"]):
        ax.plot(daily["revealed_day"], daily[f"{side}_mae"], label="champion", color="#1f5fa8", lw=1.8)
        ax.plot(daily["revealed_day"], daily[f"{side}_naive_mae"], label="naive: same hour 7 days ago", color="#999999")
        ax.set_ylabel(f"{side}\ndaily MAE")
        ax.grid(alpha=0.3)
    axes[2].bar(daily["day"], daily["capacity_segments"], color="#cc8a2e", label="segments above training capacity")
    axes[2].set_ylabel("drift alerts")
    axes[2].grid(alpha=0.3)
    colors = {"promoted": "#2a9d4b", "rejected": "#c44", "rolled back": "#7b3fa0"}
    for r in decisions[decisions["decision"].isin(colors)].itertuples():
        for ax in axes:
            ax.axvline(r.day, color=colors[r.decision], ls="--", lw=1)
    axes[0].legend(fontsize=8, frameon=False)
    axes[2].legend(fontsize=8, frameon=False)
    fig.suptitle("Live run 2023-04-16..05-31: daily MAE; dashed = retrained model went live (green) / kept old model (red) / rolled back (purple)")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--start-champion", help="registry version that is @champion when the live run starts "
                                             "(default: the newest v6 milestone)")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args(argv)
    run_dir = TR.OUTPUTS / "live_run" / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True)
    TR.connect()
    client = mlflow.MlflowClient()
    raw = data.load_raw_tables()
    train_path, _ = TR.feature_table(C.TRAIN_FEATURE_CONFIG)

    champion = a.start_champion or newest_milestone(client)
    for s in (0, 1):
        client.set_registered_model_alias(NAMES[s], "champion", champion)
    start = client.get_model_version(NAMES[0], champion)
    rounds = json.loads(client.get_run(start.run_id).data.params["refit_rounds"])
    run_ids = {}
    cutoffs = {champion: pd.Timestamp(start.tags.get("trained_through", C.TRAIN_END))}   # registry version -> data end
    limits = training_limits(train_path, cutoffs[champion] + DAY)
    models, challenger, previous = {}, None, None   # previous: the model replaced a week ago, scored until the check

    def label(version):     # registry numbers are internal ids; people read the label (e.g. "v6 retrained, data to 2023-04-21")
        tags = client.get_model_version(NAMES[0], version).tags
        return tags.get("label", tags.get("variant", f"registry #{version}"))

    def model(version, side):
        if (version, side) not in models:
            models[(version, side)] = M.load_registered(NAMES[side], version)
        return models[(version, side)]

    preds, scored, daily, decisions = [], [], [], []
    with mlflow.start_run(run_name="live run 2023-04-16 → 2023-05-31 (daily log, not a model)"):
        mlflow.log_params({"start_champion": champion, "retrain_days": json.dumps(sorted(str(d.date()) for d in RETRAIN_DAYS)),
                           "promotion_rule": f"non-inferiority: {CONFIDENCE:.0%} upper bound of error ratio < "
                                             f"{1 + MARGIN}; guardrails: each side error ratio <= {1 + MARGIN}, "
                                             f"|relative bias| +{BIAS_MARGIN:.0%} at most, valid predictions",
                           "rollback_rule": "a promoted model fails the same gate against the model it replaced "
                                            "after one week live"})
        for i, day in enumerate(DAYS):
            feats = F.features_for_day(raw, day, C.SERVING_FEATURE_CONFIG)
            roles = ([("champion", champion)] + ([("challenger", challenger["version"])] if challenger else [])
                     + ([("previous", previous["version"])] if previous else []))
            for role, version in roles:
                for s in (0, 1):
                    df = feats[feats["is_consumption"] == s]
                    preds.append(pd.DataFrame({"row_id": df["row_id"].to_numpy(), "day": day, "role": role, "version": str(version),
                                               "model": label(version), "pred": model(version, s).predict(None, df)}))
            today = preds[-len(roles) * 2]   # champion production rows of today
            prod = feats[feats["is_consumption"] == 0].merge(limits, on=SEG, how="left")
            ch_prod = prod[["row_id", *SEG, "y_max"]].merge(today, on="row_id")
            row = {"day": day, "champion": label(champion), "challenger": label(challenger["version"]) if challenger else None,
                   "capacity_segments": len(prod.loc[prod["installed_capacity"] > prod["cap_max"] + 1e-6, SEG].drop_duplicates()),
                   "pred_above_max": float((ch_prod["pred"] > ch_prod["y_max"]).mean())}

            revealed = day - 2 * DAY                      # targets of D-2 arrive with D
            if revealed >= DAYS[0]:
                p = pd.concat(preds)
                rev = E.attach_truth(p[(p["day"] == revealed) & (p["role"] == "champion")])
                scored.append(rev.assign(day=revealed))
                row["revealed_day"] = revealed
                for s, name in M.SIDE_NAMES.items():
                    g = rev[rev["is_consumption"] == s]
                    row[f"{name}_mae"] = float((g["pred"] - g["y"]).abs().mean())
                    row[f"{name}_naive_mae"] = float((g["naive_lag7"] - g["y"]).abs().mean())
                    row[f"{name}_wape"] = E.wape(g["pred"], g["y"])
                    row[f"{name}_relative_bias"] = E.relative_bias(g["pred"], g["y"])
                last7 = pd.concat(scored)
                last7 = last7[last7["day"] > revealed - 7 * DAY]
                seg = last7.assign(e=(last7["pred"] - last7["y"]).abs(), n=(last7["naive_lag7"] - last7["y"]).abs()) \
                    .groupby(SEG)[["e", "n"]].mean()
                row["segments_worse_naive"] = int((seg["e"] > seg["n"]).sum())

            if day in RETRAIN_DAYS:
                p = pd.concat(preds)

                def shadow(version, first_day):   # a model's predictions on the days since first_day with targets revealed
                    return p[(p["version"] == str(version)) & (p["day"] >= first_day) & (p["day"] <= revealed)]

                def set_champion(version):
                    for s in (0, 1):
                        client.set_registered_model_alias(NAMES[s], "champion", version)

                if previous:                              # 1. rollback check on last week's promotion
                    ok, reason, st = gate(shadow(champion, previous["first_day"]),
                                          shadow(previous["version"], previous["first_day"]))
                    decision = "confirmed after a week live" if ok else "rolled back"
                    decisions.append({"day": day, "model": label(champion), "compared_with": label(previous["version"]),
                                      "days": f"{previous['first_day'].date()}..{revealed.date()}", **st,
                                      "decision": decision, "reason": reason})
                    client.set_tag(run_ids[champion], "after_a_week_live", decision)   # client calls: the finished
                    client.set_tag(run_ids[champion], "after_a_week_live_reason", reason)   # run keeps its end time
                    TR.log(f"[live run] {day.date()} {label(champion)} {decision}: {reason}")
                    if not ok:
                        for s in (0, 1):
                            client.set_model_version_tag(NAMES[s], champion, "decision", f"rolled back: {reason}")
                        champion = previous["version"]
                        set_champion(champion)
                        limits = training_limits(train_path, cutoffs[champion] + DAY)
                    previous = None
                if challenger:                            # 2. promotion: the challenger against the model live now
                    ok, reason, st = gate(shadow(challenger["version"], challenger["first_day"]),
                                          shadow(champion, challenger["first_day"]))
                    decision = "promoted" if ok else "rejected"
                    for s in (0, 1):
                        client.set_model_version_tag(NAMES[s], challenger["version"], "decision", f"{decision}: {reason}")
                    decisions.append({"day": day, "model": label(challenger["version"]), "compared_with": label(champion),
                                      "days": f"{challenger['first_day'].date()}..{revealed.date()}", **st,
                                      "decision": decision, "reason": reason})
                    for k, v in {"shadow_mae": st["new_mae"], "champion_mae_same_days": st["ref_mae"],
                                 "shadow_wape": st["new_wape"], "champion_wape_same_days": st["ref_wape"],
                                 **{k: v for k, v in st.items() if "ratio" in k or "bias" in k}}.items():
                        client.log_metric(challenger["run_id"], k, v)
                    for k, v in {"decision": decision, "reason": reason, "shadow_days": decisions[-1]["days"]}.items():
                        client.set_tag(challenger["run_id"], k, v)
                    TR.log(f"[live run] {day.date()} {label(challenger['version'])} {decision}: {reason}")
                    if ok:
                        previous = {"version": champion, "first_day": day + DAY}
                        champion = challenger["version"]
                        set_champion(champion)
                        limits = training_limits(train_path, cutoffs[champion] + DAY)
                    challenger = None
                if day < DAYS[-1] - 7 * DAY + DAY:        # a new challenger needs a full shadow week before the end
                    cut = day - 2 * DAY
                    name = f"v6 retrained, data to {cut.date()}"
                    with mlflow.start_run(run_name=name, nested=True) as child:   # opened first: duration = training time
                        mlflow.log_params({"model": name, "trained_through": str(cut.date()),
                                           "settings": "v6 (same features and parameters)"})
                        dirs = retrain(day, rounds, train_path, a.workers, a.threads, run_dir / f"retrain_{day.date()}")
                        versions = [mlflow.register_model(M.log_side_model(dirs[s], s).model_uri, NAMES[s], tags={
                            "variant": "v6", "trained_through": str(cut.date()),
                            "label": name}).version
                            for s in (0, 1)]
                    assert versions[0] == versions[1], versions      # both registries move in lockstep
                    challenger = {"version": versions[0], "first_day": day + DAY, "run_id": child.info.run_id}
                    cutoffs[versions[0]], run_ids[versions[0]] = cut, child.info.run_id
                    TR.log(f"[live run] {day.date()} new challenger: {label(versions[0])}")
            daily.append(row)
            mlflow.log_metrics({k: v for k, v in row.items() if isinstance(v, (int, float)) and not np.isnan(v)}, step=i)
            TR.log(f"[live run] {day.date()} " + ", ".join(
                f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items() if k not in ("day", "revealed_day")))

        daily, decisions = pd.DataFrame(daily), pd.DataFrame(decisions)
        daily.to_csv(run_dir / "daily.csv", index=False)
        decisions.to_csv(run_dir / "decisions.csv", index=False)
        pd.concat(preds).to_parquet(run_dir / "predictions.parquet", index=False)
        plot(daily.dropna(subset=["revealed_day"]), decisions, run_dir / "live_run.png")
        for f in ("daily.csv", "decisions.csv", "live_run.png"):
            mlflow.log_artifact(str(run_dir / f))
        TR.log(f"[live run] done: champion {label(champion)}; outputs {run_dir}")


if __name__ == "__main__":
    main()
