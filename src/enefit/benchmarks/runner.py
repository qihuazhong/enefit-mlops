"""Unified runner and report generator for forecasting benchmarks."""
from __future__ import annotations

import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from . import base
from .. import config as C

BENCHMARK_DIR = base.DEFAULT_BENCHMARK_DIR


def find_latest_lgbm_eval() -> Path | None:
    """Find the latest milestone-v6 eval directory."""
    runs_dir = C.ROOT / "outputs" / "runs"
    v6_runs = sorted(runs_dir.glob("milestone-v6_*"))
    if v6_runs:
        eval_path = v6_runs[-1] / "eval" / "test_by_side.csv"
        if eval_path.exists():
            return eval_path
    return None


def compile_comparison_summary() -> pd.DataFrame:
    """Compile headline metrics across all benchmark models and baselines."""
    records = []

    # 1. LightGBM Champion & Naive Baseline
    lgbm_csv = find_latest_lgbm_eval()
    if lgbm_csv is not None:
        df_lgbm = pd.read_csv(lgbm_csv).set_index("side")

        # LGBM Champion
        records.append({
            "model": "LightGBM v6 (@champion)",
            "category": "Gradient Boosted Trees",
            "total_mae": df_lgbm.loc["all", "mae"],
            "total_wape": df_lgbm.loc["all", "wape"],
            "production_mae": df_lgbm.loc["production", "mae"],
            "production_wape": df_lgbm.loc["production", "wape"],
            "consumption_mae": df_lgbm.loc["consumption", "mae"],
            "consumption_wape": df_lgbm.loc["consumption", "wape"],
            "prod_rel_bias": df_lgbm.loc["production", "relative_bias"],
            "cons_rel_bias": df_lgbm.loc["consumption", "relative_bias"],
        })

        # Naive 7-Day Baseline
        records.append({
            "model": "Naive 7-Day Baseline (naive_lag7)",
            "category": "Persistence Baseline",
            "total_mae": df_lgbm.loc["all", "naive_lag7_mae"],
            "total_wape": df_lgbm.loc["all", "naive_lag7_wape"],
            "production_mae": df_lgbm.loc["production", "naive_lag7_mae"],
            "production_wape": df_lgbm.loc["production", "naive_lag7_wape"],
            "consumption_mae": df_lgbm.loc["consumption", "naive_lag7_mae"],
            "consumption_wape": df_lgbm.loc["consumption", "naive_lag7_wape"],
            "prod_rel_bias": 0.0,
            "cons_rel_bias": 0.0,
        })

    # Benchmark models to scan
    models = [
        ("basic_lgbm", "Basic LightGBM (shared features)", "Gradient Boosted Trees"),
        ("tsb_intermittent", "TSB (Teunter-Syntetos-Babai)", "Intermittent Demand"),
        ("croston_intermittent", "Croston's Method", "Intermittent Demand"),
        ("prophet", "Prophet (shared features)", "Additive / Multi-Seasonality"),
        ("sarimax", "SARIMAX (shared features)", "Seasonal State-Space"),
        ("deepar", "DeepAR (shared features)", "Global Probabilistic RNN"),
    ]

    for dir_name, display_name, category in models:
        csv_path = BENCHMARK_DIR / dir_name / f"{dir_name}_by_side.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path).set_index("side")
        records.append({
            "model": display_name,
            "category": category,
            "total_mae": df.loc["all", "mae"],
            "total_wape": df.loc["all", "wape"],
            "production_mae": df.loc["production", "mae"],
            "production_wape": df.loc["production", "wape"],
            "consumption_mae": df.loc["consumption", "mae"],
            "consumption_wape": df.loc["consumption", "wape"],
            "prod_rel_bias": df.loc["production", "relative_bias"],
            "cons_rel_bias": df.loc["consumption", "relative_bias"],
        })

    summary_df = pd.DataFrame(records)
    summary_df = summary_df.sort_values("total_mae").reset_index(drop=True)
    summary_df.to_csv(BENCHMARK_DIR / "benchmark_comparison.csv", index=False)
    return summary_df


def plot_comparison_chart(summary_df: pd.DataFrame, out_path: Path) -> None:
    """Plot comparative MAE bar chart across models and sides."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)

    metrics = [
        ("total_mae", "Overall MAE (kWh)", "#1f5fa8"),
        ("production_mae", "Production MAE (kWh)", "#cc8a2e"),
        ("consumption_mae", "Consumption MAE (kWh)", "#2a9d4b"),
    ]

    sorted_df = summary_df.sort_values("total_mae", ascending=True)
    y_pos = range(len(sorted_df))

    for ax, (col, title, color) in zip(axes, metrics):
        bars = ax.barh(y_pos, sorted_df[col], color=color, alpha=0.85, edgecolor="none", height=0.6)
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(sorted_df["model"], fontsize=9)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
        ax.grid(axis="x", alpha=0.3, linestyle="--")

        # Value labels
        for bar in bars:
            val = bar.get_width()
            ax.text(val + (max(sorted_df[col]) * 0.02), bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}", va="center", fontsize=8, fontweight="bold")

        ax.set_xlim(0, max(sorted_df[col]) * 1.18)

    fig.suptitle("Enefit Forecasting Benchmarks: Test Set Performance (2023-03-01 .. 04-15)", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run benchmarks and compile comparison.")
    parser.add_argument("--model", choices=["all", "tsb", "croston", "prophet", "sarimax", "deepar", "none"], default="none")
    parser.add_argument("--summary-only", action="store_true", help="Compile summary from existing runs without re-training")
    args = parser.parse_args()

    if not args.summary_only and args.model != "none":
        if args.model in ("all", "tsb"):
            from .croston_tsb import run_tsb_benchmark
            run_tsb_benchmark(method="tsb")
        if args.model in ("all", "croston"):
            from .croston_tsb import run_tsb_benchmark
            run_tsb_benchmark(method="croston")
        if args.model in ("all", "prophet"):
            from .prophet import run_prophet_benchmark
            run_prophet_benchmark()
        if args.model in ("all", "sarimax"):
            from .sarimax import run_sarimax_benchmark
            run_sarimax_benchmark()
        if args.model in ("all", "deepar"):
            from .deepar import run_deepar_benchmark
            run_deepar_benchmark()

    summary = compile_comparison_summary()
    plot_comparison_chart(summary, BENCHMARK_DIR / "benchmark_comparison.png")

    print("\n================== BENCHMARK COMPARISON SUMMARY ==================")
    print(summary[["model", "category", "total_mae", "production_mae", "consumption_mae"]].to_string(index=False))
    print("==================================================================\n")


if __name__ == "__main__":
    main()
