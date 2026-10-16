"""DeepAR benchmark for Enefit energy forecasting using shared features.

Implements DeepAR (Salinas et al., 2019):
- A global LSTM network trained simultaneously across all 69 segments.
- Conditioned on static entity embeddings (county, product_type, business, side).
- Conditioned on dynamic shared features:
  - Cyclical time: hour, day of week
  - Weather forecasts: temperature, direct_solar_radiation
  - Dynamic lags: lag_2_day, lag_7_day, lag_7_day_mean
  - Scale: installed_capacity
- Learns series-specific scale factors (nu_i) and outputs parametric Gaussian predictive distributions.
"""
from __future__ import annotations

import argparse
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from . import base
from .. import config as C

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class DeepAR(nn.Module):
    def __init__(
        self,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Static entity embeddings
        self.county_emb = nn.Embedding(16, 8)
        self.prod_emb = nn.Embedding(4, 4)
        self.biz_emb = nn.Embedding(2, 2)
        self.side_emb = nn.Embedding(2, 2)
        static_dim = 8 + 4 + 2 + 2  # 16

        # Dynamic features: sin/cos hour (2), sin/cos dow (2), temp (1), solar (1),
        # lag_2_day (1), lag_7_day (1), lag_7_day_mean (1), installed_capacity (1)
        dynamic_dim = 10
        target_dim = 1  # scaled lagged target y_{t-1}

        input_dim = static_dim + dynamic_dim + target_dim

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.mu_head = nn.Linear(hidden_size, 1)
        self.sigma_head = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Softplus(),
        )

    def forward(
        self,
        static_ids: torch.Tensor,     # (B, 4): county, product, is_business, is_consumption
        dyn_feats: torch.Tensor,      # (B, T, 10)
        lag_targets: torch.Tensor,    # (B, T, 1)
        h_0: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        B, T, _ = dyn_feats.shape

        c_emb = self.county_emb(static_ids[:, 0])
        p_emb = self.prod_emb(static_ids[:, 1])
        b_emb = self.biz_emb(static_ids[:, 2])
        s_emb = self.side_emb(static_ids[:, 3])
        stat = torch.cat([c_emb, p_emb, b_emb, s_emb], dim=-1).unsqueeze(1).expand(-1, T, -1)

        x = torch.cat([stat, dyn_feats, lag_targets], dim=-1)
        out, (h_n, c_n) = self.lstm(x, h_0)

        mu = self.mu_head(out)
        sigma = self.sigma_head(out) + 1e-4
        return mu, sigma, (h_n, c_n)


def extract_shared_dynamic_features(df: pd.DataFrame, scale: float) -> np.ndarray:
    """Extract and normalize all shared dynamic features for DeepAR."""
    hour = df["hour"].to_numpy(dtype=float)
    dow = df["datetime"].dt.dayofweek.to_numpy(dtype=float)
    temp = df["temperature"].fillna(0).to_numpy(dtype=float) / 20.0
    rad = df["direct_solar_radiation"].fillna(0).to_numpy(dtype=float) / 500.0

    safe_scale = max(scale, 1.0)
    lag2 = df["lag_2_day"].fillna(0).to_numpy(dtype=float) / safe_scale
    lag7 = df["lag_7_day"].fillna(0).to_numpy(dtype=float) / safe_scale
    lag7_mean = df["lag_7_day_mean"].fillna(0).to_numpy(dtype=float) / safe_scale
    cap = df["installed_capacity"].fillna(0).to_numpy(dtype=float) / safe_scale

    return np.column_stack([
        np.sin(2 * np.pi * hour / 24.0),
        np.cos(2 * np.pi * hour / 24.0),
        np.sin(2 * np.pi * dow / 7.0),
        np.cos(2 * np.pi * dow / 7.0),
        temp,
        rad,
        lag2,
        lag7,
        lag7_mean,
        cap,
    ])


class TimeSeriesWindowDataset(Dataset):
    """Dataset sampling lookback + forecast windows across all segments."""
    def __init__(
        self,
        segment_data: list[dict],
        seq_len: int = 168 + 24,
        num_samples_per_series: int = 40,
    ):
        self.samples = []
        for s in segment_data:
            total_len = len(s["y"])
            if total_len <= seq_len:
                continue
            for _ in range(num_samples_per_series):
                start = np.random.randint(0, total_len - seq_len)
                self.samples.append((s, start, start + seq_len))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        s, start, end = self.samples[idx]
        scale = s["scale"]

        y = s["y"][start:end] / scale
        dyn = s["dyn"][start:end]
        stat = s["stat"]

        lag_y = np.roll(y, 1)
        lag_y[0] = 0.0

        return {
            "stat": torch.tensor(stat, dtype=torch.long),
            "dyn": torch.tensor(dyn, dtype=torch.float32),
            "lag_y": torch.tensor(lag_y[:, None], dtype=torch.float32),
            "target": torch.tensor(y[:, None], dtype=torch.float32),
            "scale": torch.tensor(scale, dtype=torch.float32),
        }


def run_deepar_benchmark(
    epochs: int = 10,
    batch_size: int = 64,
    hidden_size: int = 64,
    lr: float = 1e-3,
    train_history_start: str = "2022-09-01",
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Train DeepAR model on shared features and evaluate on test set."""
    train_df = base.load_train_data(start=train_history_start, columns=base.SHARED_FEATURE_COLS)
    test_df = base.load_test_data(columns=base.SHARED_FEATURE_COLS)

    segment_data = []
    seg_lookup = {}
    for key, g in train_df.groupby(C.SEGMENT_KEYS, observed=True):
        g = g.sort_values("datetime")
        y = g["target"].fillna(0).to_numpy(dtype=float)
        scale = float(np.mean(np.abs(y)) + 1.0)
        dyn = extract_shared_dynamic_features(g, scale)
        stat = np.array([key[0], key[2], key[1], key[3]], dtype=np.int64)

        s_entry = {
            "key": key,
            "stat": stat,
            "dyn": dyn,
            "y": y,
            "scale": scale,
        }
        segment_data.append(s_entry)
        seg_lookup[key] = s_entry

    print(f"[deepar] Prepared {len(segment_data)} series with shared dynamic features. Training on {DEVICE}...")
    dataset = TimeSeriesWindowDataset(segment_data, seq_len=168 + 24, num_samples_per_series=40)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = DeepAR(hidden_size=hidden_size, num_layers=2, dropout=0.1).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    t0 = time.time()
    model.train()
    for ep in range(1, epochs + 1):
        total_loss = 0.0
        batches = 0
        for batch in dataloader:
            stat = batch["stat"].to(DEVICE)
            dyn = batch["dyn"].to(DEVICE)
            lag_y = batch["lag_y"].to(DEVICE)
            target = batch["target"].to(DEVICE)

            optimizer.zero_grad()
            mu, sigma, _ = model(stat, dyn, lag_y)

            var = sigma ** 2
            loss = 0.5 * torch.log(2 * np.pi * var) + ((target - mu) ** 2) / (2 * var)
            loss = loss.mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            batches += 1

        if ep % 3 == 0 or ep == epochs:
            avg_loss = total_loss / max(1, batches)
            print(f"  [deepar] Epoch {ep}/{epochs}: loss={avg_loss:.4f} ({time.time() - t0:.1f}s)")

    # ----------------- Fast Vectorized Inference on Test Set -----------------
    print("[deepar] Forecasting test set with shared features...")
    model.eval()

    test_preds_map = {}
    test_segments = []
    for key, g in test_df.groupby(C.SEGMENT_KEYS, observed=True):
        if key in seg_lookup:
            test_segments.append((key, g.sort_values("datetime")))

    num_segs = len(test_segments)
    ctx_len = 168

    all_stats = np.zeros((num_segs, 4), dtype=np.int64)
    all_scales = np.zeros((num_segs,), dtype=float)
    all_ctx_dyn = np.zeros((num_segs, ctx_len, 10), dtype=np.float32)
    all_ctx_lag = np.zeros((num_segs, ctx_len, 1), dtype=np.float32)
    last_ys = np.zeros((num_segs, 1, 1), dtype=np.float32)

    all_test_dyns = []
    test_lens = []

    for i, (key, g) in enumerate(test_segments):
        s_entry = seg_lookup[key]
        scale = s_entry["scale"]
        all_stats[i] = s_entry["stat"]
        all_scales[i] = scale

        ctx_y = s_entry["y"][-ctx_len:] / scale
        actual_len = len(ctx_y)
        all_ctx_dyn[i, -actual_len:] = s_entry["dyn"][-actual_len:]
        ctx_lag = np.roll(ctx_y, 1)
        ctx_lag[0] = 0.0
        all_ctx_lag[i, -actual_len:, 0] = ctx_lag
        last_ys[i, 0, 0] = ctx_y[-1]

        t_dyn = extract_shared_dynamic_features(g, scale)
        all_test_dyns.append(t_dyn)
        test_lens.append(len(t_dyn))

    max_test_len = max(test_lens)
    all_test_dyn_arr = np.zeros((num_segs, max_test_len, 10), dtype=np.float32)
    for i, t_dyn in enumerate(all_test_dyns):
        all_test_dyn_arr[i, :len(t_dyn)] = t_dyn

    with torch.no_grad():
        stat_t = torch.tensor(all_stats, dtype=torch.long, device=DEVICE)
        ctx_dyn_t = torch.tensor(all_ctx_dyn, dtype=torch.float32, device=DEVICE)
        ctx_lag_t = torch.tensor(all_ctx_lag, dtype=torch.float32, device=DEVICE)

        _, _, (h_n, c_n) = model(stat_t, ctx_dyn_t, ctx_lag_t)

        test_dyn_t = torch.tensor(all_test_dyn_arr, dtype=torch.float32, device=DEVICE)
        curr_y = torch.tensor(last_ys, dtype=torch.float32, device=DEVICE)

        h, c = h_n, c_n
        step_outputs = []
        for step in range(max_test_len):
            step_dyn = test_dyn_t[:, step:step+1, :]
            mu, sigma, (h, c) = model(stat_t, step_dyn, curr_y, (h, c))
            pred_clamped = torch.clamp(mu, min=0.0)
            step_outputs.append(pred_clamped[:, 0, 0].cpu().numpy())
            curr_y = pred_clamped

    step_outputs = np.array(step_outputs).T

    for i, (key, g) in enumerate(test_segments):
        n_steps = test_lens[i]
        scaled_preds = step_outputs[i, :n_steps] * all_scales[i]

        if key[3] == 0:
            solar_rad = g["direct_solar_radiation"].fillna(0).to_numpy()
            hours = g["hour"].to_numpy()
            night = (solar_rad == 0) & np.isin(hours, [20, 21, 22, 23, 0, 1, 2, 3, 4, 5])
            scaled_preds[night] = 0.0

        for row_id, p in zip(g["row_id"], scaled_preds):
            test_preds_map[row_id] = p

    pred_df = pd.DataFrame({
        "row_id": test_df["row_id"].to_numpy(dtype="int64"),
        "pred": test_df["row_id"].map(test_preds_map).fillna(0.0).to_numpy(dtype=float),
    })

    metrics = base.evaluate_predictions(pred_df, "deepar")
    return pred_df, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DeepAR benchmark on shared features.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=64)
    args = parser.parse_args()

    pred_df, metrics = run_deepar_benchmark(
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_size=args.hidden_size,
    )
    print("\n[deepar] Results on Shared Feature Set:")
    for k, v in metrics.items():
        if k.endswith("/mae") or k.endswith("/wape"):
            print(f"  {k}: {v:.2f}")


if __name__ == "__main__":
    main()
