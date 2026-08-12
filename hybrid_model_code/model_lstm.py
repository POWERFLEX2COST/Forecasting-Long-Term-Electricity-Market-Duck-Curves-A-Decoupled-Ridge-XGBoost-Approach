# =============================================================================
# BASELINE 4: LSTM (sequence-to-one), calendar-only inputs, non-recursive
#
# Purpose in the comparison: the LSTM/RNN is the most-cited neural-network
# baseline in the EPF literature (Lago et al. 2021, already cited in
# main.tex). To keep the comparison fair to the paper's own no-recursive-
# feedback design constraint (Sec. 3.2), this LSTM is deliberately built to
# take ONLY deterministic calendar/cyclic features as its input sequence
# (a 24-hour lookback window of hour/month/week/day-type encodings) — never
# past PRICE values. Because calendar features are known exactly for any
# future timestamp, this model can genuinely forecast 2035 in one shot with
# no feedback of its own predictions, unlike a canonical price-lag LSTM.
#
# Architecture: 1-layer LSTM (hidden=64) -> last hidden state -> MLP head.
# Trained with early stopping (train 2019-2023, validate 2024), matching the
# validation protocol used for the XGBoost residual component in the paper.
# =============================================================================

import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import data_utils as du

MODEL_NAME = "LSTM"
SEQ_LEN = 24
FEATURE_COLS = [
    "hora_sin", "hora_cos", "mes_sin", "mes_cos",
    "semana_sin", "semana_cos", "es_fin_semana",
    "hora_raw", "mes_raw", "anio_norm",
]
HIDDEN_SIZE = 64
BATCH_SIZE = 512
MAX_EPOCHS = 25
PATIENCE = 4
LR = 1e-3
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class SeqDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray = None):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = None if y is None else torch.as_tensor(y, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if self.y is None:
            return self.X[idx]
        return self.X[idx], self.y[idx]


class LSTMRegressor(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = HIDDEN_SIZE):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, num_layers=1, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def forward(self, x):
        out, (h_n, _) = self.lstm(x)
        return self.head(h_n[-1]).squeeze(-1)


def make_windows(feat_df: pd.DataFrame, target: np.ndarray, seq_len: int):
    """feat_df must include `seq_len - 1` extra rows BEFORE the first target
    timestamp (the lookback). Returns X of shape (len(target), seq_len, F)."""
    feats = feat_df[FEATURE_COLS].values.astype(np.float32)
    n_target = len(target)
    n_total = feats.shape[0]
    assert n_total == n_target + seq_len - 1, (
        f"expected {n_target + seq_len - 1} feature rows, got {n_total}"
    )
    F = feats.shape[1]
    X = np.zeros((n_target, seq_len, F), dtype=np.float32)
    for i in range(n_target):
        X[i] = feats[i: i + seq_len]
    return X


def standardize(feat_df: pd.DataFrame, mean: pd.Series, std: pd.Series) -> pd.DataFrame:
    df = feat_df.copy()
    for col in FEATURE_COLS:
        df[col] = (df[col] - mean[col]) / std[col]
    return df


def train_val_split_with_lookback(df_all: pd.DataFrame):
    """Chronological split mirroring the paper's XGBoost validation protocol:
    train 2019-2023, validate 2024 (early stopping)."""
    tr_mask = df_all["Año"] <= 2023
    va_mask = df_all["Año"] == 2024
    idx_tr_end = np.where(tr_mask)[0].max()
    idx_va = np.where(va_mask)[0]
    # Extend validation slice backwards by SEQ_LEN-1 rows for lookback context
    va_start = idx_va.min() - (SEQ_LEN - 1)
    df_tr = df_all.iloc[: idx_tr_end + 1].reset_index(drop=True)
    df_va = df_all.iloc[va_start: idx_va.max() + 1].reset_index(drop=True)
    return df_tr, df_va


def main():
    t0 = time.time()
    df = du.load_data()
    df = du.add_cyclic_features(df)

    # We need a contiguous calendar (with lookback) that also extends into
    # the projection horizon; build one unified calendar timeline covering
    # 2019-01-01 through 2035-12-31 using only deterministic time fields.
    base_cols = ["timestamp", "Año", "mes", "hora", "dia_semana", "semana_anio"]
    df_hist = df[base_cols + [c for c in FEATURE_COLS if c not in base_cols] + ["target"]]

    df_train_full, df_test = du.train_test_split(df_hist)
    df_train, df_val = train_val_split_with_lookback(df_train_full)

    # ---- Standardisation (fit on train only) --------------------------
    mean = df_train[FEATURE_COLS].mean()
    std = df_train[FEATURE_COLS].std().replace(0, 1.0)

    df_train_s = standardize(df_train, mean, std)
    df_val_s = standardize(df_val, mean, std)

    y_train = df_train_s["target"].values[SEQ_LEN - 1:]
    X_train = make_windows(df_train_s, y_train, SEQ_LEN)

    y_val = df_val_s["target"].values[SEQ_LEN - 1:]
    X_val = make_windows(df_val_s, y_val, SEQ_LEN)

    print(f"[LSTM] Train windows: {X_train.shape}, Val windows: {X_val.shape}")

    train_loader = DataLoader(SeqDataset(X_train, y_train), batch_size=BATCH_SIZE,
                               shuffle=True)
    val_loader = DataLoader(SeqDataset(X_val, y_val), batch_size=BATCH_SIZE * 2,
                             shuffle=False)

    model = LSTMRegressor(n_features=len(FEATURE_COLS)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    best_val = np.inf
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = model(xb)
                val_loss += loss_fn(pred, yb).item() * xb.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"[LSTM] Epoch {epoch:02d}  train_MSE={train_loss:.3f}  "
              f"val_MSE={val_loss:.3f}")

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"[LSTM] Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    print(f"[LSTM] Training done in {time.time() - t0:.1f}s. Best val MSE={best_val:.3f}")

    # ---- Refit standardisation/lookback on FULL 2019-2024 for final model -
    # (the early-stopped model above is retained: only re-used for inference)

    def predict_for_range(df_target: pd.DataFrame, df_lookback_source: pd.DataFrame):
        """df_target: rows to predict (has FEATURE_COLS + timestamp, Año...).
        df_lookback_source: full calendar (train+extension) to pull the
        SEQ_LEN-1 preceding calendar rows from."""
        start_ts = df_target["timestamp"].iloc[0]
        end_ts = df_target["timestamp"].iloc[-1]
        lb_start = start_ts - pd.Timedelta(hours=SEQ_LEN - 1)
        window_df = df_lookback_source[
            (df_lookback_source["timestamp"] >= lb_start) &
            (df_lookback_source["timestamp"] <= end_ts)
        ].reset_index(drop=True)
        window_df_s = standardize(window_df, mean, std)
        dummy_target = np.zeros(len(df_target))
        X = make_windows(window_df_s, dummy_target, SEQ_LEN)
        model.eval()
        with torch.no_grad():
            preds = model(torch.as_tensor(X, dtype=torch.float32).to(DEVICE))
        return preds.cpu().numpy()

    # ---- Backtest 2025 ------------------------------------------------
    print("[LSTM] Forecasting blind 2025 backtest ...")
    calendar_hist = df_hist  # 2019-2025, deterministic features only used
    df_test = df_test.copy()
    df_test["target_pred"] = predict_for_range(df_test, calendar_hist)

    metrics = du.compute_metrics(
        df_test["target"].values, df_test["target_pred"].values,
        du.load_data().merge(df_test[["timestamp"]], on="timestamp")["target_naive"].values,
    )
    print(f"[LSTM] Backtest 2025 metrics: {metrics}")
    du.save_predictions(
        df_test[["timestamp", "target", "target_pred"]],
        "backtest_2025_lstm.csv",
    )
    pd.DataFrame([metrics]).assign(model=MODEL_NAME).to_csv(
        f"{du.PRED_DIR}/metrics_backtest_lstm.csv", index=False
    )

    # ---- Long-horizon projection 2026-2035 (stability check) -----------
    print("[LSTM] Forecasting 2026-2035 projection (stability check) ...")
    df_fut_raw = du.build_future_calendar()
    df_fut_raw = du.add_cyclic_features(df_fut_raw)
    # Extend calendar lookback source to cover the tail of 2025 + full future
    calendar_full = pd.concat([
        calendar_hist[["timestamp"] + FEATURE_COLS],
        df_fut_raw[["timestamp"] + FEATURE_COLS],
    ]).reset_index(drop=True)

    df_fut_raw["target_pred"] = predict_for_range(df_fut_raw, calendar_full)
    du.save_predictions(
        df_fut_raw[["timestamp", "Año", "mes", "hora", "target_pred"]],
        "projection_2026_2035_lstm.csv",
    )
    print(f"[LSTM] Done. Total time {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
