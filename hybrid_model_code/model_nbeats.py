# =============================================================================
# BASELINE 5: N-BEATS-style deep residual stack, calendar-conditioned
# (Oreshkin et al. 2020, "N-BEATS: Neural basis expansion analysis for
# interpretable time series forecasting")
#
# IMPORTANT DEVIATION FROM THE PUBLISHED ARCHITECTURE (documented explicitly,
# to be stated in the paper if this baseline is included):
#   Canonical N-BEATS is a purely autoregressive univariate model: it looks
#   back at the last L observed PRICE values and forecasts the next H
#   directly. That design is fundamentally incompatible with a genuine
#   87,648-hour (10-year) blind projection: by year 3 of the projection there
#   is no more "recently observed price" to look back at, so a canonical
#   N-BEATS would have to be fed its own prior forecasts as if they were
#   observations — precisely the recursive-feedback failure mode the paper
#   argues against for XGBoost-with-lags (Sec. 3.2). Applying canonical
#   N-BEATS "honestly" at this horizon would therefore not be a meaningful
#   additional data point (it would simply recreate the lag-XGBoost failure
#   under a fancier name).
#   To keep the comparison informative rather than a foregone conclusion,
#   this implementation instead conditions the N-BEATS-style residual-stack
#   architecture on deterministic calendar features (identical feature set
#   to the LSTM baseline) rather than on lagged price — i.e. it keeps
#   N-BEATS's signature fully-connected residual "basis expansion" blocks,
#   but removes the autoregressive price input, exactly like the LSTM
#   baseline. This isolates the question "does this modern deep architecture
#   extract more from the calendar signal than Ridge+XGBoost?", which is the
#   comparison that is actually fair and decidable without recursion.
# =============================================================================

import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import data_utils as du

MODEL_NAME = "N-BEATS (calendar-conditioned)"
FEATURE_COLS = [
    "hora_sin", "hora_cos", "mes_sin", "mes_cos",
    "semana_sin", "semana_cos", "es_fin_semana",
    "hora_raw", "mes_raw", "anio_norm",
]
N_STACKS = 3
N_BLOCKS_PER_STACK = 2
HIDDEN = 128
BATCH_SIZE = 1024
MAX_EPOCHS = 40
PATIENCE = 5
LR = 1e-3
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class NBeatsBlock(nn.Module):
    """Fully-connected block with a residual (basis-expansion-like) output;
    here the 'backcast' reconstructs the input features and the 'forecast'
    contributes to the scalar prediction, following the generic (non
    trend/seasonality-specific) N-BEATS block formulation."""

    def __init__(self, n_features: int, hidden: int = HIDDEN):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.backcast = nn.Linear(hidden, n_features)
        self.forecast = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.fc(x)
        return self.backcast(h), self.forecast(h).squeeze(-1)


class NBeatsNet(nn.Module):
    def __init__(self, n_features: int, n_stacks: int = N_STACKS,
                 n_blocks: int = N_BLOCKS_PER_STACK):
        super().__init__()
        self.blocks = nn.ModuleList([
            NBeatsBlock(n_features) for _ in range(n_stacks * n_blocks)
        ])

    def forward(self, x):
        residual = x
        total_forecast = 0.0
        for block in self.blocks:
            backcast, forecast = block(residual)
            residual = residual - backcast
            total_forecast = total_forecast + forecast
        return total_forecast


class TabDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = None if y is None else torch.as_tensor(y, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if self.y is None:
            return self.X[idx]
        return self.X[idx], self.y[idx]


def standardize(df: pd.DataFrame, mean, std) -> pd.DataFrame:
    out = df.copy()
    for col in FEATURE_COLS:
        out[col] = (out[col] - mean[col]) / std[col]
    return out


def main():
    t0 = time.time()
    df = du.load_data()
    df = du.add_cyclic_features(df)

    df_train_full, df_test = du.train_test_split(df)
    df_train = df_train_full[df_train_full["Año"] <= 2023].reset_index(drop=True)
    df_val = df_train_full[df_train_full["Año"] == 2024].reset_index(drop=True)

    mean = df_train[FEATURE_COLS].mean()
    std = df_train[FEATURE_COLS].std().replace(0, 1.0)

    df_train_s = standardize(df_train, mean, std)
    df_val_s = standardize(df_val, mean, std)

    X_train = df_train_s[FEATURE_COLS].values.astype(np.float32)
    y_train = df_train_s["target"].values.astype(np.float32)
    X_val = df_val_s[FEATURE_COLS].values.astype(np.float32)
    y_val = df_val_s["target"].values.astype(np.float32)

    print(f"[N-BEATS] Train rows: {X_train.shape}, Val rows: {X_val.shape}")

    train_loader = DataLoader(TabDataset(X_train, y_train), batch_size=BATCH_SIZE,
                               shuffle=True)
    val_loader = DataLoader(TabDataset(X_val, y_val), batch_size=BATCH_SIZE * 2,
                             shuffle=False)

    model = NBeatsNet(n_features=len(FEATURE_COLS)).to(DEVICE)
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

        print(f"[N-BEATS] Epoch {epoch:02d}  train_MSE={train_loss:.3f}  "
              f"val_MSE={val_loss:.3f}")

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"[N-BEATS] Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    print(f"[N-BEATS] Training done in {time.time() - t0:.1f}s. Best val MSE={best_val:.3f}")

    def predict(df_raw: pd.DataFrame) -> np.ndarray:
        df_s = standardize(df_raw, mean, std)
        X = df_s[FEATURE_COLS].values.astype(np.float32)
        model.eval()
        with torch.no_grad():
            preds = model(torch.as_tensor(X, dtype=torch.float32).to(DEVICE))
        return preds.cpu().numpy()

    # ---- Backtest 2025 --------------------------------------------------
    print("[N-BEATS] Forecasting blind 2025 backtest ...")
    df_test = df_test.copy()
    df_test["target_pred"] = predict(df_test)

    metrics = du.compute_metrics(
        df_test["target"].values, df_test["target_pred"].values,
        df_test["target_naive"].values,
    )
    print(f"[N-BEATS] Backtest 2025 metrics: {metrics}")
    du.save_predictions(
        df_test[["timestamp", "target", "target_pred"]],
        "backtest_2025_nbeats.csv",
    )
    pd.DataFrame([metrics]).assign(model=MODEL_NAME).to_csv(
        f"{du.PRED_DIR}/metrics_backtest_nbeats.csv", index=False
    )

    # ---- Long-horizon projection 2026-2035 -------------------------------
    print("[N-BEATS] Forecasting 2026-2035 projection (stability check) ...")
    df_fut = du.build_future_calendar()
    df_fut = du.add_cyclic_features(df_fut)
    df_fut["target_pred"] = predict(df_fut)

    du.save_predictions(
        df_fut[["timestamp", "Año", "mes", "hora", "target_pred"]],
        "projection_2026_2035_nbeats.csv",
    )
    print(f"[N-BEATS] Done. Total time {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
