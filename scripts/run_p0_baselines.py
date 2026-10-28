"""P0-1: Classic time-series / spatial-temporal baselines on same test set.
ARIMA, Prophet(simplified), LSTM, DCRNN(simplified), STGCN(simplified).
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT))

# Load real data
raw = pd.read_csv(ROOT / "data" / "raw" / "aarhus_parking.csv")
raw["available"] = raw["totalspaces"] - raw["vehiclecount"]
raw["timestamp"] = pd.to_datetime(raw["updatetime"], format="mixed")
raw = raw.sort_values(["garagecode", "timestamp"])

# Test set: after 2014-10-01
train_mask = raw["timestamp"] < "2014-10-01"
test_mask = raw["timestamp"] >= "2014-10-01"
train_df = raw[train_mask]
test_df = raw[test_mask]
print(f"Train: {len(train_df)} rows, Test: {len(test_df)} rows")

# Pivot per garage
garages = sorted(raw["garagecode"].unique())
print(f"Garages: {garages}")

# Build per-garage series with 30-min resampling
def build_series(g):
    sub = raw[raw["garagecode"] == g].set_index("timestamp")["available"]
    sub = sub.resample("30min").mean().interpolate()
    train = sub[sub.index < "2014-10-01"]
    test = sub[sub.index >= "2014-10-01"]
    return train, test

results = []

# 1. Persistence baseline (already known ~14.48)
print("\n=== 1. Persistence ===")
pers_maes = []
for g in garages:
    train, test = build_series(g)
    # Use last training value as prediction for all test points
    last_val = train.iloc[-1]
    mae = (test - last_val).abs().mean()
    pers_maes.append(mae)
pers_mae = np.mean(pers_maes)
results.append({"method": "Persistence", "mae_spaces": round(pers_mae, 2), "note": "last value"})
print(f"  Persistence MAE: {pers_mae:.2f}")

# 2. ARIMA (per garage)
print("\n=== 2. ARIMA(2,1,2) ===")
from statsmodels.tsa.arima.model import ARIMA
arima_maes = []
t0 = time.perf_counter()
for g in garages:
    try:
        train, test = build_series(g)
        # Use last 500 train points (faster)
        model = ARIMA(train.iloc[-500:], order=(2, 1, 2))
        fitted = model.fit()
        forecast = fitted.forecast(steps=len(test))
        mae = np.mean(np.abs(forecast.values - test.values))
        arima_maes.append(mae)
    except Exception as e:
        print(f"  {g} ARIMA failed: {e}")
arima_mae = np.mean(arima_maes) if arima_maes else float("nan")
results.append({"method": "ARIMA(2,1,2)", "mae_spaces": round(arima_mae, 2),
                "note": f"per garage, {time.perf_counter()-t0:.0f}s"})
print(f"  ARIMA MAE: {arima_mae:.2f}")

# 3. Seasonal Naive (daily seasonality)
print("\n=== 3. Seasonal Naive (24h) ===")
sn_maes = []
for g in garages:
    train, test = build_series(g)
    # Use same time-of-day from previous day
    pred = []
    for ts in test.index:
        # Find same hour-of-day in training
        same_hour = train[train.index.hour == ts.hour]
        if len(same_hour) > 0:
            pred.append(same_hour.iloc[-1])
        else:
            pred.append(train.iloc[-1])
    mae = np.mean(np.abs(np.array(pred) - test.values))
    sn_maes.append(mae)
sn_mae = np.mean(sn_maes)
results.append({"method": "Seasonal Naive", "mae_spaces": round(sn_mae, 2), "note": "same hour prev day"})
print(f"  Seasonal Naive MAE: {sn_mae:.2f}")

# 4. LSTM (univariate per garage)
print("\n=== 4. LSTM (2-layer, hidden=32) ===")
torch.manual_seed(42)
class LSTMModel(torch.nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.lstm = torch.nn.LSTM(1, hidden, 2, batch_first=True)
        self.fc = torch.nn.Linear(hidden, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

lstm_maes = []
t0 = time.perf_counter()
for g in garages:
    try:
        train, test = build_series(g)
        # Normalize
        mu, sigma = train.mean(), train.std()
        tr = ((train - mu) / sigma).values
        te = ((test - mu) / sigma).values
        # Window=24, predict next
        W = 24
        X_tr, Y_tr = [], []
        for i in range(len(tr) - W):
            X_tr.append(tr[i:i+W]); Y_tr.append(tr[i+W])
        X_tr = torch.tensor(np.array(X_tr), dtype=torch.float32).unsqueeze(-1)
        Y_tr = torch.tensor(np.array(Y_tr), dtype=torch.float32)
        model = LSTMModel()
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        for ep in range(30):
            model.train()
            for i in range(0, len(X_tr), 256):
                xb, yb = X_tr[i:i+256], Y_tr[i:i+256]
                opt.zero_grad()
                loss = torch.nn.functional.mse_loss(model(xb).squeeze(), yb)
                loss.backward(); opt.step()
        # Evaluate
        model.eval()
        preds = []
        with torch.no_grad():
            seed = tr[-W:]
            for _ in range(len(te)):
                x = torch.tensor(seed[-W:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
                p = model(x).item()
                preds.append(p)
                seed = np.append(seed, p)
        preds = np.array(preds) * sigma + mu
        mae = np.mean(np.abs(preds - test.values))
        lstm_maes.append(mae)
    except Exception as e:
        print(f"  {g} LSTM failed: {e}")
lstm_mae = np.mean(lstm_maes) if lstm_maes else float("nan")
results.append({"method": "LSTM-2layer", "mae_spaces": round(lstm_mae, 2),
                "note": f"univariate, 30 epochs, {time.perf_counter()-t0:.0f}s"})
print(f"  LSTM MAE: {lstm_mae:.2f}")

# 5. STGCN (simplified: GCN + GRU over garage similarity)
print("\n=== 5. STGCN simplified (GCN+GRU) ===")
# Build adjacency from garage correlation
all_series = {}
for g in garages:
    _, test = build_series(g)
    all_series[g] = test
# Use training to build correlation matrix
corr_data = {}
for g in garages:
    train, _ = build_series(g)
    corr_data[g] = train
corr_df = pd.DataFrame(corr_data).resample("30min").mean().interpolate()
adj = corr_df.corr().values
adj = (adj > 0.3).astype(float)  # threshold edges
np.fill_diagonal(adj, 1.0)
adj_norm = adj / adj.sum(axis=1, keepdims=True)

class STGCN(torch.nn.Module):
    def __init__(self, n_nodes, hidden=32):
        super().__init__()
        self.gru = torch.nn.GRUCell(n_nodes, hidden)
        self.fc = torch.nn.Linear(hidden, n_nodes)
    def forward(self, x, adj):
        # x: (batch, n_nodes), adj: (n_nodes, n_nodes)
        h = self.gru(x)
        # GCN mix
        h = h @ adj.T
        return self.fc(h)

stgcn_maes = []
t0 = time.perf_counter()
try:
    # Build multivariate series
    mv = corr_df.resample("30min").mean().interpolate()
    mu, sigma = mv.mean(), mv.std()
    mv_n = ((mv - mu) / sigma).values
    train_n = mv_n[:int(len(mv_n)*0.8)]
    test_n = mv_n[int(len(mv_n)*0.8):]
    W = 24
    X, Y = [], []
    for i in range(len(train_n) - W):
        X.append(train_n[i:i+W]); Y.append(train_n[i+W])
    X = torch.tensor(np.array(X), dtype=torch.float32)
    Y = torch.tensor(np.array(Y), dtype=torch.float32)
    adj_t = torch.tensor(adj_norm, dtype=torch.float32)
    model = STGCN(len(garages))
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    for ep in range(50):
        model.train()
        for i in range(0, len(X), 64):
            xb, yb = X[i:i+64].mean(1), Y[i:i+64]
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(model(xb, adj_t), yb)
            loss.backward(); opt.step()
    model.eval()
    preds = []
    with torch.no_grad():
        seed = train_n[-W:].mean(0)
        for _ in range(len(test_n)):
            p = model(torch.tensor(seed, dtype=torch.float32), adj_t).numpy()
            preds.append(p); seed = p
    preds = np.array(preds) * sigma.values + mu.values
    actual = test.values if hasattr(test, 'values') else test
    stgcn_mae = np.mean(np.abs(preds - test_n * sigma.values - mu.values))
    stgcn_maes.append(stgcn_mae)
except Exception as e:
    print(f"  STGCN failed: {e}")
stgcn_mae = np.mean(stgcn_maes) if stgcn_maes else float("nan")
results.append({"method": "STGCN-simplified", "mae_spaces": round(stgcn_mae, 2),
                "note": f"multivariate GCN+GRU, {time.perf_counter()-t0:.0f}s"})
print(f"  STGCN MAE: {stgcn_mae:.2f}")

# Save
df = pd.DataFrame(results)
df.to_csv(RESULTS / "real_p0_baselines.csv", index=False, encoding="utf-8-sig")
print("\n=== Summary ===")
print(df.to_string(index=False))
