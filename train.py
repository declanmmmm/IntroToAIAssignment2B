import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from models import build_lstm, build_gru, build_cnn_lstm, train_model, evaluate_model, save_model

# load and prep data (same as TrafficPredictor.py)
ds = pd.read_excel("Datasets/MainSCATSDataset.xls", sheet_name="Data", engine="xlrd", header=None)
ds.columns = ds.iloc[1]
ds = ds.drop([0, 1]).reset_index(drop=True)
ds["SCATS Number"] = ds["SCATS Number"].astype(str).str.strip()
traffic_cols = [f"V{str(i).zfill(2)}" for i in range(96)]

LOOKBACK = 8

def make_sequences(values, lookback=LOOKBACK):
    X, y = [], []
    for i in range(len(values) - lookback):
        X.append(values[i:i + lookback])
        y.append(values[i + lookback])
    return np.array(X), np.array(y)


# store results for comparison at the end
results = {"lstm": [], "gru": [], "cnn_lstm": []}

sites = ds["SCATS Number"].unique()

for site_id in sites:
    print(f"\n=== site {site_id} ===")

    group = ds[ds["SCATS Number"] == site_id]
    raw = group[traffic_cols].astype(float).values.flatten()
    series = pd.Series(raw).ffill().bfill()

    if series.isna().all():
        continue

    values = series.values
    scaler = MinMaxScaler(feature_range=(0, 1))
    values_scaled = scaler.fit_transform(values.reshape(-1, 1)).flatten()

    X, y = make_sequences(values_scaled)

    if len(X) < 20:
        continue

    X = X.reshape((X.shape[0], X.shape[1], 1))
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    input_shape = (X_train.shape[1], X_train.shape[2])

    for name, model in [
        ("lstm", build_lstm(input_shape)),
        ("gru", build_gru(input_shape)),
        ("cnn_lstm", build_cnn_lstm(input_shape)),
    ]:
        print(f"\n{name.upper()}:")
        train_model(model, X_train, y_train, epochs=20, batch_size=32)
        metrics = evaluate_model(model, X_test, y_test, scaler)
        metrics["site"] = site_id
        results[name].append(metrics)
        save_model(model, f"{name}_{site_id}")


# print average results across all sites
print("\n\n=== average results across all sites ===")
for name, site_results in results.items():
    if len(site_results) == 0:
        continue
    avg_mae  = np.mean([r["mae"] for r in site_results])
    avg_rmse = np.mean([r["rmse"] for r in site_results])
    avg_mape = np.mean([r["mape"] for r in site_results])
    print(f"\n{name.upper()}")
    print(f"  avg MAE:  {avg_mae:.2f}")
    print(f"  avg RMSE: {avg_rmse:.2f}")
    print(f"  avg MAPE: {avg_mape:.2f}%")