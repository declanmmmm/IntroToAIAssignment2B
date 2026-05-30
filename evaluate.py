import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from models import load_saved_model, evaluate_model

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

os.makedirs("plots", exist_ok=True)

lstm_results = []
gru_results = []
cnn_results = []

for site_id in ds["SCATS Number"].unique():
    print(f"\nsite {site_id}")

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
    X_test = X[split:]
    y_test = y[split:]
    actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    preds = {}

    for name in ["lstm", "gru", "cnn_lstm"]:
        path = f"saved_models/{name}_{site_id}.keras"
        if not os.path.exists(path):
            print(f"  {name} model not found")
            continue
        model = load_saved_model(f"{name}_{site_id}")
        print(f"  {name}:")
        m = evaluate_model(model, X_test, y_test, scaler)
        m["site"] = site_id
        preds[name] = m["predictions"].flatten()
        if name == "lstm":
            lstm_results.append(m)
        elif name == "gru":
            gru_results.append(m)
        else:
            cnn_results.append(m)

    # plot this site
    if preds:
        plt.figure(figsize=(12, 4))
        plt.plot(actual, label="actual", color="black", linewidth=1)
        for name, p in preds.items():
            colors = {"lstm": "blue", "gru": "orange", "cnn_lstm": "green"}
            plt.plot(p, label=name, color=colors[name], alpha=0.7, linewidth=1)
        plt.title(f"site {site_id}")
        plt.xlabel("time interval")
        plt.ylabel("vehicles")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"plots/site_{site_id}.png")
        plt.close()


# comparison bar chart
print("\nresults summary:")
for name, res in [("lstm", lstm_results), ("gru", gru_results), ("cnn_lstm", cnn_results)]:
    if not res:
        continue
    print(f"\n{name}")
    print(f"  mae:  {np.mean([r['mae']  for r in res]):.2f}")
    print(f"  rmse: {np.mean([r['rmse'] for r in res]):.2f}")
    print(f"  mape: {np.mean([r['mape'] for r in res]):.2f}%")

names = []
maes = []
rmses = []
for name, res in [("lstm", lstm_results), ("gru", gru_results), ("cnn_lstm", cnn_results)]:
    if res:
        names.append(name)
        maes.append(np.mean([r["mae"] for r in res]))
        rmses.append(np.mean([r["rmse"] for r in res]))

x = np.arange(len(names))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - width/2, maes, width, label="MAE", color="steelblue")
ax.bar(x + width/2, rmses, width, label="RMSE", color="coral")
ax.set_xticks(x)
ax.set_xticklabels(names)
ax.set_ylabel("error (vehicles)")
ax.set_title("model comparison")
ax.legend()
plt.tight_layout()
plt.savefig("plots/model_comparison.png")
plt.close()
print("\nsaved plots/model_comparison.png")