import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from models import build_lstm, build_gru, build_cnn_lstm, train_model, evaluate_model, save_model

ds = pd.read_excel("Datasets/MainSCATSDataset.xls", sheet_name="Data", engine="xlrd", header=None)
ds.columns = ds.iloc[1]
ds = ds.drop([0, 1]).reset_index(drop=True)
ds["SCATS Number"] = ds["SCATS Number"].astype(str).str.strip()
traffic_cols = [f"V{str(i).zfill(2)}" for i in range(96)]

LOOKBACK = 8

def make_sequences(values, lookback=LOOKBACK):
    X = []
    y = []
    for i in range(len(values) - lookback):
        X.append(values[i:i + lookback])
        y.append(values[i + lookback])
    return np.array(X), np.array(y)


lstm_results = []
gru_results = []
cnn_results = []

sites = ds["SCATS Number"].unique()

for site_id in sites:
    print(f"\ntraining site {site_id}")

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
    X_train = X[:split]
    X_test = X[split:]
    y_train = y[:split]
    y_test = y[split:]

    input_shape = (X_train.shape[1], X_train.shape[2])

    #Train LSTM
    print("lstm:")
    lstm = build_lstm(input_shape)
    train_model(lstm, X_train, y_train)
    m = evaluate_model(lstm, X_test, y_test, scaler)
    m["site"] = site_id
    lstm_results.append(m)
    save_model(lstm, f"lstm_{site_id}")

    #Train GRU
    print("gru:")
    gru = build_gru(input_shape)
    train_model(gru, X_train, y_train)
    m = evaluate_model(gru, X_test, y_test, scaler)
    m["site"] = site_id
    gru_results.append(m)
    save_model(gru, f"gru_{site_id}")

    #Train CNN-LSTM
    print("cnn_lstm:")
    cnn = build_cnn_lstm(input_shape)
    train_model(cnn, X_train, y_train)
    m = evaluate_model(cnn, X_test, y_test, scaler)
    m["site"] = site_id
    cnn_results.append(m)
    save_model(cnn, f"cnn_lstm_{site_id}")


print("\n\nfinal results:")

print("\nlstm")
print(f"  mae:  {np.mean([r['mae']  for r in lstm_results]):.2f}")
print(f"  rmse: {np.mean([r['rmse'] for r in lstm_results]):.2f}")
print(f"  mape: {np.mean([r['mape'] for r in lstm_results]):.2f}%")

print("\ngru")
print(f"  mae:  {np.mean([r['mae']  for r in gru_results]):.2f}")
print(f"  rmse: {np.mean([r['rmse'] for r in gru_results]):.2f}")
print(f"  mape: {np.mean([r['mape'] for r in gru_results]):.2f}%")

print("\ncnn_lstm")
print(f"  mae:  {np.mean([r['mae']  for r in cnn_results]):.2f}")
print(f"  rmse: {np.mean([r['rmse'] for r in cnn_results]):.2f}")
print(f"  mape: {np.mean([r['mape'] for r in cnn_results]):.2f}%")