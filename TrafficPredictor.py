import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

ds = pd.read_excel("Datasets/MainSCATSDataset.xls", sheet_name="Data", engine="xlrd", header=None)

# row 1 is the actual header
ds.columns = ds.iloc[1]
ds = ds.drop([0, 1]).reset_index(drop=True)

ds["SCATS Number"] = ds["SCATS Number"].astype(str).str.strip()

traffic_cols = [f"V{str(i).zfill(2)}" for i in range(96)]

print("shape:", ds.shape)
print("sites:", ds["SCATS Number"].nunique())

LOOKBACK = 8

def make_sequences(values, lookback=LOOKBACK):
    X, y = [], []
    for i in range(len(values) - lookback):
        X.append(values[i:i + lookback])
        y.append(values[i + lookback])
    return np.array(X), np.array(y)


site_data = {}

for site_id, group in ds.groupby("SCATS Number"):

    raw = group[traffic_cols].astype(float).values.flatten()

    series = pd.Series(raw)
    series = series.ffill().bfill()

    if series.isna().all():
        print(f"skipping {site_id} - no data")
        continue

    values = series.values

    scaler = MinMaxScaler(feature_range=(0, 1))
    values_scaled = scaler.fit_transform(values.reshape(-1, 1)).flatten()

    X, y = make_sequences(values_scaled)

    if len(X) < 20:
        print(f"skipping {site_id} - not enough data")
        continue

    X = X.reshape((X.shape[0], X.shape[1], 1))

    split = int(len(X) * 0.8)

    site_data[site_id] = {
        "scaler":  scaler,
        "X_train": X[:split],
        "X_test":  X[split:],
        "y_train": y[:split],
        "y_test":  y[split:],
    }

print(f"processed {len(site_data)} sites")

# quick check
d = site_data["0970"]
print("X_train shape:", d["X_train"].shape)
print("X_test shape:", d["X_test"].shape)