import pandas as pd
import numpy as np

ds = pd.read_excel("Datasets/MainSCATSDataset.xls", sheet_name="Data")
    

#removing the first line (empty)
ds.columns = ds.iloc[0]
ds = ds.drop(0)

#isolating traffic columns (V0 - V95)

traffic_cols = ds.columns[-96:]
traffic_data = ds[traffic_cols]


#printing original data
print("\n\n")
print("This is the head of the data")
print(ds.head())
print("\n")
print("This is the columns")
print(ds.columns)
print("\n")
print("This is the shape")
print(ds.shape)

#printing traffic data (only V0 - V95)

print("\n\n")
print("Head from traffic data")
print(traffic_data.head())

#just testing the data with a singular site (0970)

ds["SCATS Number"] = ds["SCATS Number"].astype(str).str.strip()

site = ds[ds["SCATS Number"] == "0970"]

values = site[traffic_cols].astype(float).values.flatten()

print(values[:50])
print(len(values))

#Create sequences using 8 previous values (2 hours)

X = []
y = []

for i in range(len(values) - 8):
    X.append(values[i:i+8])
    y.append(values[i+8])

#Convert to numpy arrays
X = np.array(X)
y = np.array(y)

#Check shapes
print("X shape before reshape:", X.shape)
print("y shape:", y.shape)

#Reshape for LSTM: (samples, timesteps, features)
X = X.reshape((X.shape[0], X.shape[1], 1))

print("X shape after reshape:", X.shape)


#Spliting data 80/20

split = int(len(X) * 0.8)

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]

print("Training samples:", len(X_train))# <- Number of samples used in training
print("Testing samples:", len(X_test))# <- Number of samples used in testing