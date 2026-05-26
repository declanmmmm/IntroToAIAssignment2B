import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, GRU, Dense, Conv1D, MaxPooling1D, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error


def build_lstm(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def build_gru(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        GRU(64),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


# 3rd model - CNN extracts local patterns then LSTM handles the sequence
def build_cnn_lstm(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(64, kernel_size=3, activation="relu", padding="same"),
        MaxPooling1D(pool_size=2),
        LSTM(64),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def train_model(model, X_train, y_train, epochs=20, batch_size=32):
    early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1
    )
    return history


def evaluate_model(model, X_test, y_test, scaler):
    preds_scaled = model.predict(X_test)

    preds = scaler.inverse_transform(preds_scaled)
    actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    mae = mean_absolute_error(actual, preds)
    rmse = np.sqrt(mean_squared_error(actual, preds))

    # exclude zeros from MAPE so we don't get division by zero at quiet times
    mask = actual.flatten() != 0
    mape = np.mean(np.abs((actual[mask] - preds[mask]) / actual[mask])) * 100

    print(f"  MAE:  {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAPE: {mape:.2f}%")

    return {"mae": mae, "rmse": rmse, "mape": mape, "predictions": preds, "actual": actual}


def save_model(model, name):
    os.makedirs("saved_models", exist_ok=True)
    model.save(f"saved_models/{name}.keras")
    print(f"saved {name}")


def load_saved_model(name):
    return load_model(f"saved_models/{name}.keras")