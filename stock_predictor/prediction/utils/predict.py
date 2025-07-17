import os
import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from datetime import timedelta
import time

# Load the model class
class LSTMModel(nn.Module):
    def __init__(self, input_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, 50, batch_first=True)
        self.fc = nn.Linear(50, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# Retry wrapper for yfinance
def fetch_data_with_retry(ticker, start, end, retries=3, delay=5):
    for i in range(retries):
        try:
            data = yf.download(ticker, start=start, end=end, progress=False, threads=False)
            if not data.empty:
                return data
            else:
                raise ValueError("Empty data received.")
        except Exception as e:
            print(f"[Retry {i+1}/{retries}] Failed to download data: {e}")
            time.sleep(delay)
    raise ValueError("Failed to download stock data after several attempts.")

# Main prediction function
def run_prediction(ticker, start_date, end_date, future_days=3):
    # Load data with retry
    data_df = fetch_data_with_retry(ticker, start_date, end_date)
    data = data_df['Close'].values.reshape(-1, 1)

    # Normalize
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    def create_dataset(dataset, time_step=60):
        X, y = [], []
        for i in range(len(dataset) - time_step - 1):
            X.append(dataset[i:(i + time_step), 0])
            y.append(dataset[i + time_step, 0])
        return np.array(X), np.array(y)

    time_step = 60
    X, y = create_dataset(scaled_data, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Split
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Load model and weights
    model = LSTMModel()
    model.load_state_dict(torch.load('models\lstm_trained_weights.pth'))
    model.eval()

    # Predict
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test).view(-1, 1)
    y_pred_tensor = model(X_test_tensor)

    y_pred = scaler.inverse_transform(y_pred_tensor.detach().numpy())
    y_test_original = scaler.inverse_transform(y_test_tensor.detach().numpy())

    # ====== 📈 Predict future ======
    last_seq = scaled_data[-time_step:]
    future_preds = []

    input_seq = torch.FloatTensor(last_seq.reshape(1, time_step, 1))

    for _ in range(future_days):
        with torch.no_grad():
            next_val = model(input_seq)
        future_preds.append(next_val.item())
        next_scaled = np.append(last_seq[1:], [[next_val.item()]], axis=0)
        input_seq = torch.FloatTensor(next_scaled.reshape(1, time_step, 1))
        last_seq = next_scaled

    future_preds_unscaled = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))

    # ====== 📊 Create plot ======
    plt.figure(figsize=(14, 7))
    plt.plot(data, label='Actual Prices', color='blue')
    plt.plot(np.arange(len(data) - len(y_test_original), len(data)), y_test_original, label='Test Actual Prices', color='orange')
    plt.plot(np.arange(len(data) - len(y_test_original), len(data)), y_pred, label='Predicted Prices', color='green')

    future_range = np.arange(len(data), len(data) + future_days)
    plt.plot(future_range, future_preds_unscaled, label='Future Predictions', color='red', linestyle='--')

    plt.title(f'{ticker} Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.tight_layout()

    os.makedirs("media", exist_ok=True)
    image_path = f"media/graph.png"
    plt.savefig(image_path)
    plt.close()

    # Table of actual vs predicted
    df_results = pd.DataFrame({
        'Actual': y_test_original.flatten(),
        'Predicted': y_pred.flatten()
    })

    # Future predictions table
    future_dates = pd.date_range(end=pd.to_datetime(end_date), periods=future_days+1, freq='B')[1:]
    df_future = pd.DataFrame({
        'Date': future_dates,
        'Predicted Price': future_preds_unscaled.flatten()
    })

    return df_future, df_results, image_path
