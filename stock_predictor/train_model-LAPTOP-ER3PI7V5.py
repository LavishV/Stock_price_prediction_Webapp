import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Now you can safely import your modules
from prediction.models.lstm_model import LSTMModel
from prediction.utils.predict import create_dataset
# train_model.py

import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from prediction.models.lstm_model import LSTMModel
from prediction.utils.predict import create_dataset

# Step 1: Download stock data
ticker = 'AAPL'
print(f"📥 Downloading 10 years of data for {ticker}...")
data = yf.download(ticker, start='2015-05-16', end='2025-05-16')
data = data['Close'].values.reshape(-1, 1)

# Step 2: Normalize the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Step 3: Create LSTM sequences
time_step = 60
X, y = create_dataset(scaled_data, time_step)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Step 4: Split into training data
train_size = int(len(X) * 0.8)
X_train, y_train = X[:train_size], y[:train_size]

# Step 5: Initialize model
model = LSTMModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)

# Step 6: Train the model
print("🧠 Training the model...")
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}")

# Step 7: Save model weights
torch.save(model.state_dict(), 'prediction/models/lstm_trained_weights.pth')
print("✅ Model training complete and saved at 'prediction/models/lstm_trained_weights.pth'")

