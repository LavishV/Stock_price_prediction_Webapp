import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, 50, batch_first=True)
        self.fc = nn.Linear(50, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out