import torch
import torch.nn as nn

class SimpleLSTM(nn.Module):
    def __init__(self, input_dim=34, hidden_dim=128, num_classes=5):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        B, T, J, C = x.shape

        x = x.reshape(B, T, J * C)

        _, (h, _) = self.lstm(x)      
        h = h.squeeze(0)              

        return self.fc(h)
