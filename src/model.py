import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

class SimpleLSTM(nn.Module):
    """Modelo baseline original"""
    def __init__(self, input_dim=34, hidden_dim=128, num_classes=5):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, lengths=None):
        B, T, J, C = x.shape
        x = x.reshape(B, T, J * C)
        _, (h, _) = self.lstm(x)      
        h = h.squeeze(0)              
        return self.fc(h)


class ImprovedLSTM(nn.Module):
    """LSTM mejorado con pack_padded_sequence, dropout y más capas"""
    def __init__(self, input_dim=34, hidden_dim=256, num_layers=2, num_classes=5, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, lengths=None):
        B, T, J, C = x.shape
        x = x.reshape(B, T, J * C)

        if lengths is not None:
            # Ordenar por longitud 
            lengths_sorted, sort_idx = lengths.sort(descending=True)
            x_sorted = x[sort_idx]
            
            # Pack para ignorar padding
            packed = pack_padded_sequence(x_sorted, lengths_sorted.cpu(), batch_first=True, enforce_sorted=True)
            packed_out, (h, _) = self.lstm(packed)
            
            # Restaurar orden original
            _, unsort_idx = sort_idx.sort()
            h = h[-1][unsort_idx]  # Última capa
        else:
            _, (h, _) = self.lstm(x)
            h = h[-1]  # Última capa

        h = self.dropout(h)
        return self.fc(h)

class BaselineFC(nn.Module):
    """Baseline simple: promedio temporal + red densa"""
    def __init__(self, input_dim=34, hidden_dim=128, num_classes=5):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x, lengths=None):
        B, T, J, C = x.shape
        x = x.reshape(B, T, J * C)
        x = x.mean(dim=1)
        return self.fc(x)