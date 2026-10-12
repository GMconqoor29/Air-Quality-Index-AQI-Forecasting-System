import torch
import torch.nn as nn
from src import config

class LSTMAQIModel(nn.Module):
    def __init__(self, input_size=len(config.FEATURE_COLS), hidden_size=config.HIDDEN_SIZE, dropout=config.DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        out, (h_n, _) = self.lstm(x)
        return self.fc(self.dropout(h_n[-1]))

class StackedLSTMAQIModel(nn.Module):
    def __init__(self, input_size=len(config.FEATURE_COLS), hidden_size=config.HIDDEN_SIZE, num_layers=2, dropout=config.DROPOUT):
        super().__init__()
        # Note: dropout in nn.LSTM is only applied between layers, so if num_layers=1, it does nothing.
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        out, (h_n, _) = self.lstm(x)
        # h_n shape: (num_layers, batch, hidden_size). We take the last layer's hidden state.
        return self.fc(self.dropout(h_n[-1]))

class GRUAQIModel(nn.Module):
    def __init__(self, input_size=len(config.FEATURE_COLS), hidden_size=config.HIDDEN_SIZE, dropout=config.DROPOUT):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        out, h_n = self.gru(x)
        # For GRU, h_n is of shape (num_layers, batch, hidden_size)
        return self.fc(self.dropout(h_n[-1]))


class CNNRecurrentAttentionAQIModel(nn.Module):
    def __init__(
        self,
        recurrent_type="GRU",
        input_size=len(config.FEATURE_COLS),
        conv_channels=64,
        hidden_size=config.HIDDEN_SIZE,
        dropout=config.DROPOUT,
    ):
        super().__init__()
        recurrent_type = recurrent_type.upper()
        if recurrent_type not in {"GRU", "LSTM"}:
            raise ValueError("recurrent_type must be 'GRU' or 'LSTM'.")
        if conv_channels <= 0 or hidden_size <= 0:
            raise ValueError("conv_channels and hidden_size must be positive.")

        self.recurrent_type = recurrent_type
        self.conv = nn.Sequential(
            nn.Conv1d(input_size, conv_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        recurrent = nn.GRU if recurrent_type == "GRU" else nn.LSTM
        self.recurrent = recurrent(
            input_size=conv_channels,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2 or 1),
            nn.Tanh(),
            nn.Linear(hidden_size // 2 or 1, 1, bias=False),
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, return_attention=False):
        convolved = self.conv(x.transpose(1, 2)).transpose(1, 2)
        recurrent_outputs, _ = self.recurrent(convolved)
        attention_weights = torch.softmax(self.attention(recurrent_outputs), dim=1)
        context = torch.sum(attention_weights * recurrent_outputs, dim=1)
        prediction = self.fc(self.dropout(context))
        if return_attention:
            return prediction, attention_weights.squeeze(-1)
        return prediction


if __name__ == "__main__":
    import torch
    model = LSTMAQIModel()
    dummy_input = torch.randn(config.BATCH_SIZE, config.WINDOW_SIZE, len(config.FEATURE_COLS))
    out = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {out.shape}")
