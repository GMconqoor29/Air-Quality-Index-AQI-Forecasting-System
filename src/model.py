import torch.nn as nn
import src.config as config

class LSTMAQIModel(nn.Module):
    def __init__(self, input_size=len(config.FEATURE_COLS), hidden_size=config.HIDDEN_SIZE, dropout=config.DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        out, (h_n, _) = self.lstm(x)
        return self.fc(self.dropout(h_n[-1]))

if __name__ == "__main__":
    import torch
    model = LSTMAQIModel()
    dummy_input = torch.randn(config.BATCH_SIZE, config.WINDOW_SIZE, len(config.FEATURE_COLS))
    out = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {out.shape}")
