import torch

# Columns and targets
TIME_COL = "Local Time"
TARGET_COL = "AQI"
FEATURE_COLS = ["AQI", "PM25", "PM10", "CO", "NO2", "O3", "SO2",
                "Temperature", "Relative Humidity", "Wind Speed", "Pressure",
                "hour", "month"]

# Time series parameters
WINDOW_SIZE = 48
HORIZON = 6

# Data split ratios
TRAIN_RATIO = 0.55
VAL_RATIO = 0.25
TEST_RATIO = 0.20

# Hyperparameters
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 5e-4
HIDDEN_SIZE = 128
DROPOUT = 0.2

# Hardware
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 42

def set_seed(seed: int = 42) -> None:
    import numpy as np
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
