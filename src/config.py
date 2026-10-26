import os
import torch

# Results directory configuration
RESULTS_DIR = os.getenv("RESULTS_DIR", "results/again")
os.makedirs(RESULTS_DIR, exist_ok=True)
# Columns and targets
TIME_COL = "Local Time"
TARGET_COL = "AQI"
BASELINE_FEATURE_COLS = [
    "AQI", "PM25", "PM10", "CO", "NO2", "O3", "SO2",
    "Temperature", "Relative Humidity", "Wind Speed", "Pressure", "hour", "month",
]
_DEFAULT_FEATURE_COLS = [
    "PM25", "PM10", "Temperature", "Relative Humidity", "Pressure",
    "hour", "month",
]
_ENV_FEATURES = os.getenv("FEATURE_COLS")
if _ENV_FEATURES:
    FEATURE_COLS = [col.strip() for col in _ENV_FEATURES.split(",") if col.strip()]
else:
    FEATURE_COLS = _DEFAULT_FEATURE_COLS


# Time series parameters
WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", "48"))
HORIZON = int(os.getenv("HORIZON", "6"))
MAX_FORWARD_FILL_HOURS = 1

# Evaluation cutoff used to describe high-AQI forecast performance. This is
# an experiment threshold, not an official AQI warning threshold.
PEAK_AQI_THRESHOLD = 200.0

# Data split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TEST_RATIO = 0.2

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
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
