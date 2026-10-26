import os
import glob
from typing import Literal, overload

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import src.config as config

class AQIDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self): 
        return len(self.X)
        
    def __getitem__(self, idx): 
        return self.X[idx], self.y[idx]

@overload
def create_sequences(
    data: np.ndarray,
    window_size: int,
    horizon: int,
    target_idx: int = 0,
    *,
    target_values: np.ndarray | None = None,
    return_target_indices: Literal[False] = False,
) -> tuple[np.ndarray, np.ndarray]: ...


@overload
def create_sequences(
    data: np.ndarray,
    window_size: int,
    horizon: int,
    target_idx: int = 0,
    *,
    target_values: np.ndarray | None = None,
    return_target_indices: Literal[True],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...


def create_sequences(
    data,
    window_size,
    horizon,
    target_idx=0,
    *,
    target_values=None,
    return_target_indices=False,
):
    if target_values is not None and len(target_values) != len(data):
        raise ValueError("target_values must have the same length as data.")

    X, y, target_indices = [], [], []
    for i in range(window_size - 1, len(data) - horizon):
        sequence = data[i - window_size + 1 : i + 1]
        target_position = i + horizon
        target = (
            data[target_position, target_idx]
            if target_values is None
            else target_values[target_position]
        )
        if not np.isfinite(sequence).all() or not np.isfinite(target):
            continue
        X.append(sequence)
        y.append(target)
        target_indices.append(target_position)

    sequences = np.asarray(X, dtype=np.float32).reshape(
        -1, window_size, data.shape[1]
    )
    targets = np.asarray(y, dtype=np.float32)
    indices = np.asarray(target_indices, dtype=np.int64)
    if return_target_indices:
        return sequences, targets, indices
    return sequences, targets

def load_and_preprocess_data(data_dir='Dataset/archive'):
    csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in '{data_dir}'. "
            "Please check the data directory path."
        )
    df_list = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(df_list, ignore_index=True)
    
    df.drop_duplicates(inplace=True)
    df[config.TIME_COL] = pd.to_datetime(df[config.TIME_COL])
    df = df.sort_values(config.TIME_COL).drop_duplicates(config.TIME_COL)

    timestamps = df[config.TIME_COL]
    if (
        timestamps.dt.minute.ne(0)
        | timestamps.dt.second.ne(0)
        | timestamps.dt.microsecond.ne(0)
    ).any():
        raise ValueError("Local Time values must align to whole-hour boundaries.")

    df = df.set_index(config.TIME_COL).asfreq('h')
    df.loc[df[config.TARGET_COL] > 400, config.TARGET_COL] = np.nan
    timestamps = pd.DatetimeIndex(df.index)
    df['hour'] = timestamps.hour
    df['month'] = timestamps.month
    columns_to_keep = [config.TIME_COL] + [c for c in config.FEATURE_COLS]
    if config.TARGET_COL not in columns_to_keep:
        columns_to_keep.append(config.TARGET_COL)
    return df.reset_index()[columns_to_keep]


def _split_and_forward_fill_features(df):
    """Split chronologically, then fill missing inputs using past data only."""
    n = len(df)
    train_end = int(n * config.TRAIN_RATIO)
    val_end = train_end + int(n * config.VAL_RATIO)
    partitions = (
        df.iloc[:train_end],
        df.iloc[train_end:val_end],
        df.iloc[val_end:],
    )

    prepared = []
    previous_features = None
    previous_timestamp = None
    for partition in partitions:
        timestamps = partition[config.TIME_COL].to_numpy()
        target_values = partition[config.TARGET_COL].to_numpy(dtype=np.float64)
        features = partition[config.FEATURE_COLS].astype(np.float64)
        if (
            previous_features is not None
            and previous_timestamp is not None
            and not partition.empty
            and partition[config.TIME_COL].iloc[0] - previous_timestamp
            == pd.Timedelta(hours=1)
        ):
            first_index = features.index[0]
            features.loc[first_index] = features.iloc[0].fillna(previous_features)
        features = features.ffill(limit=config.MAX_FORWARD_FILL_HOURS)

        if not features.empty:
            previous_features = features.iloc[-1].copy()
            previous_timestamp = partition[config.TIME_COL].iloc[-1]
        prepared.append(
            (features.to_numpy(dtype=np.float64), target_values, timestamps)
        )
    return prepared


def _scale_target_values(values, scaler, target_idx):
    scaled = np.full(values.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(values)
    if hasattr(scaler, "target_scaler_"):
        scaled[valid] = scaler.target_scaler_.transform(values[valid].reshape(-1, 1)).flatten()
    else:
        scaled[valid] = (
            values[valid] * scaler.scale_[target_idx] + scaler.min_[target_idx]
        )
    return scaled


def _get_scaled_sequence_splits(data_dir, verbose=True):
    df = load_and_preprocess_data(data_dir)
    raw_splits = _split_and_forward_fill_features(df)
    train, val, test = raw_splits

    scaler = MinMaxScaler()
    complete_train_rows = np.isfinite(train[0]).all(axis=1)
    if not complete_train_rows.any():
        raise ValueError("The training period has no complete feature rows.")
    scaler.fit(train[0][complete_train_rows])

    if config.TARGET_COL in config.FEATURE_COLS:
        target_idx = config.FEATURE_COLS.index(config.TARGET_COL)
    else:
        target_scaler = MinMaxScaler()
        valid_targets = np.isfinite(train[1])
        if not valid_targets.any():
            raise ValueError("The training period has no valid target values.")
        target_scaler.fit(train[1][valid_targets].reshape(-1, 1))
        scaler.target_scaler_ = target_scaler
        target_idx = 0

    sequence_splits = []
    for features, targets, timestamps in raw_splits:
        scaled_features = scaler.transform(features)
        scaled_targets = _scale_target_values(targets, scaler, target_idx)
        X, y, target_indices = create_sequences(
            scaled_features,
            config.WINDOW_SIZE,
            config.HORIZON,
            target_idx,
            target_values=scaled_targets,
            return_target_indices=True,
        )
        sequence_splits.append((X, y, timestamps[target_indices]))

    if verbose:
        print(
            f"Data Split -> Train: {len(train[0])} | Val: {len(val[0])} "
            f"| Test: {len(test[0])}"
        )
        print(
            "Sequences -> "
            + " | ".join(
                f"{name}: {X.shape}"
                for name, (X, _, _) in zip(
                    ("Train", "Val", "Test"), sequence_splits
                )
            )
        )
    return sequence_splits, scaler


def load_raw_aqi_series(data_dir='Dataset/archive'):
    """Return the raw AQI series on the hourly grid (AQI > 400 masked as NaN)."""
    df = load_and_preprocess_data(data_dir)
    return df.set_index(config.TIME_COL)[config.TARGET_COL]


def get_test_target_timestamps(data_dir='Dataset/archive'):
    """Return the timestamps of the chronological test targets (no extra logging)."""
    sequence_splits, _ = _get_scaled_sequence_splits(data_dir, verbose=False)
    return sequence_splits[2][2]

def get_dataloaders(data_dir='Dataset/archive'):
    sequence_splits, scaler = _get_scaled_sequence_splits(data_dir)
    (X_tr, y_tr, _), (X_va, y_va, _), (X_te, y_te, _) = sequence_splits

    tr_ld = DataLoader(AQIDataset(X_tr, y_tr), batch_size=config.BATCH_SIZE, shuffle=True)
    va_ld = DataLoader(AQIDataset(X_va, y_va), batch_size=config.BATCH_SIZE, shuffle=False)
    te_ld = DataLoader(AQIDataset(X_te, y_te), batch_size=config.BATCH_SIZE, shuffle=False)
    
    return tr_ld, va_ld, te_ld, scaler

def get_xgboost_data(
    data_dir='Dataset/archive',
    *,
    include_target_timestamps=False,
    include_split_timestamps=False,
):
    if include_target_timestamps and include_split_timestamps:
        raise ValueError(
            "Choose either include_target_timestamps or include_split_timestamps."
        )

    sequence_splits, scaler = _get_scaled_sequence_splits(data_dir)
    (X_tr, y_tr, train_timestamps), (
        X_va,
        y_va,
        val_timestamps,
    ), (X_te, y_te, test_timestamps) = sequence_splits

    X_tr_flat = X_tr.reshape(X_tr.shape[0], -1)
    X_va_flat = X_va.reshape(X_va.shape[0], -1)
    X_te_flat = X_te.reshape(X_te.shape[0], -1)
    print(
        f"XGBoost Lag Features -> Train: {X_tr_flat.shape} | "
        f"Val: {X_va_flat.shape} | Test: {X_te_flat.shape}"
    )

    result = (X_tr_flat, y_tr, X_va_flat, y_va, X_te_flat, y_te, scaler)
    if include_split_timestamps:
        return result + (train_timestamps, val_timestamps, test_timestamps)
    if include_target_timestamps:
        return result + (test_timestamps,)
    return result

if __name__ == "__main__":
    # Test block to verify
    tr_ld, va_ld, te_ld, scaler = get_dataloaders()
    for batch_x, batch_y in tr_ld:
        print(f"Batch X shape: {batch_x.shape}")
        print(f"Batch Y shape: {batch_y.shape}")
        break
