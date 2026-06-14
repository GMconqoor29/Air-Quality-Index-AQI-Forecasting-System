import os
import glob
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

def create_sequences(data, window_size, horizon, target_idx=0):
    X, y = [], []
    for i in range(window_size - 1, len(data) - horizon):
        X.append(data[i - window_size + 1 : i + 1])      
        y.append(data[i + horizon, target_idx])          
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def load_and_preprocess_data(data_dir='archive'):
    # Load all CSV files in the archive directory
    csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in '{data_dir}'. "
            "Please check the data directory path."
        )
    df_list = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(df_list, ignore_index=True)
    
    # Drop full duplicates
    df.drop_duplicates(inplace=True)
    
    # Replace extreme AQI values with NaN (preserves temporal contiguity)
    df.loc[df[config.TARGET_COL] > 400, config.TARGET_COL] = np.nan
    
    # Convert Local Time to datetime
    df[config.TIME_COL] = pd.to_datetime(df[config.TIME_COL])
    
    # Extract time features
    df['hour'] = df[config.TIME_COL].dt.hour
    df['month'] = df[config.TIME_COL].dt.month
    
    # Keep only target/feature columns + time
    df = df[[config.TIME_COL] + config.FEATURE_COLS].copy()
    
    # Sort and handle duplicated timestamps
    df = df.sort_values(config.TIME_COL).drop_duplicates(config.TIME_COL).reset_index(drop=True)
    
    # Interpolate missing values properly based on time
    df = df.set_index(config.TIME_COL)
    df = df.interpolate(method='time').ffill().bfill()
    df = df.reset_index()
    
    return df

def get_dataloaders(data_dir='archive'):
    df = load_and_preprocess_data(data_dir)
    
    # Chronological Split
    n = len(df)
    train_end = int(n * config.TRAIN_RATIO)
    val_end = train_end + int(n * config.VAL_RATIO)
    
    tr = df.iloc[:train_end][config.FEATURE_COLS].values
    va = df.iloc[train_end:val_end][config.FEATURE_COLS].values
    te = df.iloc[val_end:][config.FEATURE_COLS].values
    
    print(f"Data Split -> Train: {len(tr)} | Val: {len(va)} | Test: {len(te)}")
    
    # Scale features (fit on train only!)
    scaler = MinMaxScaler()
    tr_s = scaler.fit_transform(tr)
    va_s = scaler.transform(va)
    te_s = scaler.transform(te)
    
    target_idx = config.FEATURE_COLS.index(config.TARGET_COL)
    
    # Create Sequences
    X_tr, y_tr = create_sequences(tr_s, config.WINDOW_SIZE, config.HORIZON, target_idx)
    X_va, y_va = create_sequences(va_s, config.WINDOW_SIZE, config.HORIZON, target_idx)
    X_te, y_te = create_sequences(te_s, config.WINDOW_SIZE, config.HORIZON, target_idx)
    
    print(f"Sequences -> Train: {X_tr.shape} | Val: {X_va.shape} | Test: {X_te.shape}")
    
    # DataLoaders (with multi-processing to speed up CPU/GPU feed)
    pin_mem = (config.DEVICE == "cuda")
    tr_ld = DataLoader(AQIDataset(X_tr, y_tr), batch_size=config.BATCH_SIZE, shuffle=True)
    va_ld = DataLoader(AQIDataset(X_va, y_va), batch_size=config.BATCH_SIZE, shuffle=False)
    te_ld = DataLoader(AQIDataset(X_te, y_te), batch_size=config.BATCH_SIZE, shuffle=False)
    
    return tr_ld, va_ld, te_ld, scaler

if __name__ == "__main__":
    # Test block to verify
    tr_ld, va_ld, te_ld, scaler = get_dataloaders()
    for batch_x, batch_y in tr_ld:
        print(f"Batch X shape: {batch_x.shape}")
        print(f"Batch Y shape: {batch_y.shape}")
        break
