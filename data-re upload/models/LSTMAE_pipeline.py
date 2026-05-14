import copy
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from models.LSTMAE import LSTMAE

ArrayLikeColumn = Union[int, str]

@dataclass
class LSTMAEConfig:
    value_col: ArrayLikeColumn = 4
    window_size: int = 20
    latent_dim: int = 6
    train_window_end: int = 400
    batch_size: int = 64
    n_epochs: int = 500
    learning_rate: float = 0.05
    dropout_ratio: float = 0.0
    scaler_feature_range: Tuple[float, float]
    use_act: bool = False
    device: str = "cpu"
    save_path: Optional[str] = None
    print_every: int = 10

def get_device(device: Optional[str] = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")



def extract_series(df: pd.DataFrame, value_col: ArrayLikeColumn) -> np.ndarray:
    if isinstance(value_col, int):
        series = df.iloc[:,value_col].values.reshape(-1,1)
    else:
        series = df[value_col].values.reshape(-1,1)
    
    series = series.astype(np.float32)
    return series

def series_to_Xy(arr: np.ndarray, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
    X =[]
    y = []

    for i in range(len(arr) - window_size):
        X.append(arr[i:i + window_size])
        y.append(arr[i +window_size])

    return np.ndarray(X, dtype=np.float32), np.ndarray(y, dtype=np.float32)

def fit_scaler(
    series: np.ndarray,
    train_window_end: int,
    window_size: int,
    feature_range: Tuple[float, float],
) -> MinMaxScaler:

    scaler = MinMaxScaler(feature_range=feature_range)
    fit_end = min(train_window_end + window_size + 1, len(series))
    scaler.fit(series[:fit_end])
    return scaler

def prepare_windows_from_df(
    df: pd.DataFrame,
    value_col: ArrayLikeColumn = 4,
    window_size: int = 20,
    train_window_end: int = 400,
    scaler_feature_range: Tuple[float, float] = (-1.0, 1.0),
    fit_scaler_in_train: bool = True,
) -> Dict[str, object]:
    
    series = extract_series(df, value_col)

    if fit_scaler_in_train:
        scaler = fit_scaler(
            series = series,
            train_window_end = train_window_end,
            window_size = window_size,
            feature_range = scaler_feature_range,
            )
    else:
        scaler = MinMaxScaler(feature_range = scaler_feature_range)
        scaler.fit(series)

    series_scaled = scaler.transform(series).astype(np.float32)
    X_all, y_all = series_to_Xy(series_scaled, window_size)

    X_train = X_all[:train_window_end]
    y_train = y_all[:train_window_end]

    X_val = X_all[train_window_end:]
    y_val = y_all[train_window_end:]

    return {
        "series": series,
        "series_scaled": series_scaled,
        "scaler": scaler,
        "X_all": X_all,
        "y_all": y_all,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
    }

def make_ae_loaders(
        X_train: np.ndarray,
        X_val: np.ndarray,
        batch_size: int = 64,
) -> Tuple[DataLoader, DataLoader, torch.Tensor, torch.Tensor]:
    
    X_train_torch = torch.tensor(X_train, dtype = np.float32)
    X_val_torch = torch.tensor(X_val, dtype = np.float32)

    train_dataset = TensorDataset(X_train_torch, X_train_torch)
    val_dataset = TensorDataset(X_val_torch, X_val_torch)

    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)

    return train_loader, val_loader, X_train_torch, X_val_torch

def build_lstm_ae(
        input_size: int,
        latent_dim: int,
        dropout_ratio: float,
        window_size: int,
        use_act: bool,
        device: torch.device,
) -> LSTMAE:
    
    model = LSTMAE(
        input_size = input_size,
        hidden_size = latent_dim,
        dropout_ratio = dropout_ratio,
        seq_len = window_size,
        use_act = use_act,
    )
    model = model.to(device)
    return model
