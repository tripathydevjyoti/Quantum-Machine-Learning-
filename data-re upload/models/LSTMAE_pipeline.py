import copy
from dataclasses import dataclass, asdict
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


class FixedRangeScaler:
    """
    Fixed physical-range scaler.

    Example for rainfall:
        0 mm   -> -1
        200 mm ->  0
        400 mm ->  1

    If clip=False, values above fixed_max are allowed to scale above 1.
    This preserves extreme rainfall information.
    """

    def __init__(
        self,
        fixed_min: float,
        fixed_max: float,
        feature_range: Tuple[float, float] = (-1.0, 1.0),
        clip: bool = False,
    ):
        if fixed_max <= fixed_min:
            raise ValueError("fixed_max must be greater than fixed_min.")

        self.fixed_min = float(fixed_min)
        self.fixed_max = float(fixed_max)
        self.feature_range = feature_range
        self.clip = clip

        self.data_min_ = np.array([self.fixed_min], dtype=np.float32)
        self.data_max_ = np.array([self.fixed_max], dtype=np.float32)

    def fit(self, x):
        return self

    def transform(self, x):
        x = np.asarray(x, dtype=np.float32)

        a, b = self.feature_range

        x_scaled = a + (x - self.fixed_min) * (b - a) / (
            self.fixed_max - self.fixed_min
        )

        if self.clip:
            x_scaled = np.clip(x_scaled, a, b)

        return x_scaled.astype(np.float32)

    def inverse_transform(self, x_scaled):
        x_scaled = np.asarray(x_scaled, dtype=np.float32)

        a, b = self.feature_range

        x = self.fixed_min + (x_scaled - a) * (
            self.fixed_max - self.fixed_min
        ) / (b - a)

        return x.astype(np.float32)


@dataclass
class LSTMAEConfig:
    value_col: ArrayLikeColumn = 4
    window_size: int = 20
    latent_dim: int = 6
    train_window_end: int = 400
    batch_size: int = 64
    n_epochs: int = 500
    learning_rate: float = 0.001
    dropout_ratio: float = 0.0

    # Scaling config
    scaler_type: str = "minmax_train"
    scaler_feature_range: Tuple[float, float] = (-1.0, 1.0)
    fixed_min: float = 0.0
    fixed_max: float = 400.0
    clip_fixed_scaler: bool = False

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
        series = df.iloc[:, value_col].values.reshape(-1, 1)
    else:
        series = df[value_col].values.reshape(-1, 1)

    series = series.astype(np.float32)
    return series


def series_to_Xy(arr: np.ndarray, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
    X = []
    y = []

    for i in range(len(arr) - window_size):
        X.append(arr[i:i + window_size])
        y.append(arr[i + window_size])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def fit_scaler(
    series: np.ndarray,
    train_window_end: int,
    window_size: int,
    feature_range: Tuple[float, float],
) -> MinMaxScaler:
    """
    Fits scaler only on the portion used for training windows and training labels.

    If train_window_end = 400 and window_size = 20,
    training windows include raw indices up to approximately 419,
    and y includes the next point.
    """
    scaler = MinMaxScaler(feature_range=feature_range)

    fit_end = min(train_window_end + window_size, len(series))

    scaler.fit(series[:fit_end])

    return scaler


def prepare_windows_from_df(
    df: pd.DataFrame,
    value_col: ArrayLikeColumn = 4,
    window_size: int = 20,
    train_window_end: int = 400,
    scaler_feature_range: Tuple[float, float] = (-1.0, 1.0),
    fit_scaler_in_train: bool = True,
    scaler_type: str = "minmax_train",
    fixed_min: float = 0.0,
    fixed_max: float = 400.0,
    clip_fixed_scaler: bool = False,
) -> Dict[str, object]:
    series = extract_series(df, value_col)

    if scaler_type == "fixed_range":
        scaler = FixedRangeScaler(
            fixed_min=fixed_min,
            fixed_max=fixed_max,
            feature_range=scaler_feature_range,
            clip=clip_fixed_scaler,
        )

    elif scaler_type == "minmax_train":
        if fit_scaler_in_train:
            scaler = fit_scaler(
                series=series,
                train_window_end=train_window_end,
                window_size=window_size,
                feature_range=scaler_feature_range,
            )
        else:
            scaler = MinMaxScaler(feature_range=scaler_feature_range)
            scaler.fit(series)

    else:
        raise ValueError(
            "Invalid scaler_type. Use 'minmax_train' or 'fixed_range'."
        )

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
    X_train_torch = torch.tensor(X_train, dtype=torch.float32)
    X_val_torch = torch.tensor(X_val, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_torch, X_train_torch)
    val_dataset = TensorDataset(X_val_torch, X_val_torch)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

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
        input_size=input_size,
        hidden_size=latent_dim,
        dropout_ratio=dropout_ratio,
        seq_len=window_size,
        use_act=use_act,
    )

    model = model.to(device)

    return model


def train_lstm_ae(
    model: LSTMAE,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int = 500,
    learning_rate: float = 0.001,
    device: Union[str, torch.device] = "cpu",
    print_every: int = 10,
) -> Tuple[LSTMAE, Dict[str, object]]:
    device = torch.device(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loss = []
    val_loss = []

    best_val_loss = float("inf")
    best_model_state = None

    for epoch in range(n_epochs):
        model.train()
        total_train_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            X_reconstruct = model(X_batch)

            loss = criterion(X_reconstruct, y_batch)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * X_batch.size(0)

        avg_train_loss = total_train_loss / len(train_loader.dataset)

        model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                X_reconstruct = model(X_batch)

                loss = criterion(X_reconstruct, y_batch)

                total_val_loss += loss.item() * X_batch.size(0)

        avg_val_loss = total_val_loss / len(val_loader.dataset)

        train_loss.append(avg_train_loss)
        val_loss.append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = copy.deepcopy(model.state_dict())

        if print_every is not None and print_every > 0:
            if (epoch + 1) % print_every == 0:
                print(
                    f"Epoch [{epoch + 1}/{n_epochs}] "
                    f"Train Loss: {avg_train_loss:.6f} "
                    f"Val Loss: {avg_val_loss:.6f}"
                )

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    history = {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "best_val_loss": best_val_loss,
    }

    return model, history


def extract_latents(
    model: LSTMAE,
    X: Union[np.ndarray, torch.Tensor],
    device: Union[str, torch.device] = "cpu",
) -> np.ndarray:
    device = torch.device(device)

    if isinstance(X, np.ndarray):
        X_tensor = torch.tensor(X, dtype=torch.float32)
    else:
        X_tensor = X.float()

    model.eval()

    with torch.no_grad():
        X_tensor = X_tensor.to(device)

        x_enc_repeated, _ = model.encoder(X_tensor)

        z = x_enc_repeated[:, 0, :]

    return z.cpu().numpy()


def get_reconstructions(
    model: LSTMAE,
    X: Union[np.ndarray, torch.Tensor],
    device: Union[str, torch.device] = "cpu",
) -> np.ndarray:
    device = torch.device(device)

    if isinstance(X, np.ndarray):
        X_tensor = torch.tensor(X, dtype=torch.float32)
    else:
        X_tensor = X.float()

    model.eval()

    with torch.no_grad():
        X_reconstruct = model(X_tensor.to(device)).cpu().numpy()

    return X_reconstruct


def reconstruct_series_from_windows(windows: np.ndarray) -> np.ndarray:
    """
    Converts overlapping windows back into one continuous time series
    by averaging overlapping values.

    This matches the windowing function:
        for i in range(len(arr) - window_size)

    Therefore, the recovered series length from X windows is:
        num_windows + window_size - 1
    """
    num_windows = windows.shape[0]
    window_size = windows.shape[1]

    series_length = num_windows + window_size - 1

    reconstructed = np.zeros(series_length, dtype=np.float32)
    counts = np.zeros(series_length, dtype=np.float32)

    for i in range(num_windows):
        reconstructed[i:i + window_size] += windows[i, :, 0]
        counts[i:i + window_size] += 1.0

    reconstructed = reconstructed / counts

    return reconstructed.reshape(-1, 1)


def save_checkpoint(
    save_path: str,
    model: LSTMAE,
    scaler: Union[MinMaxScaler, FixedRangeScaler],
    config: LSTMAEConfig,
    history: Dict[str, object],
) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "encoder_state_dict": model.encoder.state_dict(),
        "scaler": scaler,
        "config": asdict(config),
        "history": history,
    }

    torch.save(checkpoint, save_path)


def train_lstm_ae_latent_pipeline(
    df: pd.DataFrame,
    config: LSTMAEConfig,
    fit_scaler_in_train: bool = True,
) -> Dict[str, object]:
    device = get_device(config.device)

    data = prepare_windows_from_df(
        df=df,
        value_col=config.value_col,
        window_size=config.window_size,
        train_window_end=config.train_window_end,
        scaler_feature_range=config.scaler_feature_range,
        fit_scaler_in_train=fit_scaler_in_train,
        scaler_type=config.scaler_type,
        fixed_min=config.fixed_min,
        fixed_max=config.fixed_max,
        clip_fixed_scaler=config.clip_fixed_scaler,
    )

    train_loader, val_loader, X_train_torch, X_val_torch = make_ae_loaders(
        X_train=data["X_train"],
        X_val=data["X_val"],
        batch_size=config.batch_size,
    )

    input_size = data["X_train"].shape[-1]

    model = build_lstm_ae(
        input_size=input_size,
        latent_dim=config.latent_dim,
        dropout_ratio=config.dropout_ratio,
        window_size=config.window_size,
        use_act=config.use_act,
        device=device,
    )

    model, history = train_lstm_ae(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=config.n_epochs,
        learning_rate=config.learning_rate,
        device=device,
        print_every=config.print_every,
    )

    z_train = extract_latents(
        model=model,
        X=data["X_train"],
        device=device,
    )

    z_val = extract_latents(
        model=model,
        X=data["X_val"],
        device=device,
    )

    z_all = extract_latents(
        model=model,
        X=data["X_all"],
        device=device,
    )

    X_train_reconstructed = get_reconstructions(
        model=model,
        X=data["X_train"],
        device=device,
    )

    X_val_reconstructed = get_reconstructions(
        model=model,
        X=data["X_val"],
        device=device,
    )

    if config.save_path is not None:
        save_checkpoint(
            save_path=config.save_path,
            model=model,
            scaler=data["scaler"],
            config=config,
            history=history,
        )

    results = {
        **data,
        "model": model,
        "encoder": model.encoder,
        "history": history,
        "X_train_torch": X_train_torch,
        "X_val_torch": X_val_torch,
        "z_train": z_train,
        "z_val": z_val,
        "z_all": z_all,
        "X_train_reconstructed": X_train_reconstructed,
        "X_val_reconstructed": X_val_reconstructed,
        "device": device,
        "config": config,
    }

    return results


def plot_training_history(history: Dict[str, object]) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("LSTM Autoencoder Training Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_reconstructed_series(
    original_windows: np.ndarray,
    reconstructed_windows: np.ndarray,
    scaler: Union[MinMaxScaler, FixedRangeScaler],
    title: str = "Original vs Reconstructed Series",
) -> None:
    original_scaled_series = reconstruct_series_from_windows(original_windows)
    reconstructed_scaled_series = reconstruct_series_from_windows(reconstructed_windows)

    original_series = scaler.inverse_transform(original_scaled_series)
    reconstructed_series = scaler.inverse_transform(reconstructed_scaled_series)

    plt.figure(figsize=(12, 5))
    plt.plot(original_series, label="Original Series")
    plt.plot(reconstructed_series, label="Reconstructed Series")
    plt.xlabel("Original Time Step")
    plt.ylabel("Original Scale Value")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_latent_trajectories(
    z: np.ndarray,
    title: str = "Latent Dimensions Across Windows",
) -> None:
    plt.figure(figsize=(12, 6))

    for i in range(z.shape[1]):
        plt.plot(z[:, i], label=f"Latent dim {i + 1}")

    plt.xlabel("Window Index")
    plt.ylabel("Latent Value")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def load_lstm_ae_checkpoint(
    checkpoint_path: str,
    device: Union[str, torch.device] = "cpu",
) -> Dict[str, object]:
    device = torch.device(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    config = LSTMAEConfig(**checkpoint["config"])

    model = build_lstm_ae(
        input_size=1,
        latent_dim=config.latent_dim,
        dropout_ratio=config.dropout_ratio,
        window_size=config.window_size,
        use_act=config.use_act,
        device=device,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return {
        "model": model,
        "encoder": model.encoder,
        "scaler": checkpoint["scaler"],
        "config": config,
        "history": checkpoint["history"],
    }