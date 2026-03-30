"""
models.py
---------
PyTorch deep learning architectures for CMAPSS RUL prediction.

Classes
-------
LSTM_RUL      Stacked LSTM with dropout, fully-connected output.
CNN_LSTM_RUL  1-D CNN feature extractor followed by a single-layer LSTM.

Utility functions
-----------------
train_pytorch  Generic training loop with early stopping.
predict_pytorch  Batched inference returning NumPy array.
get_device       Return CUDA / MPS / CPU device string.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Device helper
# ---------------------------------------------------------------------------

def get_device() -> str:
    """Return the best available device: CUDA > MPS (Apple Silicon) > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Architecture 1: Stacked LSTM
# ---------------------------------------------------------------------------

class LSTM_RUL(nn.Module):
    """Stacked bidirectional LSTM for Remaining Useful Life regression.

    Architecture
    ------------
    Input  → LSTM (num_layers, hidden_size) → Dropout → Linear → scalar RUL

    The LSTM processes the full window and only the **last hidden state** is
    passed to the output head, so predictions are conditioned on the complete
    degradation trajectory within the window.

    Parameters
    ----------
    input_size  : number of features per time step
    hidden_size : LSTM hidden units per layer (default 64)
    num_layers  : stacked LSTM depth (default 2)
    dropout     : dropout probability between LSTM layers (default 0.2)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)                  # (batch, seq_len, hidden_size)
        out = self.dropout(out[:, -1, :])       # last time step
        return self.fc(out).squeeze(-1)         # (batch,)


# ---------------------------------------------------------------------------
# Architecture 2: CNN → LSTM hybrid
# ---------------------------------------------------------------------------

class CNN_LSTM_RUL(nn.Module):
    """1-D Convolutional feature extractor followed by a single-layer LSTM.

    Architecture
    ------------
    Input → Conv1d → ReLU → MaxPool → LSTM → Dropout → Linear → scalar RUL

    The CNN learns short-range local patterns in the sensor signals (e.g.
    step changes, inflection points) and passes compressed representations
    to the LSTM, which then models the longer-range temporal trend.

    Parameters
    ----------
    input_size   : number of features per time step
    cnn_filters  : number of Conv1d output channels (default 64)
    kernel_size  : convolution kernel width in cycles (default 3)
    hidden_size  : LSTM hidden units (default 64)
    num_layers   : LSTM stacking depth (default 1)
    dropout      : dropout probability before the output head (default 0.2)
    """

    def __init__(
        self,
        input_size: int,
        cnn_filters: int = 64,
        kernel_size: int = 3,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(input_size, cnn_filters, kernel_size, padding=kernel_size // 2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.lstm = nn.LSTM(
            input_size=cnn_filters,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        x = x.permute(0, 2, 1)                 # → (batch, input_size, seq_len) for Conv1d
        x = self.relu(self.conv(x))             # → (batch, cnn_filters, seq_len)
        x = self.pool(x)                        # → (batch, cnn_filters, seq_len/2)
        x = x.permute(0, 2, 1)                 # → (batch, seq_len/2, cnn_filters)
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out).squeeze(-1)


# ---------------------------------------------------------------------------
# Training and inference utilities
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Architecture 3: Pure 1-D CNN (CPU-friendly — no recurrent components)
# ---------------------------------------------------------------------------

class CNN_RUL(nn.Module):
    """Pure 1-D Convolutional Network for RUL prediction.

    Unlike LSTM-based models, all convolutions are computed in parallel across
    the time dimension, making this architecture 10–20× faster on CPU at
    equivalent parameter counts.

    Architecture
    ------------
    Input → Conv1d(feat→32, k=5) → BN → ReLU
          → Conv1d(32→32,   k=3) → BN → ReLU
          → Conv1d(32→16,   k=3) → BN → ReLU
          → AdaptiveAvgPool1d(1) → flatten
          → Linear(16 → 1)

    AdaptiveAvgPool aggregates across the sequence dimension, so the model
    handles sequences of any length (including padded test windows) without
    modification.

    Parameters
    ----------
    input_size : number of features per time step
    """

    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(input_size, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        x = x.permute(0, 2, 1)          # → (batch, input_size, seq_len)
        x = self.conv_block(x)           # → (batch, 16, seq_len)
        x = self.pool(x).squeeze(-1)     # → (batch, 16)
        return self.fc(x).squeeze(-1)    # → (batch,)


def train_pytorch(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    epochs: int = 80,
    batch_size: int = 256,
    lr: float = 1e-3,
    patience: int = 10,
    device: str | None = None,
) -> tuple[nn.Module, list, list]:
    """Train a PyTorch regression model with optional early stopping.

    Parameters
    ----------
    model      : un-trained PyTorch nn.Module (moved to *device* inside).
    X_train    : float32 ndarray, shape (n, seq, features) or (n, features).
    y_train    : float32 ndarray, shape (n,).
    X_val / y_val : optional validation arrays for early stopping.
    epochs     : maximum training epochs.
    batch_size : mini-batch size.
    lr         : Adam initial learning rate.
    patience   : early-stopping patience (epochs without val-loss improvement).
    device     : torch device string; auto-detected if None.

    Returns
    -------
    model        : best model (restored from early-stopping checkpoint).
    train_losses : list of per-epoch mean training MSE.
    val_losses   : list of per-epoch validation MSE (empty if no val data).
    """
    if device is None:
        device = get_device()

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=False
    )

    X_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)

    has_val = X_val is not None
    if has_val:
        X_v = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_v = torch.tensor(y_val, dtype=torch.float32).to(device)

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        # --- Training pass ---
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        train_losses.append(epoch_loss / len(X_train))

        # --- Validation pass ---
        if has_val:
            model.eval()
            with torch.no_grad():
                val_pred = model(X_v)
                val_loss = criterion(val_pred, y_v).item()
            val_losses.append(val_loss)
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break
        else:
            scheduler.step(train_losses[-1])

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, train_losses, val_losses


def predict_pytorch(
    model: nn.Module,
    X: np.ndarray,
    batch_size: int = 512,
    device: str | None = None,
) -> np.ndarray:
    """Run batched inference and return predictions as a NumPy array.

    Parameters
    ----------
    model      : trained PyTorch nn.Module.
    X          : float32 ndarray of input features / sequences.
    batch_size : inference batch size (tune for memory).
    device     : torch device string; auto-detected if None.

    Returns
    -------
    preds : np.ndarray, shape (n,)
    """
    if device is None:
        device = get_device()

    model = model.to(device).eval()
    X_t = torch.tensor(X, dtype=torch.float32)
    preds = []

    with torch.no_grad():
        for start in range(0, len(X_t), batch_size):
            xb = X_t[start : start + batch_size].to(device)
            preds.append(model(xb).cpu().numpy())

    return np.concatenate(preds)
