"""
preprocessing.py
----------------
Feature engineering and data partitioning helpers for CMAPSS and Naval.

Key functions
-------------
make_sequences        Build sliding-window 3-D arrays for LSTM / CNN-LSTM.
make_tabular          Flatten CMAPSS to (n_cycles × n_features) for tree / linear models.
get_test_sequences    Build one padded sequence per test engine.
get_test_tabular      Extract last cycle per test engine for tabular evaluation.
split_engines         Engine-level train / validation split (avoids data leakage).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# CMAPSS helpers
# ---------------------------------------------------------------------------

def make_sequences(
    df: pd.DataFrame,
    feature_cols: list,
    window_size: int = 30,
    step: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Create sliding-window sequences from CMAPSS time-series data.

    Each window covers ``window_size`` consecutive cycles; the label is the
    RUL at the **last** cycle of that window.  This approach allows the model
    to learn temporal degradation patterns rather than point-in-time snapshots.

    Parameters
    ----------
    df : pd.DataFrame
        CMAPSS training DataFrame with unit_id, cycle, features, and rul.
    feature_cols : list[str]
        Ordered list of column names to use as model inputs.
    window_size : int
        Number of consecutive cycles per window.
    step : int
        Stride of the sliding window (1 = maximum overlap).

    Returns
    -------
    X : np.ndarray, shape (n_sequences, window_size, n_features), float32
    y : np.ndarray, shape (n_sequences,), float32 — RUL at window end.
    """
    X_list, y_list = [], []

    for _, engine_df in df.groupby("unit_id"):
        engine_df = engine_df.sort_values("cycle")
        feats = engine_df[feature_cols].values.astype(np.float32)
        labels = engine_df["rul"].values.astype(np.float32)

        n = len(feats)
        if n < window_size:
            # Pad short engines by repeating the first row
            pad_len = window_size - n
            feats = np.vstack([np.tile(feats[0], (pad_len, 1)), feats])
            labels = np.concatenate([np.full(pad_len, labels[0]), labels])
            n = window_size

        for start in range(0, n - window_size + 1, step):
            end = start + window_size
            X_list.append(feats[start:end])
            y_list.append(labels[end - 1])

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


def make_tabular(
    df: pd.DataFrame,
    feature_cols: list,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract tabular (n_cycles × n_features) arrays from CMAPSS.

    All cycles from all engines are used as independent samples, which
    maximises training data for linear and tree-based models.

    Returns
    -------
    X : np.ndarray, shape (n_cycles, n_features), float32
    y : np.ndarray, shape (n_cycles,), float32
    """
    X = df[feature_cols].values.astype(np.float32)
    y = df["rul"].values.astype(np.float32)
    return X, y


def get_test_sequences(
    test_df: pd.DataFrame,
    feature_cols: list,
    window_size: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """Build one padded sequence per test engine (last *window_size* cycles).

    The ground-truth RUL is taken from the last observed cycle of each engine,
    matching the evaluation convention for CMAPSS test sets.

    Returns
    -------
    X : np.ndarray, shape (n_engines, window_size, n_features), float32
    y : np.ndarray, shape (n_engines,), float32
    """
    X_list, y_list = [], []

    for _, engine_df in test_df.groupby("unit_id"):
        engine_df = engine_df.sort_values("cycle")
        true_rul_rows = engine_df.dropna(subset=["rul"])
        if len(true_rul_rows) == 0:
            continue

        feats = engine_df[feature_cols].values.astype(np.float32)
        true_rul = true_rul_rows["rul"].values[-1]

        if len(feats) < window_size:
            pad_len = window_size - len(feats)
            feats = np.vstack([np.tile(feats[0], (pad_len, 1)), feats])

        X_list.append(feats[-window_size:])
        y_list.append(true_rul)

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


def get_test_tabular(
    test_df: pd.DataFrame,
    feature_cols: list,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract the last observed cycle per test engine for tabular evaluation.

    Returns
    -------
    X_last : np.ndarray, shape (n_engines, n_features), float32
    y_last : np.ndarray, shape (n_engines,), float32
    """
    valid = test_df.dropna(subset=["rul"])
    X_last = valid[feature_cols].values.astype(np.float32)
    y_last = valid["rul"].values.astype(np.float32)
    return X_last, y_last


def split_engines(
    train_df: pd.DataFrame,
    val_fraction: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Engine-level train / validation split.

    Splitting at the engine level (rather than at the cycle level) is critical
    for CMAPSS: cycles from the same engine are highly correlated, so a
    cycle-level random split would leak future degradation state into the
    validation set and badly overestimate generalisation performance.

    Parameters
    ----------
    train_df : pd.DataFrame
        Full CMAPSS training set (all engines, all cycles).
    val_fraction : float
        Fraction of engines to hold out for validation.
    random_state : int
        NumPy random seed for reproducibility.

    Returns
    -------
    df_train : pd.DataFrame  — training engines only
    df_val   : pd.DataFrame  — held-out validation engines only
    """
    rng = np.random.default_rng(random_state)
    all_engines = train_df["unit_id"].unique()
    n_val = max(1, int(len(all_engines) * val_fraction))
    val_engines = rng.choice(all_engines, size=n_val, replace=False)
    val_mask = train_df["unit_id"].isin(val_engines)
    return train_df[~val_mask].copy(), train_df[val_mask].copy()


# ---------------------------------------------------------------------------
# Window feature engineering (temporal features for tabular models)
# ---------------------------------------------------------------------------

def make_window_features(
    df: pd.DataFrame,
    feature_cols: list,
    window_size: int = 30,
    step: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract hand-crafted temporal statistics from sliding windows.

    For each window of *window_size* consecutive cycles, four statistics are
    computed per feature: last value, mean, standard deviation, and linear
    slope (least-squares fit across the window).  This converts a 3-D sequence
    array into a 2-D tabular array that any regression model can consume
    while still encoding trend information from the sensor history.

    Feature layout (4 stats × n_features):
        [feat_0_last, feat_0_mean, feat_0_std, feat_0_slope,
         feat_1_last, ..., feat_{n-1}_slope]

    Parameters
    ----------
    df : pd.DataFrame
        CMAPSS training/validation DataFrame with unit_id, cycle, features, rul.
    feature_cols : list[str]
        Ordered list of feature column names.
    window_size : int
        Number of consecutive cycles per window.
    step : int
        Sliding window stride.

    Returns
    -------
    X : np.ndarray, shape (n_windows, 4 * n_features), float32
    y : np.ndarray, shape (n_windows,), float32  — RUL at window end
    """
    n_feat = len(feature_cols)
    t = np.arange(window_size, dtype=np.float32)
    t_norm = t - t.mean()   # centred, for numerically stable slope

    X_list, y_list = [], []

    for _, engine_df in df.groupby("unit_id"):
        engine_df = engine_df.sort_values("cycle")
        feats = engine_df[feature_cols].values.astype(np.float32)
        labels = engine_df["rul"].values.astype(np.float32)

        n = len(feats)
        if n < window_size:
            pad_len = window_size - n
            feats = np.vstack([np.tile(feats[0], (pad_len, 1)), feats])
            labels = np.concatenate([np.full(pad_len, labels[0]), labels])
            n = window_size

        denom = float((t_norm ** 2).sum())   # for slope calculation

        for start in range(0, n - window_size + 1, step):
            end = start + window_size
            w = feats[start:end]             # (window_size, n_feat)

            last  = w[-1]                                       # (n_feat,)
            mean  = w.mean(axis=0)                              # (n_feat,)
            std   = w.std(axis=0)                               # (n_feat,)
            # slope: least-squares coefficient of linear trend across window
            centered = w - mean                                 # (window_size, n_feat)
            slope = (t_norm[:, None] * centered).sum(axis=0) / denom  # (n_feat,)

            # interleave: [last, mean, std, slope] per feature → (4*n_feat,)
            row = np.empty(4 * n_feat, dtype=np.float32)
            row[0::4] = last
            row[1::4] = mean
            row[2::4] = std
            row[3::4] = slope

            X_list.append(row)
            y_list.append(labels[end - 1])

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


def get_test_window_features(
    test_df: pd.DataFrame,
    feature_cols: list,
    window_size: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract window features for the last observed window of each test engine."""
    n_feat = len(feature_cols)
    t = np.arange(window_size, dtype=np.float32)
    t_norm = t - t.mean()
    denom = float((t_norm ** 2).sum())

    X_list, y_list = [], []

    for _, engine_df in test_df.groupby("unit_id"):
        engine_df = engine_df.sort_values("cycle")
        true_rul_rows = engine_df.dropna(subset=["rul"])
        if len(true_rul_rows) == 0:
            continue

        feats = engine_df[feature_cols].values.astype(np.float32)
        true_rul = true_rul_rows["rul"].values[-1]

        if len(feats) < window_size:
            pad_len = window_size - len(feats)
            feats = np.vstack([np.tile(feats[0], (pad_len, 1)), feats])

        w = feats[-window_size:]
        mean  = w.mean(axis=0)
        std   = w.std(axis=0)
        centered = w - mean
        slope = (t_norm[:, None] * centered).sum(axis=0) / denom

        row = np.empty(4 * n_feat, dtype=np.float32)
        row[0::4] = w[-1]
        row[1::4] = mean
        row[2::4] = std
        row[3::4] = slope

        X_list.append(row)
        y_list.append(true_rul)

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


# ---------------------------------------------------------------------------
# Naval helpers
# ---------------------------------------------------------------------------

def scale_naval(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Standard-scale Naval features using training statistics.

    Standard scaling (zero mean, unit variance) is used for Naval because
    the raw features span very different physical scales (rpm, bar, °C, kN·m)
    and several models (Ridge, MLP) are sensitive to feature magnitude.

    Returns
    -------
    X_train_scaled, X_test_scaled, fitted_scaler
    """
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    return X_train_sc.astype(np.float32), X_test_sc.astype(np.float32), scaler
